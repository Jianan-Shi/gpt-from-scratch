import math

import torch

from gpt import (Block, GPTConfig, GPTLanguageModel, Head, MultiHeadAttention,
                 estimate_loss, load_data)
from nnzh.shakespeare import build_vocab, get_batch, load_text, train_val_split

TINY = GPTConfig(vocab_size=65, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)
TEXT = load_text()
STOI, ITOS = build_vocab(TEXT)


def test_split_is_by_position_not_shuffled():
    """文本是连续的，打乱切分会把同一句话的两半分到两边，val loss 虚低。"""
    train_data, val_data = train_val_split(TEXT, STOI)

    assert len(train_data) + len(val_data) == len(TEXT)
    assert train_data[:20].tolist() == [STOI[c] for c in TEXT[:20]]
    assert val_data[:20].tolist() == [STOI[c] for c in TEXT[len(train_data):len(train_data) + 20]]


def test_batch_targets_are_inputs_shifted_by_one():
    """一个 (B, T) 的样本里藏着 T 个预测任务，这是 GPT 训练效率的来源。"""
    data, _ = train_val_split(TEXT, STOI)
    x, y = get_batch(data, 8, 4, generator=torch.Generator().manual_seed(0))

    assert x.shape == y.shape == (4, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_attention_cannot_see_the_future():
    """改动 t 之后的输入，不能影响 t 及之前的输出。"""
    torch.manual_seed(0)
    head = Head(TINY, head_size=8).eval()
    x = torch.randn(1, TINY.block_size, TINY.n_embd)
    y = x.clone()
    y[:, 8:] = torch.randn(1, 8, TINY.n_embd)

    with torch.no_grad():
        assert torch.allclose(head(x)[:, :8], head(y)[:, :8], atol=1e-6)


def test_attention_weights_are_a_lower_triangular_distribution():
    """每一行 softmax 出来是概率分布，且只在 0..t 上有质量。"""
    torch.manual_seed(0)
    head = Head(TINY, head_size=8).eval()
    x = torch.randn(2, TINY.block_size, TINY.n_embd)

    with torch.no_grad():
        k, q = head.key(x), head.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(head.tril == 0, float("-inf")).softmax(dim=-1)

    assert torch.allclose(wei.sum(-1), torch.ones(2, TINY.block_size), atol=1e-5)
    assert wei.triu(diagonal=1).abs().max() == 0, "上三角必须是 0"
    assert wei[0, 0, 0] == 1.0, "第一个 token 只能看自己"


def test_scaling_keeps_the_softmax_from_saturating():
    """除以 sqrt(head_size)：点积方差随 head_size 线性增长，不缩放 softmax 会饱和。"""
    torch.manual_seed(0)
    head_size = 64
    q, k = torch.randn(1, 32, head_size), torch.randn(1, 32, head_size)

    unscaled = (q @ k.transpose(-2, -1)).softmax(-1)
    scaled = (q @ k.transpose(-2, -1) * head_size**-0.5).softmax(-1)

    # 饱和的表现：概率几乎全落在一个位置上
    assert unscaled.max(-1).values.mean() > scaled.max(-1).values.mean() + 0.2


def test_multihead_splits_the_embedding_evenly():
    mha = MultiHeadAttention(TINY)
    assert len(mha.heads) == TINY.n_head
    assert mha.heads[0].key.weight.shape[0] == TINY.n_embd // TINY.n_head

    x = torch.randn(2, TINY.block_size, TINY.n_embd)
    assert mha(x).shape == x.shape, "拼接后投影回 n_embd，才能接到残差上"


def test_block_is_residual():
    """f 的输出被清零时，Block 必须是恒等映射——残差通路没接对就会露馅。"""
    torch.manual_seed(0)
    block = Block(TINY).eval()
    for p in list(block.sa.proj.parameters()) + list(block.ffwd.net[2].parameters()):
        torch.nn.init.zeros_(p)

    x = torch.randn(2, TINY.block_size, TINY.n_embd)
    with torch.no_grad():
        assert torch.allclose(block(x), x, atol=1e-6)


def test_initial_loss_is_the_uniform_baseline():
    torch.manual_seed(0)
    model = GPTLanguageModel(TINY)
    idx = torch.randint(0, TINY.vocab_size, (4, TINY.block_size))
    targets = torch.randint(0, TINY.vocab_size, (4, TINY.block_size))

    with torch.no_grad():
        _, loss = model(idx, targets)

    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.2


def test_position_embeddings_change_the_output():
    """去掉位置信息，attention 就是个词袋——这是最容易漏掉的一块。"""
    torch.manual_seed(0)
    model = GPTLanguageModel(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (1, TINY.block_size))

    with torch.no_grad():
        before = model(idx)[0]
        torch.nn.init.zeros_(model.position_embedding_table.weight)
        after = model(idx)[0]

    assert not torch.allclose(before, after, atol=1e-4)


def test_dropout_is_off_in_eval():
    torch.manual_seed(0)
    model = GPTLanguageModel(GPTConfig(vocab_size=65, block_size=16, n_embd=32,
                                       n_head=4, n_layer=2, dropout=0.5))
    idx = torch.randint(0, 65, (2, 16))

    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(idx)[0], model(idx)[0])

    model.train()
    with torch.no_grad():
        assert not torch.allclose(model(idx)[0], model(idx)[0])


def test_generate_crops_to_block_size():
    """位置嵌入表只有 block_size 行，生成时必须裁掉更早的上下文。"""
    torch.manual_seed(0)
    model = GPTLanguageModel(TINY).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)

    out = model.generate(idx, max_new_tokens=TINY.block_size + 5)

    assert out.shape == (1, TINY.block_size + 6)


def test_forward_rejects_sequences_longer_than_block_size():
    model = GPTLanguageModel(TINY)
    try:
        model(torch.zeros((1, TINY.block_size + 1), dtype=torch.long))
    except AssertionError:
        return
    raise AssertionError("超过 block_size 必须报错")
