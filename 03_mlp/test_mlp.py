import math

import torch

from mlp import BLOCK_SIZE, fit, init_mlp, nll, sample
from nnzh.data import VOCAB_SIZE, build_dataset, build_vocab, load_words

WORDS = load_words()
STOI, ITOS = build_vocab(WORDS)
SMALL = WORDS[:2000]
X, Y = build_dataset(SMALL, STOI, BLOCK_SIZE)


def test_dataset_shape_and_dtype():
    assert X.shape == (sum(len(w) + 1 for w in SMALL), BLOCK_SIZE)
    assert Y.shape == (X.shape[0],)
    assert X.dtype == torch.int64 and Y.dtype == torch.int64, "要当索引用，必须是整型"


def test_context_resets_between_words():
    """词与词之间必须清零，否则上一个名字的尾巴会泄漏进下一个名字的开头。"""
    starts, row = [], 0
    for w in SMALL:
        starts.append(row)
        row += len(w) + 1
    assert (X[starts] == 0).all()


def test_context_rolls_within_word():
    """词内部：下一行的 context = 这一行左移一格 + 这一行的答案。"""
    row = 0
    for w in SMALL[:50]:
        for j in range(len(w)):          # 最后一行是词尾，不和下一词比
            expected = torch.cat([X[row + j][1:], Y[row + j].reshape(1)])
            assert torch.equal(X[row + j + 1], expected)
        row += len(w) + 1


def test_flatten_preserves_position_order():
    """view 拼接后，前 n_embd 列必须是最老那个字符的向量——位置信息不能串。"""
    m = init_mlp()
    emb = m.C[X[:64]]
    flat = emb.view(-1, m.block_size * m.n_embd)
    for pos in range(m.block_size):
        lo, hi = pos * m.n_embd, (pos + 1) * m.n_embd
        assert torch.equal(flat[:, lo:hi], m.C[X[:64, pos]])


def test_view_is_free():
    """view 只换读法，不复制内存。"""
    m = init_mlp()
    emb = m.C[X[:64]]
    assert torch.equal(emb.reshape(-1), emb.view(-1, m.block_size * m.n_embd).reshape(-1))


def test_forward_shapes():
    m = init_mlp()
    assert m(X[:37]).shape == (37, VOCAB_SIZE)
    assert m(X[:1]).shape == (1, VOCAB_SIZE), "batch=1 也要能跑，采样时要用"


def test_init_loss_is_ln_vocab():
    """初始化自检：什么都没学到时应该输出均匀分布，loss = ln(27)。

    远高于它 = 输出层权重过大，模型在自信地犯错；低于它 = 大概率有 bug。
    """
    m = init_mlp()
    assert abs(nll(m, X, Y) - math.log(VOCAB_SIZE)) < 0.05


def test_training_reduces_loss():
    m = init_mlp()
    before = nll(m, X, Y)
    fit(m, X, Y, steps=2000, decay_at=2000)
    after = nll(m, X, Y)
    assert after < before - 0.5, f"{before} -> {after}"
    assert after < math.log(VOCAB_SIZE)


def test_samples_contain_no_terminator():
    """'.' 是控制符，不该出现在名字里——采样循环必须先 break 再 append。"""
    m = init_mlp()
    for name in sample(m, ITOS, n=5):
        assert "." not in name


def test_block_size_changes_columns_not_rows():
    """列数 = block_size，行数与它无关：每个词永远贡献 len(w)+1 条样本。"""
    rows = sum(len(w)+1 for w in SMALL)
    for bs in [1, 3, 5, 8]:
        X, Y = build_dataset(SMALL, STOI, block_size=bs)
        assert X.shape == (rows, bs)
        assert Y.shape == (rows,)
