"""Decoder-only Transformer on tiny Shakespeare (Let's build GPT).

自注意力的推导分三步，最后一步才是这里实现的版本：

1. 位置 t 只看 0..t，最朴素的做法是对前面的向量求平均（两层 for 循环）；
2. 求平均可以写成下三角矩阵乘法，一次矩阵乘搞定所有位置；
3. 权重不该是均匀的——用 Q/K 算出"谁该看谁"，softmax 掩码代替手工的 tril 归一化。

于是 attention 是一次**数据相关的**加权聚合：token 通过 query 说"我要找什么"，
通过 key 说"我有什么"，value 才是真正被搬运的内容。除以 sqrt(head_size) 是为了
让点积的方差回到 1，否则 softmax 会在初始化时就饱和成近似 one-hot。

和 06 之前几章的区别：上下文不再是固定拍平的窗口，位置之间怎么组合由数据决定，
长度也不再写死在结构里（只受 block_size 限制）。

语料换成 tiny Shakespeare，词表 65，按位置切分 90/10，所以 bpc 与 01-06 不可比，
uniform 基线从 log2(27)=4.755 变成 log2(65)=6.022。见 nnzh/shakespeare.py。
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from nnzh.shakespeare import (build_vocab, decode, get_batch, load_text,
                              train_val_split)

SEED = 1337


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256   # 上下文长度
    n_embd: int = 384
    n_head: int = 6         # head_size = 384 / 6 = 64
    n_layer: int = 6
    dropout: float = 0.2


class Head(nn.Module):
    """单头因果自注意力。"""

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        # buffer 而不是参数：掩码是常量，不参与训练，但要跟着 .to(device) 走
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        # 缩放：k 和 q 的每个分量方差为 1 时，点积方差是 head_size，
        # 不除回去 softmax 会在初始化时就退化成 one-hot，梯度消失
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # 因果掩码
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    """多个头并行，各自学不同的关注模式，拼起来再投影回 n_embd。"""

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        assert head_size * config.n_head == config.n_embd
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """逐位置的两层 MLP。attention 负责"交流"，这里负责"各自消化"。"""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """通信 + 计算，都挂在残差上，LayerNorm 前置（pre-norm）。

    x = x + f(ln(x)) 而不是 ln(x + f(x))：残差是一条干净的加法通路，
    梯度可以原样流回去，深了才训得动。
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.sa = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, config=GPTConfig()):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"{T} > block_size {self.config.block_size}"
        tok = self.token_embedding_table(idx)                                    # (B, T, C)
        pos = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = self.ln_f(self.blocks(tok + pos))
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, generator=None):
        for _ in range(max_new_tokens):
            # 只能喂最后 block_size 个 token——位置嵌入表就这么长
            logits, _ = self(idx[:, -self.config.block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1, generator=generator)], dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size=64, iters=200, device="cpu", generator=None):
    """随机采样若干 batch 求平均。单个 batch 的 loss 噪声太大，看不出趋势。"""
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = get_batch(data, block_size, batch_size, generator=generator, device=device)
        losses[k] = model(x, y)[1].item()
    model.train()
    return losses.mean().item()


def train(model, train_data, val_data, max_iters=5000, batch_size=64, lr=3e-4,
          eval_interval=500, device="cpu", seed=SEED, verbose=True):
    g = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    block_size = model.config.block_size
    history = []

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            tr = estimate_loss(model, train_data, block_size, device=device, generator=g)
            va = estimate_loss(model, val_data, block_size, device=device, generator=g)
            history.append((it, tr, va))
            if verbose:
                print(f"step {it:5d}: train {tr:.4f} val {va:.4f}", flush=True)

        x, y = get_batch(train_data, block_size, batch_size, generator=g, device=device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return history


def load_data(device="cpu"):
    text = load_text()
    stoi, itos = build_vocab(text)
    train_data, val_data = train_val_split(text, stoi)
    return train_data.to(device), val_data.to(device), stoi, itos


if __name__ == "__main__":
    import json
    import time
    from pathlib import Path

    from nnzh.data import bpc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    train_data, val_data, stoi, itos = load_data(device)
    print(f"device {device} | vocab {len(stoi)} | train {len(train_data):,} val {len(val_data):,}")

    model = GPTLanguageModel(GPTConfig(vocab_size=len(stoi))).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    t0 = time.time()
    history = train(model, train_data, val_data, device=device)
    it, tr, va = history[-1]
    print(f"final: train {tr:.4f} nats ({bpc(tr):.4f} bpc) | val {va:.4f} nats ({bpc(va):.4f} bpc)")
    print(f"{time.time() - t0:.0f}s on {device}")

    g = torch.Generator(device=device).manual_seed(SEED)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = decode(model.generate(context, 500, generator=g)[0].tolist(), itos)
    print("\n" + sample)

    out = Path(__file__).resolve().parent.parent / "experiments" / "gpt_results.json"
    out.write_text(json.dumps({
        "params": sum(p.numel() for p in model.parameters()),
        "train_nats": tr, "val_nats": va,
        "train_bpc": bpc(tr), "val_bpc": bpc(va),
        "history": history, "seconds": time.time() - t0, "device": device,
        "sample": sample,
    }, indent=1))
    torch.save(model.state_dict(), Path(__file__).resolve().parent / "gpt_shakespeare.pt")
