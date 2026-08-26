"""Bengio-style MLP character language model (makemore part 2).

Fixed-width context of `block_size` characters, each embedded into `n_embd`
dimensions, concatenated and pushed through one tanh hidden layer.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nnzh.data import (VOCAB_SIZE, bpc, build_dataset, build_vocab, load_words,
                       split_words)

BLOCK_SIZE = 3
N_EMBD = 10
N_HIDDEN = 200


@dataclass
class MLP:
    C: torch.Tensor          # (VOCAB_SIZE, n_embd)   embedding 查找表
    W1: torch.Tensor         # (block_size*n_embd, n_hidden)
    b1: torch.Tensor
    W2: torch.Tensor         # (n_hidden, VOCAB_SIZE)
    b2: torch.Tensor
    block_size: int
    n_embd: int

    def parameters(self):
        return [self.C, self.W1, self.b1, self.W2, self.b2]

    def __call__(self, X):
        """X (B, block_size) int64 -> logits (B, VOCAB_SIZE)。"""
        emb = self.C[X]                                        # (B, block_size, n_embd)
        flat = emb.view(-1, self.block_size * self.n_embd)     # 拼接，零拷贝
        h = torch.tanh(flat @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


def init_mlp(block_size=BLOCK_SIZE, n_embd=N_EMBD, n_hidden=N_HIDDEN,
             seed=2147483647, w1_scale=0.2, out_scale=0.01):
    """输出层刻意缩到近零，让初始 loss 落在 ln(VOCAB_SIZE) 而不是远高于它。

    w1_scale 目前是手调的；makemore part 3 会换成有原理的 Kaiming 初始化。
    """
    g = torch.Generator().manual_seed(seed)
    fan_in = block_size * n_embd
    model = MLP(
        C=torch.randn((VOCAB_SIZE, n_embd), generator=g),
        W1=torch.randn((fan_in, n_hidden), generator=g) * w1_scale,
        b1=torch.randn(n_hidden, generator=g) * 0.01,
        W2=torch.randn((n_hidden, VOCAB_SIZE), generator=g) * out_scale,
        b2=torch.zeros(VOCAB_SIZE),
        block_size=block_size,
        n_embd=n_embd,
    )
    for p in model.parameters():
        p.requires_grad = True
    return model


def fit(model, X, Y, steps=60000, batch_size=64, lr=0.1, decay_at=40000,
        lr_after=0.01, seed=2147483647):
    """Minibatch SGD。history 记录每步的 minibatch loss（nats）。"""
    g = torch.Generator().manual_seed(seed)
    params = model.parameters()
    history = []
    for i in range(steps):
        ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)
        loss = F.cross_entropy(model(X[ix]), Y[ix])
        for p in params:
            p.grad = None
        loss.backward()
        step_lr = lr if i < decay_at else lr_after
        for p in params:
            p.data += -step_lr * p.grad
        history.append(loss.item())
    return history


@torch.no_grad()
def nll(model, X, Y):
    """整个 split 的平均交叉熵（nats）。有标签、要标量，所以用 cross_entropy。"""
    return F.cross_entropy(model(X), Y).item()


@torch.no_grad()
def sample(model, itos, n=20, seed=2147483647 + 10):
    """自回归生成。没有标签、要分布，所以显式 softmax 再 multinomial。"""
    g = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(n):
        out, context = [], [0] * model.block_size
        while True:
            probs = F.softmax(model(torch.tensor([context])), dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            if ix == 0:                       # 结束符不进名字本身
                break
            out.append(ix)
            context = context[1:] + [ix]      # 输出接回输入
        names.append("".join(itos[i] for i in out))
    return names


if __name__ == "__main__":
    words = load_words()
    stoi, itos = build_vocab(words)
    tr_words, va_words, te_words = split_words(words)   # te 全程不碰
    Xtr, Ytr = build_dataset(tr_words, stoi, BLOCK_SIZE)
    Xva, Yva = build_dataset(va_words, stoi, BLOCK_SIZE)

    model = init_mlp()
    n_params = sum(p.nelement() for p in model.parameters())
    print(f"{len(tr_words)} train / {len(va_words)} val words, "
          f"{len(Xtr)} train examples, {n_params} params")
    print(f"step 0    loss {nll(model, Xtr[:2000], Ytr[:2000]):.4f} nats "
          f"(ln {VOCAB_SIZE} = {torch.tensor(float(VOCAB_SIZE)).log():.4f})")

    fit(model, Xtr, Ytr)
    print(f"MLP       train {bpc(nll(model, Xtr, Ytr)):.4f} bpc  "
          f"val {bpc(nll(model, Xva, Yva)):.4f} bpc")
    print(sample(model, itos, n=20))
