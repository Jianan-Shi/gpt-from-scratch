"""Bigram character-level language model, fit two ways: counting and gradient descent."""
import torch
import torch.nn.functional as F

from nnzh.data import VOCAB_SIZE, bpc, build_vocab, load_words, split_words


def build_dataset(words, stoi):
    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs), torch.tensor(ys)


def fit_counting(xs, ys, smoothing=1.0):
    """把 (i, j) 压成一维索引后用 bincount 统计，比 Python 循环快两个数量级。"""
    flat = xs * VOCAB_SIZE + ys
    N = torch.bincount(flat, minlength=VOCAB_SIZE**2).reshape(VOCAB_SIZE, VOCAB_SIZE).float()
    P = N + smoothing
    P /= P.sum(1, keepdim=True)
    return P


def nll_counting(P, xs, ys):
    return -P[xs, ys].log().mean().item()


def fit_neural(xs, ys, steps=300, lr=50.0, reg=0.01, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn((VOCAB_SIZE, VOCAB_SIZE), generator=g, requires_grad=True)
    history = []
    for _ in range(steps):
        logits = W[xs]                       # 等价于 one_hot(xs) @ W，但不用构造 27 维的 0
        loss = F.cross_entropy(logits, ys)   # = log_softmax + nll，数值稳定版
        history.append(loss.item())          # 只记数据项，正则项不进 bpc 表
        W.grad = None
        (loss + reg * (W**2).mean()).backward()
        W.data += -lr * W.grad
    return W, history


def nll_neural(W, xs, ys):
    with torch.no_grad():
        return F.cross_entropy(W[xs], ys).item()


def sample(P, itos, n=5, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(n):
        out, ix = [], 0
        while True:
            ix = torch.multinomial(P[ix], 1, replacement=True, generator=g).item()
            if ix == 0:
                break
            out.append(itos[ix])
        names.append("".join(out))
    return names


if __name__ == "__main__":
    words = load_words()
    stoi, itos = build_vocab(words)
    # te_words 全程不碰，只在最后一章评估一次。
    tr_words, va_words, te_words = split_words(words)
    xs_tr, ys_tr = build_dataset(tr_words, stoi)
    xs_va, ys_va = build_dataset(va_words, stoi)
    print(f"{len(tr_words)} train / {len(va_words)} val / {len(te_words)} test words, "
          f"{len(xs_tr)} train bigrams")

    P = fit_counting(xs_tr, ys_tr, smoothing=1.0)
    print(f"counting  train {bpc(nll_counting(P, xs_tr, ys_tr)):.4f} bpc  "
          f"val {bpc(nll_counting(P, xs_va, ys_va)):.4f} bpc")

    W, _ = fit_neural(xs_tr, ys_tr)
    print(f"neural    train {bpc(nll_neural(W, xs_tr, ys_tr)):.4f} bpc  "
          f"val {bpc(nll_neural(W, xs_va, ys_va)):.4f} bpc")

    print(sample(P, itos))
