import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mlp import BLOCK_SIZE, fit, init_mlp, nll
from nnzh.data import (ROOT, VOCAB_SIZE, bpc, build_dataset, build_vocab,
                       load_words, split_words)

FIGURES = ROOT / "figures"
BIGRAM_VAL_BPC = 3.5555          # 02_bigram 的计数法最优值，见 experiments/bpc.md

words = load_words()
stoi, itos = build_vocab(words)
tr_words, va_words, _test = split_words(words)   # test 全程不碰

# ---------- 实验 1：学习率扫描 ----------
# Karpathy 的做法：一次训练里每步换一个 lr，看 loss 在哪个量级开始发散。
# 注意这个方法的偏差：loss 天然随步数下降，所以左半段偏低。它能可靠告诉你
# 「上界在哪」，不能精确告诉你「最优是多少」。
Xtr, Ytr = build_dataset(tr_words, stoi, BLOCK_SIZE)
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

model = init_mlp()
params = model.parameters()
g = torch.Generator().manual_seed(2147483647)
losses = []
for i in range(1000):
    ix = torch.randint(0, Xtr.shape[0], (64,), generator=g)
    loss = F.cross_entropy(model(Xtr[ix]), Ytr[ix])
    for p in params:
        p.grad = None
    loss.backward()
    for p in params:
        p.data += -lrs[i].item() * p.grad
    losses.append(loss.item())

smooth = np.convolve(losses, np.ones(31) / 31, mode="valid")
best = lrs[15 + int(np.argmin(smooth))].item()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.semilogx(lrs, losses, lw=.6, alpha=.35, c="tab:blue")
ax1.semilogx(lrs[15:-15], smooth, lw=1.8, c="tab:blue", label="smoothed (31)")
ax1.axvline(best, ls="--", c="gray", lw=1, label=f"best ~ {best:.3f}")
ax1.axhline(np.log(VOCAB_SIZE), ls=":", c="crimson", lw=1, label="ln 27 (uniform)")
ax1.set_xlabel("learning rate"); ax1.set_ylabel("minibatch loss (nats)")
ax1.set_title("LR sweep: one step per candidate")
ax1.legend(fontsize=8); ax1.grid(alpha=.3)

# 右图：为什么用 10**linspace(-3,0) 而不是 linspace(0.001, 1)
naive = torch.linspace(0.001, 1, 1000)
for y, pts, lab, c in [(1, lrs, "10**linspace(-3, 0)", "tab:green"),
                       (0, naive, "linspace(0.001, 1)", "tab:red")]:
    ax2.semilogx(pts, np.full(1000, y), "|", ms=14, alpha=.25, c=c, label=lab)
    for lo, hi in [(1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1e0)]:
        n = int(((pts >= lo) & (pts < hi)).sum())
        ax2.text((lo * hi) ** .5, y + .12, f"{n}", ha="center", fontsize=9, c=c)
ax2.set_ylim(-.5, 1.6); ax2.set_yticks([])
ax2.set_xlabel("learning rate"); ax2.set_title("points per decade")
ax2.legend(loc="lower left", fontsize=8); ax2.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.savefig(FIGURES / "mlp_lr_sweep.png", dpi=150)
print(f"lr sweep: best ~ {best:.4f}")

# ---------- 实验 2：上下文长度扫描（视频里没有） ----------
rows = []
for bs in [1, 2, 3, 4, 5, 6, 8]:
    Xt, Yt = build_dataset(tr_words, stoi, block_size=bs)
    Xv, Yv = build_dataset(va_words, stoi, block_size=bs)
    m = init_mlp(block_size=bs)
    fit(m, Xt, Yt)
    n_par = sum(p.nelement() for p in m.parameters())
    tr_b, va_b = bpc(nll(m, Xt, Yt)), bpc(nll(m, Xv, Yv))
    rows.append((bs, n_par, tr_b, va_b))
    print(f"block_size={bs}  params={n_par:6d}  train {tr_b:.4f}  val {va_b:.4f} bpc")

bss, pars, trs, vas = zip(*rows)
plt.figure(figsize=(6.5, 4.2))
plt.plot(bss, trs, "o-", label="train")
plt.plot(bss, vas, "o-", label="val")
plt.axhline(BIGRAM_VAL_BPC, ls="--", c="crimson", lw=1,
            label=f"bigram counting = {BIGRAM_VAL_BPC}")
for bs, p, _, v in rows:
    plt.annotate(f"{p//1000}k", (bs, v), textcoords="offset points",
                 xytext=(0, -14), ha="center", fontsize=7, c="gray")
plt.xlabel("block_size (context length)"); plt.ylabel("bits per character")
plt.title("More context helps — until it starts overfitting")
plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(FIGURES / "mlp_block_size.png", dpi=150)

# ---------- 实验 3：二维 embedding 里学到了什么 ----------
Xt, Yt = build_dataset(tr_words, stoi, BLOCK_SIZE)
Xv, Yv = build_dataset(va_words, stoi, BLOCK_SIZE)
m2 = init_mlp(n_embd=2)
fit(m2, Xt, Yt)
print(f"n_embd=2  val {bpc(nll(m2, Xv, Yv)):.4f} bpc")

C = m2.C.detach()
vowels = set("aeiou")
plt.figure(figsize=(6, 6))
for i in range(VOCAB_SIZE):
    ch = itos[i]
    c = "crimson" if ch in vowels else ("black" if ch == "." else "tab:blue")
    plt.scatter(C[i, 0], C[i, 1], s=180, c=c, alpha=.25)
    plt.text(C[i, 0], C[i, 1], ch, ha="center", va="center", fontsize=11, c=c)
plt.title("2-D character embeddings (red = vowels, black = end token)")
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(FIGURES / "mlp_embeddings.png", dpi=150)
