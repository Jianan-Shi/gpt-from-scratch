import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bigram import (bpc, build_dataset, build_vocab, fit_counting, fit_neural,
                    load_words, nll_counting, nll_neural, split_words, VOCAB_SIZE)

words = load_words()
stoi, itos = build_vocab(words)
tr_words, va_words = split_words(words)
xs_tr, ys_tr = build_dataset(tr_words, stoi)
xs_va, ys_va = build_dataset(va_words, stoi)

# ---------- 实验 1：平滑量 × 训练集大小 ----------
# 全量数据上平滑只有偏差、没有方差可对冲（验证集里 0 个 bigram 是训练集没见过的），
# 曲线单调。把训练集饿瘦，未见 bigram 出现，k→0 才会爆炸，U 形才现身。
ks = np.logspace(-3, 2, 30)
sizes = [500, 2000, len(tr_words)]

plt.figure(figsize=(6.5, 4.2))
for n in sizes:
    xs_n, ys_n = build_dataset(tr_words[:n], stoi)
    N = np.bincount((xs_n * VOCAB_SIZE + ys_n).numpy(), minlength=VOCAB_SIZE**2)
    unseen = (N.reshape(VOCAB_SIZE, VOCAB_SIZE)[xs_va.numpy(), ys_va.numpy()] == 0).mean()
    curve = [bpc(nll_counting(fit_counting(xs_n, ys_n, smoothing=float(k)), xs_va, ys_va))
             for k in ks]
    b = int(np.argmin(curve))
    line, = plt.semilogx(ks, curve, lw=1.6,
                         label=f"{n} words — best k={ks[b]:.3g}, {curve[b]:.3f} bpc")
    plt.plot(ks[b], curve[b], "o", c=line.get_color(), ms=5)
    print(f"{n:>6} train words  unseen val bigrams {unseen:6.2%}  "
          f"best k={ks[b]:.4g}  val={curve[b]:.4f} bpc")

plt.axhline(np.log2(VOCAB_SIZE), ls=":", c="crimson", lw=1, label="uniform (4.755)")
plt.xlabel("smoothing k"); plt.ylabel("validation bits per character")
plt.title("Smoothing sweep — the U-curve is a small-data phenomenon")
plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("../figures/bigram_smoothing_sweep.png", dpi=150)

# ---------- 实验 2：梯度下降收敛到解析解 ----------
W, hist = fit_neural(xs_tr, ys_tr, steps=300, lr=50.0, reg=0.0)
P0 = fit_counting(xs_tr, ys_tr, smoothing=0.0)
floor = bpc(nll_counting(P0, xs_tr, ys_tr))

plt.figure(figsize=(6, 4))
plt.plot([bpc(h) for h in hist], label="gradient descent")
plt.axhline(floor, ls="--", c="crimson", lw=1, label=f"counting MLE = {floor:.4f}")
plt.xlabel("step"); plt.ylabel("train bits per character")
plt.title("Two methods, one optimum")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("../figures/bigram_convergence.png", dpi=150)
print(f"final gap = {bpc(nll_neural(W, xs_tr, ys_tr)) - floor:.5f} bpc")

# ---------- unigram 基线 ----------
flat = xs_tr * VOCAB_SIZE + ys_tr
N = np.bincount(flat.numpy(), minlength=VOCAB_SIZE**2).reshape(VOCAB_SIZE, VOCAB_SIZE)
q = (N.sum(0) + 1) / (N.sum() + VOCAB_SIZE)
uni_tr = -np.log(q[ys_tr.numpy()]).mean()
uni_va = -np.log(q[ys_va.numpy()]).mean()
print(f"unigram   train {bpc(uni_tr):.4f}  val {bpc(uni_va):.4f} bpc")
