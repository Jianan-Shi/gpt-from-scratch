"""第三集的实验：初始化消融、深度 × BatchNorm、update:data 比、BN 的 batch 耦合。

用法：python experiments_bn.py   （约 25 分钟，CPU）
所有训练用同一套预算和同一个 split（nnzh.data，seed=42），数字可直接进 bpc.md。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bn import (BLOCK_SIZE, N_HIDDEN, Tanh, activation_stats, fit, grad_stats,
                init_deep, nll, sample)
from nnzh.data import (ROOT, VOCAB_SIZE, bpc, build_dataset, build_vocab,
                       load_words, split_words)

FIGURES = ROOT / "figures"
STEPS = 100000
BATCH = 32
MLP_VAL_BPC = 3.0648             # 03_mlp 的基线，见 experiments/bpc.md

words = load_words()
stoi, itos = build_vocab(words)
tr_words, va_words, _test = split_words(words)          # test 全程不碰
Xtr, Ytr = build_dataset(tr_words, stoi, BLOCK_SIZE)
Xva, Yva = build_dataset(va_words, stoi, BLOCK_SIZE)


def train_and_score(label, steps=STEPS, **kw):
    m = init_deep(**kw)
    start = nll(m, Xtr[:20000], Ytr[:20000])
    hist = fit(m, Xtr, Ytr, steps=steps, batch_size=BATCH, decay_at=steps // 2)
    tr_b, va_b = bpc(nll(m, Xtr, Ytr)), bpc(nll(m, Xva, Yva))
    n_par = sum(p.nelement() for p in m.parameters())
    print(f"{label:28s} params={n_par:6d}  step0 {start:.4f} nats  "
          f"train {tr_b:.4f}  val {va_b:.4f} bpc", flush=True)
    return m, hist, dict(label=label, params=n_par, step0=start, train=tr_b, val=va_b)


# ---------- 实验 1：初始化诊断（不训练，只看 step 0） ----------
# 三种设置下，五个 tanh 层的激活分布。要看的不是"哪条曲线好看"，而是
# 深层曲线有没有从第一层的位置漂走——漂走就意味着深度本身在改变尺度。
settings = [("gain=1, no BN", dict(batchnorm=False, gain=1.0)),
            ("gain=5/3, no BN", dict(batchnorm=False)),
            ("gain=1 + BN", dict(gain=1.0)),
            ("gain=5/3 + BN", dict())]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
init_rows = []
for label, kw in settings:
    m = init_deep(**kw)
    stats = activation_stats(m, Xtr[:5000])
    grads = grad_stats(init_deep(**kw), Xtr[:5000], Ytr[:5000])
    init_rows.append((label, [s for _, _, s, _ in stats], [s for _, _, s, _ in stats][-1],
                      [g for _, _, g in grads]))
    axes[0].plot([s for _, _, s, _ in stats], "o-", label=label)
    axes[1].semilogy([g for _, _, g in grads], "o-", label=label)
    # 只给两个极端画直方图，四条会糊成一团
    if label in ("gain=1, no BN", "gain=5/3 + BN"):
        t = [l.out for l in m.layers if isinstance(l, Tanh)][-1]
        hy, hx = torch.histogram(t.detach(), density=True)
        axes[2].plot(hx[:-1], hy, label=f"layer 5 · {label}")

axes[0].set_xlabel("tanh layer"); axes[0].set_ylabel("activation std")
axes[0].set_title("activation scale vs depth"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
axes[1].set_xlabel("tanh layer"); axes[1].set_ylabel("grad std (log)")
axes[1].set_title("gradient scale vs depth"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
axes[2].set_xlabel("activation"); axes[2].set_title("last tanh layer, distribution")
axes[2].legend(fontsize=8); axes[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIGURES / "bn_init_stats.png", dpi=150)
print("--- init diagnostics ---")
for label, stds, last, gstds in init_rows:
    print(f"{label:18s} act std {['%.3f' % s for s in stds]}  "
          f"grad std {['%.1e' % g for g in gstds]}", flush=True)

# ---------- 实验 2：一次加一项修正（消融） ----------
print("\n--- ablation ---", flush=True)
ablation = []
curves = {}
for label, kw in [("naive (gain=1, loud out)", dict(batchnorm=False, gain=1.0, out_scale=1.0)),
                  ("+ squashed output layer", dict(batchnorm=False, gain=1.0)),
                  ("+ Kaiming gain 5/3", dict(batchnorm=False)),
                  ("+ BatchNorm", dict()),
                  ("BatchNorm, gain=1", dict(gain=1.0))]:
    m, hist, row = train_and_score(label, **kw)
    ablation.append(row)
    curves[label] = hist
    if label == "+ BatchNorm":
        final = m

plt.figure(figsize=(7.5, 4.5))
for label, hist in curves.items():
    smooth = np.convolve(hist, np.ones(1000) / 1000, mode="valid")
    plt.plot(np.arange(len(smooth)), smooth, lw=1.4, label=label)
plt.axhline(np.log(VOCAB_SIZE), ls=":", c="crimson", lw=1, label="ln 27")
plt.xlabel("step"); plt.ylabel("minibatch loss (nats, 1k-step mean)")
plt.title("one fix at a time"); plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIGURES / "bn_ablation.png", dpi=150)

# ---------- 实验 3：深度 × BatchNorm（视频里没有） ----------
# 问题：BN 到底买到了什么？拿深度当自变量，看没有 BN 时从第几层开始训不动。
print("\n--- depth x batchnorm ---", flush=True)
depth_rows = []
for n_layers in [2, 4, 6, 10]:
    for use_bn in [False, True]:
        tag = f"{n_layers} layers {'+BN' if use_bn else '   '}"
        _, _, row = train_and_score(tag, steps=STEPS // 2, n_layers=n_layers, batchnorm=use_bn)
        depth_rows.append((n_layers, use_bn, row["val"]))

plt.figure(figsize=(6.5, 4.2))
for use_bn, style in [(False, "o--"), (True, "o-")]:
    xs = [d for d, b, _ in depth_rows if b == use_bn]
    ys = [v for _, b, v in depth_rows if b == use_bn]
    plt.plot(xs, ys, style, label="with BatchNorm" if use_bn else "no BatchNorm")
plt.axhline(MLP_VAL_BPC, ls=":", c="gray", lw=1, label="03_mlp baseline")
plt.xlabel("layers"); plt.ylabel("val bpc"); plt.title("depth x BatchNorm (50k steps)")
plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIGURES / "bn_depth.png", dpi=150)

# ---------- 实验 4：update:data ratio ----------
# 判断学习率是否合适的最快方法，比扫 lr 便宜得多：只要看这条线落在 1e-3 附近。
print("\n--- update:data ratio ---", flush=True)
plt.figure(figsize=(7.5, 4.2))
for lr, c in [(0.001, "tab:blue"), (0.1, "tab:green"), (1.0, "tab:red")]:
    m = init_deep()
    _, ud = fit(m, Xtr, Ytr, steps=1000, batch_size=BATCH, lr=lr, track_ud=True)
    ud = np.array(ud)
    idx = [i for i, p in enumerate(m.parameters()) if p.ndim == 2]
    med = np.median(ud[:, idx], axis=1)
    plt.plot(med, lw=1.2, c=c, label=f"lr={lr}")
    print(f"lr={lr}: median log10(update:data) over last 500 steps = "
          f"{np.median(med[500:]):.2f}", flush=True)
plt.axhline(-3, ls="--", c="k", lw=1, label="1e-3 (rule of thumb)")
plt.xlabel("step"); plt.ylabel("log10 (lr*grad).std() / data.std()")
plt.title("update:data ratio, median over weight matrices"); plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIGURES / "bn_update_ratio.png", dpi=150)

# ---------- 实验 5：BN 的 batch 耦合（视频里只提了一句） ----------
# 忘记切 eval 时到底错多少？训练模式下同一批数据的 loss 会随 batch 大小抖动。
print("\n--- batch coupling ---", flush=True)
print(f"eval mode (running stats): val {bpc(nll(final, Xva, Yva)):.4f} bpc", flush=True)
g = torch.Generator().manual_seed(0)
for bs in [2, 8, 32, 256]:
    final.train()
    losses = []
    for _ in range(50):
        ix = torch.randint(0, Xva.shape[0], (bs,), generator=g)
        with torch.no_grad():
            losses.append(F.cross_entropy(final(Xva[ix]), Yva[ix]).item())
    print(f"train mode, batch={bs:4d}: val {bpc(float(np.mean(losses))):.4f} "
          f"+- {bpc(float(np.std(losses))):.4f} bpc (50 batches)", flush=True)
final.eval()

print("\nsamples:", sample(final, itos, n=20))
print("\n--- summary for bpc.md ---")
for row in ablation:
    print(f"| {row['label']} | {row['params']} | {row['train']:.4f} | {row['val']:.4f} |")
for n_layers, use_bn, val in depth_rows:
    print(f"| {n_layers} layers {'BN' if use_bn else 'no BN'} (50k) | | | {val:.4f} |")
