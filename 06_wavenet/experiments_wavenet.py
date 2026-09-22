"""06 的三个实验：树状 vs 拍平、上下文长度、以及 BatchNorm 那个 dim 写错的代价。

    python experiments_wavenet.py          # 跑全部，约 20 分钟（CPU）
    python experiments_wavenet.py tree     # 只跑一个

结果写进 experiments/wavenet_results.json，图写进 figures/。
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from nnzh.data import bpc
from wavenet import (BLOCK_SIZE, BatchNorm1d, fit, init_wavenet, load_splits,
                     nll)

ROOT = Path(__file__).resolve().parent.parent
STEPS = 200_000
# 通过 scripts/validate_palette.js 的三色：蓝 / 橙 / 紫
BLUE, ORANGE, PURPLE = "#1f77b4", "#e8710a", "#9467bd"


class BuggyBatchNorm1d(BatchNorm1d):
    """故意写错的版本：3 维输入也只在 dim=0 上统计。

    每个 (T, C) 位置各自攒一套 running 统计量。训练时用的是 batch 统计量，所以
    train loss 完全看不出问题；只有 eval() 切到 running buffer 之后才开始用错。
    """

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True, unbiased=True)
        else:
            xmean, xvar = self.running_mean, self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze()
        return self.out


def match_params(target, block_size, **kw):
    """给拍平结构挑一个 n_hidden，让参数量和树状结构对齐。

    不对齐参数量的比较没有意义：树状结构多了两层 Linear，本来就更大。
    """
    best, best_gap = None, float("inf")
    for n_hidden in range(32, 768, 2):
        model = init_wavenet(block_size=block_size, fan_in=block_size, n_hidden=n_hidden, **kw)
        n = sum(p.nelement() for p in model.parameters())
        if abs(n - target) < best_gap:
            best, best_gap = (n_hidden, n), abs(n - target)
    return best


def run(name, model, splits, steps=STEPS, seed=None):
    (Xtr, Ytr), (Xva, Yva) = splits
    n_params = sum(p.nelement() for p in model.parameters())
    print(f"\n=== {name} ({n_params:,} params) ===", flush=True)
    history = fit(model, Xtr, Ytr, steps=steps, log_every=50_000,
                  **({} if seed is None else {"seed": seed}))
    tr, va = nll(model, Xtr, Ytr, batch=10_000), nll(model, Xva, Yva, batch=10_000)
    print(f"{name}: train {bpc(tr):.4f} bpc | val {bpc(va):.4f} bpc", flush=True)
    return {"name": name, "params": n_params, "train_bpc": bpc(tr), "val_bpc": bpc(va),
            "train_nats": tr, "val_nats": va, "history": history[::100]}


def main(which=None):
    (Xtr, Ytr), (Xva, Yva), _, _ = load_splits(BLOCK_SIZE)
    splits8 = ((Xtr, Ytr), (Xva, Yva))
    results = {}

    tree = init_wavenet()
    target = sum(p.nelement() for p in tree.parameters())

    if which in (None, "tree"):
        results["tree"] = run("tree, context 8", tree, splits8)

    if which in (None, "flat"):
        n_hidden, n = match_params(target, BLOCK_SIZE)
        print(f"拍平结构用 n_hidden={n_hidden} 对齐参数量：{n:,} vs {target:,}")
        flat = init_wavenet(block_size=BLOCK_SIZE, fan_in=BLOCK_SIZE, n_hidden=n_hidden)
        results["flat"] = run("flat, context 8 (params matched)", flat, splits8)

    if which in (None, "context3"):
        (X3, Y3), (Xv3, Yv3), _, _ = load_splits(4)
        small = init_wavenet(block_size=4, fan_in=2)
        results["context4"] = run("tree, context 4", small, ((X3, Y3), (Xv3, Yv3)))

    if which in (None, "buggy"):
        buggy = init_wavenet()
        buggy.layers = [BuggyBatchNorm1d(l.gamma.shape[0]) if isinstance(l, BatchNorm1d) else l
                        for l in buggy.layers]
        results["buggy_bn"] = run("tree, BatchNorm over dim=0 (the bug)", buggy, splits8)

    if which == "seeds":
        # BatchNorm 那个 dim 写错到底值多少 bpc？单次对比差 0.009，不够下结论，
        # 换两个种子各跑一遍——04 章也是这么区分"真差距"和"种子噪声"的。
        for seed in (1, 2):
            correct = init_wavenet(seed=seed)
            results[f"tree_seed{seed}"] = run(f"tree, seed {seed}", correct, splits8, seed=seed)
            buggy = init_wavenet(seed=seed)
            buggy.layers = [BuggyBatchNorm1d(l.gamma.shape[0]) if isinstance(l, BatchNorm1d) else l
                            for l in buggy.layers]
            results[f"buggy_seed{seed}"] = run(f"buggy BN, seed {seed}", buggy, splits8, seed=seed)

    out = ROOT / "experiments" / "wavenet_results.json"
    merged = json.loads(out.read_text()) if out.exists() else {}
    merged.update(results)
    out.write_text(json.dumps(merged, indent=1))
    print(f"\nwrote {out}")
    if len(merged) >= 2:
        plot(merged)


def smooth(xs, k=50):
    t = torch.tensor(xs).view(-1, k).mean(1)
    return t.tolist()


def plot(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = {"tree": BLUE, "flat": ORANGE, "buggy_bn": PURPLE, "context4": "#888888"}
    for key in ("tree", "flat"):
        if key in results:
            ys = smooth(results[key]["history"])
            xs = [i * 100 * 50 for i in range(len(ys))]
            ax1.plot(xs, ys, color=colors[key], linewidth=2, label=results[key]["name"])
    ax1.set_xlabel("step")
    ax1.set_ylabel("train loss (nats)")
    ax1.set_title("Same parameter budget, different wiring")
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False)

    # 差距只有 0.03 bpc，柱状图从 0 起会把差别压没——用点图配合放大的坐标轴
    keys = [k for k in ("tree", "flat", "context4", "buggy_bn") if k in results]
    vals = [results[k]["val_bpc"] for k in keys]
    labels = [results[k]["name"].replace(", ", "\n") for k in keys]
    ys = range(len(keys))
    ax2.scatter(vals, ys, s=90, color=[colors[k] for k in keys], zorder=3)
    for y, v in zip(ys, vals):
        ax2.text(v, y - 0.28, f"{v:.4f}", ha="center", fontsize=10)

    # 同一个变体换种子重跑的结果，用来判断上面的差距是不是噪声
    seeds = {"tree": [results[k]["val_bpc"] for k in results if k.startswith("tree_seed")],
             "buggy_bn": [results[k]["val_bpc"] for k in results if k.startswith("buggy_seed")]}
    for y, k in zip(ys, keys):
        for v in seeds.get(k, []):
            ax2.scatter([v], [y], s=45, facecolors="none", edgecolors=colors[k], zorder=2)

    ax2.set_yticks(list(ys), labels, fontsize=9)
    ax2.set_ylim(len(keys) - 0.5, -0.5)
    lo, hi = min(vals + sum(seeds.values(), [])), max(vals + sum(seeds.values(), []))
    pad = max(hi - lo, 0.01) * 0.35
    ax2.set_xlim(lo - pad, hi + pad)
    ax2.set_xlabel("val bpc (lower is better) — note the zoomed axis")
    ax2.set_title("Validation, bits per character\n(hollow = same variant, other seeds)", fontsize=11)
    ax2.grid(alpha=0.25, axis="x", linewidth=0.6)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    path = ROOT / "figures" / "wavenet_tree_vs_flat.png"
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
