"""experiments/ablation_results.json -> figures/ablation.png

左图：5 亿 token 上每个模块单独的效果，带噪声底线。
右图：同一套现代配置在 5 亿和 100 亿 token 上的效应量——这是整组实验的主结论。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, PURPLE, GREY = "#1f77b4", "#e8710a", "#9467bd", "#8a8a8a"

data = json.loads((ROOT / "experiments" / "ablation_results.json").read_text())
runs = {r["tag"]: r for r in data["runs_500M"]}
ten = {r["tag"]: r for r in data["runs_10B"]}
noise = data["noise_floor_500M"]
base = runs["base_s1337"]["val"] # 同种子基线，消融都用 seed 1337

# 左图用的行：单模块，加合并版和最终配置
rows = [
    ("QK-norm", "qknorm", PURPLE),
    ("GQA (4 KV heads)", "gqa4", GREY),
    ("RMSNorm", "rmsnorm", GREY),
    ("SwiGLU", "swiglu", BLUE),
    ("RoPE", "rope", BLUE),
    ("all four", "combined", ORANGE),
    ("Muon (lr 0.02)", "muon_lr02", BLUE),
    ("all four + Muon", "combined_muon", ORANGE),
]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2),
                                    gridspec_kw={"width_ratios": [1.5, 1, 1]})

# ---------------------------------------------------------------- 左图
deltas = [runs[tag]["val"] - base for _, tag, _ in rows]
ys = range(len(rows))
ax1.axvspan(-noise, noise, color=GREY, alpha=0.18, zorder=0)
ax1.axvline(0, color="#444444", linewidth=1, zorder=1)
ax1.scatter(deltas, ys, s=95, color=[c for _, _, c in rows], zorder=3)
for y, d in zip(ys, deltas):
    ax1.text(d, y - 0.32, f"{d:+.3f}", ha="center", fontsize=9.5)
ax1.annotate(f"grey band = seed noise, ±{noise:.4f}", (0, -0.55), xytext=(-8, 0),
             textcoords="offset points", ha="right", fontsize=9, color="#555555")

ax1.set_yticks(list(ys), [label for label, _, _ in rows], fontsize=10)
ax1.set_ylim(-0.7, len(rows) - 0.3)
ax1.set_xlabel("change in val loss vs the GPT-2 baseline (nats, lower is better)")
ax1.set_title("One change at a time, 500M tokens", fontsize=12)
ax1.grid(alpha=0.25, axis="x", linewidth=0.6)
ax1.set_axisbelow(True)

# ---------------------------------------------------------------- 右图
small = runs["combined_muon"]["val"] - base
large = ten["modern10b"]["val_1p31M"] - ten["base10b"]["val_1p31M"]

ax2.plot([0, 1], [small, large], color=ORANGE, linewidth=2.5, marker="o", markersize=11, zorder=3)
ax2.axhline(0, color="#444444", linewidth=1)
ax2.axhspan(-noise, noise, color=GREY, alpha=0.18)
ax2.annotate(f"{small:+.3f}", (0, small), textcoords="offset points", xytext=(14, -4), fontsize=11)
ax2.annotate(f"{large:+.3f}", (1, large), textcoords="offset points", xytext=(-52, -4), fontsize=11)
ax2.annotate("94% of the apparent\ngain was the budget,\nnot the architecture",
             (0.5, (small + large) / 2), textcoords="offset points", xytext=(10, -6),
             fontsize=10, color="#444444")

ax2.set_xticks([0, 1], ["500M tokens\n(954 steps)", "10B tokens\n(19,073 steps)"], fontsize=10)
ax2.set_xlim(-0.25, 1.35)
ax2.set_ylabel("change in val loss vs baseline (nats)")
ax2.set_title("The same configuration, two budgets", fontsize=12)
ax2.grid(alpha=0.25, axis="y", linewidth=0.6)
ax2.set_axisbelow(True)

# ---------------------------------------------------------------- 右图：10B 上的拆分
if ten["arch10b"]["val_1p31M"]:
    b = ten["base10b"]["val_1p31M"]
    steps = [("GPT-2\nbaseline", b, GREY),
             ("+ architecture\n(AdamW)", ten["arch10b"]["val_1p31M"], BLUE),
             ("+ Muon", ten["modern10b"]["val_1p31M"], ORANGE)]
    xs = range(len(steps))
    ax3.plot(xs, [v for _, v, _ in steps], color="#bbbbbb", linewidth=2, zorder=1)
    ax3.scatter(xs, [v for _, v, _ in steps], s=110, color=[c for _, _, c in steps], zorder=3)
    for x, (_, v, _) in zip(xs, steps):
        ax3.annotate(f"{v:.4f}", (x, v), textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=10)
    for x in (0, 1): # 每一段的增量
        d0, d1 = steps[x][1], steps[x + 1][1]
        ax3.annotate(f"{d1 - d0:+.4f}", (x + 0.5, (d0 + d1) / 2), textcoords="offset points",
                     xytext=(6, 4), fontsize=10, color="#444444")
    ax3.set_xticks(list(xs), [label for label, _, _ in steps], fontsize=9.5)
    ax3.set_xlim(-0.4, len(steps) - 0.5)
    ax3.set_ylabel("val loss on 1.31M tokens (nats)")
    ax3.set_title("Where the 10B gain comes from\n(the two halves are nearly equal)", fontsize=12)
    ax3.grid(alpha=0.25, axis="y", linewidth=0.6)
    ax3.set_axisbelow(True)

fig.tight_layout()
out = ROOT / "figures" / "ablation.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
print(f"500M: {small:+.4f}   10B: {large:+.4f}   shrinkage {100 * (1 - large / small):.0f}%")
