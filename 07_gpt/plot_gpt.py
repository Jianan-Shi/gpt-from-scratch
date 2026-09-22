"""画 07 的训练曲线：experiments/gpt_results.json -> figures/gpt_shakespeare.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE = "#1f77b4", "#e8710a" # 通过 dataviz 的配色校验

d = json.loads((ROOT / "experiments" / "gpt_results.json").read_text())
steps, train, val = zip(*d["history"])
best = min(range(len(val)), key=lambda i: val[i])

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(steps, train, color=BLUE, linewidth=2, marker="o", markersize=4, label="train")
ax.plot(steps, val, color=ORANGE, linewidth=2, marker="o", markersize=4, label="val")
ax.scatter([steps[best]], [val[best]], s=140, facecolors="none", edgecolors=ORANGE, linewidth=2)
ax.annotate(f"best val {val[best]:.4f}\n@ step {steps[best]}", (steps[best], val[best]),
            textcoords="offset points", xytext=(12, 18), fontsize=9)
ax.annotate(f"final {val[-1]:.4f}", (steps[-1], val[-1]),
            textcoords="offset points", xytext=(-46, 10), fontsize=9)
ax.set_xlabel("step")
ax.set_ylabel("loss (nats/char)")
ax.set_title("10.8M-param GPT on tiny Shakespeare — val bottoms out at step 2000")
ax.grid(alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False)
fig.tight_layout()
out = ROOT / "figures" / "gpt_shakespeare.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
