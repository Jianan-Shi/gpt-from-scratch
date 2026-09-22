# parse and visualize the logfile (adapted from build-nanogpt/play.ipynb)
# usage: python plot_log.py [log/run_xxx]  -> writes <run>/curves.png; safe to run mid-training
# with no argument it plots the newest log/run_* directory
import glob, os, sys
import matplotlib
matplotlib.use("Agg") # write a png, no display needed under WSL
import matplotlib.pyplot as plt

sz = "124M"
loss_baseline = 3.2799 # OpenAI GPT-2 (124M), measured here by eval_gpt2_baseline.py (Karpathy quotes 3.2924)
hella2_baseline = 0.2976 # HellaSwag for GPT-2 (124M), measured here (Karpathy quotes 0.294463)
hella3_baseline = 0.337 # HellaSwag for GPT-3 (124M)

run_dir = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob("log/run_*"), key=os.path.getmtime)
print(f"plotting {run_dir}")

# parse the individual lines, group by stream (train,val,hella)
streams = {}
with open(os.path.join(run_dir, "log.txt"), "r") as f:
    for line in f:
        parts = line.split()
        if len(parts) != 3: # the last line may be half-written while training is running
            continue
        step, stream, val = parts
        streams.setdefault(stream, {})[int(step)] = float(val)

# convert each stream from {step: val} to (steps[], vals[]) so it's easier for plotting
streams_xy = {k: list(zip(*sorted(v.items()))) for k, v in streams.items()}

has_noise = "bsimple" in streams_xy
plt.figure(figsize=(21, 6) if has_noise else (16, 6))
cols = 3 if has_noise else 2

# Panel 1: losses: both train and val
plt.subplot(1, cols, 1)
for name in ("train", "val"):
    if name in streams_xy:
        xs, ys = streams_xy[name]
        plt.plot(xs, ys, marker="o" if name == "val" else None, label=f"nanogpt ({sz}) {name} loss")
        print(f"Min {name} loss: {min(ys):.4f} (step {xs[ys.index(min(ys))]})")
plt.axhline(y=loss_baseline, color="r", linestyle="--", label=f"OpenAI GPT-2 ({sz}) checkpoint val loss")
plt.xlabel("steps")
plt.ylabel("loss")
plt.yscale("log")
# no ylim(top=4.0) like the original: a 2h run stays well above 4, it would clip the whole curve
plt.legend()
plt.title("Loss")

# Panel 2: HellaSwag eval
plt.subplot(1, cols, 2)
if "hella" in streams_xy:
    xs, ys = streams_xy["hella"]
    plt.plot(xs, ys, marker="o", label=f"nanogpt ({sz})")
    print(f"Max HellaSwag: {max(ys):.4f}")
plt.axhline(y=hella2_baseline, color="r", linestyle="--", label=f"OpenAI GPT-2 ({sz}) checkpoint")
plt.axhline(y=hella3_baseline, color="g", linestyle="--", label=f"OpenAI GPT-3 ({sz}) checkpoint")
plt.axhline(y=0.25, color="gray", linestyle=":", label="random guess (1 in 4)")
plt.xlabel("steps")
plt.ylabel("accuracy")
plt.legend()
plt.title("HellaSwag eval")

# Panel 3: gradient noise scale, if the run measured it
if has_noise:
    plt.subplot(1, cols, 3)
    xs, ys = streams_xy["bsimple"]
    plt.plot(xs, ys, color="#9467bd", linewidth=2, label="B_simple (EMA)")
    # the two batch sizes this project actually compared
    plt.axhline(y=65536, color="#1f77b4", linestyle="--", label="2**16 batch (the 9h run)")
    plt.axhline(y=524288, color="#e8710a", linestyle="--", label="2**19 batch (the lecture)")
    plt.xlabel("steps")
    plt.ylabel("tokens")
    plt.yscale("log")
    plt.title("Gradient noise scale\n(a batch above the curve is buying little)")
    plt.legend()
    print(f"B_simple: {ys[0]:,.0f} -> {ys[-1]:,.0f} tokens")

plt.tight_layout()
out = os.path.join(run_dir, "curves.png")
plt.savefig(out, dpi=100)
print(f"saved {out}")
