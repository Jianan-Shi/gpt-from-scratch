# parse and visualize the logfile (adapted from build-nanogpt/play.ipynb)
# usage: python plot_log.py [log/run_xxx]  -> writes <run>/curves.png; safe to run mid-training
# with no argument it plots the newest log/run_* directory
import glob, os, sys
import matplotlib
matplotlib.use("Agg") # write a png, no display needed under WSL
import matplotlib.pyplot as plt

sz = "124M"
loss_baseline = 3.2924 # OpenAI GPT-2 (124M) checkpoint val loss on FineWeb-Edu
hella2_baseline = 0.294463 # HellaSwag for GPT-2 (124M)
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

plt.figure(figsize=(16, 6))

# Panel 1: losses: both train and val
plt.subplot(121)
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
plt.subplot(122)
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

plt.tight_layout()
out = os.path.join(run_dir, "curves.png")
plt.savefig(out, dpi=100)
print(f"saved {out}")
