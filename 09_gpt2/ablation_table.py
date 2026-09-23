"""把 log/run_*/ 汇总成一张消融表。

    python ablation_table.py                 # 所有 run
    python ablation_table.py --steps 954     # 只看这个预算的 run（消融和 10B 别混）

val loss 取最后一次评测；tok/s 和显存从 stdout 日志里找（<tag>.log），找不到就留空。
最后一行给出同配置不同种子之间的差距，也就是**噪声底线**——消融表里小于它的差距
都不能声称是改进。
"""
import argparse
import glob
import os
import re
import statistics


def parse_run(run_dir):
    """log.txt 是 `{step} {stream} {value}` 三列。"""
    streams = {}
    path = os.path.join(run_dir, "log.txt")
    if not os.path.exists(path):
        return None
    for line in open(path):
        parts = line.split()
        if len(parts) != 3:
            continue
        step, stream, value = int(parts[0]), parts[1], float(parts[2])
        streams.setdefault(stream, {})[step] = value
    if "val" not in streams:
        return None

    # glob 给的目录带尾斜杠，basename 会得到空串——normpath 先去掉它
    name = os.path.basename(os.path.normpath(run_dir))
    parts = name.split("_", 3) # run_YYYYmmdd_HHMMSS_tag
    tag = parts[3] if len(parts) > 3 else name
    last_step = max(streams.get("train", {0: 0}))
    val = streams["val"]
    hella = streams.get("hella", {})
    out = {
        "run": name,
        "tag": tag,
        "steps": last_step + 1,
        "val": val[max(val)],
        "val_best": min(val.values()),
        "hella": hella[max(hella)] if hella else None,
        "bsimple": streams.get("bsimple", {}).get(max(streams.get("bsimple", {0: 0}))),
        "tok_per_sec": None,
        "mem": None,
    }

    # tok/s 和显存只在 stdout 里：按 tag 找同名日志
    for candidate in (f"{tag}.log", os.path.join(run_dir, "stdout.log")):
        if candidate and os.path.exists(candidate):
            rates, mems = [], []
            for line in open(candidate):
                if m := re.search(r"tok/sec: ([\d.]+)", line):
                    rates.append(float(m.group(1)))
                if m := re.search(r"mem: ([\d.]+)GB", line):
                    mems.append(float(m.group(1)))
            if rates:
                out["tok_per_sec"] = statistics.median(rates[1:] or rates) # 第一步含编译预热
                out["mem"] = max(mems) if mems else None
            break
    return out


def main(steps_filter):
    runs = [r for r in (parse_run(d) for d in sorted(glob.glob("log/run_*/"))) if r]
    if steps_filter:
        runs = [r for r in runs if r["steps"] == steps_filter]
    if not runs:
        print("no runs found")
        return
    runs.sort(key=lambda r: r["val"])

    print(f"| {'run':<34} | steps | val loss | best  | HellaSwag | tok/s   | mem    | B_simple |")
    print(f"|{'-' * 36}|-------|----------|-------|-----------|---------|--------|----------|")
    for r in runs:
        hella = f"{r['hella']:.4f}" if r["hella"] is not None else "   -   "
        rate = f"{r['tok_per_sec']:,.0f}" if r["tok_per_sec"] else "   -   "
        mem = f"{r['mem']:.1f}GB" if r["mem"] else "  -   "
        bs = f"{r['bsimple']:,.0f}" if r["bsimple"] else "   -   "
        print(f"| {r['tag'] or r['run']:<34} | {r['steps']:>5} | {r['val']:.4f}   | "
              f"{r['val_best']:.4f} | {hella:>9} | {rate:>7} | {mem:>6} | {bs:>8} |")

    # 噪声底线：tag 以 base 开头的那些 run 之间的差距
    baselines = [r for r in runs if r["tag"].startswith("base")]
    if len(baselines) >= 2:
        spread = max(b["val"] for b in baselines) - min(b["val"] for b in baselines)
        print(f"\n噪声底线: {len(baselines)} 个基线种子, val loss 极差 {spread:.4f}")
        print(f"          小于这个数的差距不能算改进")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None)
    main(ap.parse_args().steps)
