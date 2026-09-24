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
        "params": streams.get("params", {}).get(0),
        "kvcache": streams.get("kvcache_kb", {}).get(0),
    }

    # 优先用 log.txt 里的（自包含）；旧的 run 没有这两个流，再退回 stdout 文件
    rates = sorted(streams.get("toks", {}).items())[1:] # 第一步含编译预热
    if rates:
        out["tok_per_sec"] = statistics.median(v for _, v in rates)
        out["mem"] = max(streams.get("mem", {0: 0}).values()) or None
    else:
        log = f"{tag}.log"
        if os.path.exists(log):
            r, m = [], []
            for line in open(log):
                if hit := re.search(r"tok/sec: ([\d.]+)", line):
                    r.append(float(hit.group(1)))
                if hit := re.search(r"mem: ([\d.]+)GB", line):
                    m.append(float(hit.group(1)))
            if r:
                out["tok_per_sec"], out["mem"] = statistics.median(r[1:] or r), (max(m) if m else None)
                out["stale"] = True # 来自按 tag 拼的文件名，同 tag 跑过两次就可能张冠李戴
    return out


def main(steps_filter):
    runs = [r for r in (parse_run(d) for d in sorted(glob.glob("log/run_*/"))) if r]
    if steps_filter:
        runs = [r for r in runs if r["steps"] == steps_filter]
    if not runs:
        print("no runs found")
        return
    runs.sort(key=lambda r: r["val"])
    tags = [r["tag"] for r in runs] # 用于检测同 tag 的重复运行

    print(f"| {'run':<24} | steps | val loss | HellaSwag | tok/s   | mem    | params  | KV/tok |")
    print(f"|{'-' * 26}|-------|----------|-----------|---------|--------|---------|--------|")
    for r in runs:
        hella = f"{r['hella']:.4f}" if r["hella"] is not None else "   -   "
        rate = f"{r['tok_per_sec']:,.0f}" if r["tok_per_sec"] else "   -   "
        mem = f"{r['mem']:.1f}GB" if r["mem"] else "  -   "
        bs = f"{r['bsimple']:,.0f}" if r["bsimple"] else "   -   "
        params = f"{r['params'] / 1e6:.1f}M" if r["params"] else "   -   "
        kv = f"{r['kvcache']:.0f}KB" if r["kvcache"] else "  -   "
        label = r["tag"] if tags.count(r["tag"]) == 1 else r["run"][4:] # 撞名就显示时间戳
        rate = rate + "?" if r.get("stale") and r["tok_per_sec"] else rate
        print(f"| {label:<24} | {r['steps']:>5} | {r['val']:.4f}   | "
              f"{hella:>9} | {rate:>7} | {mem:>6} | {params:>7} | {kv:>6} |")

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
