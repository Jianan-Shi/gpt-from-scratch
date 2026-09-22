"""08 的实验：压缩率怎么随词表增长、正则切分值多少、以及 tokenization 造成的怪现象。

    python experiments_bpe.py

结果写进 experiments/bpe_results.json，图写进 figures/bpe_compression.png。
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tiktoken

from bpe import BasicTokenizer, RegexTokenizer, compression_ratio
from nnzh.shakespeare import load_text

ROOT = Path(__file__).resolve().parent.parent
TEXT = load_text()[:100_000]
VOCAB_SIZES = [300, 512, 768, 1024, 1536]
BLUE, ORANGE, PURPLE = "#1f77b4", "#e8710a", "#9467bd"

SAMPLES = {
    "English": "The quick brown fox jumps over the lazy dog. Machine learning models "
               "are trained on large collections of text gathered from the internet.",
    "Chinese": "机器学习模型是在从互联网上收集的大量文本上训练出来的。分词器决定了"
            "同样一段话要花多少个 token，这直接影响上下文长度和推理成本。",
    "Python": "def train(model, data, lr=3e-4):\n    for x, y in data:\n"
              "        loss = model(x, y)\n        loss.backward()\n",
}


def sweep():
    """压缩率随词表增长：每多一个 merge，就多一个能代表更长字节串的 token。"""
    rows = []
    for v in VOCAB_SIZES:
        for name, cls in (("regex", RegexTokenizer), ("basic", BasicTokenizer)):
            t0 = time.time()
            tok = cls().train(TEXT, v)
            rows.append({"vocab_size": v, "kind": name,
                         "ratio": compression_ratio(tok, TEXT),
                         "seconds": time.time() - t0})
            print(f"{name:5s} vocab {v:5d}: {rows[-1]['ratio']:.3f} bytes/token "
                  f"({rows[-1]['seconds']:.0f}s)", flush=True)
    return rows


def against_tiktoken():
    """和 09 章实际用的 tiktoken gpt2（50257）比，也顺带量不同语言的代价差。"""
    enc = tiktoken.get_encoding("gpt2")
    ours = RegexTokenizer().train(TEXT, 1536)

    rows = []
    for name, text in SAMPLES.items():
        n_bytes = len(text.encode("utf-8"))
        rows.append({
            "sample": name,
            "chars": len(text),
            "bytes": n_bytes,
            "gpt2_tokens": len(enc.encode(text)),
            "gpt2_bytes_per_token": n_bytes / len(enc.encode(text)),
            "ours_tokens": len(ours.encode(text)),
            "ours_bytes_per_token": n_bytes / len(ours.encode(text)),
        })
        print(f"{name:8s} gpt2 {rows[-1]['gpt2_bytes_per_token']:.2f} B/tok | "
              f"ours(1536) {rows[-1]['ours_bytes_per_token']:.2f} B/tok", flush=True)
    return rows


def weirdness():
    """LLM 的一堆怪毛病，根源都在这里。用 09 章那个 tokenizer 演示。"""
    enc = tiktoken.get_encoding("gpt2")
    show = lambda s: [enc.decode([i]) for i in enc.encode(s)]

    cases = {
        "前导空格改变一切": {s: show(s) for s in ("hello", " hello", "Hello")},
        "数字切得没有规律": {s: show(s) for s in ("677", "6773", "67730", "1234567")},
        "行尾空格是另一个 token": {s: show(s) for s in ("hello world", "hello world ")},
        "拼写任务看不到字母": {s: show(s) for s in ("strawberry", "ubiquitous")},
        "中文按字节碎掉": {s: show(s) for s in ("你好", "机器学习")},
    }
    for title, group in cases.items():
        print(f"\n{title}")
        for s, toks in group.items():
            print(f"  {s!r:16s} -> {len(toks):2d} tokens {toks}")
    return {k: {s: v for s, v in g.items()} for k, g in cases.items()}


def plot(sweep_rows, lang_rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for kind, color in (("regex", BLUE), ("basic", ORANGE)):
        xs = [r["vocab_size"] for r in sweep_rows if r["kind"] == kind]
        ys = [r["ratio"] for r in sweep_rows if r["kind"] == kind]
        ax1.plot(xs, ys, marker="o", markersize=5, linewidth=2, color=color,
                 label=f"{kind} split")
    ax1.set_xlabel("vocab size")
    ax1.set_ylabel("bytes per token (higher = better compression)")
    ax1.set_title("Compression vs vocabulary size, tiny Shakespeare")
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False)

    names = [r["sample"] for r in lang_rows]
    gpt2 = [r["gpt2_bytes_per_token"] for r in lang_rows]
    xs = range(len(names))
    bars = ax2.bar(xs, gpt2, color=[BLUE, ORANGE, PURPLE], width=0.55)
    ax2.set_xticks(list(xs), names)
    ax2.set_ylabel("bytes per token, GPT-2 tokenizer")
    ax2.set_title("The same tokenizer is not equally efficient everywhere")
    ax2.grid(alpha=0.25, axis="y", linewidth=0.6)
    ax2.set_axisbelow(True)
    for bar, v in zip(bars, gpt2):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.2f}",
                 ha="center", fontsize=10)

    fig.tight_layout()
    path = ROOT / "figures" / "bpe_compression.png"
    fig.savefig(path, dpi=120)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    sweep_rows = sweep()
    lang_rows = against_tiktoken()
    cases = weirdness()

    out = ROOT / "experiments" / "bpe_results.json"
    out.write_text(json.dumps({"sweep": sweep_rows, "languages": lang_rows,
                               "weirdness": cases}, indent=1, ensure_ascii=False))
    print(f"wrote {out}")
    plot(sweep_rows, lang_rows)
