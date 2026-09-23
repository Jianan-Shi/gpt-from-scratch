"""Tokenize FineWeb-Edu into the .npy shards the training script reads.

Karpathy's fineweb.py downloads the whole 28GB sample-10BT before writing anything,
which is wrong twice over here: verifying DDP needs one shard, not a hundred, and on a
rented box that download is billed. This streams instead, so `--shards 2` costs a few
hundred MB and a couple of minutes.

    export HF_ENDPOINT=https://hf-mirror.com     # if huggingface.co is unreachable
    python prep_shards.py --shards 2             # enough to smoke-test DDP
    python prep_shards.py                        # all 100 shards, ~10B tokens, ~20GB

Shard 0 is the validation split and the rest are training, matching the layout and the
filenames fineweb.py produces, so existing runs and existing shards stay valid.
"""
import argparse
import os
import sys

import numpy as np
import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"] # delimits documents; every shard starts with one


def encode_batch(texts, workers):
    """tiktoken 的批量接口在 Rust 侧多线程并释放 GIL，比 multiprocessing 简单也快。

    用 mp.Pool 喂这个流式 dataset 会在解释器退出时崩（feeder 线程和 datasets 自己的
    线程撞在一起），而且提前停止时更难收拾。
    """
    for ids in enc.encode_ordinary_batch(texts, num_threads=workers):
        tokens = np.array([EOT] + ids, dtype=np.int64) # EOT 分隔文档，每个分片以它开头
        assert (0 <= tokens).all() and (tokens < 2**16).all(), "token id 超出 uint16"
        yield tokens.astype(np.uint16)


def write_shard(path, tokens):
    np.save(path, tokens)
    print(f"wrote {path} ({len(tokens):,} tokens)", flush=True)


def main(out_dir, shards, shard_size, workers, batch_docs=512):
    os.makedirs(out_dir, exist_ok=True)
    # streaming=True: pull documents as needed instead of downloading the full 28GB sample
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                           split="train", streaming=True)

    shard_index, filled = 0, 0
    buffer = np.empty((shard_size,), dtype=np.uint16)
    texts = []

    def flush(texts):
        nonlocal shard_index, filled
        for tokens in encode_batch(texts, workers):
            while len(tokens):
                take = min(shard_size - filled, len(tokens))
                buffer[filled:filled + take] = tokens[:take]
                filled += take
                tokens = tokens[take:]
                if filled == shard_size:
                    split = "val" if shard_index == 0 else "train" # shard 0 is validation
                    write_shard(os.path.join(out_dir, f"edufineweb_{split}_{shard_index:06d}.npy"), buffer)
                    shard_index, filled = shard_index + 1, 0
                    if shards is not None and shard_index >= shards:
                        return True
        return False

    for doc in dataset:
        texts.append(doc["text"])
        if len(texts) == batch_docs:
            if flush(texts):
                return
            texts = []
    if texts and flush(texts):
        return

    if filled: # trailing partial shard
        split = "val" if shard_index == 0 else "train"
        write_shard(os.path.join(out_dir, f"edufineweb_{split}_{shard_index:06d}.npy"), buffer[:filled])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.environ.get("GPT2_DATA_ROOT", "edu_fineweb10B"))
    parser.add_argument("--shards", type=int, default=None, help="stop after N shards (default: all)")
    parser.add_argument("--shard-size", type=int, default=100_000_000)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    a = parser.parse_args()
    print(f"writing to {a.out}, {a.shards or 'all'} shards of {a.shard_size:,} tokens, {a.workers} workers")
    main(a.out, a.shards, a.shard_size, a.workers)

    # The shards are written and closed by now. A normal exit would tear down the HF
    # streaming iterator's background threads and abort with "PyGILState_Release:
    # auto-releasing thread-state" — a crash *after* the work, which looks exactly like
    # a failed run. Leave the way we came in instead.
    print("done", flush=True)
    sys.stdout.flush()
    os._exit(0)
