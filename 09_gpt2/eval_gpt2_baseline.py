"""Evaluate OpenAI's released GPT-2 (124M) with the same code that evaluates our own runs.

The README used to quote Karpathy's numbers (3.29 val loss, 0.2945 HellaSwag). Measuring
them here instead makes the comparison a controlled one: same val shard, same number of
tokens, same HellaSwag scoring, same dtype, same machine.

usage: python eval_gpt2_baseline.py [model_name_or_checkpoint.pt]
       defaults to "gpt2" (the 124M checkpoint from OpenAI via HuggingFace)
"""
import glob
import sys
import time

import numpy as np
import torch

from hellaswag import iterate_examples, render_example

# gpt2_follow.py starts training at import time, so exec only the definitions above the
# training section (this is also how test_gpt2.py gets at the model).
_src = open("gpt2_follow.py").read()
_defs = {"__name__": "gpt2_defs"}
exec(_src[: _src.index("# run the training loop")], _defs)
GPT, GPTConfig = _defs["GPT"], _defs["GPTConfig"]
get_most_likely_row = _defs["get_most_likely_row"]

VAL_SHARD = sorted(glob.glob("../build-nanogpt/edu_fineweb10B/*val*"))[0]
B, T = 4, 1024
VAL_STEPS = 320 # 320 * 4 * 1024 = 1.31M tokens, the same slice Karpathy's 20 steps at B=64 covered


def evaluate(model, device="cuda"):
    model.to(device).eval()

    tokens = np.load(VAL_SHARD).astype(np.int32)
    tokens = torch.tensor(tokens, dtype=torch.long)
    val_loss = 0.0
    t0 = time.time()
    with torch.no_grad():
        for i in range(VAL_STEPS):
            buf = tokens[i * B * T : i * B * T + B * T + 1].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(buf[:-1].view(B, T), buf[1:].view(B, T))
            val_loss += loss.item() / VAL_STEPS
    print(f"val loss: {val_loss:.4f}  ({VAL_STEPS * B * T / 1e6:.2f}M tokens, {time.time() - t0:.0f}s)")

    n_correct = n_total = 0
    t0 = time.time()
    for example in iterate_examples("val"):
        _, tokens, mask, label = render_example(example)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(tokens.to(device))
        n_total += 1
        n_correct += int(get_most_likely_row(tokens.to(device), mask.to(device), logits) == label)
    print(f"HellaSwag: {n_correct}/{n_total}={n_correct / n_total:.4f}  ({time.time() - t0:.0f}s)")
    return val_loss, n_correct / n_total


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    torch.set_float32_matmul_precision("high")
    if target.endswith(".pt"):
        import __main__ # checkpoints pickled their config as __main__.GPTConfig
        __main__.GPTConfig = GPTConfig
        ckpt = torch.load(target, weights_only=False)
        cfg = ckpt["config"]
        model = GPT(cfg if not isinstance(cfg, dict) else GPTConfig(**cfg))
        model.load_state_dict(ckpt["model"])
        print(f"loaded {target} (step {ckpt['step']}, logged val loss {ckpt['val_loss']:.4f})")
    else:
        model = GPT.from_pretrained(target)
    evaluate(model)
