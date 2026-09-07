"""Tiny Shakespeare, character level —— 第 07 章的语料。

和 data.py 的 names.txt 是两套东西，不要混用：
  - 语料不同：1,115,394 个字符的连续文本 vs 32,033 个独立的名字
  - 词表不同：65 vs 27
  - 切分不同：按位置切前 90%/后 10%（文本是连续的，不能打乱），
    names 那边是按词 80/10/10 打乱切

所以 07 的 bpc 和前五章不可比，uniform 基线从 log2(27)=4.755 变成 log2(65)=6.022。
experiments/bpc.md 里另起一张表。
"""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "tinyshakespeare.txt"
TRAIN_FRAC = 0.9


def load_text(path=DATA_PATH):
    return Path(path).read_text(encoding="utf-8")


def build_vocab(text):
    """字符级词表。没有 <unk>——词表就是语料里出现过的全部字符。"""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    return stoi, {i: ch for ch, i in stoi.items()}


def encode(s, stoi):
    return [stoi[c] for c in s]


def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)


def train_val_split(text, stoi, frac=TRAIN_FRAC):
    """按位置切，不打乱。

    打乱会把同一句话的前后半分到两边——验证集里的上下文模型在训练时见过，
    val loss 会虚低。names 那边按词打乱没这个问题，因为词与词本来就独立。
    """
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    k = int(frac * len(data))
    return data[:k], data[k:]


def get_batch(data, block_size, batch_size, generator=None, device="cpu"):
    """随机取 batch_size 个起点，每个截 block_size 个字符。

    y 是 x 右移一位：一个 (B, T) 的样本里其实藏着 T 个预测任务，
    位置 t 的目标是 x[t+1]，这是 GPT 训练效率的来源。
    """
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
