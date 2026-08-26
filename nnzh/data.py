"""Shared across all chapters so bpc numbers stay comparable."""
import math
from pathlib import Path

import torch

VOCAB_SIZE = 27
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "names.txt"
SPLIT_SEED = 42


def load_words(path=DATA_PATH):
    return Path(path).read_text().splitlines()


def build_vocab(words):
    chars = sorted(set("".join(words)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    return stoi, {i: ch for ch, i in stoi.items()}


def split_words(words, seed=SPLIT_SEED):
    """80/10/10 by word. Never change the seed — every chapter depends on it."""
    g = torch.Generator().manual_seed(seed)
    shuffled = [words[i] for i in torch.randperm(len(words), generator=g).tolist()]
    n1, n2 = int(0.8 * len(shuffled)), int(0.9 * len(shuffled))
    return shuffled[:n1], shuffled[n1:n2], shuffled[n2:]


def build_dataset(words, stoi, block_size):
    """滑动窗口展开成 (X, Y)：X (N, block_size) 是上下文，Y (N,) 是下一个字符。

    词与词之间 context 重置为全 0，所以一个名字的尾巴不会泄漏进下一个名字的开头。
    block_size 没有默认值——各章用的长度不同，必须显式写出来，否则迟早会漂。
    block_size=1 时等价于 02_bigram 的数据集（squeeze 掉第二维即可）。
    """
    xs, ys = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            xs.append(context)
            ys.append(ix)
            context = context[1:] + [ix]      # 重新绑定，不是就地修改
    return torch.tensor(xs), torch.tensor(ys)


def bpc(nats):
    return nats / math.log(2)
