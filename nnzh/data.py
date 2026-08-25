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


def bpc(nats):
    return nats / math.log(2)
