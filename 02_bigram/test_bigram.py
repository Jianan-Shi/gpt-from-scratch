import math

import torch
import torch.nn.functional as F

from bigram import (build_dataset, fit_counting, fit_neural, nll_counting,
                    nll_neural)
from nnzh.data import VOCAB_SIZE, build_vocab, load_words

WORDS = load_words()
STOI, ITOS = build_vocab(WORDS)
XS, YS = build_dataset(WORDS, STOI)


def test_vocab_round_trip():
    assert len(STOI) == VOCAB_SIZE and STOI["."] == 0
    for ch, i in STOI.items():
        assert ITOS[i] == ch


def test_rows_are_distributions():
    """漏掉 keepdim=True 会变成按列归一化，这个测试专门抓它。"""
    P = fit_counting(XS, YS, smoothing=1.0)
    assert torch.allclose(P.sum(1), torch.ones(VOCAB_SIZE), atol=1e-5)
    assert (P > 0).all(), "平滑之后不该有 0 概率"


def test_row_sums_equal_col_sums():
    """每个 token 的入度 = 出度：所有名字都从 . 出发、回到 . 结束。"""
    flat = XS * VOCAB_SIZE + YS
    N = torch.bincount(flat, minlength=VOCAB_SIZE**2).reshape(VOCAB_SIZE, VOCAB_SIZE)
    assert torch.equal(N.sum(1), N.sum(0))


def test_onehot_matmul_equals_row_lookup():
    W = torch.randn((VOCAB_SIZE, VOCAB_SIZE))
    idx = XS[:64]
    assert torch.allclose(F.one_hot(idx, VOCAB_SIZE).float() @ W, W[idx])


def test_manual_loss_matches_cross_entropy():
    W = torch.randn((VOCAB_SIZE, VOCAB_SIZE))
    idx, tgt = XS[:1000], YS[:1000]
    logits = W[idx]
    probs = logits.exp() / logits.exp().sum(1, keepdim=True)
    manual = -probs[torch.arange(len(idx)), tgt].log().mean()
    assert torch.allclose(manual, F.cross_entropy(logits, tgt), atol=1e-5)


def test_beats_uniform():
    P = fit_counting(XS, YS, smoothing=1.0)
    assert nll_counting(P, XS, YS) < math.log(VOCAB_SIZE)


def test_two_methods_converge():
    """计数法是这个凸问题的解析最优解，梯度下降只能逼近、不可能超越。"""
    P = fit_counting(XS, YS, smoothing=0.0)
    W, _ = fit_neural(XS, YS, steps=500, lr=50.0, reg=0.0)
    gap = nll_neural(W, XS, YS) - nll_counting(P, XS, YS)
    assert -1e-4 < gap < 0.02, f"gap = {gap}"


# TODO(你自己想一个测试，和 micrograd 那次一样)。备选思路：
#   - sample() 生成的名字里不含 "."，且长度 >= 1
#   - smoothing 越大，P 的行熵越高（k -> inf 时趋近均匀分布）
#   - split_words 的两侧无交集，且并集等于原词表
