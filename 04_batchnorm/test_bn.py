import copy
import math

import torch

from bn import (BLOCK_SIZE, TANH_GAIN, BatchNorm1d, Linear, Tanh,
                activation_stats, calibrate_bn, fit, init_deep, nll, sample)
from nnzh.data import VOCAB_SIZE, build_dataset, build_vocab, load_words

WORDS = load_words()
STOI, ITOS = build_vocab(WORDS)
SMALL = WORDS[:2000]
X, Y = build_dataset(SMALL, STOI, BLOCK_SIZE)


def test_forward_shapes():
    m = init_deep()
    assert m(X[:37]).shape == (37, VOCAB_SIZE)


def test_init_loss_is_ln_vocab():
    """输出层被 out_scale 压扁后，step-0 应该正好是均匀分布的 ln(27)。

    高于它 = 模型在自信地犯错，前几百步全花在把 logits 压回 0。
    """
    assert abs(nll(init_deep(), X, Y) - math.log(VOCAB_SIZE)) < 0.05


def test_out_scale_actually_squashes_logits():
    """对照组：不压输出层，初始 loss 就高于 ln(27)（这里约 +0.26）。

    幅度之所以温和，是因为 Linear 本身已经除了 sqrt(fan_in)；视频里那个
    「初始 loss 27」的惨状是连这一步都没有时才会出现的。
    """
    assert nll(init_deep(out_scale=1.0), X, Y) > math.log(VOCAB_SIZE) + 0.2


def test_batchnorm_standardises_each_column():
    """gamma=1, beta=0 时，BN 的输出每一列都是零均值单位方差——这就是它的定义。"""
    bn = BatchNorm1d(64)
    out = bn(torch.randn(512, 64) * 7 + 3)
    assert out.mean(0).abs().max() < 1e-5
    assert (out.std(0) - 1).abs().max() < 1e-3


def test_batchnorm_absorbs_any_preceding_bias():
    """BN 前面那层的 bias 完全无效：加个常数进去，输出一模一样。

    所以 init_deep(batchnorm=True) 里 Linear 一律 bias=False——留着不是错，
    是白算一遍梯度。视频里这一步是「b1 的 grad 全是 0」的那个发现。
    """
    bn = BatchNorm1d(32)
    x = torch.randn(256, 32)
    plain = bn(x).clone()
    shifted = bn(x + torch.randn(32) * 5)
    assert torch.allclose(plain, shifted, atol=1e-4)


def test_batchnorm_layers_carry_no_bias():
    m = init_deep(batchnorm=True)
    assert all(l.bias is None for l in m.layers if isinstance(l, Linear))
    assert all(l.bias is not None for l in init_deep(batchnorm=False).layers
               if isinstance(l, Linear))


def test_gain_one_makes_activations_shrink_with_depth():
    """没有 BN、gain=1：每过一层 tanh 就窄一点，五层下来只剩一半。"""
    stds = [s for _, _, s, _ in activation_stats(init_deep(batchnorm=False, gain=1.0), X[:5000])]
    assert all(a > b for a, b in zip(stds, stds[1:])), f"应当逐层单调收缩: {stds}"
    assert stds[-1] / stds[0] < 0.6, stds


def test_kaiming_gain_keeps_activations_flat():
    """gain=5/3 补偿 tanh 的压缩，衰减基本停住（第一层偏高是因为输入本就是单位方差）。"""
    stds = [s for _, _, s, _ in activation_stats(init_deep(batchnorm=False), X[:5000])]
    assert stds[-1] / stds[0] > 0.8, stds
    assert max(stds[1:]) - min(stds[1:]) < 0.05, stds


def test_batchnorm_rescues_a_bad_gain():
    """有 BN 时初始化不再关键：故意用 gain=1，五层激活的 std 照样是一条平线。

    这正是 BN 的卖点——把「初始化必须刚刚好」这件事从人手里拿走。
    """
    stds = [s for _, _, s, _ in activation_stats(init_deep(gain=1.0), X[:5000])]
    assert max(stds) - min(stds) < 0.02, stds


def test_batchnorm_makes_the_forward_pass_scale_invariant():
    """比"平"更强的说法：整个前向对权重的整体缩放**完全不敏感**。

    gain=1 和 gain=5/3 只差一个常数因子，BN 把它原样除掉了，两个模型的 logits
    逐点相同（1e-6 量级的浮点噪声）。没有 BN 时同样的对比差 0.15。
    所以「BN 之后还要不要调 gain」这个问题，答案是不用——它在数学上没有效果。
    """
    with_bn = (init_deep(gain=1.0)(X[:512]) - init_deep(gain=TANH_GAIN)(X[:512])).abs().max()
    without = (init_deep(batchnorm=False, gain=1.0)(X[:512])
               - init_deep(batchnorm=False, gain=TANH_GAIN)(X[:512])).abs().max()
    assert with_bn < 1e-4, with_bn
    assert without > 1e-2, without


def test_saturation_is_low_at_init():
    """|h| > 0.97 的比例应该只有个位数百分比；接近 100% 意味着这一层已经死了。"""
    for _, _, _, sat in activation_stats(init_deep(), X[:5000]):
        assert sat < 10.0


def test_eval_decouples_examples_within_a_batch():
    """训练模式下同一个样本的 logits 会随同批邻居变化，eval 模式下不会。

    这是 BN 唯一真正的副作用。推理时若忘了切 eval，预测结果会依赖你恰好
    把哪些样本凑在一起——一个不会报错、只会让指标微妙变差的 bug。
    """
    m = init_deep()
    one, other = X[:1], X[100:200]
    m.eval()
    assert torch.allclose(m(one), m(torch.cat([one, other]))[:1], atol=1e-6)
    m.train()
    assert not torch.allclose(m(one), m(torch.cat([one, other]))[:1], atol=1e-3)


def test_train_mode_cannot_handle_a_single_example():
    """batch=1 时无偏方差是 nan，所以采样必须走 eval 分支。"""
    assert torch.isfinite(init_deep().eval()(X[:1])).all()
    m = init_deep()                      # 换一个全新的模型：nan 会顺手污染 buffer
    m.train()
    assert torch.isnan(m(X[:1])).any()


def test_running_stats_match_explicit_calibration():
    """滑动平均攒出来的 buffer，应该和拿整个训练集显式标定的结果几乎一致。

    视频里两种做法都讲了。momentum 写反、或者忘了在 no_grad 里更新时，
    这条测试是唯一会响的警报——loss 曲线一切正常，只有验证分数偷偷变差。
    """
    m = init_deep()
    fit(m, X, Y, steps=2000)
    running = nll(m, X, Y)
    calibrated = nll(calibrate_bn(copy.deepcopy(m), X), X, Y)
    assert abs(running - calibrated) < 0.05, (running, calibrated)


def test_training_reduces_loss():
    m = init_deep()
    before = nll(m, X, Y)
    fit(m, X, Y, steps=2000)
    after = nll(m, X, Y)
    assert after < before - 0.5, (before, after)
    assert after < math.log(VOCAB_SIZE)


def test_samples_contain_no_terminator():
    for name in sample(init_deep(), ITOS, n=5):
        assert "." not in name


def test_layer_stack_matches_the_declared_depth():
    """6 层 = 5 个 Linear+BN+Tanh 块 + 1 个 Linear+BN 输出块，不多不少。"""
    m = init_deep(n_layers=6)
    assert sum(isinstance(l, Linear) for l in m.layers) == 6
    assert sum(isinstance(l, BatchNorm1d) for l in m.layers) == 6
    assert sum(isinstance(l, Tanh) for l in m.layers) == 5
    assert isinstance(m.layers[-1], BatchNorm1d), "输出层后面不能再有非线性"
