import math

import torch

from nnzh.data import VOCAB_SIZE
from wavenet import (BLOCK_SIZE, BatchNorm1d, Embedding, FlattenConsecutive,
                     Linear, Sequential, Tanh, init_flat, init_wavenet,
                     load_splits, nll)

(XTR, YTR), _, _, _ = load_splits()


def test_flatten_consecutive_pairs_adjacent_positions():
    """(B, T, C) -> (B, T//n, C*n)，相邻 n 个位置拼到通道维，顺序不能乱。"""
    x = torch.arange(2 * 4 * 3).float().view(2, 4, 3)
    out = FlattenConsecutive(2)(x)

    assert out.shape == (2, 2, 6)
    assert torch.equal(out[0, 0], torch.cat([x[0, 0], x[0, 1]]))
    assert torch.equal(out[0, 1], torch.cat([x[0, 2], x[0, 3]]))


def test_flatten_by_block_size_reduces_to_the_flat_mlp():
    """n = block_size 时退化成整条拍平，也就是 03/04 章的做法。"""
    x = torch.randn(5, BLOCK_SIZE, 4)
    out = FlattenConsecutive(BLOCK_SIZE)(x)

    assert out.shape == (5, BLOCK_SIZE * 4)
    assert torch.equal(out, x.view(5, BLOCK_SIZE * 4))


def test_batchnorm_statistics_are_per_channel_on_3d_input():
    """本章最容易写错的地方：3 维输入要在 dim=(0, 1) 上统计。

    写成 dim=0 的话每个 (T, C) 位置各自统计，running buffer 形状会变成
    (T, C)——不报错，训练也正常（训练用的是 batch 统计量），只有切到
    eval() 之后才开始悄悄用错的数字。
    """
    C = 7
    bn = BatchNorm1d(C)
    x = torch.randn(32, 4, C)
    bn(x)

    assert bn.running_mean.shape == (C,), "每个通道一个均值，和 T 无关"
    assert bn.running_var.shape == (C,)

    # 对照：错误写法的形状，换一个 block_size 就再也对不上
    wrong = x.mean(0, keepdim=True).squeeze()
    assert wrong.shape == (4, C)


def test_batchnorm_normalises_across_batch_and_time():
    bn = BatchNorm1d(5)
    x = torch.randn(64, 4, 5) * 3 + 10
    out = bn(x)

    assert out.mean(dim=(0, 1)).abs().max() < 1e-5
    assert (out.std(dim=(0, 1)) - 1).abs().max() < 0.05


def test_eval_mode_switches_to_running_statistics():
    """eval 时每个样本必须独立，不能被同 batch 的邻居影响。"""
    model = init_wavenet()
    model.train()
    model(XTR[:256]) # 攒一点 running 统计量

    model.eval()
    x = XTR[:16]
    alone = model(x[:1])
    with_others = model(x)[:1]

    assert torch.allclose(alone, with_others, atol=1e-6)


def test_train_mode_couples_examples_in_a_batch():
    """对照上一条：train 模式下同一个样本的输出取决于和谁同批。"""
    model = init_wavenet()
    model.train()
    x = XTR[:16]

    alone = model(x[:1])
    with_others = model(x)[:1]

    assert not torch.allclose(alone, with_others, atol=1e-3)


def test_tree_has_one_level_per_halving():
    """block_size=8, fan_in=2 -> 3 层融合，加上输出层共 4 个 Linear。"""
    model = init_wavenet(block_size=8, fan_in=2)
    n_linear = sum(isinstance(l, Linear) for l in model.layers)
    n_flatten = sum(isinstance(l, FlattenConsecutive) for l in model.layers)

    assert n_flatten == 3 == int(math.log2(8))
    assert n_linear == 4


def test_initial_loss_is_the_uniform_baseline():
    """输出层被压扁，step-0 的 loss 应该落在 ln(27) 附近而不是 20+。"""
    model = init_wavenet()
    loss = nll(model, XTR[:5000], YTR[:5000])

    assert abs(loss - math.log(VOCAB_SIZE)) < 0.05


def test_parameters_include_every_layer():
    model = init_wavenet()
    params = model.parameters()

    assert any(p.shape == (VOCAB_SIZE, 24) for p in params), "Embedding 的表也是参数"
    assert sum(p.nelement() for p in params) == 76_579


def test_eval_only_touches_batchnorm():
    """Tanh/Linear/Embedding 没有 training 状态，eval() 不该给它们塞属性。"""
    model = init_wavenet().eval()

    for layer in model.layers:
        if isinstance(layer, BatchNorm1d):
            assert layer.training is False
        else:
            assert not hasattr(layer, "training")


def test_flat_and_tree_see_the_same_context():
    """两种结构吃的是同一批输入，差别只在怎么组合——否则比较就没意义。"""
    tree, flat = init_wavenet(), init_flat()

    assert tree.layers[0].weight.shape == flat.layers[0].weight.shape
    assert tree(XTR[:8]).shape == flat(XTR[:8]).shape == (8, VOCAB_SIZE)
