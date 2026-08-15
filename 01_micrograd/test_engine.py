"""micrograd 引擎与 PyTorch 的梯度对拍。

思路：同一个表达式分别喂给自己的 Value 和 torch.Tensor，
断言前向值和所有梯度一致。PyTorch 在这里充当"标准答案"。
"""
import random
import pytest
import torch
from engine import Value


def expr(a, b, c):
    """鸭子类型：Value 和 Tensor 都支持这些运算符，同一份代码能跑两遍。"""
    d = a * b + b ** 3
    d = d + d * 2 + (b + a).relu()
    e = d * c - c / 2.0
    f = (e + a) * (b - c)
    return f.relu() + e


def run_micrograd(vals):
    xs = [Value(v) for v in vals]
    out = expr(*xs)
    out.backward()
    return out.data, [x.grad for x in xs]


def run_torch(vals):
    xs = [torch.tensor(v, dtype=torch.double, requires_grad=True) for v in vals]
    out = expr(*xs)
    out.backward()
    return out.item(), [x.grad.item() for x in xs]


@pytest.mark.parametrize("vals", [
    (-4.0, 2.0, 1.5),
    (3.0, -1.0, 0.5),
    (0.7, 0.3, -2.0),
])
def test_forward_and_grad_match(vals):
    """固定用例：前向值和梯度都要和 PyTorch 一致。"""
    d_mg, g_mg = run_micrograd(vals)
    d_pt, g_pt = run_torch(vals)
    assert d_mg == pytest.approx(d_pt, rel=1e-9)
    for a, b in zip(g_mg, g_pt):
        assert a == pytest.approx(b, rel=1e-9)


def test_random_dags():
    """200 组随机输入，防止固定用例碰巧通过。"""
    random.seed(0)
    for _ in range(200):
        vals = tuple(random.uniform(-3, 3) for _ in range(3))
        _, g_mg = run_micrograd(vals)
        _, g_pt = run_torch(vals)
        for a, b in zip(g_mg, g_pt):
            assert a == pytest.approx(b, rel=1e-8, abs=1e-10)


def test_grad_accumulates_on_reuse():
    """节点复用时梯度必须累加。
    d = a + a  =>  dd/da == 2；_backward 用 = 而非 += 的实现这里会得到 1。"""
    a = Value(3.0)
    d = a + a
    d.backward()
    assert a.grad == pytest.approx(2.0)


def test_pow_grad_with_reuse():
    """x**2 + x 里 x 被复用，dy/dx = 2x + 1。
    __pow__ 的 _backward 若写成 = 而非 +=，这里会得到 2x，静默错误。"""
    x = Value(3.0)
    y = x ** 2 + x
    y.backward()
    assert x.grad == pytest.approx(7.0)   # 2*3 + 1


def test_leaf_backward_is_noop():
    """叶子节点的 _backward 是空函数，调用不应报错。"""
    a = Value(1.0)
    a._backward()