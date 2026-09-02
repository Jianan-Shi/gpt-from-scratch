"""Deep MLP with principled initialisation and BatchNorm (makemore part 3).

第三集的主题不是"更强的模型"，而是"能训得动的模型"：
1. 压扁输出层 -> 初始 loss 落在 ln(27)，不再浪费前几百步纠正瞎自信；
2. Kaiming 增益 -> 每层激活的方差不随深度衰减，tanh 不饱和；
3. BatchNorm  -> 直接强制每个 batch 的 preactivation 标准化，代价是样本之间耦合。

层的接口和 PyTorch 一致（__call__ / parameters() / training），每层把输出存在
self.out，诊断代码（激活/梯度直方图）才能事后取到中间量。
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nnzh.data import (VOCAB_SIZE, bpc, build_dataset, build_vocab, load_words,
                       split_words)

BLOCK_SIZE = 3
N_EMBD = 10
N_HIDDEN = 100
N_LAYERS = 6                  # 5 个隐藏层 + 1 个输出层
TANH_GAIN = 5 / 3             # Kaiming for tanh: torch.nn.init.calculate_gain('tanh')


class Linear:
    def __init__(self, fan_in, fan_out, bias=True, gain=1.0, generator=None):
        # 除以 sqrt(fan_in)：fan_in 个独立项求和会把方差放大 fan_in 倍，这里除回去。
        self.weight = torch.randn((fan_in, fan_out), generator=generator) * gain / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out = self.out + self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    """按列（特征维）标准化，gamma/beta 学回想要的尺度和偏移。

    training=True 用当前 batch 的统计量，同时以动量更新 running buffer；
    training=False 用 running buffer——推理时每个样本必须独立，不能被同 batch 的
    邻居影响。这是本章唯一一处「训练和推理行为不同」的地方，也最容易写错。
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)          # 参数：反向传播学
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)  # buffer：滑动平均攒，不参与反传
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True, unbiased=True)
        else:
            xmean, xvar = self.running_mean, self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


@dataclass
class DeepMLP:
    """C 查表 -> 拼接 -> [Linear (BatchNorm) Tanh] * k -> Linear (BatchNorm)。"""
    C: torch.Tensor
    layers: list
    block_size: int
    n_embd: int

    def parameters(self):
        return [self.C] + [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, X):
        x = self.C[X].view(X.shape[0], self.block_size * self.n_embd)
        for layer in self.layers:
            x = layer(x)
        return x

    def train(self, mode=True):
        for layer in self.layers:
            if isinstance(layer, BatchNorm1d):
                layer.training = mode
        return self

    def eval(self):
        return self.train(False)


def init_deep(block_size=BLOCK_SIZE, n_embd=N_EMBD, n_hidden=N_HIDDEN,
              n_layers=N_LAYERS, batchnorm=True, gain=TANH_GAIN, out_scale=0.1,
              seed=2147483647):
    """out_scale 压扁最后一层，让 step-0 的 loss 等于 ln(27) 而不是 20+。

    有 BN 时压 gamma，没 BN 时压 weight——两处是同一件事：让 logits 起步接近 0。
    gain=5/3 是 tanh 的 Kaiming 增益，补偿 tanh 对方差的压缩；gain=1 会让激活
    逐层变窄（见 experiments_bn.py 的第一张图）。
    """
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((VOCAB_SIZE, n_embd), generator=g)

    sizes = [block_size * n_embd] + [n_hidden] * (n_layers - 1) + [VOCAB_SIZE]
    layers = []
    for i in range(n_layers):
        last = i == n_layers - 1
        # 有 BN 时 Linear 的 bias 是多余的：BN 会减掉均值，把它整个抵消掉。
        layers.append(Linear(sizes[i], sizes[i + 1], bias=not batchnorm,
                             gain=1.0 if last else gain, generator=g))
        if batchnorm:
            layers.append(BatchNorm1d(sizes[i + 1]))
        if not last:
            layers.append(Tanh())

    with torch.no_grad():
        if batchnorm:
            layers[-1].gamma *= out_scale
        else:
            layers[-1].weight *= out_scale

    model = DeepMLP(C=C, layers=layers, block_size=block_size, n_embd=n_embd)
    for p in model.parameters():
        p.requires_grad = True
    return model


def fit(model, X, Y, steps=100000, batch_size=32, lr=0.1, decay_at=None,
        lr_after=0.01, seed=2147483647, track_ud=False):
    """Minibatch SGD。track_ud 时额外返回 update:data ratio 的 log10。

    decay_at 默认是 steps//2，和 experiments_bn.py / bpc.md 里那张表用的是同一套预算。

    track_ud 的那个比值 (lr*grad).std() / p.data.std() 是本章最实用的诊断：它回答的是
    「这一步把参数改动了百分之几」，经验上应该稳定在 1e-3 附近。远低于它说明
    学习率太小（白等），远高于说明太大（在原地乱跳）。
    """
    decay_at = steps // 2 if decay_at is None else decay_at   # 默认跑到一半降 lr
    g = torch.Generator().manual_seed(seed)
    params = model.parameters()
    model.train()
    history, ud = [], []
    for i in range(steps):
        ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)
        loss = F.cross_entropy(model(X[ix]), Y[ix])
        for p in params:
            p.grad = None
        loss.backward()
        step_lr = lr if i < decay_at else lr_after
        for p in params:
            p.data += -step_lr * p.grad
        history.append(loss.item())
        if track_ud:
            with torch.no_grad():
                ud.append([(step_lr * p.grad.std() / p.data.std()).log10().item()
                           for p in params])
    return (history, ud) if track_ud else history


@torch.no_grad()
def calibrate_bn(model, X, chunk=20000):
    """显式标定：拿整个训练集过一遍，把每个 BN 的 mean/var 直接测出来。

    视频里先讲这个再讲 running buffer。两条路应该给出几乎一样的数——
    test_bn.py 有一条测试专门盯着这件事，因为 momentum 写错时它是唯一的报警。
    """
    model.train()                                  # 需要用 batch 统计量前传
    acts = {i: [] for i, l in enumerate(model.layers) if isinstance(l, BatchNorm1d)}
    for start in range(0, X.shape[0], chunk):
        x = model.C[X[start:start + chunk]].view(-1, model.block_size * model.n_embd)
        for i, layer in enumerate(model.layers):
            if isinstance(layer, BatchNorm1d):
                acts[i].append(x)                  # 记的是 BN 的输入
            x = layer(x)
    for i, parts in acts.items():
        pre = torch.cat(parts, dim=0)
        model.layers[i].running_mean = pre.mean(0, keepdim=True)
        model.layers[i].running_var = pre.var(0, keepdim=True, unbiased=True)
    return model


@torch.no_grad()
def nll(model, X, Y, chunk=20000):
    """整个 split 的平均交叉熵（nats）。必须在 eval 模式下算。

    用训练模式评估会作弊：BN 用的是「这一批」的统计量，分数偏好，而且同一个
    样本换个 batch 就换个结果。
    """
    model.eval()
    total, n = 0.0, X.shape[0]
    for start in range(0, n, chunk):
        xb, yb = X[start:start + chunk], Y[start:start + chunk]
        total += F.cross_entropy(model(xb), yb).item() * xb.shape[0]
    return total / n


@torch.no_grad()
def sample(model, itos, n=20, seed=2147483647 + 10):
    """自回归生成。batch=1，所以必须 eval——训练模式下单样本的方差是 nan。"""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(n):
        out, context = [], [0] * model.block_size
        while True:
            probs = F.softmax(model(torch.tensor([context])), dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            if ix == 0:
                break
            out.append(ix)
            context = context[1:] + [ix]
        names.append("".join(itos[i] for i in out))
    return names


@torch.no_grad()
def activation_stats(model, X):
    """每个 tanh 层输出的 (层号, mean, std, 饱和比例)。饱和 = |h| > 0.97。

    饱和的神经元梯度 1 - h**2 接近 0，反向传播到这里就断了。
    """
    model.train()
    model(X)
    return [(i, l.out.mean().item(), l.out.std().item(),
             (l.out.abs() > 0.97).float().mean().item() * 100)
            for i, l in enumerate(model.layers) if isinstance(l, Tanh)]


def grad_stats(model, X, Y):
    """每个 tanh 层输出上的梯度 (层号, mean, std)。前传时要 retain_grad 才拿得到。"""
    model.train()
    x = model.C[X].view(X.shape[0], model.block_size * model.n_embd)
    for layer in model.layers:
        x = layer(x)
        x.retain_grad()
    for p in model.parameters():
        p.grad = None
    F.cross_entropy(x, Y).backward()
    return [(i, l.out.grad.mean().item(), l.out.grad.std().item())
            for i, l in enumerate(model.layers) if isinstance(l, Tanh)]


if __name__ == "__main__":
    words = load_words()
    stoi, itos = build_vocab(words)
    tr_words, va_words, _te = split_words(words)      # test 全程不碰
    Xtr, Ytr = build_dataset(tr_words, stoi, BLOCK_SIZE)
    Xva, Yva = build_dataset(va_words, stoi, BLOCK_SIZE)

    model = init_deep()
    print(f"{sum(p.nelement() for p in model.parameters())} params, "
          f"{N_LAYERS} layers of {N_HIDDEN}")
    print(f"step 0    loss {nll(model, Xtr[:2000], Ytr[:2000]):.4f} nats "
          f"(ln {VOCAB_SIZE} = {torch.tensor(float(VOCAB_SIZE)).log():.4f})")
    for i, m, s, sat in activation_stats(model, Xtr[:5000]):
        print(f"  layer {i:2d} tanh: mean {m:+.2f}  std {s:.2f}  saturated {sat:.2f}%")

    fit(model, Xtr, Ytr)
    print(f"deep+BN   train {bpc(nll(model, Xtr, Ytr)):.4f} bpc  "
          f"val {bpc(nll(model, Xva, Yva)):.4f} bpc")
    print(sample(model, itos, n=20))
