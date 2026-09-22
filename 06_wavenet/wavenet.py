"""WaveNet-style hierarchical MLP (makemore part 5).

这一章做两件事：

1. 把 04 章那堆散装张量重构成 PyTorch 风格的层，并且能用 Sequential 串起来。
   层的接口（__call__ / parameters() / training / self.out）和 04 保持一致，
   新增 Embedding、FlattenConsecutive、Sequential 三个。

2. 把"8 个字符一次性拍平送进一层 Linear"换成**树状逐层两两融合**：
   8 -> 4 -> 2 -> 1，每层只让相邻的一对字符交互，再把结果往上送。
   这就是 WaveNet 的扩张因果卷积在全连接网络里的样子。

   拍平的做法里，第一层要一次性吸收 block_size * n_embd 维输入，字符之间
   怎么组合全压在一个矩阵里；树状结构让"两两融合"重复 log2(block_size) 次，
   同样的参数量能吃更长的上下文。

本章最容易踩的坑在 BatchNorm1d：输入变成 3 维 (B, T, C) 之后，统计量必须在
dim=(0, 1) 上求。写成 dim=0 不会报错，也能正常训练，但 running_mean 的形状会
变成 (1, T, C) 而不是 (1, C)——训练用 batch 统计量所以看不出来，一旦 eval()
切到 running buffer，形状还能广播回去，数字却是错的。test_wavenet.py 里有断言。
"""
import torch
import torch.nn.functional as F

from nnzh.data import (VOCAB_SIZE, bpc, build_dataset, build_vocab, load_words,
                       split_words)

BLOCK_SIZE = 8
N_EMBD = 24
N_HIDDEN = 128
SEED = 2147483647


class Linear:
    def __init__(self, fan_in, fan_out, bias=True, generator=None):
        self.weight = torch.randn((fan_in, fan_out), generator=generator) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out = self.out + self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    """和 04 章的同名类只差一处：3 维输入要在 (0, 1) 两个维度上统计。

    (B, C)    -> dim=(0,)   每个特征一个均值
    (B, T, C) -> dim=(0, 1) 每个特征一个均值，T 个位置共享
    写成 dim=0 的话每个 (T, C) 位置各自统计，running buffer 形状变成 (1, T, C)，
    参数量凭空多出 T 倍，而且换一个 block_size 就再也对不上了。
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            dim = 0 if x.ndim == 2 else (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True, unbiased=True)
        else:
            xmean, xvar = self.running_mean, self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze()
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    """查表。之前写成 C[X]，现在包成一层，才能进 Sequential。"""

    def __init__(self, num_embeddings, embedding_dim, generator=None):
        self.weight = torch.randn((num_embeddings, embedding_dim), generator=generator)

    def __call__(self, idx):
        self.out = self.weight[idx]
        return self.out

    def parameters(self):
        return [self.weight]


class FlattenConsecutive:
    """把相邻的 n 个位置合并到通道维：(B, T, C) -> (B, T // n, C * n)。

    n = block_size 时退化成整条拍平，也就是 03/04 章的做法（此时 T // n == 1，
    多出来的那一维会被 squeeze 掉，形状回到 (B, C * block_size)）。
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self, mode=True):
        for layer in self.layers:
            if isinstance(layer, BatchNorm1d):
                layer.training = mode
        return self

    def eval(self):
        return self.train(False)


def init_wavenet(block_size=BLOCK_SIZE, n_embd=N_EMBD, n_hidden=N_HIDDEN,
                 fan_in=2, out_scale=0.1, seed=SEED):
    """树状结构：每层 FlattenConsecutive(fan_in) 把上下文长度砍掉 fan_in 倍。

    block_size=8, fan_in=2 时是 8 -> 4 -> 2 -> 1 三层。
    fan_in=block_size 时只有一层，等价于 03/04 的拍平 MLP（见 init_flat）。
    """
    g = torch.Generator().manual_seed(seed)
    n_levels = 0
    t = block_size
    while t > 1:
        assert t % fan_in == 0, f"block_size={block_size} 必须能被 fan_in={fan_in} 整除"
        t //= fan_in
        n_levels += 1

    layers = [Embedding(VOCAB_SIZE, n_embd, generator=g)]
    fan = n_embd
    for _ in range(n_levels):
        layers += [FlattenConsecutive(fan_in),
                   Linear(fan * fan_in, n_hidden, bias=False, generator=g),
                   BatchNorm1d(n_hidden),
                   Tanh()]
        fan = n_hidden
    layers.append(Linear(n_hidden, VOCAB_SIZE, generator=g))

    model = Sequential(layers)
    with torch.no_grad():
        # 压扁输出层，让 step-0 的 loss 落在 ln(27)——和 04 章同一个理由
        model.layers[-1].weight *= out_scale
    return model


def init_flat(block_size=BLOCK_SIZE, **kwargs):
    """对照组：一次性拍平，一个隐藏层。参数量对齐时用来衡量树状结构值多少。"""
    return init_wavenet(block_size=block_size, fan_in=block_size, **kwargs)


def nll(model, X, Y, batch=None, generator=None):
    """平均负对数似然（nats）。batch=None 时一次算完整个集合。"""
    model.eval()
    with torch.no_grad():
        if batch is None:
            return F.cross_entropy(model(X), Y).item()
        total, n = 0.0, 0
        for i in range(0, len(X), batch):
            logits = model(X[i : i + batch])
            total += F.cross_entropy(logits, Y[i : i + batch], reduction="sum").item()
            n += len(logits)
        return total / n


def fit(model, Xtr, Ytr, steps=200_000, batch_size=32, lr=0.1, lr_drop_at=0.75,
        seed=SEED, log_every=10_000, verbose=True):
    """Karpathy 的配置：200k 步，后 1/4 把学习率降一个数量级。"""
    g = torch.Generator().manual_seed(seed)
    params = model.parameters()
    for p in params:
        p.requires_grad_(True)

    history = []
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        model.train()
        logits = model(Xtr[ix])
        loss = F.cross_entropy(logits, Ytr[ix])

        for p in params:
            p.grad = None
        loss.backward()
        step_lr = lr if i < steps * lr_drop_at else lr * 0.1
        with torch.no_grad():
            for p in params:
                p -= step_lr * p.grad

        history.append(loss.item())
        if verbose and i % log_every == 0:
            print(f"{i:7d}/{steps} loss {loss.item():.4f}")
    return history


@torch.no_grad()
def sample(model, itos, n=10, block_size=BLOCK_SIZE, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    model.eval()
    out = []
    for _ in range(n):
        context, word = [0] * block_size, ""
        while True:
            logits = model(torch.tensor([context]))
            ix = torch.multinomial(F.softmax(logits, dim=1), 1, generator=g).item()
            if ix == 0:
                break
            word += itos[ix]
            context = context[1:] + [ix]
        out.append(word)
    return out


def load_splits(block_size=BLOCK_SIZE):
    words = load_words()
    stoi, itos = build_vocab(words)
    tr, va, te = split_words(words)
    return (build_dataset(tr, stoi, block_size),
            build_dataset(va, stoi, block_size),
            build_dataset(te, stoi, block_size), itos)


if __name__ == "__main__":
    (Xtr, Ytr), (Xva, Yva), _, itos = load_splits()
    print(f"train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}")

    model = init_wavenet()
    n_params = sum(p.nelement() for p in model.parameters())
    print(f"{n_params:,} parameters")
    print(f"step-0 loss {nll(model, Xtr[:5000], Ytr[:5000]):.4f}  (ln 27 = 3.2958)")

    fit(model, Xtr, Ytr)
    tr_nats, va_nats = nll(model, Xtr, Ytr, batch=10_000), nll(model, Xva, Yva, batch=10_000)
    print(f"train {tr_nats:.4f} nats = {bpc(tr_nats):.4f} bpc")
    print(f"val   {va_nats:.4f} nats = {bpc(va_nats):.4f} bpc")
    print("samples:", " ".join(sample(model, itos)))
