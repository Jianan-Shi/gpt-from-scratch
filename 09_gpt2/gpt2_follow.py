import os
import math
import time
import inspect
from dataclasses import asdict, dataclass, replace
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # [cleanup] the causal mask buffer ("bias") is gone: flash attention takes
        # is_causal=True instead, and the buffer only wasted ~48MB of VRAM (12 x 1024x1024 fp32).
        # from_pretrained already filters out the HF checkpoint's .attn.bias keys.

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention (materializes th elarge(T, T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # (needed the removed bias buffer)
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projectioin
        y = self.c_proj(y)
        return y



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


def swiglu_hidden(n_embd, multiple_of=None):
    """SwiGLU 有三个矩阵而不是两个，所以隐藏层取 8/3 倍才和原来参数量相同。

    原 MLP:  n_embd -> 4*n_embd -> n_embd          = 8 * n_embd^2
    SwiGLU: n_embd -> h (两路) -> n_embd            = 3 * n_embd * h
    令两者相等：h = 8/3 * n_embd。n_embd=768 时正好是 2048。

    照搬 4 倍会让这一层凭空多出 33% 的参数，那时"SwiGLU 更好"就只是"参数更多更好"，
    消融就白做了。Llama 也是这么取的（再向上对齐到 256 的倍数）。
    """
    # 对齐到 64 的倍数是为了照顾 tensor core，但在小维度上这个对齐会反客为主：
    # n_embd=32 时理想值 85.3 被抬到 128，参数量多 27%，消融就不公平了。
    # 所以对齐粒度跟着维度走，大模型仍是 64。
    if multiple_of is None:
        multiple_of = min(64, max(8, n_embd // 8))
    h = int(8 * n_embd / 3)
    return ((h + multiple_of - 1) // multiple_of) * multiple_of


class SwiGLU(nn.Module):
    """用一路去门控另一路：SwiGLU(x) = (SiLU(x W_gate) * (x W_up)) W_down

    原来的 MLP 是"升维 -> 非线性 -> 降维"，非线性对每个通道独立作用。门控把它换成
    **两路相乘**：一路过 SiLU 当作"开关"，另一路是"内容"，逐元素相乘之后再降维。
    于是某个通道要不要往下传，取决于输入本身，而不是一个固定的形状。

    SiLU(x) = x * sigmoid(x)，和 GELU 形状几乎一样，所以差别不在这个非线性，
    而在那个乘法。Shazeer (2020) 的原话是 "we offer no explanation for the
    improvement"——它是实验上站住的，不是推导出来的。Llama/PaLM/Qwen 都用。

    代价：三个矩阵意味着三次 matmul 而不是两次，同参数量下通常略慢一点。
    这一列要和 loss 一起看。
    """

    def __init__(self, config):
        super().__init__()
        hidden = swiglu_hidden(config.n_embd)
        self.c_gate = nn.Linear(config.n_embd, hidden) # 开关那一路
        self.c_up = nn.Linear(config.n_embd, hidden)   # 内容那一路
        self.c_proj = nn.Linear(hidden, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # 和原 MLP 一样：残差路径上的投影要缩小初始化

    def forward(self, x):
        return self.c_proj(F.silu(self.c_gate(x)) * self.c_up(x))


def make_mlp(config):
    return SwiGLU(config) if config.mlp == "swiglu" else MLP(config)


class RMSNorm(nn.Module):
    """LayerNorm 去掉减均值和 bias，只按均方根缩放。

    LayerNorm:  (x - mean) / sqrt(var + eps) * gamma + beta
    RMSNorm:    x / sqrt(mean(x^2) + eps) * gamma

    它回答的问题是"减均值到底有没有必要"。Zhang & Sennrich (2019) 的观察是：
    LayerNorm 起作用的是**缩放不变性**（让每层输出的尺度稳定），而不是**平移不变性**，
    所以减均值那一步可以省掉。省下的是每层两次全量归约（求均值、再求方差）中的一次，
    以及 n_embd 个 bias 参数。Llama、Qwen、Gemma 现在都用它。

    代价是失去了对输入常数偏移的免疫：LayerNorm(x + c) == LayerNorm(x)，
    但 RMSNorm(x + c) != RMSNorm(x)。实践中残差流的均值本来就接近 0，
    所以这个代价很小——这正是消融要量的东西。
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 在 fp32 里做归约：bf16 下 x^2 容易丢精度，而这一步每层都要过。
        # 乘完 gain 之后再转回输入的 dtype——先转回去的话 bf16 会被 fp32 的 weight
        # 提升回 fp32，后面的 matmul 跟着退回 fp32，训练变慢而没有任何报错。
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def make_norm(config):
    return RMSNorm(config.n_embd) if config.norm == "rmsnorm" else nn.LayerNorm(config.n_embd)


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config)
        self.mlp = make_mlp(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    # --- architecture switches, one per ablation ---------------------------------------
    # One codebase, one training path, one flag apart between any two runs: that is what
    # makes the comparison a controlled experiment rather than two different programs.
    norm: str = "layernorm" # layernorm | rmsnorm
    mlp: str = "gelu"       # gelu | swiglu

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = make_norm(config),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # note: grouping is by dim() >= 2, so RMSNorm's gain lands in the no-decay group
        # automatically, exactly like LayerNorm's gamma/beta did.
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer



# -----------------------------------------------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # [fix] shards are uint16, which older torch can't turn into a tensor
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        # [portable] a path into an untracked local clone is not reproducible on another
        # machine. Order: $GPT2_DATA_ROOT, then shards made here by prep_shards.py, then
        # the build-nanogpt clone this project originally borrowed them from.
        data_root = next((d for d in (os.environ.get("GPT2_DATA_ROOT"), "edu_fineweb10B",
                                      "../build-nanogpt/edu_fineweb10B") if d and os.path.isdir(d)), None)
        assert data_root is not None, "no shards found; run: python prep_shards.py --shards 2"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        # uint16 tokens plus a small npy header: close enough to count tokens from size
        self.total_tokens = sum(os.path.getsize(s) // 2 for s in shards)
        if master_process:
            print(f"found {len(shards)} shards for split {split} ({self.total_tokens:,} tokens)")
        self.reset() # [cleanup] the shard/position init lives only in reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def state_dict(self):
        # 只有 master 存盘，而各 rank 的位置相差 rank * B * T（见 reset），
        # 所以恢复时按自己的 rank 加回偏移即可。
        return {"current_shard": self.current_shard, "current_position": self.current_position}

    def load_state_dict(self, sd, rank_offset=0):
        self.current_shard = sd["current_shard"]
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = sd["current_position"] + rank_offset

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm
# -----------------------------------------------------------------------------------------------------------------
# gradient noise scale (McCandlish et al., An Empirical Model of Large-Batch Training)
#
# 每个 batch 的梯度 = 全体数据的真实方向 + 这批样本带来的随机偏差。B_simple 就是
# 两者量级相当时的 batch 大小（按 token 计）：远小于它，方向被噪声淹没；远大于它，
# 多出来的样本只是在反复确认一个已经清楚的方向。
#
# 用两个 batch 大小就能估：小的用第一个 micro step 的梯度，大的用累积完的梯度，
# 后者 clip_grad_norm_ 本来就返回了。所以额外开销只有一次求范数，没有多余的前反向。

def gradient_noise_scale(g_small_sq, g_big_sq, b_small, b_big):
    """返回 (|G|^2 的估计, tr(Sigma) 的估计, B_simple)，单位都按 token 算。

    |g_B|^2 的期望是 |G|^2 + tr(Sigma)/B，两个不同的 B 联立即可解出两个未知量。
    单步估计噪声很大（分母可能为负），所以调用方要对分子分母各做 EMA 再相除。
    """
    assert b_big > b_small > 0
    g2 = (b_big * g_big_sq - b_small * g_small_sq) / (b_big - b_small)
    s = (g_small_sq - g_big_sq) / (1.0 / b_small - 1.0 / b_big)
    # 两个估计都必须为正才有意义：g2 <= 0 说明连真实方向都没测出来，
    # s < 0 说明小 batch 的范数反而更小——纯粹是这一步的噪声，不能用。
    b_simple = s / g2 if (g2 > 0 and s >= 0) else float("nan")
    return g2, s, b_simple


# -----------------------------------------------------------------------------------------------------------------
# checkpoints that can actually resume
#
# 原版只存权重，机器半夜掉线就得从头再来。续跑还需要：优化器状态（AdamW 的 m/v，
# 丢了等于重新预热）、数据读到哪了、步数、以及 RNG 状态。

def save_checkpoint(path, *, model, optimizer, step, val_loss, loader_state, meta, keep=2, noise_ema=None):
    torch.save({
        "model": model.state_dict(),
        # asdict, not the dataclass: pickling the object makes the checkpoint loadable
        # only where GPTConfig is importable (that is why eval_gpt2_baseline.py needed
        # to graft the class onto __main__). A dict loads anywhere.
        "config": asdict(model.config),
        "optimizer": optimizer.state_dict(), # ~1GB for 124M: AdamW keeps two moments per param
        "step": step,
        "val_loss": val_loss,
        "loader": loader_state,
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "noise_ema": noise_ema, # else the noise-scale EMA restarts from scratch on every resume
        "meta": meta, # preset name, world size, batch geometry — checked on resume
    }, path)
    prune_checkpoints(os.path.dirname(path), keep)
    return path


def prune_checkpoints(log_dir, keep):
    """只留最近 keep 个。每个约 1.5GB，一整晚的训练会把盘写满。"""
    ckpts = sorted(f for f in os.listdir(log_dir) if f.startswith("ckpt_") and f.endswith(".pt"))
    for stale in ckpts[:-keep] if keep > 0 else ckpts:
        os.remove(os.path.join(log_dir, stale))
    return ckpts[-keep:] if keep > 0 else []


def find_latest_checkpoint(log_root="log"):
    """--resume auto 用：拿所有 run 目录里最新的一个 ckpt。"""
    candidates = []
    for run in os.listdir(log_root) if os.path.isdir(log_root) else []:
        run_dir = os.path.join(log_root, run)
        if not os.path.isdir(run_dir):
            continue
        candidates += [os.path.join(run_dir, f) for f in os.listdir(run_dir)
                       if f.startswith("ckpt_") and f.endswith(".pt")]
    return max(candidates, key=os.path.getmtime) if candidates else None


# -----------------------------------------------------------------------------------------------------------------
# simple lunch:
# python gpt2_follow.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 gpt2_follow.py
# torchrun --standalone --nproc_per_node=1 gpt2_follow.py

# run the training loop
import argparse

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# [guard] refuse to start a second run *on the same GPUs*. Two runs sharing one card
# spill VRAM into host RAM (~5x slower); two runs on different cards are exactly how
# ablations get run in parallel, so the lock is per CUDA_VISIBLE_DEVICES, not per script.
# flock is released by the OS when the process exits, even on a crash or kill -9.
import fcntl
_lock_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "_")
if master_process:
    _run_lock = open(f"/tmp/gpt2_follow.gpu{_lock_devices}.lock", "w") # held for the whole run
    try:
        fcntl.flock(_run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another gpt2_follow.py is already training on GPU {_lock_devices} "
                         f"(check: pgrep -af gpt2_follow), refusing to start")

# [fix] autocast and synchronize want the device *type* ("cuda"), not "cuda:0" as under DDP
device_type = "cuda" if device.startswith("cuda") else "cpu"

enc = tiktoken.get_encoding("gpt2")

# -----------------------------------------------------------------------------------------------------------------
# [presets] one machine, one entry. Switch with --preset instead of editing the source:
# editing hyperparameters on a rented, per-hour GPU is how you pay to debug.
#
# val_tokens rather than val_loss_steps: the number of *steps* means a different amount
# of data on every machine (B and world size differ), and the val loss is only comparable
# across runs when the slice is the same. The 2h run on the 4060 read 0.045 nats low
# because 20 steps at B=4 scored 82K tokens instead of the intended 10.5M.

@dataclass
class Preset:
    total_batch_size: int   # tokens per optimizer step, summed over all processes
    B: int                  # micro batch size, i.e. what one forward pass holds
    warmup_steps: int
    max_steps: int
    eval_interval: int
    val_tokens: int         # how much of the val shard each eval scores
    use_compile: bool
    memory_fraction: float = None # cap the allocator so WSL raises OOM instead of spilling to host RAM
    T: int = 1024
    checkpoint_every: int = 1000  # a rented box dying at hour 7 should cost one interval, not the night
    keep_checkpoints: int = 2     # ~1.5GB each (weights + AdamW moments)
    noise_every: int = 10         # gradient noise scale: one extra grad-norm every N steps

PRESETS = {
    # one RTX 4060 8GB: 5.7GB peak at B=4, ~3 s/step, ~9h -> 655M tokens
    "4060": Preset(total_batch_size=2**16, B=4, warmup_steps=300, max_steps=10_000,
                   eval_interval=500, val_tokens=327_680, use_compile=False,
                   memory_fraction=0.95),
    # the original recipe: 2 x A800-80G, B=64, grad_accum 4, one epoch of 10B tokens
    "a800": Preset(total_batch_size=2**19, B=64, warmup_steps=715, max_steps=19_073,
                   eval_interval=250, val_tokens=10_485_760, use_compile=True),
    # 2 x RTX 4090-24G: same recipe, but 24GB only fits B=16, so grad_accum goes 4 -> 16.
    # Gradient accumulation is numerically identical to one big batch, just slower, so
    # total_batch_size stays at the GPT-3 paper's 2**19 and nothing else changes.
    # Estimated 16.9GB at B=16 from the 4060's measured 5.72GB at B=4 — confirm with
    # --preset smoke before starting the long run, and drop to 12 if it is tight.
    "4090x2": Preset(total_batch_size=2**19, B=16, warmup_steps=715, max_steps=19_073,
                     eval_interval=250, val_tokens=10_485_760, use_compile=True),
    # 50 steps for the first rented hour: check throughput, checkpointing, DDP
    "smoke": Preset(total_batch_size=2**16, B=4, warmup_steps=5, max_steps=50,
                    eval_interval=25, val_tokens=81_920, use_compile=False,
                    memory_fraction=0.95, checkpoint_every=25, noise_every=5),
}

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="4060", choices=sorted(PRESETS))
parser.add_argument("--compile", dest="use_compile", default=None, action=argparse.BooleanOptionalAction,
                    help="override the preset, e.g. --compile with --preset smoke to check the compiled path")
parser.add_argument("--resume", default=None, metavar="PATH|auto",
                    help="continue from a checkpoint; writes back into that run's directory")
# overrides for shakedown runs: use the real batch geometry of a preset but stop early,
# e.g. --preset 4090x2 --max-steps 20 measures true peak memory and throughput in a minute
parser.add_argument("--max-steps", type=int, default=None)
# B is the micro batch, not the batch: total_batch_size stays put and grad_accum_steps
# absorbs the change, so this trades kernel-launch overhead against peak memory and
# changes nothing about the math.
parser.add_argument("--micro-batch", "-B", type=int, default=None)
parser.add_argument("--warmup-steps", type=int, default=None,
                    help="keep it near 3.75%% of max-steps, as in the original 715/19073")
# The data order is fixed by the loader, so --seed varies initialisation only. That is
# the noise floor an ablation table needs: two runs of the same config, different seed.
parser.add_argument("--norm", choices=["layernorm", "rmsnorm"], default=None,
                    help="architecture switch; default keeps the GPT-2 baseline")
parser.add_argument("--mlp", choices=["gelu", "swiglu"], default=None)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--tag", default=None, help="appended to the run directory name")
parser.add_argument("--eval-interval", type=int, default=None)
args = parser.parse_args()
preset = PRESETS[args.preset]
if args.use_compile is not None:
    preset = replace(preset, use_compile=args.use_compile)
if args.micro_batch is not None:
    preset = replace(preset, B=args.micro_batch)
if args.max_steps is not None:
    preset = replace(preset, max_steps=args.max_steps)
if args.warmup_steps is not None:
    preset = replace(preset, warmup_steps=args.warmup_steps)
if args.eval_interval is not None:
    preset = replace(preset, eval_interval=args.eval_interval)

total_batch_size, B, T = preset.total_batch_size, preset.B, preset.T
warmup_steps, max_steps = preset.warmup_steps, preset.max_steps
eval_interval, use_compile = preset.eval_interval, preset.use_compile
if total_batch_size % (B * T * ddp_world_size) != 0:
    # B has to divide total_batch_size / (T * world). Saying which values do beats
    # making the reader factorise 524288 at the command line.
    ok = [b for b in (1, 2, 4, 8, 16, 32, 64, 128, 256) if total_batch_size % (b * T * ddp_world_size) == 0]
    raise SystemExit(f"B={B} does not divide total_batch_size={total_batch_size} "
                     f"at T={T} x {ddp_world_size} process(es). Valid: {ok}")
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
val_loss_steps = max(1, preset.val_tokens // (B * T * ddp_world_size)) # per process
torch.manual_seed(args.seed) # after argparse: --seed has to be known first
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
if preset.memory_fraction is not None and device_type == "cuda":
    torch.cuda.set_per_process_memory_fraction(preset.memory_fraction)
if master_process:
    print(f"preset: {args.preset}")
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    print(f"=> val loss over {val_loss_steps * B * T * ddp_world_size:,} tokens ({val_loss_steps} steps/process)")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
# [guard] the loader wraps silently at the end of the last shard, so a run with too few
# shards is not a short run — it is the same tokens over and over. Karpathy's recipe is
# one epoch of 10B; a partial download turns that into 100 epochs of 100M without a word.
tokens_needed = max_steps * total_batch_size
epochs = tokens_needed / train_loader.total_tokens
if master_process:
    print(f"=> {tokens_needed:,} tokens needed, {train_loader.total_tokens:,} available "
          f"=> {epochs:.2f} epochs")
    if epochs > 1.05:
        print(f"!! WARNING: this run repeats the data {epochs:.1f}x. For the 10B baseline "
              f"you want ~100 shards; run prep_shards.py or copy them in.")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# [resume] load first: the checkpoint decides which run directory we append to
resume_path = find_latest_checkpoint() if args.resume == "auto" else args.resume
resume_ckpt = None
if resume_path is not None:
    resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    meta = resume_ckpt["meta"]
    # the data position is stored per-step and depends on the batch geometry, so a
    # resume onto a different machine shape would silently re-read or skip tokens
    assert meta["world_size"] == ddp_world_size, f"checkpoint ran on {meta['world_size']} processes"
    assert (meta["B"], meta["T"], meta["total_batch_size"]) == (B, T, total_batch_size), \
        f"batch geometry changed: {meta} vs B={B} T={T} total={total_batch_size}"
    if master_process:
        print(f"resuming from {resume_path} at step {resume_ckpt['step']}")

# create model
arch = {k: v for k, v in (("norm", args.norm), ("mlp", args.mlp)) if v is not None}
if master_process and arch:
    print(f"architecture overrides: {arch}")
model = GPT(GPTConfig(vocab_size=50304, **arch))
model.to(device)
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
# [compile] torch.compile returns an OptimizedModule wrapper; ._orig_mod is the original
# model and *shares the same parameters*, so this costs no memory and is never stale.
# Anything with a changing input shape goes through eval_model: HellaSwag feeds a
# different T per example and generation grows T by one each step, so the compiled
# version would recompile constantly. val loss keeps a fixed shape and stays compiled.
# The old code instead skipped both evals whenever compile was on, silently.
eval_model = raw_model._orig_mod if use_compile else raw_model
if resume_ckpt is not None:
    eval_model.load_state_dict(resume_ckpt["model"])

max_lr = 6e-4
min_lr = max_lr * 0.1
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

start_step = 0
if resume_ckpt is not None:
    optimizer.load_state_dict(resume_ckpt["optimizer"]) # AdamW's m/v: dropping them means re-warming up
    train_loader.load_state_dict(resume_ckpt["loader"], rank_offset=ddp_rank * B * T)
    torch.set_rng_state(resume_ckpt["rng"])
    if resume_ckpt["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(resume_ckpt["cuda_rng"])
    # [off-by-one] the checkpoint is written in the eval block at the *start* of step S:
    # weights are from the end of S-1 and the loader is positioned at S's first batch.
    # So "step": S means "about to run S", and resuming re-runs S rather than skipping it.
    # Getting this wrong silently drops one optimizer step and replays its data.
    start_step = resume_ckpt["step"]

# create the log directory we will write checkpoints to and log to
# [keep runs] each run gets its own log/run_YYYYmmdd_HHMMSS/ instead of clearing log/log.txt,
# and a copy of this script goes in with it so you can tell which settings produced which curves
if resume_ckpt is not None:
    log_dir = os.path.dirname(resume_path) # append to the same curve instead of starting a new one
else:
    log_dir = os.path.join("log", time.strftime("run_%Y%m%d_%H%M%S")
                           + (f"_{args.tag}" if args.tag else ""))
log_file = os.path.join(log_dir, "log.txt")
if master_process:
    os.makedirs(log_dir, exist_ok=resume_ckpt is not None) # never write into someone else's run
    import shutil
    shutil.copy(__file__, log_dir)
    print(f"logging to {log_dir}")

# [noise scale] single-step estimates are far too noisy to use directly, so the
# numerator and denominator are smoothed separately and only then divided.
noise_ema = (resume_ckpt or {}).get("noise_ema") or {"g2": None, "s": None}
NOISE_BETA = 0.95

for step in range(start_step, max_steps):
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % eval_interval == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(val_loss_steps): # from preset.val_tokens, see above
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % preset.checkpoint_every == 0 or last_step):
                save_checkpoint(
                    os.path.join(log_dir, f"ckpt_{step:06d}.pt"),
                    model=eval_model, # not raw_model: compiled keys carry an _orig_mod. prefix
                    optimizer=optimizer,
                    step=step, # = "about to run this step"; see the resume note above
                    val_loss=val_loss_accum.item(),
                    loader_state=train_loader.state_dict(),
                    meta={"preset": args.preset, "world_size": ddp_world_size,
                          "B": B, "T": T, "total_batch_size": total_batch_size},
                    keep=preset.keep_checkpoints,
                    noise_ema=noise_ema)

    # once in a while evaluate hellaswag
    if step % eval_interval == 0 or last_step:
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = eval_model(tokens) # uncompiled: T varies per example
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if (step > 0 and step % eval_interval == 0) or last_step:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = eval_model(xgen) # uncompiled: T grows by one each step
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # training loop
    # [timing] t0 starts here, not at the top of the step: on eval steps the old
    # placement folded the ~95 s of val + HellaSwag + sampling into dt, reporting
    # 650 tok/s for a step that actually ran at 22,000.
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # [fix] set this BEFORE the forward pass, not before backward: DDP's forward() reads
        # it to decide whether to arm the gradient all-reduce for the upcoming backward().
        # Only the last micro step should sync; the others just accumulate grads locally.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation, 
        # because the gradients just add on each successive backward().
        # additon of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
        if micro_step == 0 and step % preset.noise_every == 0:
            # the small-batch gradient: one micro batch, this rank only (DDP has not
            # synced yet). It is scaled by 1/grad_accum_steps because the loss was,
            # so multiply it back out.
            with torch.no_grad():
                small_norm = torch.norm(torch.stack([
                    p.grad.norm() for p in model.parameters() if p.grad is not None])) * grad_accum_steps
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # returns the pre-clip norm
    b_simple = float("nan")
    if step % preset.noise_every == 0:
        # big batch = the whole optimizer step; small batch = one micro batch on one rank
        g2, s_est, _ = gradient_noise_scale(
            g_small_sq=(small_norm ** 2).item(), g_big_sq=(norm ** 2).item(),
            b_small=B * T, b_big=total_batch_size)
        for key, val in (("g2", g2), ("s", s_est)):
            prev = noise_ema[key]
            noise_ema[key] = val if prev is None else NOISE_BETA * prev + (1 - NOISE_BETA) * val
        if noise_ema["g2"] > 0:
            b_simple = noise_ema["s"] / noise_ema["g2"]
        if master_process and math.isfinite(b_simple):
            with open(log_file, "a") as f:
                f.write(f"{step} bsimple {b_simple:.1f}\n")
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # [fix] wait for the GPU; guarded so CPU/MPS runs don't crash
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        noise_col = f" | B_simple: {b_simple:,.0f}" if math.isfinite(b_simple) else ""
        # peak memory answers "does this B fit on this card" straight from the log,
        # which is the first thing you need to know on a machine you just rented
        mem_col = f" | mem: {torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if device_type == "cuda" else ""
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}{mem_col}{noise_col}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()







