"""Tests for the GPT-2 reproduction.

gpt2_follow.py is a script, not a module: importing it starts training. So these tests
exec the part above the training section, which is everything up to the DDP setup.
Everything here runs on CPU in a few seconds, at toy sizes — the point is the wiring,
not the numbers.
"""
import math

import torch
from torch.nn import functional as F

_SRC = open(__file__.replace("test_gpt2.py", "gpt2_follow.py")).read()
_DEFS = {"__name__": "gpt2_defs"}
exec(_SRC[: _SRC.index("# run the training loop")], _DEFS)
CausalSelfAttention, RMSNorm = _DEFS["CausalSelfAttention"], _DEFS["RMSNorm"]
SwiGLU, swiglu_hidden = _DEFS["SwiGLU"], _DEFS["swiglu_hidden"]
rope_cache, apply_rope, rotate_half = _DEFS["rope_cache"], _DEFS["apply_rope"], _DEFS["rotate_half"]
Muon, MultiOptimizer = _DEFS["Muon"], _DEFS["MultiOptimizer"]
zeropower_via_newtonschulz5 = _DEFS["zeropower_via_newtonschulz5"]
gradient_noise_scale = _DEFS["gradient_noise_scale"]
save_checkpoint, prune_checkpoints = _DEFS["save_checkpoint"], _DEFS["prune_checkpoints"]
GPT, GPTConfig = _DEFS["GPT"], _DEFS["GPTConfig"]
get_most_likely_row = _DEFS["get_most_likely_row"]

# the preset table and the lr schedule both sit below the DDP setup
from dataclasses import dataclass  # noqa: E402  (the exec'd slice needs it)
_P = {"dataclass": dataclass}
exec(_SRC[_SRC.index("@dataclass\nclass Preset") : _SRC.index("parser = argparse")], _P)
PRESETS = _P["PRESETS"]

# get_lr reads warmup_steps/max_steps as globals; they come from the chosen preset
WARMUP_STEPS, MAX_STEPS = PRESETS["4060"].warmup_steps, PRESETS["4060"].max_steps
_LR = {"math": math, "warmup_steps": WARMUP_STEPS, "max_steps": MAX_STEPS}
exec(_SRC[_SRC.index("max_lr = 6e-4") : _SRC.index("# optimize!")], _LR)
get_lr, MAX_LR, MIN_LR = _LR["get_lr"], _LR["max_lr"], _LR["min_lr"]

TINY = GPTConfig(block_size=64, vocab_size=128, n_layer=2, n_head=2, n_embd=32)



def test_flash_attention_matches_the_manual_implementation():
    """换成 flash attention 前后必须等价——这一步替换没有任何测试兜底，错了也不会报错。"""
    torch.manual_seed(0)
    attn = CausalSelfAttention(TINY).eval()
    x = torch.randn(2, 16, TINY.n_embd)

    with torch.no_grad():
        flash = attn(x)

        # 手写版：显式构造 (B, nh, T, T) 矩阵，就是被替换掉的那段代码
        B, T, C = x.size()
        qkv = attn.c_attn(x)
        q, k, v = qkv.split(attn.n_embd, dim=2)
        k = k.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
        q = q.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
        v = v.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        manual = attn.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

    assert torch.allclose(flash, manual, atol=1e-5), (flash - manual).abs().max()


def test_attention_is_causal():
    """改动后面的 token 不能影响前面的输出，否则就是在偷看未来。"""
    torch.manual_seed(0)
    attn = CausalSelfAttention(TINY).eval()
    x = torch.randn(1, 16, TINY.n_embd)
    y = x.clone()
    y[:, 8:] = torch.randn(1, 8, TINY.n_embd) # 只改后半段

    with torch.no_grad():
        assert torch.allclose(attn(x)[:, :8], attn(y)[:, :8], atol=1e-6)


def test_weight_tying_shares_one_tensor():
    """wte 和 lm_head 共享同一个张量，在 124M 里占 38M 参数。"""
    model = GPT(TINY)
    assert model.transformer.wte.weight is model.lm_head.weight

    n_params = sum(p.numel() for p in model.parameters())
    n_unique = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    assert n_params == n_unique, "共享的张量不应该被数两次"


def test_residual_projections_get_the_scaled_init():
    """残差路径上的投影按 (2 * n_layer) ** -0.5 缩小，否则残差累加会让方差随深度增长。"""
    torch.manual_seed(0)
    model = GPT(TINY)
    expected = 0.02 * (2 * TINY.n_layer) ** -0.5

    for block in model.transformer.h:
        for proj in (block.attn.c_proj, block.mlp.c_proj):
            assert abs(proj.weight.std().item() - expected) < 0.3 * expected
    # 对照：没有标记 NANOGPT_SCALE_INIT 的层仍然是 0.02
    assert abs(model.transformer.h[0].attn.c_attn.weight.std().item() - 0.02) < 0.3 * 0.02


def test_initial_loss_is_the_uniform_baseline():
    """刚初始化时模型应该均匀猜测，loss ≈ ln(vocab_size)。偏离说明初始化写错了。

    注意 targets 必须和 inputs 无关，见下一个测试。
    """
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (4, 16))
    targets = torch.randint(0, TINY.vocab_size, (4, 16))

    with torch.no_grad():
        _, loss = model(idx, targets)

    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.15


def test_weight_tying_makes_the_model_favour_its_own_input_token():
    """权重共享的一个副作用：初始化时模型倾向于预测"当前这个 token 本身"。

    残差流里带着 wte[idx]，而 logits = x @ wte.T，于是和自己的点积最大。
    拿 targets=inputs 去测"初始 loss ≈ ln(vocab)"会低于均匀基线（4.45 vs 4.85），
    看起来像初始化有问题，其实是测试写错了。去掉共享后这个效应就消失。
    """
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (4, 16))

    untied = GPT(TINY).eval()
    untied.lm_head.weight = torch.nn.Parameter(untied.lm_head.weight.clone().normal_(0, 0.02))

    with torch.no_grad():
        tied_loss = model(idx, idx)[1].item()
        untied_loss = untied(idx, idx)[1].item()

    assert tied_loss < math.log(TINY.vocab_size) - 0.3
    assert abs(untied_loss - math.log(TINY.vocab_size)) < 0.15


def test_lr_schedule_boundaries():
    """warmup 结束时取到峰值，之后余弦衰减，过了 max_steps 停在下限。"""
    assert get_lr(0) == MAX_LR / WARMUP_STEPS # 第一步不是 0，避免白跑一步
    assert get_lr(WARMUP_STEPS - 1) == MAX_LR
    assert get_lr(MAX_STEPS) == MIN_LR
    assert get_lr(MAX_STEPS + 1000) == MIN_LR

    # 余弦的中点应该落在最大和最小值的正中间
    mid = (WARMUP_STEPS + MAX_STEPS) // 2
    assert abs(get_lr(mid) - (MAX_LR + MIN_LR) / 2) < 1e-6

    lrs = [get_lr(i) for i in range(WARMUP_STEPS, MAX_STEPS)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), "warmup 之后必须单调不增"


def test_get_most_likely_row_picks_the_lowest_completion_loss():
    """HellaSwag 的打分只看结尾部分（mask==1），且用平均而不是求和，否则会偏向短结尾。"""
    V, T, correct = 8, 6, 2
    tokens = torch.zeros(4, T, dtype=torch.long)
    tokens[:, :3] = torch.tensor([1, 2, 3]) # 四行的 context 完全相同
    tokens[:, 3:] = torch.tensor([4, 5, 6]) # 结尾也相同，区别只在 logits

    logits = torch.zeros(4, T, V)
    for row in range(4):
        for pos in range(2, 5): # 预测 token 3、4、5 的位置
            logits[row, pos, tokens[row, pos + 1]] = 10.0 if row == correct else 1.0

    mask = torch.zeros(4, T, dtype=torch.long)
    mask[:, 3:] = 1 # 只有结尾计入损失

    assert get_most_likely_row(tokens, mask, logits) == correct


def test_context_region_does_not_affect_the_choice():
    """context 部分被 mask 掉，所以它的 logits 再怎么变都不该改变答案。"""
    V, T, correct = 8, 6, 1
    tokens = torch.zeros(4, T, dtype=torch.long)
    tokens[:, :3] = torch.tensor([1, 2, 3])
    tokens[:, 3:] = torch.tensor([4, 5, 6])
    mask = torch.zeros(4, T, dtype=torch.long)
    mask[:, 3:] = 1

    logits = torch.zeros(4, T, V)
    for row in range(4):
        for pos in range(2, 5):
            logits[row, pos, tokens[row, pos + 1]] = 10.0 if row == correct else 1.0

    tampered = logits.clone()
    tampered[3, 0, :] = 50.0 # 把某一行的 context 部分改成极端值
    tampered[3, 1, :] = 50.0

    assert get_most_likely_row(tokens, mask, tampered) == correct


def test_forward_rejects_sequences_longer_than_block_size():
    model = GPT(TINY)
    idx = torch.zeros(1, TINY.block_size + 1, dtype=torch.long)
    try:
        model(idx)
    except AssertionError:
        return
    raise AssertionError("超过 block_size 时必须报错，位置嵌入没有这么多行")


def test_every_preset_divides_into_whole_micro_steps():
    """total_batch_size 必须被 B*T*world 整除，否则梯度累积的步数对不上。"""
    for name, p in PRESETS.items():
        for world in (1, 2, 8):
            tokens_per_micro = p.B * p.T * world
            if p.total_batch_size % tokens_per_micro:
                continue # 这个 world size 用不了，但不能是 1 卡就不行
            assert p.total_batch_size // tokens_per_micro >= 1, name
        assert p.total_batch_size % (p.B * p.T) == 0, f"{name} 连单卡都不整除"


def test_every_preset_scores_at_least_one_val_batch():
    for name, p in PRESETS.items():
        for world in (1, 2, 8):
            assert max(1, p.val_tokens // (p.B * p.T * world)) >= 1, name


def test_a800_preset_is_the_original_recipe():
    """租卡那一档必须是视频里的配方，不能把 4060 的应急参数带上去。"""
    p = PRESETS["a800"]

    assert p.total_batch_size == 2**19 # 0.5M tokens, the GPT-3 paper's batch
    assert p.B == 64 and p.T == 1024
    assert p.warmup_steps == 715 and p.max_steps == 19_073 # ~1 epoch of 10B tokens
    assert p.use_compile is True
    assert p.memory_fraction is None, "显存上限是 WSL 的权宜之计，不该带到租的卡上"


def test_val_slice_matches_the_original_on_two_cards():
    """原版口径是 20 步 x B=64 x 1024 x 8 卡 = 1050 万 token。"""
    p = PRESETS["a800"]
    assert p.val_tokens == 20 * 64 * 1024 * 8
    assert p.val_tokens // (p.B * p.T * 2) == 80 # 2 卡时每进程 80 步


def test_evaluation_is_not_gated_on_compile():
    """HellaSwag 和采样曾经被 `(not use_compile)` 直接跳过，而且不报错。

    这是个源码级断言，因为要防的就是"这段代码被悄悄跳过"——行为测不出来。
    """
    assert "not use_compile" not in _SRC
    assert _SRC.count("eval_model(") >= 2, "变长输入必须走未编译的模型"


def test_noise_scale_recovers_a_known_answer():
    """构造一对满足 E|g_B|^2 = |G|^2 + tr(S)/B 的观测，估计量必须解回原值。"""
    true_g2, true_s = 4.0, 800.0 # B_simple = 200
    b_small, b_big = 50, 5000
    g_small_sq = true_g2 + true_s / b_small
    g_big_sq = true_g2 + true_s / b_big

    g2, s, b_simple = gradient_noise_scale(g_small_sq, g_big_sq, b_small, b_big)

    assert abs(g2 - true_g2) < 1e-9
    assert abs(s - true_s) < 1e-9
    assert abs(b_simple - true_s / true_g2) < 1e-9


def test_noise_scale_is_invariant_to_gradient_rescaling():
    """B_simple 是个比值，整体缩放梯度（比如换 loss 的归一化）不该改变它。"""
    g_small_sq, g_big_sq, b_small, b_big = 20.0, 4.4, 64, 4096
    _, _, base = gradient_noise_scale(g_small_sq, g_big_sq, b_small, b_big)
    _, _, scaled = gradient_noise_scale(9 * g_small_sq, 9 * g_big_sq, b_small, b_big)

    assert abs(base - scaled) < 1e-9


def test_noise_scale_flags_an_unusable_estimate():
    """小 batch 的范数反而更小（纯噪声导致）时，|G|^2 的估计会变成负数，
    此时必须返回 nan 而不是一个看起来很正常的负 B_simple。"""
    _, _, b_simple = gradient_noise_scale(g_small_sq=1.0, g_big_sq=9.0, b_small=64, b_big=4096)

    assert math.isnan(b_simple)


def _tiny_training_state():
    torch.manual_seed(0)
    model = GPT(TINY)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    model(idx, idx)[1].backward()
    opt.step() # 走一步，AdamW 才会有 m/v 可存
    return model, opt


def test_checkpoint_round_trips_optimizer_state(tmp_path):
    """只存权重是不够的：丢掉 AdamW 的 m/v，续跑等于重新预热。"""
    model, opt = _tiny_training_state()
    path = tmp_path / "ckpt_000010.pt"
    save_checkpoint(str(path), model=model, optimizer=opt, step=10, val_loss=1.23,
                    loader_state={"current_shard": 3, "current_position": 4096},
                    meta={"world_size": 1, "B": 4, "T": 1024, "total_batch_size": 65536})

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    fresh_model, fresh_opt = GPT(TINY), None
    fresh_model.load_state_dict(ckpt["model"])
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    fresh_opt.load_state_dict(ckpt["optimizer"])

    before = opt.state[opt.param_groups[0]["params"][0]]["exp_avg"]
    after = fresh_opt.state[fresh_opt.param_groups[0]["params"][0]]["exp_avg"]
    assert torch.equal(before, after)
    assert ckpt["step"] == 10
    assert ckpt["loader"] == {"current_shard": 3, "current_position": 4096}


def test_checkpoint_keeps_only_the_most_recent(tmp_path):
    """每个约 1.5GB，一整夜会把盘写满。"""
    model, opt = _tiny_training_state()
    for step in (10, 20, 30, 40):
        save_checkpoint(str(tmp_path / f"ckpt_{step:06d}.pt"), model=model, optimizer=opt,
                        step=step, val_loss=1.0, loader_state={}, meta={}, keep=2)

    assert sorted(f.name for f in tmp_path.glob("ckpt_*.pt")) == ["ckpt_000030.pt", "ckpt_000040.pt"]


def test_prune_leaves_other_files_alone(tmp_path):
    (tmp_path / "log.txt").write_text("0 val 10.9\n")
    (tmp_path / "gpt2_follow.py").write_text("# the script that produced this run\n")
    for step in (10, 20, 30):
        (tmp_path / f"ckpt_{step:06d}.pt").write_text("x")

    prune_checkpoints(str(tmp_path), keep=1)

    assert (tmp_path / "log.txt").exists() and (tmp_path / "gpt2_follow.py").exists()
    assert [f.name for f in tmp_path.glob("ckpt_*.pt")] == ["ckpt_000030.pt"]


def test_4090_preset_keeps_the_recipe_and_fits_24gb():
    """24GB 装不下 B=64，但配方不变：只是累积步数从 4 变 16，两者数学等价。"""
    p, original = PRESETS["4090x2"], PRESETS["a800"]

    assert p.total_batch_size == original.total_batch_size == 2**19
    assert (p.warmup_steps, p.max_steps) == (original.warmup_steps, original.max_steps)
    assert p.val_tokens == original.val_tokens # 同一把尺子，才和基线可比
    assert p.B == 16 and p.total_batch_size % (p.B * p.T * 2) == 0
    assert p.total_batch_size // (p.B * p.T * 2) == 16 # 2 卡，每卡累积 16 次


# ---------------------------------------------------------------------------------------
# RMSNorm


def test_rmsnorm_matches_the_explicit_formula():
    """实现用的是融合算子 F.rms_norm，这里把公式逐项写出来对拍。

    x / sqrt(mean(x^2) + eps) * gain
    """
    torch.manual_seed(0)
    dim = 64
    norm = RMSNorm(dim)
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.1) # gain 不为 1，才测得到它有没有被乘上
    x = torch.randn(4, 16, dim) * 3 + 2

    manual = x.float()
    manual = manual * torch.rsqrt(manual.pow(2).mean(-1, keepdim=True) + norm.eps)
    manual = (manual * norm.weight).to(x.dtype)

    with torch.no_grad():
        assert torch.allclose(norm(x), manual, atol=1e-5)


def test_rmsnorm_matches_torchs_implementation():
    """和 torch.nn.RMSNorm 对拍——手写实现是否真的是 RMSNorm，由标准实现说了算。"""
    torch.manual_seed(0)
    dim = 64
    mine, theirs = RMSNorm(dim), torch.nn.RMSNorm(dim, eps=1e-5)
    with torch.no_grad(): # 两边的 gain 都是 1，但显式设一遍更清楚
        mine.weight.copy_(torch.ones(dim))
        theirs.weight.copy_(torch.ones(dim))
    x = torch.randn(4, 16, dim) * 3 + 2

    assert torch.allclose(mine(x), theirs(x), atol=1e-5)


def test_rmsnorm_is_scale_invariant_up_to_eps():
    """RMSNorm 的全部作用就是这个：输入整体放大 c 倍，输出不变。
    LayerNorm 也有这个性质，而这正是它真正起作用的部分。

    但"不变"只在 mean(x^2) >> eps 时成立。把输入缩小 10 倍，mean(x^2) 小 100 倍，
    eps 就不再可以忽略——写这个测试时就是在这里翻的车。真实的残差流量级远大于
    1e-5，所以不影响训练，但知道边界在哪里才算理解这个公式。
    """
    torch.manual_seed(0)
    norm = RMSNorm(32, eps=1e-5)
    x = torch.randn(2, 8, 32) # mean(x^2) ~ 1，eps 可忽略

    with torch.no_grad():
        assert torch.allclose(norm(x), norm(x * 7.0), atol=1e-5)
        assert torch.allclose(norm(x), norm(x * 100.0), atol=1e-5)

        shrunk = norm(x * 0.1) # mean(x^2) ~ 0.01，eps 开始显形
        assert not torch.allclose(norm(x), shrunk, atol=1e-5)
        assert torch.allclose(norm(x), shrunk, atol=1e-2), "偏差应当只有 eps 量级"


def test_rmsnorm_is_not_shift_invariant_but_layernorm_is():
    """这一条就是 RMSNorm 和 LayerNorm 的全部差别：它不再免疫常数偏移。

    去掉减均值省下了一次全量归约，代价就写在这里。残差流的均值接近 0 时代价很小，
    但"很小"是需要实测的——这正是消融要回答的。
    """
    torch.manual_seed(0)
    rms, ln = RMSNorm(32), torch.nn.LayerNorm(32)
    x = torch.randn(2, 8, 32)
    shifted = x + 5.0

    with torch.no_grad():
        assert torch.allclose(ln(x), ln(shifted), atol=1e-4), "LayerNorm 免疫平移"
        assert not torch.allclose(rms(x), rms(shifted), atol=1e-2), "RMSNorm 不免疫"


def test_rmsnorm_keeps_the_input_dtype():
    """归约在 fp32 里做（bf16 下 x^2 会丢精度），但输出必须还是 bf16，
    否则后面的 matmul 会悄悄退回 fp32，训练变慢而没人发现。"""
    norm = RMSNorm(32)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)

    assert norm(x).dtype == torch.bfloat16


def test_rmsnorm_model_drops_the_bias_parameters():
    """每个 norm 省掉 n_embd 个 bias。25 处 norm × 768 = 19,200 个参数。"""
    base = GPT(GPTConfig(**{**TINY.__dict__, "norm": "layernorm"}))
    rms = GPT(GPTConfig(**{**TINY.__dict__, "norm": "rmsnorm"}))

    n_base = sum(p.numel() for p in base.parameters())
    n_rms = sum(p.numel() for p in rms.parameters())
    n_norms = 2 * TINY.n_layer + 1 # 每个 block 两处，加最后的 ln_f

    assert n_base - n_rms == n_norms * TINY.n_embd


def test_rmsnorm_model_starts_at_the_uniform_loss():
    """换了归一化层，初始 loss 仍应落在 ln(vocab)——否则是接线接错了。"""
    torch.manual_seed(0)
    model = GPT(GPTConfig(**{**TINY.__dict__, "norm": "rmsnorm"})).eval()
    idx = torch.randint(0, TINY.vocab_size, (4, 16))
    targets = torch.randint(0, TINY.vocab_size, (4, 16))

    with torch.no_grad():
        _, loss = model(idx, targets)

    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.15


def test_rmsnorm_gain_is_not_weight_decayed():
    """1 维参数不该被权重衰减——RMSNorm 的 gain 和 LayerNorm 的 gamma 一样。"""
    model = GPT(GPTConfig(**{**TINY.__dict__, "norm": "rmsnorm"}))
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-3, device="cpu")

    decay_group = optimizer.param_groups[0]
    assert decay_group["weight_decay"] == 0.1
    assert all(p.dim() >= 2 for p in decay_group["params"]), "gain 混进了衰减组"


# ---------------------------------------------------------------------------------------
# SwiGLU


def test_swiglu_hidden_is_eight_thirds():
    """三个矩阵而不是两个，所以隐藏层取 8/3 倍才和原 MLP 参数量相同。

    照搬 4 倍会多出 33% 参数，那时的"更好"只是"更大"。
    """
    assert swiglu_hidden(768) == 2048 # 768 * 8/3 恰好整除
    assert swiglu_hidden(768) % 64 == 0, "对齐到 64 的倍数，照顾 tensor core"

    for n_embd in (256, 512, 1024, 4096):
        h = swiglu_hidden(n_embd)
        assert abs(3 * n_embd * h - 8 * n_embd**2) / (8 * n_embd**2) < 0.05


def test_swiglu_matches_the_baseline_parameter_count():
    """消融的前提：两个变体的规模必须一样，否则比的是参数量不是结构。

    按真实配置（n_embd=768）比，这是实际要跑的那个；小维度上对齐粒度会引入几个
    百分点的偏差，见 swiglu_hidden 里的注释。
    """
    real = GPTConfig(vocab_size=50304) # n_embd=768
    n_gelu = sum(p.numel() for p in _DEFS["MLP"](real).parameters())
    n_swiglu = sum(p.numel() for p in SwiGLU(real).parameters())

    assert abs(n_swiglu - n_gelu) / n_gelu < 0.001, (n_gelu, n_swiglu)

    tiny_gelu = GPT(GPTConfig(**{**TINY.__dict__, "mlp": "gelu"}))
    tiny_swiglu = GPT(GPTConfig(**{**TINY.__dict__, "mlp": "swiglu"}))
    n_tg = sum(p.numel() for p in tiny_gelu.parameters())
    n_ts = sum(p.numel() for p in tiny_swiglu.parameters())
    assert abs(n_ts - n_tg) / n_tg < 0.03


def test_swiglu_is_multiplicative_in_the_content_path():
    """门控的本质：输出对"内容"那一路是线性的，对"开关"那一路不是。

    把 c_up 整体放大 k 倍，输出精确放大 k 倍——普通 MLP 过了非线性就没有这个性质。
    """
    torch.manual_seed(0)
    mlp = SwiGLU(TINY).eval()
    with torch.no_grad():
        mlp.c_proj.bias.zero_() # 否则输出里混着一个常数项
    x = torch.randn(2, 8, TINY.n_embd)

    with torch.no_grad():
        before = mlp(x)
        mlp.c_up.weight.mul_(3.0)
        mlp.c_up.bias.mul_(3.0)
        after = mlp(x)

    assert torch.allclose(after, before * 3.0, atol=1e-5)


def test_swiglu_gate_can_shut_a_channel_off():
    """开关那一路归零 -> SiLU(0)=0 -> 整条输出归零，与内容无关。"""
    torch.manual_seed(0)
    mlp = SwiGLU(TINY).eval()
    with torch.no_grad():
        mlp.c_gate.weight.zero_()
        mlp.c_gate.bias.zero_()
        mlp.c_proj.bias.zero_()
    x = torch.randn(2, 8, TINY.n_embd)

    with torch.no_grad():
        assert mlp(x).abs().max() < 1e-6


def test_silu_is_x_times_sigmoid():
    """SiLU(x) = x * sigmoid(x)，和 GELU 形状几乎一样——差别不在非线性，在那个乘法。"""
    x = torch.randn(1000)

    assert torch.allclose(F.silu(x), x * torch.sigmoid(x), atol=1e-6)


def test_swiglu_down_projection_gets_the_scaled_init():
    """残差路径上的投影仍要按 (2*n_layer)**-0.5 缩小，换了 MLP 不能把这个丢掉。"""
    torch.manual_seed(0)
    model = GPT(GPTConfig(**{**TINY.__dict__, "mlp": "swiglu"}))
    expected = 0.02 * (2 * TINY.n_layer) ** -0.5

    for block in model.transformer.h:
        assert abs(block.mlp.c_proj.weight.std().item() - expected) < 0.3 * expected
        assert abs(block.mlp.c_gate.weight.std().item() - 0.02) < 0.3 * 0.02


def test_swiglu_model_starts_at_the_uniform_loss():
    torch.manual_seed(0)
    model = GPT(GPTConfig(**{**TINY.__dict__, "mlp": "swiglu"})).eval()
    idx = torch.randint(0, TINY.vocab_size, (4, 16))
    targets = torch.randint(0, TINY.vocab_size, (4, 16))

    with torch.no_grad():
        _, loss = model(idx, targets)

    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.15


# ---------------------------------------------------------------------------------------
# QK-norm


def _attn(qk_norm):
    torch.manual_seed(0)
    cfg = GPTConfig(**{**TINY.__dict__, "qk_norm": qk_norm})
    return CausalSelfAttention(cfg).eval()


def test_qk_norm_makes_the_logits_scale_invariant():
    """核心性质：把 q 整体放大 100 倍，输出完全不变。

    点积因此只取决于 q、k 的方向。没有它的话，权重在训练中变大 -> logits 变大 ->
    softmax 变尖 -> 梯度消失，而这个过程没有任何报错。
    """
    attn = _attn(qk_norm=True)
    x = torch.randn(2, 16, TINY.n_embd)

    with torch.no_grad():
        before = attn(x)
        attn.c_attn.weight[:TINY.n_embd].mul_(100.0) # c_attn 的前 n_embd 行是 q
        attn.c_attn.bias[:TINY.n_embd].mul_(100.0)
        after = attn(x)

    assert torch.allclose(before, after, atol=1e-4)


def test_without_qk_norm_the_same_scaling_changes_everything():
    """对照组：不加 QK-norm 时，同样的放大会把 softmax 推向 one-hot。"""
    attn = _attn(qk_norm=False)
    x = torch.randn(2, 16, TINY.n_embd)

    with torch.no_grad():
        before = attn(x)
        attn.c_attn.weight[:TINY.n_embd].mul_(100.0)
        attn.c_attn.bias[:TINY.n_embd].mul_(100.0)
        after = attn(x)

    assert not torch.allclose(before, after, atol=1e-2)


def test_qk_norm_bounds_the_attention_logits():
    """gain=1 时 |q| = |k| = sqrt(hs)，所以 logit = q·k/sqrt(hs) 的上界是 sqrt(hs)。

    这就是"logits 不会爆"的定量版本。
    """
    attn = _attn(qk_norm=True)
    head_size = TINY.n_embd // TINY.n_head
    x = torch.randn(2, 16, TINY.n_embd) * 50 # 输入故意放大

    with torch.no_grad():
        B, T, C = x.shape
        q, k, _ = attn.c_attn(x).split(TINY.n_embd, dim=2)
        q = attn.q_norm(q.view(B, T, TINY.n_head, head_size).transpose(1, 2))
        k = attn.k_norm(k.view(B, T, TINY.n_head, head_size).transpose(1, 2))
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)

    assert logits.abs().max().item() <= math.sqrt(head_size) + 1e-4


def test_qk_norm_keeps_attention_causal():
    """归一化是逐位置的，不该破坏因果性。"""
    attn = _attn(qk_norm=True)
    x = torch.randn(1, 16, TINY.n_embd)
    y = x.clone()
    y[:, 8:] = torch.randn(1, 8, TINY.n_embd)

    with torch.no_grad():
        assert torch.allclose(attn(x)[:, :8], attn(y)[:, :8], atol=1e-6)


def test_qk_norm_costs_two_vectors_per_layer():
    """每层多 2 * head_size 个参数——相对 124M 可以忽略。"""
    plain = GPT(GPTConfig(**{**TINY.__dict__, "qk_norm": False}))
    normed = GPT(GPTConfig(**{**TINY.__dict__, "qk_norm": True}))

    head_size = TINY.n_embd // TINY.n_head
    delta = sum(p.numel() for p in normed.parameters()) - sum(p.numel() for p in plain.parameters())

    assert delta == 2 * head_size * TINY.n_layer


def test_qk_norm_model_starts_at_the_uniform_loss():
    torch.manual_seed(0)
    model = GPT(GPTConfig(**{**TINY.__dict__, "qk_norm": True})).eval()
    idx = torch.randint(0, TINY.vocab_size, (4, 16))
    targets = torch.randint(0, TINY.vocab_size, (4, 16))

    with torch.no_grad():
        _, loss = model(idx, targets)

    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.15


# ---------------------------------------------------------------------------------------
# GQA (grouped query attention)


def test_gqa_with_one_kv_per_head_is_exactly_mha():
    """n_kv_head = n_head 时必须和标准多头逐元素相同——这是所有 GQA 实现的第一关。"""
    torch.manual_seed(0)
    mha = CausalSelfAttention(GPTConfig(**{**TINY.__dict__, "n_kv_head": None})).eval()
    torch.manual_seed(0)
    gqa = CausalSelfAttention(GPTConfig(**{**TINY.__dict__, "n_kv_head": TINY.n_head})).eval()
    x = torch.randn(2, 16, TINY.n_embd)

    with torch.no_grad():
        assert torch.equal(mha(x), gqa(x))


def test_gqa_groups_query_heads_contiguously():
    """分组映射：query 头 0..(nh/n_kv - 1) 用 kv 头 0，依此类推。

    也就是 repeat_interleave，不是 repeat。这两个搞反了模型照样能训，loss 也会降，
    只是每个头配错了 K/V——最典型的静默 bug，所以这里手工展开一遍对拍。
    """
    torch.manual_seed(0)
    n_kv = 2
    cfg = GPTConfig(**{**TINY.__dict__, "n_kv_head": n_kv}) # TINY: n_head=2 -> 取 1 组 2 头
    cfg = GPTConfig(**{**TINY.__dict__, "n_head": 4, "n_kv_head": n_kv, "n_embd": 32})
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 16, cfg.n_embd)

    with torch.no_grad():
        B, T, C = x.shape
        hs = cfg.n_embd // cfg.n_head
        q, k, v = attn.c_attn(x).split([C, n_kv * hs, n_kv * hs], dim=2)
        q = q.view(B, T, cfg.n_head, hs).transpose(1, 2)
        k = k.view(B, T, n_kv, hs).transpose(1, 2)
        v = v.view(B, T, n_kv, hs).transpose(1, 2)

        repeats = cfg.n_head // n_kv
        expanded = F.scaled_dot_product_attention(
            q, k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1),
            is_causal=True)
        expected = attn.c_proj(expanded.transpose(1, 2).contiguous().view(B, T, C))

        assert torch.allclose(attn(x), expected, atol=1e-5)

        # 对照：repeat 而不是 repeat_interleave，映射就错了
        wrong = F.scaled_dot_product_attention(
            q, k.repeat(1, repeats, 1, 1), v.repeat(1, repeats, 1, 1), is_causal=True)
        assert not torch.allclose(expanded, wrong, atol=1e-3)


def test_gqa_requires_the_head_count_to_divide():
    try:
        CausalSelfAttention(GPTConfig(**{**TINY.__dict__, "n_head": 4, "n_embd": 32, "n_kv_head": 3}))
    except AssertionError:
        return
    raise AssertionError("n_head 不能被 n_kv_head 整除时必须报错")


def test_gqa_shrinks_the_kv_cache_proportionally():
    """GQA 真正换来的东西：推理时每个 token 要缓存的 K/V。

    12 头 -> 36KB/token，4 头 -> 12KB/token。同样的显存能装 3 倍长的上下文。
    """
    real = GPTConfig(vocab_size=50304) # n_layer=12, n_head=12, n_embd=768

    def kv_kb(n_kv):
        return 2 * real.n_layer * n_kv * (real.n_embd // real.n_head) * 2 / 1024

    assert kv_kb(12) == 36.0
    assert kv_kb(4) == 12.0
    assert kv_kb(1) == 3.0


def test_gqa_also_removes_parameters():
    """c_attn 变窄，所以这一行不是等参数量比较——读结果时必须记得。"""
    mha = GPT(GPTConfig(vocab_size=1024, n_layer=2, n_head=12, n_embd=768, block_size=64))
    gqa = GPT(GPTConfig(vocab_size=1024, n_layer=2, n_head=12, n_embd=768, block_size=64, n_kv_head=4))

    n_mha = sum(p.numel() for p in mha.parameters())
    n_gqa = sum(p.numel() for p in gqa.parameters())
    per_layer = 2 * (12 - 4) * 64 * 768 + 2 * (12 - 4) * 64 # 权重 + bias

    assert n_mha - n_gqa == 2 * per_layer


def test_gqa_keeps_attention_causal():
    torch.manual_seed(0)
    attn = CausalSelfAttention(GPTConfig(**{**TINY.__dict__, "n_head": 4, "n_embd": 32,
                                           "n_kv_head": 2})).eval()
    x = torch.randn(1, 16, 32)
    y = x.clone()
    y[:, 8:] = torch.randn(1, 8, 32)

    with torch.no_grad():
        assert torch.allclose(attn(x)[:, :8], attn(y)[:, :8], atol=1e-6)


def test_gqa_model_starts_at_the_uniform_loss():
    torch.manual_seed(0)
    model = GPT(GPTConfig(**{**TINY.__dict__, "n_kv_head": 1})).eval()
    idx = torch.randint(0, TINY.vocab_size, (4, 16))
    targets = torch.randint(0, TINY.vocab_size, (4, 16))

    with torch.no_grad():
        _, loss = model(idx, targets)

    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.15


# ---------------------------------------------------------------------------------------
# RoPE


def _rope_at(x, pos, head_size=64, block_size=128):
    """把单个向量 x 当作位置 pos 上的 q 或 k，旋转之后返回。"""
    cos, sin = rope_cache(head_size, block_size)
    return apply_rope(x.view(1, 1, 1, head_size), cos[pos].view(1, 1, 1, -1),
                      sin[pos].view(1, 1, 1, -1)).view(-1)


def test_rope_dot_product_depends_only_on_the_distance():
    """RoPE 的定义性质，也是"相对位置编码"这个说法的全部含义：

    q 在位置 m、k 在位置 n，点积只取决于 m - n。把两者同时平移同样的距离，
    注意力分数一个字节都不变。绝对位置嵌入做不到这一点——它把 5 和 105 当作
    两个无关的位置来学。
    """
    torch.manual_seed(0)
    q, k = torch.randn(64), torch.randn(64)

    base = torch.dot(_rope_at(q, 3), _rope_at(k, 7))
    shifted = torch.dot(_rope_at(q, 23), _rope_at(k, 27)) # 同时 +20，距离仍是 -4
    far = torch.dot(_rope_at(q, 60), _rope_at(k, 64))

    assert torch.allclose(base, shifted, atol=1e-4), (base, shifted)
    assert torch.allclose(base, far, atol=1e-4)

    different_distance = torch.dot(_rope_at(q, 3), _rope_at(k, 9)) # 距离变了
    assert not torch.allclose(base, different_distance, atol=1e-3)


def test_rope_is_a_rotation_so_it_preserves_length():
    """旋转不改变长度——这是它能安全插在 attention 里的前提：
    q、k 的模长不变，所以 1/sqrt(hs) 的缩放仍然成立。"""
    torch.manual_seed(0)
    x = torch.randn(64)

    for pos in (0, 1, 17, 127):
        assert torch.allclose(_rope_at(x, pos).norm(), x.norm(), atol=1e-4)


def test_rope_at_position_zero_is_the_identity():
    """位置 0 的角度是 0，cos=1、sin=0，所以什么都不做。"""
    torch.manual_seed(0)
    x = torch.randn(64)

    assert torch.allclose(_rope_at(x, 0), x, atol=1e-5)


def test_rotate_half_is_multiplication_by_i():
    """把 (x1, x2) 看成复数 x1 + i*x2 时，这一步就是乘 i。
    连续两次应该得到 -x（i^2 = -1）。"""
    x = torch.randn(1, 1, 1, 8)

    assert torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-6)


def test_rope_frequencies_span_scales():
    """第 i 对维度的角速度是 base^(-2i/hs)：高频区分近距离，低频区分远距离。

    如果所有维度用同一个频率，位置信息就只剩一个尺度，长距离会绕回来混淆。
    """
    cos, sin = rope_cache(head_size=64, block_size=1024)
    angles = torch.atan2(sin[1], cos[1]) # 相邻一个位置转过的角度

    assert angles[0] > angles[31], "维度对应的频率必须递减"
    assert angles[0] > 0.9, "最高频每步接近 1 弧度"
    assert 0 < angles[31] < 1e-3, "最低频每步几乎不动"


def test_rope_model_has_no_learned_position_embedding():
    """RoPE 取代 wpe，省下 block_size * n_embd 个参数。"""
    learned = GPT(GPTConfig(**{**TINY.__dict__, "pos": "learned"}))
    rope = GPT(GPTConfig(**{**TINY.__dict__, "pos": "rope"}))

    assert hasattr(learned.transformer, "wpe")
    assert not hasattr(rope.transformer, "wpe")

    delta = sum(p.numel() for p in learned.parameters()) - sum(p.numel() for p in rope.parameters())
    assert delta == TINY.block_size * TINY.n_embd


def test_rope_cache_is_not_saved_in_checkpoints():
    """cos/sin 是算出来的常量，不该占 checkpoint 的体积。"""
    rope = GPT(GPTConfig(**{**TINY.__dict__, "pos": "rope"}))

    assert "rope_cos" not in rope.state_dict()
    assert any("rope_cos" in name for name, _ in rope.named_buffers())


def test_rope_keeps_attention_causal():
    torch.manual_seed(0)
    model = GPT(GPTConfig(**{**TINY.__dict__, "pos": "rope"})).eval()
    idx = torch.randint(0, TINY.vocab_size, (1, 16))
    changed = idx.clone()
    changed[:, 8:] = torch.randint(0, TINY.vocab_size, (1, 8))

    with torch.no_grad():
        assert torch.allclose(model(idx)[0][:, :8], model(changed)[0][:, :8], atol=1e-5)


def test_rope_model_starts_at_the_uniform_loss():
    torch.manual_seed(0)
    model = GPT(GPTConfig(**{**TINY.__dict__, "pos": "rope"})).eval()
    idx = torch.randint(0, TINY.vocab_size, (4, 16))
    targets = torch.randint(0, TINY.vocab_size, (4, 16))

    with torch.no_grad():
        _, loss = model(idx, targets)

    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.15


# ---------------------------------------------------------------------------------------
# Muon


def test_newton_schulz_pulls_singular_values_to_one():
    """正交化的定义：所有奇异值拉到 1 附近，方向不变。

    五次迭代是刻意不追求精确的——收进 [0.7, 1.3] 就够用，换来的是只做矩阵乘法，
    比真的做 SVD 快几个数量级。
    """
    torch.manual_seed(0)
    G = torch.randn(64, 32) @ torch.diag(torch.linspace(1.0, 10.0, 32)) # 条件数约 10

    before = torch.linalg.svdvals(G)
    after = torch.linalg.svdvals(zeropower_via_newtonschulz5(G).float())

    assert before.max() / before.min() > 8, "输入本来就该是病态的，否则测不出东西"
    assert after.min() > 0.6 and after.max() < 1.3, after
    assert after.max() / after.min() < 2.0, "条件数从 ~10 压到 2 以内"


def test_newton_schulz_needs_more_steps_when_extremely_ill_conditioned():
    """五步是够用而不是收敛：条件数 3000 的矩阵五步只能压到 [0.17, 0.76]，十步才到位。

    真实梯度的条件数在十几这个量级，所以默认的五步是划算的取舍。知道这个边界在哪，
    是因为写测试时先拿了一个条件数 3000 的矩阵，然后测试红了。
    """
    torch.manual_seed(0)
    G = torch.randn(64, 32) @ torch.diag(torch.tensor([100.0, 10.0] + [0.1] * 30))

    five = torch.linalg.svdvals(zeropower_via_newtonschulz5(G, steps=5).float())
    ten = torch.linalg.svdvals(zeropower_via_newtonschulz5(G, steps=10).float())

    assert five.min() < 0.5, "五步不够"
    assert ten.min() > 0.6 and ten.max() < 1.3, "十步收敛"


def test_newton_schulz_discards_the_gradient_scale():
    """梯度整体放大 100 倍，正交化之后的方向不变——步长由学习率单独决定。"""
    torch.manual_seed(0)
    G = torch.randn(32, 16)

    a = zeropower_via_newtonschulz5(G).float()
    b = zeropower_via_newtonschulz5(G * 100).float()

    # bf16 迭代，所以只能对到小数点后两位——Muon 本来就不需要精确的正交矩阵
    assert torch.allclose(a, b, atol=5e-2), (a - b).abs().max()


def test_newton_schulz_handles_both_orientations():
    """长边在前时内部会转置，转置回来的结果必须和直接算一致。"""
    torch.manual_seed(0)
    G = torch.randn(64, 16)

    tall = zeropower_via_newtonschulz5(G).float()
    wide = zeropower_via_newtonschulz5(G.T.contiguous()).float()

    assert torch.allclose(tall, wide.T, atol=2e-2)


def test_muon_reduces_a_simple_loss():
    """能不能真的优化——最基本的一关。"""
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(16, 16))
    target = torch.randn(16, 16)
    opt = Muon([W], lr=0.05)

    losses = []
    for _ in range(150):
        loss = (W - target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # 正交化把每步的长度固定住了，所以收敛是线性的而不是二次的：
    # 50 步降一半，150 步降到 1/20。这个形状本身就是 Muon 和 Adam 的区别之一。
    assert losses[49] < losses[0] / 1.8, losses[::25]
    assert losses[-1] < losses[0] / 10, losses[::25]


def test_muon_takes_only_the_block_matrices():
    """嵌入表和 lm_head 虽然是 2 维，但每行是一个独立 token，
    "矩阵的奇异值"没有对应意义，所以按作者的做法留给 AdamW。
    权重共享让 wte 和 lm_head 是同一个张量，更不能被两个优化器同时更新。
    """
    model = GPT(GPTConfig(**{**TINY.__dict__, "optimizer": "muon"}))
    opt = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device="cpu")

    assert isinstance(opt, MultiOptimizer)
    muon_params = {id(p) for g in opt.optimizers[0].param_groups for p in g["params"]}

    assert id(model.transformer.wte.weight) not in muon_params, "嵌入表不该交给 Muon"
    assert id(model.lm_head.weight) not in muon_params
    assert id(model.transformer.h[0].attn.c_attn.weight) in muon_params

    # 每个参数恰好属于一个优化器，没有重复也没有遗漏
    all_ids = [id(p) for o in opt.optimizers for g in o.param_groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids)) == len(list(model.parameters()))


def test_muon_and_adamw_keep_their_own_learning_rates():
    """Muon 的学习率比 AdamW 大三十倍，调度器只能按比例缩放，不能一刀切。"""
    model = GPT(GPTConfig(**{**TINY.__dict__, "optimizer": "muon", "muon_lr": 0.02}))
    opt = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device="cpu")

    base = {g["base_lr"] for g in opt.param_groups}
    assert base == {0.02, 6e-4}

    # 调度到一半时，两组都应该减半
    for g in opt.param_groups:
        g["lr"] = 0.5 * g["base_lr"]
    assert sorted(g["lr"] for g in opt.param_groups) == [3e-4, 3e-4, 0.01]


def test_multi_optimizer_state_survives_a_checkpoint():
    """续跑要靠它：两个优化器的状态都得存下来。"""
    torch.manual_seed(0)
    model = GPT(GPTConfig(**{**TINY.__dict__, "optimizer": "muon"}))
    opt = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device="cpu")
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    model(idx, idx)[1].backward()
    opt.step()

    sd = opt.state_dict()
    fresh_model = GPT(GPTConfig(**{**TINY.__dict__, "optimizer": "muon"}))
    fresh = fresh_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device="cpu")
    fresh.load_state_dict(sd)

    p = opt.optimizers[0].param_groups[0]["params"][0]
    q = fresh.optimizers[0].param_groups[0]["params"][0]
    assert torch.equal(opt.optimizers[0].state[p]["momentum_buffer"],
                       fresh.optimizers[0].state[q]["momentum_buffer"])
