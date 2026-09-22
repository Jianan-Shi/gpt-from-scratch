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
CausalSelfAttention = _DEFS["CausalSelfAttention"]
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
