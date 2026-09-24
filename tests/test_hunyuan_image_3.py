"""Step 2 unit tests for comfy/ldm/hunyuan_image_3: attention block, MoE FFN and the 2D rope.

Weights are random; these pin shapes, dtypes and the semantics the reference implementation
defines (interleaved qkv split, rope before qk norm, contiguous SwiGLU chunk, weight-dtype router).
"""
import math

import pytest
import torch
import torch.nn.functional as F

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ops
import comfy.quant_ops
from op_module_init import init_op_modules
from comfy.ldm.hunyuan_image_3.model import (
    HunyuanImage3Attention,
    HunyuanImage3DecoderLayer,
    HunyuanImage3MoE,
    HunyuanImage3Params,
    HunyuanImage3MoEGate,
    _split_qkv,
    _swiglu,
    build_rope_freqs,
)


def ops_for(dtype):
    return comfy.ops.mixed_precision_ops({}, dtype)


def make_params(**overrides):
    params = dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_head_dim=8,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        max_position_embeddings=1024,
        attention_bias=False,
        mlp_bias=False,
        moe_intermediate_size=16,
        num_experts=8,
        moe_topk=2,
        num_shared_expert=1,
        cfg_distilled=True,
        use_meanflow=True,
        model_type="instruct_distil",
        sequence_template="instruct",
        pad_token_id=0,
        image_token_id=128006,
        patch_size=1,
        patch_embed_hidden_dim=8,
        image_base_size=1024,
        vae_latent_channels=4,
        vae_downsample_factor=(16, 16),
        vit_aligner={"projector_type": "mlp_gelu", "input_dim": 8, "n_embed": 8, "depth": 2},
    )
    params.update(overrides)
    return HunyuanImage3Params(**params)


# ---------------------------------------------------------------- rope


def reference_rope_angles(seq_len, head_dim, image_section=None):
    """Independent re-derivation of the reference's 2D rope angles: (seq_len, head_dim // 2)."""
    pairs = head_dim // 2
    angle = torch.zeros(seq_len, pairs, dtype=torch.float64)
    for token in range(seq_len):
        if image_section is not None and image_section[0].start <= token < image_section[0].stop:
            start = image_section[0].start
            height, width = image_section[1]
            offset = token - start
            y = start + (width * height - height) / 2 + offset // width
            x = start + (width * height - width) / 2 + offset % width
        else:
            y = x = token
        for k in range(pairs):
            position = y if k % 2 == 0 else x
            angle[token, k] = position * 10000.0 ** (-2 * k / head_dim)
    return angle


def apply_reference_rope(x, angles):
    """Rotate dims (k, k + head_dim // 2) by angles[:, k] — the split-half convention."""
    half = x.shape[-1] // 2
    cos = torch.cos(angles).to(x.dtype)
    sin = torch.sin(angles).to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def test_rope_freqs_match_reference():
    head_dim, seq_len = 16, 12
    section = (slice(4, 10), (2, 3))
    freqs = build_rope_freqs(seq_len, head_dim, [section], 10000.0)
    assert freqs.shape == (1, 1, seq_len, head_dim // 2, 2, 2)

    angles = reference_rope_angles(seq_len, head_dim, image_section=section)
    cos = torch.cos(angles).to(freqs.dtype)
    sin = torch.sin(angles).to(freqs.dtype)
    assert torch.allclose(freqs[0, 0, :, :, 0, 0], cos, atol=1e-6)
    assert torch.allclose(freqs[0, 0, :, :, 0, 1], -sin, atol=1e-6)
    assert torch.allclose(freqs[0, 0, :, :, 1, 0], sin, atol=1e-6)
    assert torch.allclose(freqs[0, 0, :, :, 1, 1], cos, atol=1e-6)


def test_rope_rotation_matches_reference_scheme():
    head_dim, seq_len = 16, 12
    section = (slice(4, 10), (2, 3))
    freqs = build_rope_freqs(seq_len, head_dim, [section], 10000.0)
    angles = reference_rope_angles(seq_len, head_dim, image_section=section)

    x = torch.randn(1, 2, seq_len, head_dim)
    import comfy.quant_ops
    rotated, _ = comfy.quant_ops.ck.apply_rope_split_half(x, x, freqs)
    expected = apply_reference_rope(x, angles)
    assert torch.allclose(rotated, expected, atol=1e-5)


# ---------------------------------------------------------------- attention


def test_qkv_split_is_group_interleaved():
    kv_heads, groups, head_dim = 2, 4, 4
    qkv = torch.arange((groups + 2) * kv_heads * head_dim, dtype=torch.float32).reshape(1, 1, -1)
    query, key, value = _split_qkv(qkv, kv_heads, groups, head_dim)

    def row(kv_head, slot, dim):
        return kv_head * (groups + 2) * head_dim + slot * head_dim + dim

    # q head (kv_head * groups + slot) takes rows from its own group
    for kv_head in range(kv_heads):
        for slot in range(groups):
            head = kv_head * groups + slot
            for dim in range(head_dim):
                assert query[0, 0, head, dim].item() == row(kv_head, slot, dim)
        for dim in range(head_dim):
            assert key[0, 0, kv_head, dim].item() == row(kv_head, groups, dim)
            assert value[0, 0, kv_head, dim].item() == row(kv_head, groups + 1, dim)

    # the contiguous convention (all q, then all k, then all v) would place these rows elsewhere
    heads = kv_heads * groups
    contiguous_q_head_4 = groups * head_dim
    contiguous_k_head_1 = heads * head_dim + head_dim
    contiguous_v_head_0 = (heads + kv_heads) * head_dim
    assert query[0, 0, groups, 0].item() != contiguous_q_head_4
    assert key[0, 0, 1, 0].item() != contiguous_k_head_1
    assert value[0, 0, 0, 0].item() != contiguous_v_head_0
    assert key[0, 0, 1, 0].item() == row(1, groups, 0) == 40
    assert value[0, 0, 0, 0].item() == row(0, groups + 1, 0) == 20


def reference_attention_forward(module, hidden_states, angles, attention_mask=None):
    """Independent implementation of the reference attention math (fp32), given rope angles."""
    bsz, q_len, _ = hidden_states.shape
    qkv = torch.nn.functional.linear(hidden_states, module.qkv_proj.weight)
    query, key, value = _split_qkv(qkv, module.num_key_value_heads, module.num_key_value_groups, module.head_dim)
    query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    query = apply_reference_rope(query, angles)
    key = apply_reference_rope(key, angles)

    query = F.rms_norm(query, (module.head_dim,), weight=module.query_layernorm.weight, eps=module.query_layernorm.eps)
    key = F.rms_norm(key, (module.head_dim,), weight=module.key_layernorm.weight, eps=module.key_layernorm.eps)

    key = key.repeat_interleave(module.num_key_value_groups, dim=1)
    value = value.repeat_interleave(module.num_key_value_groups, dim=1)

    scores = query @ key.transpose(-1, -2) / math.sqrt(module.head_dim)
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :q_len, :q_len]
    out = torch.softmax(scores.float(), dim=-1).to(value.dtype) @ value
    out = out.transpose(1, 2).reshape(bsz, q_len, -1)
    return torch.nn.functional.linear(out, module.o_proj.weight)


def test_attention_forward_matches_reference_math():
    torch.manual_seed(0)
    config = make_params()
    dtype = torch.float32
    module = HunyuanImage3Attention(config, dtype=dtype, operations=ops_for(dtype))
    init_op_modules(module, dtype)

    seq_len, image = 10, (slice(3, 9), (2, 3))
    hidden_states = torch.randn(1, seq_len, config.hidden_size)
    freqs = build_rope_freqs(seq_len, config.attention_head_dim, [image], config.rope_theta)
    angles = reference_rope_angles(seq_len, config.attention_head_dim, image_section=image)

    got = module(hidden_states, freqs)
    expected = reference_attention_forward(module, hidden_states, angles)
    assert got.shape == hidden_states.shape
    assert got.dtype == hidden_states.dtype
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-4)


def test_kv_cache_reproduces_the_masked_full_sequence_attention():
    """The autoregressive stage's cache must be an optimisation, not a behaviour change.

    The diffusion path recomputes its whole sequence every step — that is its semantics — while the text
    stage keeps keys and values and feeds one token at a time. If the two disagree the CoT stage produces a
    fluent continuation of the wrong thing, and nothing downstream would attribute that to the cache. The
    cached tensors are the per-key/value-head ones, stored before the GQA expansion.
    """
    torch.manual_seed(0)
    config = make_params()
    dtype = torch.float32
    module = HunyuanImage3Attention(config, dtype=dtype, operations=ops_for(dtype))
    init_op_modules(module, dtype)

    seq_len = 10
    hidden_states = torch.randn(1, seq_len, config.hidden_size)
    freqs = build_rope_freqs(seq_len, config.attention_head_dim, [], config.rope_theta)
    mask = torch.zeros(1, 1, seq_len, seq_len).masked_fill(
        ~torch.ones(seq_len, seq_len, dtype=torch.bool).tril(), float("-inf"))

    whole = module(hidden_states, freqs, mask)

    cache = [None, None]
    steps = [module(hidden_states[:, :1], freqs[:, :, :1], None, cache)]
    for position in range(1, seq_len):
        steps.append(module(hidden_states[:, position:position + 1], freqs[:, :, position:position + 1],
                            None, cache))
    stepwise = torch.cat(steps, dim=1)

    assert torch.allclose(whole, stepwise, atol=1e-5, rtol=1e-4)
    assert cache[0].shape[-2] == seq_len and cache[1].shape[-2] == seq_len
    assert cache[0].shape[1] == config.num_key_value_heads


def test_attention_honours_additive_mask():
    torch.manual_seed(0)
    config = make_params()
    dtype = torch.float32
    module = HunyuanImage3Attention(config, dtype=dtype, operations=ops_for(dtype))
    init_op_modules(module, dtype)

    seq_len, blocked = 6, 2
    hidden_states = torch.randn(1, seq_len, config.hidden_size)
    freqs = build_rope_freqs(seq_len, config.attention_head_dim, [], config.rope_theta)
    angles = reference_rope_angles(seq_len, config.attention_head_dim)

    mask = torch.zeros(1, 1, seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i or (j == blocked and i != blocked):   # causal, plus one blocked key
                mask[0, 0, i, j] = float("-inf")
    assert not (mask[0, 0] == float("-inf")).all(dim=-1).any()  # no fully masked query row

    got = module(hidden_states, freqs, mask)
    expected = reference_attention_forward(module, hidden_states, angles, mask)
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-4)

    # changing the blocked token's input must not change any other position's output
    masked_input = hidden_states.clone()
    masked_input[0, blocked] = torch.randn(config.hidden_size)
    after = module(masked_input, freqs, mask)
    keep = [i for i in range(seq_len) if i != blocked]
    assert torch.allclose(after[0, keep], got[0, keep], atol=1e-6)
    assert not torch.allclose(after[0, blocked], got[0, blocked], atol=1e-6)

    # without the mask the change does propagate
    unmasked_after = module(masked_input, freqs)
    assert not torch.allclose(unmasked_after[0, keep], module(hidden_states, freqs)[0, keep], atol=1e-6)


def test_attention_real_head_configuration_shapes():
    torch.manual_seed(0)
    config = make_params(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8, attention_head_dim=128)
    dtype = torch.bfloat16
    module = HunyuanImage3Attention(config, dtype=dtype, operations=ops_for(dtype))
    init_op_modules(module, dtype)

    seq_len = 8
    hidden_states = torch.randn(1, seq_len, config.hidden_size, dtype=dtype)
    freqs = build_rope_freqs(seq_len, config.attention_head_dim, [], config.rope_theta)
    out = module(hidden_states, freqs)
    assert out.shape == (1, seq_len, config.hidden_size)
    assert out.dtype == dtype


# ---------------------------------------------------------------- moe


def test_swiglu_uses_second_half_of_the_contiguous_chunk():
    first = torch.randn(2, 5)
    second = torch.randn(2, 5)
    combined_input = torch.cat([first, second], dim=-1)
    assert torch.allclose(_swiglu(combined_input), first * F.silu(second))
    # the swapped and interleaved conventions must not match
    assert not torch.allclose(_swiglu(combined_input), second * F.silu(first))
    interleaved = torch.stack([first, second], dim=-1).reshape(2, -1)
    assert not torch.allclose(_swiglu(interleaved), first * F.silu(second))


def reference_moe_forward(module, hidden_states):
    """Independent dense implementation: every token through its own top-k experts, in fp32."""
    bsz, seq_len, hidden_size = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_size).float()
    logits = F.linear(flat, module.gate.wg.weight.float())
    probabilities = logits.softmax(dim=-1)
    top_k_weights, top_k_index = torch.topk(probabilities, module.top_k, dim=-1)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    routed = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for slot in range(module.top_k):
            expert = int(top_k_index[token, slot])
            gate_up = flat[token] @ module.experts_gate_up_proj.weight[expert].float().t()
            down = _swiglu(gate_up) @ module.experts_down_proj.weight[expert].float().t()
            routed[token] += top_k_weights[token, slot] * down
    shared = module.shared_mlp(flat.to(hidden_states.dtype)).float()
    return (routed + shared).reshape(bsz, seq_len, hidden_size)


def test_moe_matches_dense_reference():
    torch.manual_seed(0)
    config = make_params()
    dtype = torch.float32
    module = HunyuanImage3MoE(config, dtype=dtype, operations=ops_for(dtype))
    init_op_modules(module, dtype, seed=1)

    hidden_states = torch.randn(1, 7, config.hidden_size)
    got = module(hidden_states)
    expected = reference_moe_forward(module, hidden_states)
    assert got.shape == hidden_states.shape
    assert torch.allclose(got.float(), expected, atol=1e-4, rtol=1e-4)


def test_moe_output_dtype_is_preserved():
    torch.manual_seed(0)
    config = make_params()
    dtype = torch.bfloat16
    module = HunyuanImage3MoE(config, dtype=dtype, operations=ops_for(dtype))
    init_op_modules(module, dtype, seed=2)

    hidden_states = torch.randn(1, 5, config.hidden_size, dtype=dtype)
    out = module(hidden_states)
    assert out.shape == hidden_states.shape
    assert out.dtype == dtype


def test_router_gemm_follows_the_weight_dtype():
    """The reference upcasts the router input only while `wg.weight` is fp32, so a bf16 load routes in
    bf16 and an fp32 load routes in fp32.

    This is the policy, not a detail: upcasting unconditionally makes the ops Linear cast the weight and
    run the GEMM in fp32, which changes 10-12% of expert selections. It is also invisible to any test
    that forces one dtype on both sides — including the fp32 comparison that verified the blocks.
    """
    config = make_params()
    dtype = torch.bfloat16
    gate = HunyuanImage3MoEGate(config.hidden_size, config.num_experts, config.moe_topk, dtype=dtype, operations=ops_for(dtype))
    init_op_modules(gate, dtype)
    # Seeded random data: `arange/3` and `2^-9` fixtures both turned out to be exactly representable
    # through the bf16 path, so the two policies produced identical logits and the precondition could not
    # hold. Random values make the bf16 GEMM's rounding observable, and the seed keeps it deterministic.
    torch.manual_seed(0)
    weight = torch.randn(config.num_experts, config.hidden_size, dtype=torch.float32) + 1.0
    gate.wg.weight = torch.nn.Parameter(weight.to(dtype))
    torch.manual_seed(1)
    hidden = torch.randn(1, config.hidden_size, dtype=torch.float32).to(dtype)

    logits_bf16 = F.linear(hidden, weight.to(dtype))
    logits_fp32 = F.linear(hidden.float(), weight)      # the fp32 weight, not the bf16 one upcast
    assert not torch.equal(logits_bf16.float(), logits_fp32), "the two policies must be distinguishable"

    weights, index = gate(hidden)

    # The indices are discrete and must agree exactly; the weights come from a softmax chain that need
    # not be bit-identical to my reconstruction, so a tight tolerance is the correct criterion here —
    # demanding bit-exactness across differently-ordered operations is the mistake the fp32 block
    # comparison already taught us.
    assert torch.equal(index, logits_bf16.float().topk(config.moe_topk, dim=-1).indices), \
        "bf16 weight must route in bf16"
    expected = torch.softmax(logits_bf16.float(), dim=-1)
    picked = expected.gather(-1, index)
    picked = picked / picked.sum(dim=-1, keepdim=True)
    assert torch.allclose(weights, picked, atol=1e-6), "the gate must use the bf16 GEMM, not a cast weight"

    fp32_gate = HunyuanImage3MoEGate(config.hidden_size, config.num_experts, config.moe_topk,
                                     dtype=torch.float32, operations=ops_for(torch.float32))
    fp32_gate.wg.weight = torch.nn.Parameter(weight)
    fp32_weights, fp32_index = fp32_gate(hidden.float())
    assert torch.equal(fp32_index, logits_fp32.topk(config.moe_topk, dim=-1).indices), \
        "fp32 weight must route in fp32"
    fp32_expected = torch.softmax(logits_fp32, dim=-1)
    fp32_picked = fp32_expected.gather(-1, fp32_index)
    fp32_picked = fp32_picked / fp32_picked.sum(dim=-1, keepdim=True)
    assert torch.allclose(fp32_weights, fp32_picked, atol=1e-6)


# ---------------------------------------------------------------- block


def test_decoder_layer_shapes_and_dtype():
    torch.manual_seed(0)
    config = make_params()
    dtype = torch.bfloat16
    module = HunyuanImage3DecoderLayer(config, dtype=dtype, operations=ops_for(dtype))
    init_op_modules(module, dtype, seed=3)

    seq_len, image = 9, (slice(2, 8), (2, 3))
    hidden_states = torch.randn(1, seq_len, config.hidden_size, dtype=dtype)
    freqs = build_rope_freqs(seq_len, config.attention_head_dim, [image], config.rope_theta)
    mask = torch.zeros(1, 1, seq_len, seq_len)
    mask[:, :, 2:8, 2:8] = 0.0
    out = module(hidden_states, freqs, mask)
    assert out.shape == hidden_states.shape
    assert out.dtype == dtype
    assert torch.isfinite(out.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_decoder_layer_matches_cpu_on_cuda():
    """The rope kernel, attention and the expert banks all take a different path on CUDA."""
    torch.manual_seed(0)
    config = make_params()
    dtype = torch.float32
    module = HunyuanImage3DecoderLayer(config, dtype=dtype, operations=ops_for(dtype))
    init_op_modules(module, dtype, seed=5)

    seq_len, image = 9, (slice(2, 8), (2, 3))
    hidden_states = torch.randn(1, seq_len, config.hidden_size)
    freqs = build_rope_freqs(seq_len, config.attention_head_dim, [image], config.rope_theta)
    mask = torch.zeros(1, 1, seq_len, seq_len)
    mask[:, :, :, 5:] = float("-inf")
    mask[:, :, 5:, 5:] = 0.0

    out_cpu = module(hidden_states, freqs, mask)

    module.to("cuda")
    out_cuda = module(hidden_states.cuda(), freqs.cuda(), mask.cuda())
    assert out_cuda.shape == out_cpu.shape
    assert out_cuda.dtype == out_cpu.dtype
    assert torch.allclose(out_cuda.cpu(), out_cpu, atol=1e-4, rtol=1e-4)


def test_expert_cast_paths_agree():
    """The per-expert cast and the whole-bank cast must produce identical output.

    A whole sequence routes to nearly every expert, so casting the bank once serves all of them; a decode
    step routes to a few (`top_k` of `num_experts`), and casting the bank for it would stream every expert
    in the layer rather than the ones the token uses (spec §51). The paths differ only in what gets moved,
    so the numbers must be bit-identical — this is the invariant the choice between them rests on.
    """
    operations = comfy.ops.mixed_precision_ops({"mixed_ops": True}, torch.bfloat16)
    experts = operations.MoEExperts(8, 32, 48, bias=False, device=torch.device("cpu"), dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(0)
    # the weight slot is filled from a checkpoint, so it starts empty — the test supplies it
    experts.weight = torch.nn.Parameter(
        torch.randn(8, 48, 32, generator=generator).to(torch.bfloat16) * 0.05, requires_grad=False)

    x = torch.randn(1, 32, generator=generator).to(torch.bfloat16)
    for expert in (0, 3, 7):
        per_expert = experts.expert_linear(x, expert)
        with experts.bank_resident(x) as bank:
            from_bank = bank.expert_linear(x, expert)
        assert torch.equal(per_expert, from_bank), f"expert {expert} differs between the cast paths"


def test_wide_image_grids_rescale_the_rope_base():
    """A grid wider than 64x64 raises the rope base, as the reference's `base_rescale_factor` does.

    Image positions grow with w*h — L+2016 at 64x64, L+4560 at 96x96 — and the reference never asks
    for a grid beyond its resolution table, so an unscaled wider grid sits at rope phases the model
    only ever encountered for text. Measured at 1536x1536: unscaled rendered a plausible image that
    ignored the prompt, and the rescaled base rendered the prompt itself.
    """
    start, head_dim, slowest = 100, 128, 63

    def block_angle(height, width):
        section = type("Section", (), {"start": start})()
        freqs = build_rope_freqs(start + height * width, head_dim, [(section, (height, width))],
                                 10000.0, torch.device("cpu"))
        first = freqs[0, 0, start, slowest]
        return float(torch.atan2(first[1, 0], first[0, 0]))

    raised = 10000.0 * (96 / 64) ** (head_dim / (head_dim - 2))
    assert raised == pytest.approx(15096.9, rel=1e-5)
    assert block_angle(64, 64) == pytest.approx(2116.0 * 10000.0 ** (-2 * slowest / head_dim), abs=1e-5)
    assert block_angle(96, 96) == pytest.approx(4660.0 * raised ** (-2 * slowest / head_dim), abs=1e-4)
    assert block_angle(96, 96) < 4660.0 * 10000.0 ** (-2 * slowest / head_dim)


def test_unquantized_bank_moves_one_expert_and_agrees_with_the_resident_path():
    """The pack's own bf16 bank: the per-expert path must index before it moves, and agree exactly.

    It used to call `.to()` on the whole bank and index afterwards, which streams every expert in the
    layer to use one — the docstring said otherwise. With a GPU the input goes there, so the bank has
    to cross and the old code is caught calling the bank's `.to`; on a CPU-only machine nothing moves
    and only the numbers are checked.
    """
    from comfy.ldm.hunyuan_image_3.ops import MoEExperts, expert_linear_sliced

    experts = MoEExperts(8, 32, 48, bias=True, device=torch.device("cpu"), dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(0)
    experts.weight.data.copy_(torch.randn(8, 48, 32, generator=generator) * 0.05)
    experts.bias.data.copy_(torch.randn(8, 48, generator=generator) * 0.05)

    moved = []
    bank_to = experts.weight.to
    experts.weight.to = lambda *a, **k: (moved.append(a), bank_to(*a, **k))[1]   # noqa: E731

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x = torch.randn(1, 32, generator=generator).to(device, torch.bfloat16)
    for expert in (0, 3, 7):
        per_expert = experts.expert_linear(x, expert)
        via_sliced = expert_linear_sliced(experts, x, expert)      # not quantized: falls through
        assert not moved, "the per-expert path called .to() on the whole bank"
        with experts.bank_resident(x) as bank:                     # moving the bank is its job
            from_bank = bank.expert_linear(x, expert)
        moved.clear()
        assert torch.equal(per_expert, from_bank), f"expert {expert} differs between the cast paths"
        assert torch.equal(via_sliced, from_bank)


def _populated_quantized_model():
    """A tiny model on the mixed-precision ops with every weight filled, as a loaded checkpoint is."""
    from comfy.ldm.hunyuan_image_3.model import HunyuanImage3

    params = make_params(vocab_size=64, patch_embed_hidden_dim=64)
    operations = comfy.ops.mixed_precision_ops({"mixed_ops": True}, torch.bfloat16)
    model = HunyuanImage3(params, dtype=torch.bfloat16, device="cpu", operations=operations)
    state_dict = {}
    for name, module in model.named_modules():
        prefix = f"{name}." if name else ""
        shape = getattr(module, "_orig_shape", None)
        if shape is None and hasattr(module, "in_features") and hasattr(module, "out_features"):
            shape = (module.out_features, module.in_features)
        if shape is not None:
            state_dict[f"{prefix}weight"] = torch.randn(shape).to(torch.bfloat16)
            if getattr(module, "bias", None) is not None:
                state_dict[f"{prefix}bias"] = torch.zeros(module.bias.shape, dtype=torch.bfloat16)
    for key, value in model.state_dict().items():
        state_dict.setdefault(key, torch.randn(value.shape).to(value.dtype) if value.is_floating_point() else value)
    model.load_state_dict(state_dict, strict=False)
    return params, model
