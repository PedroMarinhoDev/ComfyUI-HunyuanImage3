"""The grouped MoE path must reproduce the per-expert loop bit for bit.

`grouped_moe` replaces ~4,100 per-expert GEMM launches per pass with one grouped launch per
projection. Its whole claim is that nothing changes but the speed, so these compare it against the
comfy-kitchen kernels the loop calls — `int8_linear` for int8 ConvRot banks, `w4a8_int8_linear` and
the CUDA int4 decode for W4A8 banks — on synthetic weights, and require exact equality. Eight sampling
steps turn a few differing outputs per million into a visibly different image, so "close" is a failure.

Needs CUDA (sm_80+), Triton and comfy-kitchen's CUDA backend; skipped otherwise.
"""
import pytest
import torch
import torch.nn.functional as F

import comfy.ldm.hunyuan_image_3.grouped_moe as grouped_moe

pytestmark = pytest.mark.skipif(
    not grouped_moe.enabled() or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() < (8, 0),
    reason="needs CUDA sm_80+ and Triton")

GROUP = 256


def _swiglu(x):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def _routing(num_tokens, num_experts, top_k, generator, device):
    """The sort `HunyuanImage3MoE.forward` performs: rows grouped by expert, slot-major within one."""
    logits = torch.randn(num_tokens, num_experts, generator=generator).to(device)
    weights, index = torch.topk(logits.softmax(-1), top_k, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)
    routing = index.t().reshape(-1)
    order = torch.sort(routing, stable=True).indices
    counts = torch.bincount(routing, minlength=num_experts).tolist()
    token_sorted = order % num_tokens
    slot_sorted = order // num_tokens
    return (token_sorted, token_sorted * top_k + slot_sorted, weights[token_sorted, slot_sorted, None], counts)


def _loop(flat, routing, top_k, linear_gu, linear_dn):
    """The per-expert loop of `HunyuanImage3MoE.forward`, with the given per-expert linears."""
    token_sorted, dest_sorted, weight_sorted, counts = routing
    combined = torch.zeros((flat.shape[0] * top_k, flat.shape[1]), dtype=flat.dtype, device=flat.device)
    start = 0
    for e, count in enumerate(counts):
        if not count:
            continue
        rows = slice(start, start + count)
        start += count
        out = linear_dn(_swiglu(linear_gu(flat[token_sorted[rows]], e)), e)
        combined[dest_sorted[rows]] = (out * weight_sorted[rows]).to(combined.dtype)
    return combined.view(flat.shape[0], top_k, flat.shape[1]).sum(dim=1)


def _activations(num_tokens, hidden, generator, device):
    x = torch.randn(num_tokens, hidden, generator=generator)
    x[:, torch.randperm(hidden, generator=generator)[:8]] *= 20   # outlier channels, as real activations have
    return x.to(device, torch.bfloat16)


def test_int8_convrot_matches_int8_linear_loop():
    device = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    E, hidden, inter, tokens, top_k = 8, 512, 512, 300, 2
    w_gu = torch.randint(-127, 128, (E, 2 * inter, hidden), generator=g, dtype=torch.int8).to(device)
    w_dn = torch.randint(-127, 128, (E, hidden, inter), generator=g, dtype=torch.int8).to(device)
    s_gu = (torch.rand(E, 2 * inter, generator=g) * 1e-3 + 1e-4).to(device)
    s_dn = (torch.rand(E, hidden, generator=g) * 1e-3 + 1e-4).to(device)
    flat = _activations(tokens, hidden, g, device)
    routing = _routing(tokens, E, top_k, g, device)

    def int8_linear(w, s):
        return lambda x, e: torch.ops.comfy_kitchen.int8_linear(x, w[e], s[e], None, 2, True, GROUP)

    expected = _loop(flat, routing, top_k, int8_linear(w_gu, s_gu), int8_linear(w_dn, s_dn))
    got = grouped_moe.moe_int8(flat, *routing, (w_gu, s_gu, GROUP), (w_dn, s_dn, GROUP), top_k, _swiglu)
    assert torch.equal(got, expected), f"{(got != expected).sum().item()} of {got.numel()} outputs differ"


def _w4a8_bank(E, rows, K, g, device):
    q = torch.randint(-128, 128, (E * rows, K // 2), generator=g, dtype=torch.int8).to(device)   # packed nibbles
    scale = (torch.rand(E * rows, K // 16, generator=g) * 60 + 20).to(torch.float8_e4m3fn).to(device)
    codebook = torch.sort(torch.randn(E, 16, generator=g).clamp(-2, 2) / 2, dim=-1).values.to(device)
    s_channel = (torch.rand(E, rows, generator=g) * 1e-3 + 1e-4).to(device)
    return q, scale, codebook, s_channel


def test_w4a8_decode_matches_comfy_kitchen():
    from comfy_kitchen.backends import cuda as CB
    device = torch.device("cuda")
    g = torch.Generator().manual_seed(1)
    E, rows, K = 4, 64, 512
    q, scale, codebook, _ = _w4a8_bank(E, rows, K, g, device)
    decoded = grouped_moe.decode_w4a8(q, scale, codebook, rows).view(E, rows, K)
    for e in range(E):
        ref = torch.empty(rows, K, dtype=torch.int8, device=device)
        CB._C.dequant_int4_grouped_to_int8_e4m3(
            CB._wrap_for_dlpack(q.view(E, rows, -1)[e].contiguous()),
            CB._wrap_for_dlpack(scale.view(E, rows, -1)[e].contiguous().view(torch.uint8)),
            CB._wrap_for_dlpack(codebook[e].contiguous()), CB._wrap_for_dlpack(ref), 16,
            torch.cuda.current_stream().cuda_stream)
        assert torch.equal(decoded[e], ref), f"expert {e}: {(decoded[e] != ref).sum().item()} values differ"


def test_w4a8_matches_w4a8_int8_linear_loop():
    from comfy_kitchen.backends import cuda as CB
    device = torch.device("cuda")
    g = torch.Generator().manual_seed(2)
    E, hidden, inter, tokens, top_k = 8, 512, 512, 300, 2
    gu = _w4a8_bank(E, 2 * inter, hidden, g, device)
    dn = _w4a8_bank(E, hidden, inter, g, device)
    flat = _activations(tokens, hidden, g, device)
    routing = _routing(tokens, E, top_k, g, device)

    def w4a8_linear(bank, rows):
        q, scale, codebook, s_channel = bank
        q, scale = q.view(E, rows, -1), scale.view(E, rows, -1)
        return lambda x, e: CB.w4a8_int8_linear(x, q[e], scale[e], s_channel[e], codebook=codebook[e],
                                                group_size=16, convrot_groupsize=GROUP, out_dtype=torch.bfloat16)

    def decoded(bank, rows, K):
        q, scale, codebook, s_channel = bank
        return grouped_moe.decode_w4a8(q, scale, codebook, rows).view(E, rows, K), s_channel, GROUP

    expected = _loop(flat, routing, top_k, w4a8_linear(gu, 2 * inter), w4a8_linear(dn, hidden))
    got = grouped_moe.moe_int8(flat, *routing, decoded(gu, 2 * inter, hidden), decoded(dn, hidden, inter),
                               top_k, _swiglu)
    assert torch.equal(got, expected), f"{(got != expected).sum().item()} of {got.numel()} outputs differ"
