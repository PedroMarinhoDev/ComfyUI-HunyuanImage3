"""Grouped MoE for int8 ConvRot and W4A8 expert banks: every expert in one kernel launch per projection.

The per-expert loop in `HunyuanImage3MoE.forward` issues one small GEMM per expert and projection —
about 4,100 launches per pass at 64 experts x 32 layers — and the GPU idles while Python feeds them.
Here each projection is a single Triton grouped GEMM over the expert-sorted rows, with

* each token rotated and quantized ONCE, before it is gathered to its top-k experts (the loop
  re-quantizes the same token for every expert it routes to);
* the routing weight and the scatter into the combine buffer fused into the down projection.

The result is bit-identical to the loop. Activations go through the same fused rotate + row-quantize
kernel comfy-kitchen's CUDA `int8_linear` uses, the GEMM is exact int32 accumulation, and the epilogue
multiplies in `int8_linear`'s order — `(acc * x_scale) * w_scale` — because any other order differs in
a few outputs per million, which eight sampling steps amplify into a visibly different image. W4A8
banks are decoded to their int8 grid in one launch per bank (the same values comfy-kitchen's decode
kernel produces) and then take the same GEMM.

Measured on an RTX PRO 6000 Blackwell, 1024x1024, warm: Instruct-Distil int8 8.0 -> ~4.0 s per image,
W4A8 7.3 -> 4.8 s; Instruct W4A8 1.66 -> 1.06 s/it. Images are pixel-identical to the loop.

`routed()` returns None whenever this path cannot take a layer — no CUDA or Triton, a GPU without int8
tensor cores, another bank format, a comfy-kitchen without the kernels it calls — and the caller falls
back to the loop. A failure on first use disables the path for the process (logged once). Set
`HY3_GROUPED_MOE=0` to force the loop.
"""
import logging
import os
import time

import torch

import comfy.quant_ops

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the install
    triton = None

_disabled_reason = None


def enabled():
    if _disabled_reason is not None or triton is None or os.environ.get("HY3_GROUPED_MOE", "1") == "0":
        return False
    return torch.cuda.is_available()


def _disable(reason):
    global _disabled_reason
    if _disabled_reason is None:
        _disabled_reason = reason
        logging.warning("HunyuanImage3: grouped MoE path disabled (%s); using the per-expert loop", reason)


def _device_ok(device):
    # int8 tl.dot needs int8 tensor cores (sm_80+)
    return device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 0)


# ---------------------------------------------------------------------------------------------------
# Grouped int8 GEMM: rows sorted by expert, one program per (row tile, column tile); a row tile never
# straddles two experts, so each program reads exactly one expert's weights.

if triton is not None:
    @triton.jit
    def _grouped_int8_kernel(
        xq_ptr, xs_ptr, w_ptr, ws_ptr, out_ptr, offsets_ptr,
        tile_expert_ptr, tile_row_ptr, num_m_tiles,
        rw_ptr, dest_ptr,
        N, K,
        stride_we, stride_wn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
        SCATTER: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_n = tl.cdiv(N, BN)
        group_size = GROUP * num_n
        gid = pid // group_size
        first_m = gid * GROUP
        gm = tl.minimum(num_m_tiles - first_m, GROUP)
        pid_m = first_m + (pid % group_size) % gm
        pid_n = (pid % group_size) // gm

        e = tl.load(tile_expert_ptr + pid_m)
        r0 = tl.load(tile_row_ptr + pid_m)
        r_end = tl.load(offsets_ptr + e + 1)

        rows = r0 + tl.arange(0, BM)
        cols = pid_n * BN + tl.arange(0, BN)
        rmask = rows < r_end
        cmask = cols < N
        ks = tl.arange(0, BK)

        a_ptrs = xq_ptr + rows[:, None].to(tl.int64) * K + ks[None, :]
        b_ptrs = w_ptr + e.to(tl.int64) * stride_we + cols[None, :].to(tl.int64) * stride_wn + ks[:, None]
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for k in range(0, K, BK):
            a = tl.load(a_ptrs, mask=rmask[:, None], other=0)
            b = tl.load(b_ptrs, mask=cmask[None, :], other=0)
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            a_ptrs += BK
            b_ptrs += BK

        xs = tl.load(xs_ptr + rows, mask=rmask, other=0.0)
        ws = tl.load(ws_ptr + e.to(tl.int64) * N + cols, mask=cmask, other=0.0)
        # (acc * x_scale) * w_scale: int8_linear's epilogue order (see the module docstring)
        out = ((acc.to(tl.float32) * xs[:, None]) * ws[None, :]).to(tl.bfloat16)
        if SCATTER:
            rw = tl.load(rw_ptr + rows, mask=rmask, other=0.0)
            out = (out.to(tl.float32) * rw.to(tl.float32)[:, None]).to(tl.bfloat16)
            dst = tl.load(dest_ptr + rows, mask=rmask, other=0)
            o_ptrs = out_ptr + dst[:, None].to(tl.int64) * N + cols[None, :]
        else:
            o_ptrs = out_ptr + rows[:, None].to(tl.int64) * N + cols[None, :]
        tl.store(o_ptrs, out, mask=rmask[:, None] & cmask[None, :])

    @triton.jit
    def _w4a8_decode_kernel(q_ptr, s_ptr, cb_ptr, out_ptr, rows_per_expert, K, G,
                            BR: tl.constexpr, BKH: tl.constexpr):
        # q [R, K/2] packed (low nibble = even column), s [R, G] fp8-e4m3 bits, cb [E, 16] f32, out [R, K]
        pid_r = tl.program_id(0)
        pid_k = tl.program_id(1)
        r = pid_r * BR + tl.arange(0, BR)
        kh = pid_k * BKH + tl.arange(0, BKH)
        e = r // rows_per_expert
        p = tl.load(q_ptr + r[:, None].to(tl.int64) * (K // 2) + kh[None, :]).to(tl.int32) & 0xFF
        cbl = tl.load(cb_ptr + e[:, None] * 16 + (p & 0xF))
        cbh = tl.load(cb_ptr + e[:, None] * 16 + ((p >> 4) & 0xF))
        sb = tl.load(s_ptr + r[:, None].to(tl.int64) * G + ((kh * 2) // 16)[None, :])
        s = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        # rint is round-half-to-even, as torch.round and the CUDA decode are
        vl = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(cbl * s), -127.0), 127.0).to(tl.int8)
        vh = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(cbh * s), -127.0), 127.0).to(tl.int8)
        v = tl.join(vl, vh).reshape(BR, 2 * BKH)
        k = pid_k * 2 * BKH + tl.arange(0, 2 * BKH)
        tl.store(out_ptr + r[:, None].to(tl.int64) * K + k[None, :], v)


CONFIGS = [  # (BM, BN, BK, warps, stages); the fastest that fits is picked per shape on first use
    (128, 128, 128, 8, 3), (128, 128, 128, 4, 4), (64, 128, 128, 4, 4),
    (128, 256, 128, 8, 3), (64, 256, 128, 8, 3), (128, 128, 64, 4, 4),
    (256, 128, 128, 8, 3),
]
_TILE_CACHE = {}
_BEST = {}


def _tiles(counts, bm, device):
    key = (tuple(counts), bm, str(device))
    t = _TILE_CACHE.get(key)
    if t is None:
        te, tr = [], []
        start = 0
        for e, c in enumerate(counts):
            for r in range(start, start + c, bm):
                te.append(e)
                tr.append(r)
            start += c
        t = (torch.tensor(te, dtype=torch.int32, device=device),
             torch.tensor(tr, dtype=torch.int32, device=device), len(te))
        if len(_TILE_CACHE) > 256:
            _TILE_CACHE.clear()
        _TILE_CACHE[key] = t
    return t


def _run(cfg, xq, xs, w, ws, out, offsets, counts, rw, dest):
    bm, bn, bk, nw, ns = cfg
    te, tr, n = _tiles(counts, bm, xq.device)
    N, K = w.shape[1], w.shape[2]
    scatter = rw is not None
    _grouped_int8_kernel[(n * triton.cdiv(N, bn),)](
        xq, xs, w, ws, out, offsets, te, tr, n,
        rw if scatter else xs, dest if scatter else offsets,
        N, K, w.stride(0), w.stride(1),
        BM=bm, BN=bn, BK=bk, GROUP=8, SCATTER=scatter, num_warps=nw, num_stages=ns)


def _tune(args):
    best, best_t = None, float("inf")
    for cfg in CONFIGS:
        try:
            _run(cfg, *args)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                _run(cfg, *args)
            torch.cuda.synchronize()
            t = time.perf_counter() - t0
        except Exception:  # e.g. out of shared memory on this card
            continue
        if t < best_t:
            best, best_t = cfg, t
    if best is None:
        raise RuntimeError("no grouped GEMM configuration compiled on this device")
    return best


def grouped_int8_linear(xq, xs, w, ws, counts, offsets, route_weight=None, dest=None, out=None):
    """out[r] = (xq[r] @ w[e(r)].T) * xs[r] * ws[e(r)] for expert-sorted rows; with `route_weight`
    the row is also scaled by it and written to row `dest[r]` of `out` instead of row r."""
    R, K = xq.shape
    E, N, _ = w.shape
    if out is None:
        out = torch.empty((R, N), dtype=torch.bfloat16, device=xq.device)
    key = (N, K, route_weight is not None, 1 << max(0, (R // E).bit_length()), str(xq.device))
    args = (xq, xs, w, ws, out, offsets, counts, route_weight, dest)
    cfg = _BEST.get(key)
    if cfg is None:
        cfg = _BEST[key] = _tune(args)  # tuning writes `out`; the launch below rewrites all of it
    _run(cfg, *args)
    return out


# ---------------------------------------------------------------------------------------------------
# Activation quantization: exactly the branch the CUDA int8_linear takes for ConvRot weights, so the
# grouped GEMM sees the same int8 activations the loop does. (The registry's
# `quantize_and_rotate_rowwise` rounds the rotated rows to bf16 first, which is not bit-identical.)

_HADAMARD = {}


def _qrot(x, group):
    from comfy_kitchen.backends import cuda as CB
    m, k = x.shape
    if (group == 256 and k % 256 == 0 and 256 <= k <= CB._CONVROT_FUSED_MAX_K
            and CB._convrot_fused_shared_memory_fits(x, k, group)):
        q = torch.empty((m, k), dtype=torch.int8, device=x.device)
        s = torch.empty((m, 1), dtype=torch.float32, device=x.device)
        CB._C.quantize_int8_rowwise_convrot64(
            CB._wrap_for_dlpack(x), CB._wrap_for_dlpack(q), CB._wrap_for_dlpack(s), group, False,
            CB._input_act_code(None), 0,
            CB._wrap_for_dlpack(CB._act_weight_arg(None, None, x.device, x.dtype)), 0.0,
            torch.cuda.current_stream(x.device).cuda_stream)
    elif CB._should_use_convrot_fused_kernel(x, k, group):
        q, s = CB.quantize_int8_rowwise_convrot(x, group)
    else:
        key = (group, str(x.device), x.dtype)
        h = _HADAMARD.get(key)
        if h is None:
            from comfy_kitchen.backends.eager.quantization import _build_hadamard
            h = _HADAMARD[key] = _build_hadamard(group, device=x.device, dtype=x.dtype)
        q, s = CB.quantize_and_rotate_rowwise(x, h, group)
    return q, s.reshape(-1).float().contiguous()


# ---------------------------------------------------------------------------------------------------
# Bank detection. Each returns None when the bank is not one this path reproduces exactly.

def _resident_quantized(experts, layout):
    resident = getattr(experts, "_resident_bank", None)
    if resident is None or resident[1] is not None or getattr(experts, "_full_precision_mm", False):
        return None
    weight = resident[0]
    if not isinstance(weight, comfy.quant_ops.QuantizedTensor) or weight._layout_cls != layout:
        return None
    return weight


def int8_bank(experts, num_experts):
    """(qdata [E, N, K] int8, scale [E, N] f32, ConvRot group) of a resident int8 ConvRot bank."""
    weight = _resident_quantized(experts, "TensorWiseINT8Layout")
    if weight is None:
        return None
    p = weight._params
    q = weight._qdata
    if not getattr(p, "convrot", False) or q.ndim != 2 or q.dtype != torch.int8 or q.shape[0] % num_experts:
        return None
    out_f = q.shape[0] // num_experts
    scale = p.scale.reshape(num_experts, out_f).float().contiguous()
    return q.view(num_experts, out_f, q.shape[1]), scale, p.convrot_groupsize


def w4a8_bank(experts, num_experts):
    """A resident W4A8 bank (4-bit codebook indices, fp8 group scales, per-channel scale) decoded to
    its int8 grid in one launch, in the same (qdata, scale, group) form `int8_bank` returns."""
    weight = _resident_quantized(experts, "AsymW4A8Int8Layout")
    if weight is None:
        return None
    p = weight._params
    q = weight._qdata
    if (p.correction is not None or p.codebook is None or p.group_size != 16 or p.convrot_groupsize != 256
            or getattr(p, "transposed", False) or p.scale.dtype != torch.float8_e4m3fn or q.ndim != 2):
        return None
    R, KH = q.shape
    if R % num_experts or R % 16 or KH % 256:
        return None
    cb = p.codebook
    if cb.numel() == 16:
        cb = cb.reshape(1, 16).expand(num_experts, 16)
    elif cb.numel() == num_experts * 16:
        cb = cb.reshape(num_experts, 16)
    else:
        return None
    decoded = decode_w4a8(q, p.scale, cb, R // num_experts)
    s_channel = p.s_channel.reshape(num_experts, R // num_experts).float().contiguous()
    return decoded.view(num_experts, R // num_experts, KH * 2), s_channel, p.convrot_groupsize


def decode_w4a8(q, scale, codebook, rows_per_expert):
    """Packed 4-bit indices [R, K/2] + fp8-e4m3 per-16 scales [R, K/16] + codebook [E, 16] -> int8 [R, K]:
    round(codebook[index] * scale) clamped to +-127, the values comfy-kitchen's decode kernel produces."""
    R, KH = q.shape
    decoded = torch.empty((R, KH * 2), dtype=torch.int8, device=q.device)
    BR, BKH = 16, 256
    _w4a8_decode_kernel[(R // BR, KH // BKH)](
        q.contiguous(), scale.contiguous().view(torch.uint8), codebook.float().contiguous(), decoded,
        rows_per_expert, KH * 2, KH * 2 // 16, BR=BR, BKH=BKH, num_warps=8)
    return decoded


def moe_int8(flat, token_sorted, dest_sorted, weight_sorted, counts, gate_up, down, top_k, swiglu):
    """Routed-expert output [tokens, hidden] from two (qdata, scale, group) banks."""
    w_gu, s_gu, g_gu = gate_up
    w_dn, s_dn, g_dn = down
    num_tokens, hidden = flat.shape
    offsets = torch.zeros(len(counts) + 1, dtype=torch.int32)
    offsets[1:] = torch.tensor(counts, dtype=torch.int32).cumsum(0)
    offsets = offsets.to(flat.device, non_blocking=True)

    xq_t, xs_t = _qrot(flat.contiguous(), g_gu)
    g = grouped_int8_linear(xq_t[token_sorted], xs_t[token_sorted], w_gu, s_gu, counts, offsets)
    hq, hs = _qrot(swiglu(g).contiguous(), g_dn)
    del g
    combined = torch.empty((num_tokens * top_k, hidden), dtype=flat.dtype, device=flat.device)
    grouped_int8_linear(hq, hs, w_dn, s_dn, counts, offsets,
                        route_weight=weight_sorted.reshape(-1).to(flat.dtype).contiguous(),
                        dest=dest_sorted.to(torch.int32).contiguous(), out=combined)
    return combined.view(num_tokens, top_k, hidden).sum(dim=1)


def routed(gate_up_experts, down_experts, num_experts, flat, token_sorted, dest_sorted, weight_sorted,
           counts, top_k, swiglu):
    """The routed-expert sum for one layer, or None if the caller should run its per-expert loop."""
    if not enabled() or not _device_ok(flat.device) or flat.dtype != torch.bfloat16:
        return None
    try:
        gate_up = int8_bank(gate_up_experts, num_experts)
        down = int8_bank(down_experts, num_experts) if gate_up is not None else None
        if down is None:
            gate_up = w4a8_bank(gate_up_experts, num_experts)
            down = w4a8_bank(down_experts, num_experts) if gate_up is not None else None
        if down is None:
            return None
        return moe_int8(flat, token_sorted, dest_sorted, weight_sorted, counts, gate_up, down, top_k, swiglu)
    except Exception as error:  # an unsupported comfy-kitchen / Triton / GPU: fall back for good
        _disable(f"{type(error).__name__}: {error}")
        return None

