# https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil
import logging
import math
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.quant_ops
import contextlib

from comfy.ldm.modules.attention import optimized_attention_masked

from .lookahead import LayerLookahead
from .ops import expert_linear_sliced


@dataclass
class HunyuanImage3Params:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    attention_bias: bool
    mlp_bias: bool
    moe_intermediate_size: int
    num_experts: int
    moe_topk: int
    num_shared_expert: int
    cfg_distilled: bool
    use_meanflow: bool
    model_type: str
    sequence_template: str
    pad_token_id: int
    image_token_id: int
    patch_size: int
    patch_embed_hidden_dim: int
    image_base_size: int
    vae_latent_channels: int
    vae_downsample_factor: tuple
    vit_aligner: dict


_EXPERT_WEIGHT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_and_up_proj|down_proj)\.weight$")
_EXPERT_BANK = {"gate_and_up_proj": "experts_gate_up_proj", "down_proj": "experts_down_proj"}

# not built here: the vae arrives from its own file (the converter strips the prefix), the vision tower
# is a separate clip_vision checkpoint the node reuses, and the text head (`lm_head` + `ln_f`) belongs
# to the prompt-rewriting stage, which loads it from its own file (`rewrite.py`). The aligner *is*
# built — it lives in this checkpoint as `vision_aligner.*`.
HEAD_PREFIXES = ("lm_head.", "model.ln_f.")
DEFERRED_PREFIXES = ("vae.", "vision_model.") + HEAD_PREFIXES


def _uniform_layers(value, name):
    if isinstance(value, (list, tuple)):
        unique = set(value)
        if len(unique) != 1:
            raise ValueError(f"{name} is per layer in this checkpoint ({sorted(unique)}); uniform per-layer values required")
        return value[0]
    return value


def params_from_config(config):
    """Build HunyuanImage3Params from a HunyuanImage-3 config.json dict.

    The three asserts guard the assumptions the model code is written against: a checkpoint that
    flips any of them would otherwise build a wrong model instead of failing.
    """
    if config.get("moe_layer_num_skipped", 0) != 0:
        raise ValueError(f"unsupported moe_layer_num_skipped {config['moe_layer_num_skipped']}, expected 0")
    if config.get("use_mixed_mlp_moe") is not True:
        raise ValueError("unsupported use_mixed_mlp_moe, expected true")
    if config.get("hidden_act") != "silu":
        raise ValueError(f"unsupported hidden_act {config.get('hidden_act')}, expected silu")

    return HunyuanImage3Params(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        attention_head_dim=config["attention_head_dim"],
        rms_norm_eps=config["rms_norm_eps"],
        rope_theta=config["rope_theta"],
        max_position_embeddings=config["max_position_embeddings"],
        attention_bias=config["attention_bias"],
        mlp_bias=config["mlp_bias"],
        moe_intermediate_size=_uniform_layers(config["moe_intermediate_size"], "moe_intermediate_size"),
        num_experts=_uniform_layers(config["num_experts"], "num_experts"),
        moe_topk=_uniform_layers(config["moe_topk"], "moe_topk"),
        num_shared_expert=_uniform_layers(config["num_shared_expert"], "num_shared_expert"),
        cfg_distilled=config.get("cfg_distilled", False),
        use_meanflow=config.get("use_meanflow", False),
        model_type=config.get("model_type", "instruct_distil"),
        # the loader writes this alongside model_type from its MODEL_TYPES table; the fallback is for a
        # config read without it, where only the base checkpoint wants the pretrain rendering
        sequence_template=config.get(
            "sequence_template",
            "pretrain" if config.get("model_type") == "base" else "instruct"),
        pad_token_id=config["pad_token_id"],
        image_token_id=config["image_token_id"],
        patch_size=config.get("patch_size", 1),
        patch_embed_hidden_dim=config.get("patch_embed_hidden_dim", 1024),
        image_base_size=config.get("image_base_size", 1024),
        vae_latent_channels=config["vae"]["latent_channels"],
        vit_aligner=config["vit_aligner"],
        vae_downsample_factor=tuple(config["vae_downsample_factor"]),
    )


def remap_key(key):
    """Checkpoint key -> this model's key, with the expert index for the routed expert tensors.

    The 64 routed experts of a layer are stacked into one MoEExperts bank each, so their tensors
    become rows of `model.layers.N.mlp.experts_gate_up_proj.weight` (shape (64, 6144, 4096)) or
    `model.layers.N.mlp.experts_down_proj.weight` (shape (64, 4096, 3072)), with the expert index
    preserved as the leading dimension. Every other key maps to itself.
    """
    match = _EXPERT_WEIGHT.match(key)
    if match is None:
        return key, None
    layer, expert, projection = match.groups()
    return f"model.layers.{layer}.mlp.{_EXPERT_BANK[projection]}.weight", int(expert)


def skipped_keys(state_dict):
    """Keys the diffusion model does not build: the vae, the vision tower and the text head each
    travel in their own file."""
    return [key for key in state_dict if key.startswith(DEFERRED_PREFIXES)]


def remap_state_dict(state_dict):
    """Checkpoint state dict -> this model's state dict, stacking the per-expert tensors.

    Returns (state_dict, skipped). Expert index order is preserved and a bank that is missing any
    expert tensor fails instead of leaving an uninitialised row.

    Whole-dict: it materialises a full-size bank per layer, so it is for small dicts and tests. The
    real checkpoint is read bank by bank instead — 57 of the 64 banks have their expert tensors
    spread over two shard files, so iterating shards cannot build them. Each bank is assembled with
    `safetensors.safe_open` per tensor, using the index to locate each expert, which caps peak
    memory at one bank (3.22 GB gate/up, 1.61 GB down).
    """
    skipped = set(skipped_keys(state_dict))
    bank_sizes = {}
    bank_counts = {}
    for key in state_dict:
        if key in skipped:
            continue
        target, expert = remap_key(key)
        if expert is not None:
            bank_sizes[target] = max(bank_sizes.get(target, 0), expert + 1)
            bank_counts[target] = bank_counts.get(target, 0) + 1

    remapped = {}
    for key, tensor in state_dict.items():
        if key in skipped:
            continue
        target, expert = remap_key(key)
        if expert is None:
            remapped[target] = tensor
            continue
        if target not in remapped:
            remapped[target] = torch.empty((bank_sizes[target], *tensor.shape), dtype=tensor.dtype, device=tensor.device)
        remapped[target][expert] = tensor

    for target, size in bank_sizes.items():
        if bank_counts[target] != size:
            raise ValueError(f"{target}: expected {size} expert tensors, got {bank_counts[target]}")

    return remapped, sorted(skipped)


def _as_list(value):
    """A conditioning value as a per-image list: one image may arrive as a bare tensor."""
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _sequence_from_ids(ids, latents, config, cond_latent=None, cond_patch_grid=None):
    """The per-step sequence geometry, derived from the token ids and the latents' grids.

    `build_sequence` writes `<boi>, <img_size>, <img_ratio>, <timestep>` — then `<guidance>` and
    `<timestep_r>` where the checkpoint has their embedders — immediately before the generated block,
    and the block itself is a contiguous run of `image_token_id`. So the block's start locates
    everything: the grid is the latent's own (patch size 1 puts one token per latent cell), and the
    embedder slots are the tokens ending there. `test_hunyuan_image_3_wiring` pins those offsets against
    `build_sequence`, which is what keeps this derivation and the builder from drifting apart.

    Each conditioning image adds two runs ahead of the generated one — its VAE latent's patches, then
    its tower patches — separated by `<joint_img_sep>` inside one bidirectional block (spec §44), so
    `cond_latent` and `cond_patch_grid` are per-image lists in sequence order. The mask gets each
    image's joint range while the RoPE gets each run with its own grid. `latents=None` is the prompt
    rewriting stage's prefill: the same conditioning blocks, and no generated block.

    Deriving the geometry rather than passing it alongside the ids is also what keeps a corrupted id
    set audible: mangled ids mean the `<img>` runs are not found, and this raises. Geometry carried as
    its own conditioning value would still look correct while the ids were wrong. That is not
    hypothetical — `_apply_model` casts `context` to the compute dtype, bf16 at ~128k has a spacing of
    512, and routing the ids that way collapses six of the seven tokens this model uses onto each
    other. The ids arrive as an int conditioning value instead, which `convert_tensor` only moves
    across devices. Do not split the geometry back out.
    """
    ids = ids.to(torch.long)
    is_image = (ids == config.image_token_id).tolist()
    runs, run_start = [], None
    for index, flag in enumerate(is_image):
        if flag and run_start is None:
            run_start = index
        elif not flag and run_start is not None:
            runs.append(slice(run_start, index))
            run_start = None
    if run_start is not None:
        runs.append(slice(run_start, len(is_image)))

    cond_latents, cond_grids = _as_list(cond_latent), _as_list(cond_patch_grid)
    if len(cond_latents) != len(cond_grids):
        raise ValueError(f"{len(cond_latents)} conditioning latents but {len(cond_grids)} tower grids")
    expected = 2 * len(cond_latents) + (0 if latents is None else 1)
    if len(runs) != expected:
        raise ValueError(f"the token sequence carries {len(runs)} image runs; {len(cond_latents)} "
                         f"conditioning images{'' if latents is None else ' and a generated block'} need "
                         f"{expected}. The conditioning images did not arrive with their sequence.")

    step = {"ids": ids, "cond_blocks": [], "full_attention_slices": [], "rope_image_info": []}
    for index, (latent, grid) in enumerate(zip(cond_latents, cond_grids)):
        vae, vit = runs[2 * index], runs[2 * index + 1]
        height, width = latent.shape[-2], latent.shape[-1]
        patch_height, patch_width = (int(value) for value in grid)
        vae = slice(vae.start, vae.start + height * width)
        step["cond_blocks"].append({
            "vae_slice": vae, "vit_slice": vit, "timestep_position": vae.start - 1,
            "token_height": height, "token_width": width,
            "patch_height": patch_height, "patch_width": patch_width,
        })
        # `<joint_img_sep>` sits between the two runs and inside the joint span
        step["full_attention_slices"].append(slice(vae.start, vit.stop))
        step["rope_image_info"] += [(vae, (height, width)), (vit, (patch_height, patch_width))]

    if latents is not None:
        token_height, token_width = latents.shape[-2], latents.shape[-1]
        gen = runs[-1]
        # `<timestep>` always precedes the block; `<guidance>` and `<timestep_r>` only where the
        # checkpoint has their embedders, in the order `build_sequence` writes them. The same gate as
        # the builder, so the two cannot disagree about which slots the sequence actually carries.
        position = gen.start - 1 - (1 if config.cfg_distilled else 0) - (1 if config.use_meanflow else 0)
        step.update({
            "image_slice": slice(gen.start, gen.start + token_height * token_width),
            "token_height": token_height,
            "token_width": token_width,
            "timestep_position": position,
            "guidance_position": position + 1 if config.cfg_distilled else None,
            "timestep_r_position": position + 1 + int(config.cfg_distilled) if config.use_meanflow else None,
        })
        step["full_attention_slices"].append(gen)
        step["rope_image_info"].append((gen, (token_height, token_width)))
    return step


# The image grid a 1024 request produces: the checkpoint's resolution table base (1024) over the 16x
# patch size. The 2D positions of an image block grow with w*h — L+2016 at 64x64, L+4560 at 96x96 —
# and the reference only ever asks for sizes inside its table, so a wider grid would sit at rope
# phases the model never saw for image tokens. Legal, but untrained: renders stay plausible while
# following the prompt less and less. The rope base is rescaled by the grid's growth factor instead,
# which is what the reference's own `base_rescale_factor` does.
_TRAINED_GRID = 64


def build_rope_freqs(seq_len, head_dim, rope_image_info, base, device=None):
    """2D rope rotation matrices for one sequence, shape (1, 1, seq_len, head_dim // 2, 2, 2).

    Text and special tokens use their sequence index for both axes. An image block starting at
    token L with token grid (height, width) uses a row major meshgrid over
    y = L + (w * h - h) / 2 + row and x = L + (w * h - w) / 2 + column. Rope pair k uses the y
    position for even k and the x position for odd k, at frequency base ** (-2 * k / head_dim). A
    grid wider than the trained one raises the base before the frequencies are built.
    """
    pairs = head_dim // 2
    positions = torch.zeros(seq_len, 2, dtype=torch.float32, device=device)
    text_positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    grid_scale = max(1.0, max((math.sqrt(height * width) / _TRAINED_GRID
                               for _, (height, width) in rope_image_info), default=1.0))
    last_pos = 0
    for section, (height, width) in rope_image_info:
        start = section.start
        if last_pos < start:
            positions[last_pos:start, 0] = text_positions[last_pos:start]
            positions[last_pos:start, 1] = text_positions[last_pos:start]
        beta_y = start + (width * height - height) / 2
        beta_x = start + (width * height - width) / 2
        index = torch.arange(height * width, dtype=torch.float32, device=device)
        positions[start:start + height * width, 0] = beta_y + torch.div(index, width, rounding_mode="floor")
        positions[start:start + height * width, 1] = beta_x + index % width
        last_pos = start + height * width
    positions[last_pos:, 0] = text_positions[last_pos:]
    positions[last_pos:, 1] = text_positions[last_pos:]

    if grid_scale > 1.0:
        base *= grid_scale ** (head_dim / (head_dim - 2))
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    angles = positions[:, torch.arange(pairs, device=device) % 2] * theta
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    freqs = torch.stack([torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2)
    return freqs.reshape(1, 1, seq_len, pairs, 2, 2)


def _bank_linear(experts, input, i):
    """Expert `i` out of a bank already cast by `bank_resident`."""
    return experts.expert_linear(input, i)


def _swiglu(gate_up):
    # chunk(2, dim=-1) is a contiguous split, and the activation goes on the second half
    # (the fused projection is named gate_and_up_proj but the reference does x1 * silu(x2))
    x1, x2 = gate_up.chunk(2, dim=-1)
    return x1 * F.silu(x2)


def _split_qkv(qkv, num_key_value_heads, num_key_value_groups, head_dim):
    """(b, s, (groups + 2) * kv_heads * head_dim) -> q (b, s, heads, head_dim), k, v (b, s, kv_heads, head_dim).

    The fused projection is laid out per key/value head as [q_head * num_key_value_groups, k_head,
    v_head] — not as all q, then all k, then all v.
    """
    bsz, q_len, _ = qkv.shape
    qkv = qkv.reshape(bsz, q_len, num_key_value_heads, num_key_value_groups + 2, head_dim)
    query, key, value = torch.split(qkv, [num_key_value_groups, 1, 1], dim=3)
    query = query.reshape(bsz, q_len, num_key_value_heads * num_key_value_groups, head_dim)
    key = key.reshape(bsz, q_len, num_key_value_heads, head_dim)
    value = value.reshape(bsz, q_len, num_key_value_heads, head_dim)
    return query, key, value


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal timestep embedding, ported from the reference (modeling:151-178)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class LightProjector(nn.Module):
    """The vision tower's aligner: `mlp_gelu`, an MLP from the tower's width to the model's hidden size.

    The checkpoint's `vit_mapping_type` says `resampler`, but the reference constructs this from
    `config.vit_aligner` (`distil_modeling:1725`) and that config says `projector_type: mlp_gelu` with
    `depth: 2` — two Linears with a GELU between them, which is exactly the four `vision_aligner.*`
    keys the checkpoint carries. The name is config drift, not architecture (spec §46).
    """

    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        if config["projector_type"] != "mlp_gelu":
            raise ValueError(f"unsupported vit_aligner projector_type {config['projector_type']!r}")
        layers = [operations.Linear(config["input_dim"], config["n_embed"], dtype=dtype, device=device)]
        for _ in range(1, config["depth"]):
            layers.append(nn.GELU())
            layers.append(operations.Linear(config["n_embed"], config["n_embed"], dtype=dtype, device=device))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000, dtype=None, device=None, operations=None):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.GELU(),  # exact (erf) GELU, the reference default act_layer, not gelu_tanh
            operations.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )

    def forward(self, t):
        # the sinusoids are computed in fp32 and cast to the caller's compute dtype, matching the
        # reference's `.type(self.mlp[0].weight.dtype)`
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).to(t.dtype)
        return self.mlp(t_freq)


class ResBlock(nn.Module):
    def __init__(self, in_channels, emb_channels, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.in_layers = nn.Sequential(
            operations.GroupNorm(32, in_channels, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dtype=dtype, device=device),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            operations.Linear(emb_channels, 2 * out_channels, dtype=dtype, device=device),
        )
        self.out_layers = nn.Sequential(
            operations.GroupNorm(32, out_channels, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Identity(),  # reference dropout slot, kept as a no-op so the conv stays at index 3
            operations.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dtype=dtype, device=device),
        )
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = operations.Conv2d(in_channels, out_channels, kernel_size=1, dtype=dtype, device=device)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # adaptive group norm: the timestep embedding supplies the per-channel scale and shift
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_layers[0](h) * (1.0 + scale) + shift
        h = self.out_layers[1:](h)
        return self.skip_connection(x) + h


class UNetDown(nn.Module):
    def __init__(self, in_channels, emb_channels, hidden_channels, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.model = nn.ModuleList([
            operations.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, dtype=dtype, device=device),
            ResBlock(hidden_channels, emb_channels, out_channels, dtype=dtype, device=device, operations=operations),
        ])

    def forward(self, x, t):
        x = self.model[0](x)
        x = self.model[1](x, t)
        token_h, token_w = x.shape[-2:]
        return x.flatten(2).transpose(1, 2), token_h, token_w


class UNetUp(nn.Module):
    def __init__(self, in_channels, emb_channels, hidden_channels, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.model = nn.ModuleList([
            ResBlock(in_channels, emb_channels, hidden_channels, dtype=dtype, device=device, operations=operations),
            nn.Sequential(
                operations.GroupNorm(32, hidden_channels, dtype=dtype, device=device),
                nn.SiLU(),
                operations.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, dtype=dtype, device=device),
            ),
        ])

    def forward(self, x, t, token_h, token_w):
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], token_h, token_w)
        x = self.model[0](x, t)
        return self.model[1](x)


class HunyuanImage3Attention(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.attention_head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        hidden_size_q = self.head_dim * self.num_heads
        hidden_size_kv = self.head_dim * self.num_key_value_heads

        # fused q|k|v, laid out as [q_head * num_key_value_groups, k_head, v_head] per key/value head
        self.qkv_proj = operations.Linear(config.hidden_size, hidden_size_q + 2 * hidden_size_kv, bias=config.attention_bias, dtype=dtype, device=device)
        self.o_proj = operations.Linear(hidden_size_q, config.hidden_size, bias=config.attention_bias, dtype=dtype, device=device)
        self.query_layernorm = operations.RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, device=device)
        self.key_layernorm = operations.RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, device=device)

    def forward(self, hidden_states, freqs, attention_mask=None, kv_cache=None):
        bsz, q_len, _ = hidden_states.shape

        query, key, value = _split_qkv(self.qkv_proj(hidden_states), self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # rope goes in before the qk norm here, unlike the other models in this repo
        query, key = comfy.quant_ops.ck.apply_rope_split_half(query, key, freqs)
        query = self.query_layernorm(query)
        key = self.key_layernorm(key)

        # Autoregressive decode: append this step's keys and values to the cache the caller owns, before
        # the GQA expansion. What gets stored is what the norms produced, one head-set per key/value
        # head; caching after `repeat_kv_for_gqa` would multiply the stored tensors by the group size,
        # and the keys already carry the rotation their own position gave them. `freqs` therefore holds
        # only the positions being processed — every row of the prompt on the first call, one row per
        # token after that — and `attention_mask` is None when a single query attends to all keys.
        if kv_cache is not None:
            if kv_cache[0] is None:
                kv_cache[0], kv_cache[1] = key, value
            else:
                key = torch.cat([kv_cache[0], key], dim=2)
                value = torch.cat([kv_cache[1], value], dim=2)
                kv_cache[0], kv_cache[1] = key, value

        key, value = comfy.ops.repeat_kv_for_gqa(key, value, self.num_heads, -3)
        out = optimized_attention_masked(query, key, value, self.num_heads, attention_mask, skip_reshape=True, skip_output_reshape=True)
        return self.o_proj(out.transpose(1, 2).reshape(bsz, q_len, -1))


class HunyuanImage3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.gate_and_up_proj = operations.Linear(hidden_size, 2 * intermediate_size, bias=bias, dtype=dtype, device=device)
        self.down_proj = operations.Linear(intermediate_size, hidden_size, bias=bias, dtype=dtype, device=device)

    def forward(self, hidden_states):
        return self.down_proj(_swiglu(self.gate_and_up_proj(hidden_states)))


class HunyuanImage3MoEGate(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k, dtype=None, device=None, operations=None):
        super().__init__()
        self.top_k = top_k
        self.wg = operations.Linear(hidden_size, num_experts, bias=False, dtype=dtype, device=device)

    def forward(self, hidden_states):
        # The router GEMM runs in the *weight's* dtype, exactly as the reference does: it upcasts the
        # input only while the weight is fp32, so a bf16 load (the deployment, and ComfyUI's default
        # compute dtype) routes in bf16. Upcasting unconditionally would make the ops Linear cast the
        # weight to match and run the GEMM in fp32 instead — measured at 10-12% different expert
        # selections versus bf16, i.e. a deviation from the released model's behaviour.
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        probabilities = self.wg(hidden_states).float().softmax(dim=-1)
        top_k_weights, top_k_index = torch.topk(probabilities, self.top_k, dim=-1)
        weight_sums = top_k_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return top_k_weights / weight_sums, top_k_index


class HunyuanImage3MoE(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.moe_topk
        self.shared_mlp = HunyuanImage3MLP(
            config.hidden_size, config.moe_intermediate_size * config.num_shared_expert,
            bias=config.mlp_bias, dtype=dtype, device=device, operations=operations,
        )
        self.gate = HunyuanImage3MoEGate(
            config.hidden_size, self.num_experts, self.top_k,
            dtype=dtype, device=device, operations=operations,
        )
        self.experts_gate_up_proj = operations.MoEExperts(
            self.num_experts, config.hidden_size, 2 * config.moe_intermediate_size,
            bias=False, device=device, dtype=dtype,
        )
        self.experts_down_proj = operations.MoEExperts(
            self.num_experts, config.moe_intermediate_size, config.hidden_size,
            bias=False, device=device, dtype=dtype,
        )

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_size)
        top_k_weights, top_k_index = self.gate(flat)
        top_k_weights = top_k_weights.to(hidden_states.dtype)

        # one pass per expert that any token was routed to
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        combined = torch.zeros((flat.shape[0] * self.top_k, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
        # A whole sequence routes to nearly every expert, so one cast of the bank serves all of them, and
        # that is what the bank context is for. A decode step routes to a handful of experts: casting the
        # bank for it would stream every expert in the layer rather than the few the token uses, which at
        # 64 experts is most of the model per token (spec §51). Both paths are numerically identical —
        # `test_expert_cast_paths_agree` asserts it — so the choice is purely about what gets moved.
        # A bank the layer prefetch already faulted in costs nothing more to use whole, so the bank path
        # takes it whatever the routing says.
        full_bank = (len(expert_hit) * 2 >= self.num_experts
                     or hasattr(self.experts_gate_up_proj, "_prefetch"))
        gate_up_bank = (self.experts_gate_up_proj.bank_resident(flat) if full_bank
                        else contextlib.nullcontext(self.experts_gate_up_proj))
        down_bank = (self.experts_down_proj.bank_resident(flat) if full_bank
                     else contextlib.nullcontext(self.experts_down_proj))
        # outside a resident bank, fetch each routed expert's slice rather than letting the core cast
        # the bank for every one of them (see `ops.expert_linear_sliced`)
        linear = _bank_linear if full_bank else expert_linear_sliced
        with gate_up_bank as gate_up_experts, down_bank as down_experts:
            for expert in expert_hit:
                expert_index = int(expert.item())
                top_k_pos, token_index = torch.where(expert_mask[expert_index])
                gate_up = linear(gate_up_experts, flat[token_index], expert_index)
                expert_out = linear(down_experts, _swiglu(gate_up), expert_index)
                combined[token_index * self.top_k + top_k_pos] = (expert_out * top_k_weights[token_index, top_k_pos, None]).to(combined.dtype)

        # (N * top_k, hidden) then sum, not index_add_ into (N, hidden): measured 2.34e-03 vs 4.00e-03
        # relative error against fp64 for the same inputs, because the reduction accumulates internally
        # while index_add_ rounds into bf16 on every step. The buffer is 268 MB at 4096 tokens (the
        # whole activation share of the peak); the 0.63 GiB it costs is cheaper than the accuracy.
        routed = combined.view(bsz, seq_len, self.top_k, hidden_size).sum(dim=2)
        return self.shared_mlp(hidden_states) + routed


class HunyuanImage3DecoderLayer(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.input_layernorm = operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = HunyuanImage3Attention(config, dtype=dtype, device=device, operations=operations)
        self.post_attention_layernorm = operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)
        self.mlp = HunyuanImage3MoE(config, dtype=dtype, device=device, operations=operations)

    def forward(self, hidden_states, freqs, attention_mask=None, kv_cache=None):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), freqs, attention_mask, kv_cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class HunyuanImage3Model(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.wte = operations.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id, dtype=dtype, device=device)
        self.layers = nn.ModuleList([HunyuanImage3DecoderLayer(config, dtype=dtype, device=device, operations=operations) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, freqs, attention_mask=None, kv_cache=None, transformer_options=None):
        """`kv_cache` is one `[keys, values]` pair per layer, owned by the caller.

        The diffusion path passes none and recomputes the whole sequence every step (that is its
        semantics — see `HunyuanImage3.forward`). Autoregressive text decode passes a list and keeps it
        across steps, which is the difference the CoT stage needs: token-by-token generation would
        otherwise re-run the entire prefix for every token.

        `transformer_options` arrives only from the sampler path, and with it a one-layer lookahead
        (`lookahead.LayerLookahead`): while layer N computes, layer N+1's weights — its two ~0.5 GiB
        expert banks included — are faulted in on an offload stream. The CoT decode passes none on
        purpose: it fetches single expert slices, and faulting whole banks ahead for it would be exactly the
        traffic `expert_linear_sliced` avoids.
        """
        # The CoT stage (it passes a kv cache) runs without: a lookahead over its non-bank modules was
        # measured at 861 vs 872 ms per token, i.e. nothing, and its whole-bank form is what
        # `expert_linear_sliced` exists to avoid.
        lookahead = LayerLookahead(
            self.layers, hidden_states.device,
            kv_cache is None and (transformer_options or {}).get("prefetch_dynamic_vbars", False))
        try:
            for index, layer in enumerate(self.layers):
                lookahead.before(index)
                hidden_states = layer(hidden_states, freqs, attention_mask,
                                      None if kv_cache is None else kv_cache[index])
                lookahead.after(index)
        finally:
            lookahead.abort()
        return hidden_states


class HunyuanImage3(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        if config.patch_size != 1:
            raise ValueError(f"unsupported patch size {config.patch_size}, only 1 is supported")

        self.config = config
        # `BaseModel.get_dtype()` reads this off the diffusion model, the way every in-tree model here
        # exposes it; the pipeline used to resolve the compute dtype separately and keep its own copy
        self.dtype = dtype
        self.model = HunyuanImage3Model(config, dtype=dtype, device=device, operations=operations)
        self.vision_aligner = LightProjector(config.vit_aligner, dtype=dtype, device=device,
                                             operations=operations)

        self.patch_embed = UNetDown(config.vae_latent_channels, config.hidden_size, config.patch_embed_hidden_dim, config.hidden_size, dtype=dtype, device=device, operations=operations)
        self.final_layer = UNetUp(config.hidden_size, config.hidden_size, config.patch_embed_hidden_dim, config.vae_latent_channels, dtype=dtype, device=device, operations=operations)

        self.time_embed = TimestepEmbedder(config.hidden_size, dtype=dtype, device=device, operations=operations)
        self.time_embed_2 = TimestepEmbedder(config.hidden_size, dtype=dtype, device=device, operations=operations)
        self.timestep_emb = TimestepEmbedder(config.hidden_size, dtype=dtype, device=device, operations=operations)
        if config.cfg_distilled:
            self.guidance_emb = TimestepEmbedder(config.hidden_size, dtype=dtype, device=device, operations=operations)
        if config.use_meanflow:
            self.timestep_r_emb = TimestepEmbedder(config.hidden_size, dtype=dtype, device=device, operations=operations)

    def forward(self, x, timestep, context=None, transformer_options=None, guidance=None, ids=None,
                cond_latent=None, cond_vit=None, cond_vit_grid=None, **kwargs):
        """One denoising step: rebuild the input sequence and return the velocity.

        This is what `BaseModel._apply_model` calls and what a stock sampler drives. `x` is the LATENT
        — `(b, c, t, h, w)` with a single frame, since a `latent_dim == 3` VAE's LATENT carries a
        temporal dim — and the velocity comes back in that same shape, because `calculate_denoised`
        broadcasts against it.

        The token ids arrive as `ids`, an extra conditioning value rather than `context`, because
        `_apply_model` casts `context` to the compute dtype and bf16 cannot hold a token id: the
        vocabulary runs to 152k and bf16 carries 8 mantissa bits, so ids would silently collide.
        `convert_tensor` only moves int/long tensors across devices, which is why this route is safe.

        The sequence geometry is derived from the ids rather than carried: the image block is a
        contiguous run of `image_token_id`, its grid is the latent's own (patch size 1), and the three
        embedder slots are the last three tokens before the block — `<timestep>`, `<guidance>`,
        `<timestep_r>`, in the order `build_sequence` writes them. A test pins that ordering against
        `build_sequence` so the two cannot drift apart.

        `timestep` has already been through the sampling class's `timestep(sigma) = sigma * 1000`, the
        same mapping the pipeline's `sigmas[:-1] * 1000` applies. `timestep_r` comes from the sampler's
        own schedule (`transformer_options["sample_sigmas"]`), the in-tree pattern for meanflow models
        and a better source than the pipeline's assumed schedule — it is the schedule actually run.

        The mask and the RoPE frequencies are rebuilt per step rather than cached. They are ~114 MB and
        ~0.2 s at 5348 tokens against a 5.6 s forward, and this codebase's rule is that temporary
        caches belong to a single execution rather than to the model, which a per-step rebuild
        satisfies without an owner to invalidate.

        With a `hy3_spectrum` state in `transformer_options` (the Spectrum node) the block stack is
        skipped on the steps that state's plan calls skips, and the image hidden states are forecast
        from the steps that did run. Without one, nothing here changes.
        """
        from .pipeline import NUM_TRAIN_TIMESTEPS, build_attention_mask, build_input_embeddings

        transformer_options = transformer_options or {}
        if ids is None:
            raise ValueError("HunyuanImage3 requires token ids; the conditioning must carry them")

        latents = x[:, :, 0] if x.dim() == 5 else x
        cond_latent = _as_list(cond_latent)
        step = _sequence_from_ids(ids.flatten(), latents, self.config, cond_latent=cond_latent,
                                  cond_patch_grid=cond_vit_grid)
        seq_len = step["ids"].shape[0]
        device, dtype = x.device, x.dtype

        sample_sigmas = transformer_options.get("sample_sigmas")
        sigmas = transformer_options.get("sigmas")
        if sample_sigmas is None or sigmas is None:
            raise ValueError("HunyuanImage3 needs the sampler's schedule (sample_sigmas) to resolve timestep_r")
        index = torch.where(sigmas.flatten()[0] == sample_sigmas)[0]
        timestep_r = (sample_sigmas[index[0] + 1] * NUM_TRAIN_TIMESTEPS).reshape(1).to(device=device, dtype=torch.float32)

        # Spectrum skips the block stack on some steps and forecasts the image hidden states instead.
        # The step index is the sampler's own, taken from the schedule it is actually running, so a
        # sampler that calls the model more than once per step still follows one plan and both guidance
        # passes agree on which steps run.
        spectrum = transformer_options.get("hy3_spectrum")
        step_index = int(index[0])
        key = t = None
        run_stack = True
        if spectrum is not None:
            spectrum.begin(int(sample_sigmas.numel()) - 1)
            run_stack = spectrum.should_run(step_index)
            spectrum.note_step(step_index, run_stack)
            key = (transformer_options.get("cond_or_uncond") or [0])[0]
            t = spectrum.time_for(step_index, sigma=float(sigmas.flatten()[0]),
                                  sigma_range=(float(sample_sigmas.min()), float(sample_sigmas.max())))
            if spectrum.verbose:
                logging.info("[Spectrum] step %d: latent batch %d, cond_or_uncond %s, %s", step_index,
                             x.shape[0], transformer_options.get("cond_or_uncond"),
                             "forward" if run_stack else "skip")

        # the embedders take a batch of timesteps; the sampler hands sigma over in the latent's shape
        timestep = timestep.flatten()[:1]
        if spectrum is not None and not run_stack and not spectrum.validate:
            # a skipped step needs none of the stack's inputs: the mask, the rope table and the input
            # embeddings are rebuilt every step (~0.2 s at 5k tokens), which on a skip was nearly its
            # whole remaining cost
            image_hidden = spectrum.predict(key, t, step_index)
        else:
            mask = build_attention_mask(step, seq_len, dtype, device)
            freqs = build_rope_freqs(seq_len, self.config.attention_head_dim, step["rope_image_info"],
                                     self.config.rope_theta, device=device)
            embeds = build_input_embeddings(self, step, latents.to(dtype), timestep, guidance, timestep_r,
                                            cond_latent=cond_latent, cond_vit=cond_vit)
            hidden = self.model(embeds, freqs, mask, transformer_options=transformer_options)
            image_hidden = hidden[0, step["image_slice"]]
            if spectrum is not None:
                if run_stack:
                    spectrum.store(key, t, image_hidden)
                else:
                    # validate: the real forward has already run, so only its forecast error is kept
                    spectrum.log_validation(step_index, spectrum.predict(key, t, step_index), image_hidden)
        pred = self.final_layer(image_hidden.unsqueeze(0), self.time_embed_2(timestep),
                                step["token_height"], step["token_width"])
        return pred.unsqueeze(2) if x.dim() == 5 else pred


class HunyuanImage3Generator:
    """Owns one HunyuanImage3, the operations it was built with, and its ModelPatcher.

    The model does not fit the denoiser contract that `BaseModel.apply_model` and KSampler expect:
    it next-token predicts text and diffusion-predicts image tokens in the same stack, driven by its
    own loop. So it is wired the way this repo's other self-contained models are — CLIP, the VAE,
    the hunyuan video upsampler: a plain nn.Module wrapped in a ModelPatcher, with loading and
    inference owned by the class that needs them.
    """

    def __init__(self, config, quant_config=None):
        self.params = params_from_config(config)
        self.quant_config = quant_config
        self.load_device = comfy.model_management.get_torch_device()
        self.offload_device = comfy.model_management.unet_offload_device()
        self.dtype = comfy.model_management.unet_dtype(supported_dtypes=[torch.bfloat16, torch.float16, torch.float32])

        # pick_operations reads .quant_config off whatever it is handed; the supported-models config
        # classes carry that for denoisers and this model has no supported-models entry of its own.
        # A quantized checkpoint selects mixed_precision_ops, which is also the only ops object with
        # the expert bank class the MoE needs.
        operations = comfy.ops.pick_operations(self.dtype, self.dtype, load_device=self.load_device, model_config=self)
        self.model = HunyuanImage3(self.params, dtype=self.dtype, device=self.offload_device, operations=operations)
        self.model.requires_grad_(False)
        self.model.eval()
        comfy.model_management.archive_model_dtypes(self.model)

        # attribute access is deliberate: main.py rebinds comfy.model_patcher.CoreModelPatcher to
        # ModelPatcherDynamic at startup when comfy-aimdo works, and a from-import here would capture
        # the plain ModelPatcher alias before that happens — silently, with no log line, on exactly
        # the large offloaded models this class exists for
        self.patcher = comfy.model_patcher.CoreModelPatcher(self.model, load_device=self.load_device, offload_device=self.offload_device)

    def load_sd(self, state_dict):
        return self.model.load_state_dict(state_dict, strict=False, assign=self.patcher.is_dynamic())

    def get_sd(self):
        return self.model.state_dict()
