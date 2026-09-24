"""Step 4a gate: the module tree is visible to ComfyUI's memory manager.

No checkpoint is involved — this is about how the tree is built (which ops object, castable
weights, expert banks), which is what a lowvram/offload decision depends on. The real-weights load
measurement needs the int8 checkpoint and belongs to Step 4b.
"""
import json
import functools
import os

import pytest
import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management
import comfy.model_patcher
import comfy.ops
from comfy.ldm.hunyuan_image_3.model import HunyuanImage3Generator
from op_module_init import init_op_modules

MODEL_DIR = os.environ.get("HUNYUAN_IMAGE_3_MODEL_DIR")
needs_model_dir = pytest.mark.skipif(not MODEL_DIR, reason="set HUNYUAN_IMAGE_3_MODEL_DIR to the checkpoint directory")

QUANT_CONFIG = {"mixed_ops": True}  # what detect_layer_quantization returns for a comfy_quant checkpoint

# the smallest tree that still exercises every module type in the model
SMALL_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "attention_head_dim": 16,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 512,
    "attention_bias": False,
    "mlp_bias": False,
    "moe_intermediate_size": 32,
    "num_experts": 8,
    "moe_topk": 2,
    "num_shared_expert": 1,
    "use_mixed_mlp_moe": True,
    "hidden_act": "silu",
    "cfg_distilled": True,
    "use_meanflow": True,
    "pad_token_id": 0,
    "image_token_id": 128006,
    "vae_downsample_factor": [16, 16],
    "vae": {"latent_channels": 32},
    "vit_aligner": {"projector_type": "mlp_gelu", "input_dim": 8, "n_embed": 32, "depth": 2},
}


def _model_dir():
    if not MODEL_DIR:
        pytest.skip("set HUNYUAN_IMAGE_3_MODEL_DIR to the checkpoint directory")
    return MODEL_DIR


def _real_config():
    with open(os.path.join(_model_dir(), "config.json")) as handle:
        return json.load(handle)


def parameter_bearing(module):
    """Weight-holding module types, by class rather than by attribute names.

    `_orig_shape` is the marker for a mixed-precision Linear: it is deliberately not an
    `nn.Linear` subclass (`class Linear(torch.nn.Module, MixedPrecisionOp)`), so an isinstance check
    alone would leave the quantized path — the one that matters most — out of the census. The expert
    bank is matched by both attributes, because the MoE container stores `num_experts` for its
    children and would otherwise look parameter-bearing.
    """
    return (isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LayerNorm, nn.GroupNorm, nn.RMSNorm, nn.Embedding))
            or getattr(module, "_orig_shape", None) is not None
            or (hasattr(module, "num_experts") and hasattr(module, "in_features")))


def has_own_parameters(module):
    return any(True for _ in module.parameters(recurse=False))


def hidden_from_the_offloader(model):
    """Modules that hold weights but carry no comfy_cast_weights, so lowvram would never see them."""
    return [name for name, m in model.named_modules()
            if not hasattr(m, "comfy_cast_weights") and (parameter_bearing(m) or has_own_parameters(m))]


def test_quantized_config_selects_mixed_precision_ops():
    generator = HunyuanImage3Generator(SMALL_CONFIG, quant_config=QUANT_CONFIG)

    # mixed_precision_ops is the only ops object that both quantizes and provides the expert banks
    assert isinstance(generator.model.model.layers[0].self_attn.o_proj, comfy.ops.MixedPrecisionOp)
    assert isinstance(generator.model.model.layers[0].mlp.experts_gate_up_proj, comfy.ops.MixedPrecisionOp)
    assert sum(1 for m in generator.model.modules() if isinstance(m, comfy.ops.MixedPrecisionOp)) > 0
    assert generator.quant_config is QUANT_CONFIG


def test_every_parameter_bearing_module_can_be_offloaded():
    generator = HunyuanImage3Generator(SMALL_CONFIG, quant_config=QUANT_CONFIG)
    model = generator.model

    # fill the parameters that only exist after a load, so the census covers them too
    init_op_modules(model, generator.dtype)

    n_total = sum(1 for _ in model.modules())
    n_castable = sum(1 for m in model.modules() if hasattr(m, "comfy_cast_weights"))
    assert hidden_from_the_offloader(model) == []
    assert has_own_parameters(model.model.layers[0].mlp.experts_gate_up_proj)
    assert has_own_parameters(model.model.layers[0].self_attn.o_proj)
    print(f"modules: {n_total} total, {n_castable} with comfy_cast_weights")


def test_construction_without_quant_config_fails_loudly():
    # bf16 carries no quant_config, so pick_operations returns disable_weight_init/manual_cast, and
    # neither has MoEExperts: the MoE needs the bank class, so construction fails instead of quietly
    # building 64 separate Linears (Addendum 2 §B1)
    with pytest.raises(AttributeError, match="MoEExperts"):
        HunyuanImage3Generator(SMALL_CONFIG, quant_config=None)


def test_generator_wraps_the_model_in_a_patcher():
    generator = HunyuanImage3Generator(SMALL_CONFIG, quant_config=QUANT_CONFIG)

    assert isinstance(generator.patcher, comfy.model_patcher.ModelPatcher)
    assert generator.patcher.load_device == generator.load_device
    assert generator.patcher.offload_device == generator.offload_device
    # ModelPatcher sets it when the module has no device attribute of its own
    assert generator.model.device == generator.offload_device
    assert generator.get_sd() is not None
    assert generator.load_sd(generator.model.state_dict()) is not None


def test_patcher_class_is_resolved_late(monkeypatch):
    """main.py rebinds CoreModelPatcher to ModelPatcherDynamic at startup.

    pytest never runs main.py, so the production patcher cannot be exercised here. What can be
    pinned is that the generator resolves the name when it builds the patcher: a from-import would
    capture the plain alias instead, with no error and no log line.
    """
    assert comfy.model_patcher.CoreModelPatcher is comfy.model_patcher.ModelPatcher  # no main.py in this process

    class SentryPatcher(comfy.model_patcher.ModelPatcher):
        pass

    monkeypatch.setattr(comfy.model_patcher, "CoreModelPatcher", SentryPatcher)
    generator = HunyuanImage3Generator(SMALL_CONFIG, quant_config=QUANT_CONFIG)
    assert isinstance(generator.patcher, SentryPatcher)


@needs_model_dir
def test_real_config_census():
    generator = HunyuanImage3Generator(_real_config(), quant_config=QUANT_CONFIG)
    model = generator.model
    params = generator.params

    banks = [m for m in model.modules() if hasattr(m, "num_experts") and hasattr(m, "out_features")]
    assert len(banks) == 2 * params.num_hidden_layers
    for layer in range(params.num_hidden_layers):
        assert model.model.layers[layer].mlp.experts_gate_up_proj._orig_shape == (params.num_experts, 2 * params.moe_intermediate_size, params.hidden_size)
        assert model.model.layers[layer].mlp.experts_down_proj._orig_shape == (params.num_experts, params.hidden_size, params.moe_intermediate_size)

    # no init_op_modules here: at the real geometry that would materialise ~155 GB of banks
    assert hidden_from_the_offloader(model) == []
    # the text head is not part of the diffusion model: prompt rewriting loads its own
    assert not hasattr(model, "lm_head") and not hasattr(model.model, "ln_f")
    assert isinstance(model.model.layers[0].mlp.experts_gate_up_proj, comfy.ops.MixedPrecisionOp)
    print(f"real config: {sum(1 for _ in model.modules())} modules, {len(banks)} expert banks")


@needs_model_dir
def test_sequence_geometry_is_derived_from_the_ids():
    """The model derives its sequence layout from the token ids; `build_sequence` writes that layout.

    On the native path nothing carries the geometry across the conditioning boundary — the model reads
    the image block's start out of the ids and takes the three embedder slots as the three tokens
    before it. If that derivation and `build_sequence` ever disagree, the timestep, guidance, and
    timestep_r embeddings land in the wrong positions and the model conditions on the wrong tokens,
    with no error to show for it. This pins the two together.
    """
    from tokenizers import Tokenizer

    from comfy.ldm.hunyuan_image_3.model import _sequence_from_ids, params_from_config
    from comfy.ldm.hunyuan_image_3.system_prompt import UNIFIED_SYSTEM_PROMPT_EN
    from comfy.ldm.hunyuan_image_3.tokenizer import build_sequence as _build_sequence

    # the reference capture is from the Instruct-Distil checkpoint, whose sequence carries
    # `<guidance>` and `<timestep_r>`; the pack requires that contract explicitly rather than
    # defaulting it, because rendering one checkpoint under another's is silent
    build_sequence = functools.partial(_build_sequence, cfg_distilled=True, use_meanflow=True)

    params = params_from_config(_real_config())
    tokenizer = Tokenizer.from_file(os.path.join(_model_dir(), "tokenizer.json"))
    built = build_sequence(tokenizer, "a red fox asleep in tall grass", "1024x1024",
                           UNIFIED_SYSTEM_PROMPT_EN, base_size=params.image_base_size)
    latents = torch.zeros(1, params.vae_latent_channels, built["token_height"], built["token_width"])
    derived = _sequence_from_ids(built["ids"], latents, params)

    for key in ("image_slice", "token_height", "token_width",
                "timestep_position", "guidance_position", "timestep_r_position"):
        assert derived[key] == built[key], f"{key}: derived {derived[key]!r}, built {built[key]!r}"

    # the image block really is one contiguous run, which is what lets the start locate the slice
    image_ids = (built["ids"] == params.image_token_id).nonzero().flatten()
    assert int(image_ids.numel()) == built["token_height"] * built["token_width"]
    assert built["ids"][built["image_slice"]].unique().tolist() == [params.image_token_id]


@needs_model_dir
def test_off_table_size_keeps_its_own_grid_and_snaps_only_the_ratio_token():
    """A custom size generates at the size asked for; only the `<img_ratio_N>` hint snaps.

    Both halves matter, and they are allowed to disagree because nothing derives geometry from the
    token. If the grid were snapped, 1152x896 would quietly generate at a table row; if the ratio token
    were derived from the grid, the model would receive a hint it cannot express. The width-first
    convention in the label is load-bearing too — the `x` form used to be parsed height-first, which a
    square test could never see.
    """
    from tokenizers import Tokenizer

    from comfy.ldm.hunyuan_image_3.system_prompt import UNIFIED_SYSTEM_PROMPT_EN
    from comfy.ldm.hunyuan_image_3.tokenizer import (PRESET_CUSTOM, ImageGeometry, ResolutionGroup,
                                                     build_sequence as _build_sequence,
                                                     preset_options, resolve_size)
    build_sequence = functools.partial(_build_sequence, cfg_distilled=True, use_meanflow=True)

    group = ResolutionGroup(1024)
    # 1152x896 turned out to *be* a staircase row, so this uses a size the table cannot express
    width, height = resolve_size(PRESET_CUSTOM, 1200, 880)
    assert (width, height) == (1200, 880)
    assert (width, height) not in {(resolution.width, resolution.height) for resolution in group.data}
    assert (width // 16, height // 16) == (75, 55)

    geometry = ImageGeometry(f"{width}x{height}")
    assert (geometry.width, geometry.height) == (1200, 880)              # the grid is the request
    assert (geometry.token_width, geometry.token_height) == (75, 55)     # 1200/16, 880/16
    assert group.get_target_size(width, height) != (1200, 880)           # the nearest row is elsewhere
    assert geometry.ratio_index == group.nearest_index(width, height)

    tokenizer = Tokenizer.from_file(os.path.join(_model_dir(), "tokenizer.json"))
    built = build_sequence(tokenizer, "a red fox asleep in tall grass", f"{width}x{height}",
                           UNIFIED_SYSTEM_PROMPT_EN, base_size=1024)
    assert (built["token_width"], built["token_height"]) == (75, 55)
    assert (built["image_width"], built["image_height"]) == (1200, 880)
    assert built["image_slice"].stop - built["image_slice"].start == 75 * 55

    # a preset still resolves to its own table row, and `custom` is offered alongside them
    assert resolve_size(f"{group[0].width}x{group[0].height}", 0, 0) == (group[0].width, group[0].height)
    assert PRESET_CUSTOM in preset_options(1024, with_custom=True)
    assert PRESET_CUSTOM not in preset_options(1024)   # the custom entry is opt-in


@needs_model_dir
def test_a_sequence_past_the_position_limit_is_rejected_with_both_numbers():
    """The guard names the request and the limit rather than truncating it silently.

    A 3072x2048 request is a 192x128 grid: 24576 image tokens plus ~1260 of prompt and meta against a
    limit of 22800. Truncating instead would generate something other than what was asked for.
    """
    from tokenizers import Tokenizer

    from comfy.ldm.hunyuan_image_3.system_prompt import UNIFIED_SYSTEM_PROMPT_EN
    from comfy.ldm.hunyuan_image_3.tokenizer import build_sequence as _build_sequence
    build_sequence = functools.partial(_build_sequence, cfg_distilled=True, use_meanflow=True)

    tokenizer = Tokenizer.from_file(os.path.join(_model_dir(), "tokenizer.json"))
    with pytest.raises(ValueError) as error:
        build_sequence(tokenizer, "a red fox asleep in tall grass", "3072x2048",
                       UNIFIED_SYSTEM_PROMPT_EN, base_size=1024, max_position_embeddings=22800)
    message = str(error.value)
    assert "3072x2048" in message and "22800" in message and "24576" in message

