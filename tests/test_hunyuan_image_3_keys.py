"""Step 3 tests: config parsing, the checkpoint -> module key map, and the 64:1 expert stacking.

The real `config.json` and `model.safetensors.index.json` are read only when
`HUNYUAN_IMAGE_3_MODEL_DIR` points at a directory holding them (the index alone is 447 KB; no
weights are needed). Everything else is built from fixtures so the file always runs.
"""
import json
import os

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ops
from op_module_init import init_op_modules
from comfy.ldm.hunyuan_image_3.model import (
    DEFERRED_PREFIXES,
    HunyuanImage3,
    HunyuanImage3Params,
    params_from_config,
    remap_key,
    remap_state_dict,
    skipped_keys,
)

MODEL_DIR = os.environ.get("HUNYUAN_IMAGE_3_MODEL_DIR")
needs_model_dir = pytest.mark.skipif(not MODEL_DIR, reason="set HUNYUAN_IMAGE_3_MODEL_DIR to the checkpoint directory")
# the two index tests read the HuggingFace repo's shard index, which a converted single-file checkpoint's
# folder does not carry; pointing the variable at that folder is the common case, so skip rather than fail
INDEX_NAME = "model.safetensors.index.json"
needs_hf_index = pytest.mark.skipif(
    not (MODEL_DIR and os.path.isfile(os.path.join(MODEL_DIR, INDEX_NAME))),
    reason=f"set HUNYUAN_IMAGE_3_MODEL_DIR to the original HuggingFace directory (with {INDEX_NAME})")


def _model_dir():
    if not MODEL_DIR:
        pytest.skip("set HUNYUAN_IMAGE_3_MODEL_DIR to the checkpoint directory")
    return MODEL_DIR

# The released config, trimmed: values for the fields the model reads, plus the lists that the
# real config spells out per layer.
CONFIG = {
    "vocab_size": 133120,
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "attention_head_dim": 128,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 22800,
    "attention_bias": False,
    "mlp_bias": False,
    "moe_intermediate_size": [3072] * 32,
    "num_experts": 64,
    "moe_topk": [8] * 32,
    "num_shared_expert": [1] * 32,
    "cfg_distilled": True,
    "use_meanflow": True,
    "pad_token_id": 128009,
    "image_token_id": 128006,
    "hidden_act": "silu",
    "use_mixed_mlp_moe": True,
    "moe_layer_num_skipped": 0,
    "vae_downsample_factor": [16, 16],
    "vae": {"latent_channels": 32},
    "vit_aligner": {"projector_type": "mlp_gelu", "input_dim": 8, "n_embed": 32, "depth": 2},
}


def make_config(**overrides):
    config = dict(CONFIG)
    config.update(overrides)
    return config


def test_config_parses_to_the_expected_params():
    params = params_from_config(make_config())
    assert params == HunyuanImage3Params(
        vocab_size=133120, hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=8, attention_head_dim=128, rms_norm_eps=1e-5, rope_theta=10000.0,
        max_position_embeddings=22800, attention_bias=False, mlp_bias=False,
        moe_intermediate_size=3072, num_experts=64, moe_topk=8, num_shared_expert=1,
        cfg_distilled=True, use_meanflow=True, model_type="instruct_distil",
        sequence_template="instruct", pad_token_id=128009, image_token_id=128006, patch_size=1,
        patch_embed_hidden_dim=1024, image_base_size=1024, vae_latent_channels=32,
        vae_downsample_factor=(16, 16),
        vit_aligner={"projector_type": "mlp_gelu", "input_dim": 8, "n_embed": 32, "depth": 2},
    )


@pytest.mark.parametrize("override", [
    {"moe_layer_num_skipped": 2},          # asserts the model is MoE in every layer
    {"use_mixed_mlp_moe": False},          # asserts the shared expert is always added
    {"hidden_act": "gelu"},                # asserts the fused SwiGLU shapes
    {"moe_topk": [8] * 31 + [4]},          # asserts per layer uniformity
])
def test_config_asserts_reject_a_different_checkpoint(override):
    with pytest.raises(ValueError):
        params_from_config(make_config(**override))


def test_key_map_remaps_only_the_expert_tensors():
    assert remap_key("model.layers.7.self_attn.qkv_proj.weight") == ("model.layers.7.self_attn.qkv_proj.weight", None)
    assert remap_key("patch_embed.model.1.emb_layers.1.bias") == ("patch_embed.model.1.emb_layers.1.bias", None)
    assert remap_key("model.layers.7.mlp.experts.63.gate_and_up_proj.weight") == ("model.layers.7.mlp.experts_gate_up_proj.weight", 63)
    assert remap_key("model.layers.0.mlp.experts.0.down_proj.weight") == ("model.layers.0.mlp.experts_down_proj.weight", 0)


def test_stacking_preserves_expert_order_and_shapes():
    state_dict = {}
    for layer in range(2):
        for expert in range(3):
            state_dict[f"model.layers.{layer}.mlp.experts.{expert}.gate_and_up_proj.weight"] = torch.full((4, 2), layer * 10 + expert)
            state_dict[f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"] = torch.full((2, 4), -(layer * 10 + expert))
    state_dict["lm_head.weight"] = torch.zeros(1)
    state_dict["vae.decoder.conv_out.weight"] = torch.zeros(1)

    remapped, skipped = remap_state_dict(state_dict)

    # the text head is the prompt-rewriting stage's, loaded from its own file
    assert skipped == ["lm_head.weight", "vae.decoder.conv_out.weight"]
    for layer in range(2):
        gate_up = remapped[f"model.layers.{layer}.mlp.experts_gate_up_proj.weight"]
        down = remapped[f"model.layers.{layer}.mlp.experts_down_proj.weight"]
        assert gate_up.shape == (3, 4, 2)
        assert down.shape == (3, 2, 4)
        for expert in range(3):
            assert torch.equal(gate_up[expert], torch.full((4, 2), layer * 10 + expert))
            assert torch.equal(down[expert], torch.full((2, 4), -(layer * 10 + expert)))


def test_stacking_rejects_a_missing_expert():
    state_dict = {f"model.layers.0.mlp.experts.{expert}.down_proj.weight": torch.zeros(2, 4) for expert in (0, 1, 3)}
    with pytest.raises(ValueError, match="expected 4 expert tensors, got 3"):
        remap_state_dict(state_dict)


def test_deferred_keys_are_reported_not_unexpected():
    state_dict = {
        "model.layers.0.self_attn.o_proj.weight": torch.zeros(2, 2),
        "lm_head.weight": torch.zeros(2, 2),
        "model.ln_f.weight": torch.zeros(2),
        "vae.encoder.conv_in.weight": torch.zeros(2),
        "vision_model.embeddings.patch_embedding.weight": torch.zeros(2),
        # the aligner is *not* deferred: it lives in this checkpoint and the image-to-image path runs
        # it, so it loads like any other weight
        "vision_aligner.layers.0.weight": torch.zeros(2),
    }
    # the text head is deferred too: prompt rewriting loads it from its own file
    kept = ("model.layers.0.self_attn.o_proj.weight", "vision_aligner.layers.0.weight")
    assert skipped_keys(state_dict) == [key for key in state_dict if key not in kept]

    remapped, skipped = remap_state_dict(state_dict)
    assert set(remapped) == set(kept)
    assert len(skipped) == 4                              # the head, the vae and the vision tower
    assert all(prefix in DEFERRED_PREFIXES for prefix in ("vae.", "vision_model.", "lm_head.", "model.ln_f."))


@needs_hf_index
def test_key_map_accounts_for_every_key_of_the_real_index():
    with open(os.path.join(_model_dir(), "model.safetensors.index.json")) as handle:
        index = json.load(handle)["weight_map"]
    keys = list(index)
    params = params_from_config(_real_config())

    # Dummy tensors stay empty: this test is about which keys map where and how many of them
    # collapse into banks. Stacking tensors of the real shapes would allocate ~300 GB.
    remapped, skipped = remap_state_dict({key: torch.empty(0) for key in keys})

    deferred = [key for key in skipped if key.startswith(DEFERRED_PREFIXES)]
    loaded = [key for key in keys if key not in set(skipped)]
    expert_keys = [key for key in loaded if remap_key(key)[1] is not None]
    bank_targets = {remap_key(key)[0] for key in expert_keys}

    # the whole index, accounted for: every key is either deliberately skipped or mapped, and the
    # 4096 per-expert tensors collapse into one bank per layer per projection
    assert len(keys) == 5169
    assert len(loaded) + len(skipped) == len(keys)
    assert len(deferred) == len(skipped)                              # vae + vision tower + head
    assert len(expert_keys) == 4096
    assert len(bank_targets) == 64
    assert len(loaded) - len(expert_keys) + len(bank_targets) == len(remapped)
    # 409 while the text head (lm_head + ln_f) was part of the diffusion model; it is the prompt
    # rewriting stage's own file now
    assert len(remapped) == 407

    # every remapped key must name a real module and a real parameter slot
    with torch.device("meta"):
        model = HunyuanImage3(params, dtype=torch.bfloat16, device="meta", operations=comfy.ops.mixed_precision_ops({}, torch.bfloat16))
    modules = dict(model.named_modules())
    for key in remapped:
        parent, _, attribute = key.rpartition(".")
        assert parent in modules, f"{key}: no module {parent}"
        assert attribute in ("weight", "bias"), key

    # the stacked banks must have the geometry the model itself builds
    for layer in range(params.num_hidden_layers):
        assert model.model.layers[layer].mlp.experts_gate_up_proj._orig_shape == (params.num_experts, 2 * params.moe_intermediate_size, params.hidden_size)
        assert model.model.layers[layer].mlp.experts_down_proj._orig_shape == (params.num_experts, params.hidden_size, params.moe_intermediate_size)


@needs_hf_index
def test_real_config_and_index_agree_with_the_params():
    params = params_from_config(_real_config())
    assert params.num_hidden_layers == 32
    assert params.num_experts == 64 and params.moe_topk == 8
    assert params.cfg_distilled is True and params.use_meanflow is True
    assert params.vae_latent_channels == 32 and params.vae_downsample_factor == (16, 16)
    assert params.num_key_value_heads == 8 and params.attention_head_dim == 128

    with open(os.path.join(_model_dir(), "model.safetensors.index.json")) as handle:
        index = json.load(handle)["weight_map"]
    assert len(index) == 5169
    assert len(set(index.values())) == 32


def _real_config():
    with open(os.path.join(_model_dir(), "config.json")) as handle:
        return json.load(handle)
