"""Load a HunyuanImage-3.0 checkpoint into a native `HunyuanImage3Model`.

The loader returns a `ModelPatcher` wrapping a `comfy.model_base.BaseModel`, which is what a MODEL
socket carries, so the stock sampler drives this model the same way it drives any other. Phase 1 builds
that model from here rather than through `comfy.sd` and a `supported_models` signature: `MODEL` is
nominal, so a node can hand out a genuine one without a registry entry.

The checkpoint converted for ComfyUI carries no config. Tencent's `config.json` and `tokenizer.json` ship with
this package (`tencent/`): one of each serves all three checkpoints (see `tencent/README.md`).
"""
import hashlib
import json
import logging
import os

import torch

import comfy.latent_formats
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
from ..latent_formats import HunyuanImage3

TENCENT_FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tencent")
CONFIG_PATH = os.path.join(TENCENT_FILES, "config.json")
TOKENIZER_PATH = os.path.join(TENCENT_FILES, "tokenizer.json")

# the sampling shift our checkpoint was distilled with; see `model_base.HunyuanImage3Sampling`
SHIFT = 3.0

# The sampling contract each checkpoint was trained with. `cfg_distilled` and `use_meanflow` decide
# whether the sequence carries the `<guidance>`/`<timestep_r>` meta tokens and which flow the sampler
# runs, so they have to match the weights rather than whichever `config.json` sits beside them: that file
# is easy to leave in place while the checkpoint changes, and the two disagreeing is silent.
#
# The shift is held at the distilled value for all three. Whether the base and Instruct want the
# flux-style shift the reference's `use_flux_shift` selects is not established — it is one number to
# revisit once their weights are here to compare against.
# `sequence_template` belongs to the same contract and for the same reason: each checkpoint's own
# `generation_config.json` names it (`instruct` for the Instruct and the Instruct-Distil, `pretrain` for the
# base), `prepare_model_inputs` reads it with `getattr(..., 'pretrain')`, and rendering one checkpoint
# under the other's template is silent — the sequence is well-formed either way and only the model
# notices. The base was previously rendered as `instruct`, i.e. with role scaffolding and
# `<answer>`/`</answer>` it was never trained on.
MODEL_TYPES = {
    "instruct_distil": {"cfg_distilled": True, "use_meanflow": True,
                        "sequence_template": "instruct"},
    "instruct": {"cfg_distilled": False, "use_meanflow": False, "sequence_template": "instruct"},
    "base": {"cfg_distilled": False, "use_meanflow": False, "sequence_template": "pretrain"},
}


# `model.layers.0.input_layernorm.weight` in each checkpoint, which every format keeps at bf16 (see
# `weight_fingerprint`). The Instruct-Distil is also recognisable by structure — it alone carries the
# `guidance_emb`/`timestep_r_emb` embedders — but the Instruct and the base share every key and differ
# only in their values, so the weights themselves are what tells them apart.
MODEL_FINGERPRINTS = {
    "ddba8fb657790c51": "instruct_distil",
    "47a6bac6de8a5785": "instruct",
    "38576cd111049364": "base",
}
FINGERPRINT_KEY = "model.layers.0.input_layernorm.weight"
FILENAME_TOKENS = (("distil", "instruct_distil"), ("instruct", "instruct"), ("base", "base"))


def weight_fingerprint(tensor):
    """A short, dtype-exact hash of one small tensor: what tells two checkpoints' weights apart."""
    raw = tensor.detach().to("cpu").contiguous()
    if raw.dtype == torch.bfloat16:
        raw = raw.view(torch.int16)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()[:16]


def detect_model_type(state_dict, checkpoint_path):
    """Which of the three checkpoints these weights are: by their fingerprint, else by the file name.

    The answer selects the sampling contract (`MODEL_TYPES`), which cannot be left to `config.json`:
    that file is shared by all three and stays behind when the weights change. A file this port has no
    fingerprint for (a fine-tune, say) falls back to its name, and structure has the last word on the
    Instruct-Distil, because running the distilled contract without its embedders — or the plain one with them —
    is not a choice but a broken render.
    """
    distilled = "guidance_emb.mlp.0.weight" in state_dict
    if FINGERPRINT_KEY in state_dict:
        known = MODEL_FINGERPRINTS.get(weight_fingerprint(state_dict[FINGERPRINT_KEY]))
        if known is not None:
            return known
    name = os.path.basename(checkpoint_path).lower()
    named = next((kind for token, kind in FILENAME_TOKENS if token in name), None)
    if distilled:
        detected = "instruct_distil"
    elif named in ("instruct", "base"):
        detected = named
    else:
        detected = "instruct"
    logging.warning("HunyuanImage3: %s is not one of the known checkpoints; treating it as %s (%s)",
                    os.path.basename(checkpoint_path), detected,
                    "it carries the distilled embedders" if distilled else
                    "from its file name" if named == detected else "the default for an undistilled model")
    return detected


class _OpsConfig:
    """`pick_operations` reads `.quant_config` off whatever it is handed, the way a supported-models
    config carries it for denoisers. Quantized checkpoints select `mixed_precision_ops`, which is also
    the only ops object with the expert bank class the MoE needs."""

    def __init__(self, quant_config):
        self.quant_config = quant_config


def _patcher_factory(checkpoint_path, disable_dynamic=False):
    """What `cached_patcher_init` calls: a freshly loaded patcher, nothing else.

    `ModelPatcher.deepclone_multigpu` (the Select Model Device node, multi-GPU sampling) rebuilds the
    model from its loader rather than copying a loaded one, and passes `disable_dynamic=True` when it
    needs a non-dynamic delegate. Without this, retargeting failed with "the loader that produced this
    model does not support multigpu" and the model silently stayed on the default GPU.
    """
    return load_hunyuan_image_3(checkpoint_path, disable_dynamic=disable_dynamic)[0]


def load_hunyuan_image_3(checkpoint_path, disable_dynamic=False):
    """Returns `(model_patcher, tokenizer)`.

    Which checkpoint the weights are is read off the weights (`detect_model_type`), and sets the
    config's `cfg_distilled`/`use_meanflow`: the shared config cannot say which checkpoint it is next to. A checkpoint that still carries the text head (`lm_head`,
    `ln_f`) loads too; the head is left out, since prompt rewriting loads its own (`rewrite.py`).
    """
    from tokenizers import Tokenizer

    from .model import HEAD_PREFIXES, params_from_config
    from .model_base import HunyuanImage3Model, HunyuanImage3ModelConfig

    state_dict = comfy.utils.load_torch_file(checkpoint_path)
    head_keys = [key for key in state_dict if key.startswith(HEAD_PREFIXES)]
    for key in head_keys:
        del state_dict[key]
    model_type = detect_model_type(state_dict, checkpoint_path)
    logging.info("HunyuanImage3: %s is the %s checkpoint%s", os.path.basename(checkpoint_path), model_type,
                 "; its built-in text head is not used" if head_keys else "")

    with open(CONFIG_PATH) as handle:
        config = json.load(handle)
    config.update(MODEL_TYPES[model_type])
    config["model_type"] = model_type
    params = params_from_config(config)

    quant_config = comfy.utils.detect_layer_quantization(state_dict, "")
    dtype = comfy.model_management.unet_dtype(supported_dtypes=[torch.bfloat16, torch.float16, torch.float32])
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    operations = comfy.ops.pick_operations(dtype, dtype, load_device=load_device, model_config=_OpsConfig(quant_config))
    if quant_config is None:
        # a non-quantized checkpoint gets the plain ops, which carry no expert bank on their own
        from .ops import QuantlessMoEOps
        operations = QuantlessMoEOps
    model_config = HunyuanImage3ModelConfig(params, HunyuanImage3(), {"shift": SHIFT},
                                           quant_config=quant_config, dtype=dtype, custom_operations=operations)
    model = HunyuanImage3Model(model_config, device=offload_device)

    # attribute access is deliberate: main.py rebinds comfy.model_patcher.CoreModelPatcher to
    # ModelPatcherDynamic at startup when comfy-aimdo works, and a from-import here would capture the
    # plain ModelPatcher alias before that happens — silently, with no log line, on exactly the large
    # offloaded models this loader exists for
    patcher_class = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
    patcher = patcher_class(model, load_device=load_device, offload_device=offload_device)
    patcher.cached_patcher_init = (_patcher_factory, (checkpoint_path,))

    # the converted file carries this port's key names already
    loaded = model.diffusion_model.load_state_dict(state_dict, strict=False, assign=patcher.is_dynamic())
    del state_dict
    if loaded.unexpected_keys:
        raise ValueError(f"checkpoint keys this model does not have: {loaded.unexpected_keys[:8]}"
                         f" ({len(loaded.unexpected_keys)} total)")

    model.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    return patcher, model.tokenizer
