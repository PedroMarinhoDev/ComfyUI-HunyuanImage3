"""HunyuanImage-3.0 nodes: a MODEL socket, its conditioning, and a latent at any size.

The graph is stock from here on. This file hands out a genuine `MODEL` (a `ModelPatcher` wrapping a
`comfy.model_base.BaseModel`), `CONDITIONING`, and a `LATENT`, so `KSampler`, `SamplerCustom`, and
`VAEDecode` drive it the way they drive every other model. Nothing here samples.

**The sampling settings are part of this model's contract, not preferences.** The schedule ComfyUI
reproduces bit-for-bit is `ModelSamplingDiscreteFlow(shift=3.0)` under the `simple` scheduler — the
frontend's usual `normal` default drifts by 2.9e-1, which is silent quality loss. The rest follows from
which checkpoint the weights are, which the loader reads off the weights. Instruct-Distil is 8-step
distilled with meanflow: sample it at 8 steps with `cfg` 1.0, where ComfyUI skips the unconditional
pass, and expect other step counts, other schedulers and `denoise < 1.0` to drift from its trajectory.
Instruct and the base run an ordinary schedule — the step count and guidance there are yours to choose.
The workflows in `workflows/` preselect each checkpoint's settings.
"""
import contextlib
import logging
import math

import torch
from typing_extensions import override

import comfy.clip_model
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.patcher_extension
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, IO

from .hunyuan_image_3.vae import HunyuanImage3VAE

from .hunyuan_image_3.loader import load_hunyuan_image_3
from .hunyuan_image_3.pipeline import DEFAULT_GUIDANCE_SCALE
from .hunyuan_image_3.rewrite import (AUTO, REWRITE_ONLY, THINK_AND_REWRITE, RewriteSettings,
                                      head_candidates, run_rewrite)
from .hunyuan_image_3.spectrum import SpectrumState
from .hunyuan_image_3.system_prompt import SYSTEM_PROMPTS
from .latent_formats import HunyuanImage3
from .hunyuan_image_3.tokenizer import (COND_PATCH_LIMIT, MAX_COND_IMAGES, ResolutionGroup, build_sequence,
                                        preset_options)

BASE_SIZE = 1024
LATENT_FORMAT = HunyuanImage3()

# A latent_dim == 3 VAE's LATENT carries a temporal dim; this model generates a single frame per call.
LATENT_T = 1

# The reference's table is offered by the resolutions node; the encoders and the empty latent take plain
# numbers, which is also the path for anything the table does not list.
DEFAULT_RESOLUTION = f"{BASE_SIZE}x{BASE_SIZE}"

SIZE_TOOLTIP = (
    "The size to generate at; wire the same width and height into the empty latent. Any aspect ratio "
    "works, and sizes round down to a multiple of 16. The model's own table (the Resolutions node) is "
    "what it was trained on; sizes well above ~1 Mpx follow the prompt less and less."
)

PRESET_TOOLTIP = (
    "A size from the model's own resolution table. Wire width and height into the encoder and the empty "
    "latent; both must be the same size, since the latent grid and the sequence grid are one grid."
)

REWRITE_TYPE = IO.Custom("HUNYUAN_IMAGE_3_REWRITE")

REWRITING_TOOLTIP = ("Optional: a HunyuanImage 3.0 Prompt Rewriting node, to let the model rewrite the "
                     "prompt before drawing. Leave it unconnected, or disable it on that node, to skip it.")
CUSTOM_SYSTEM_PROMPT_TOOLTIP = (
    "Optional: replace the checkpoint's built-in system prompt with your own text. Unconnected, the "
    "model's own is used — the unified instruction for Instruct and Instruct-Distil, none for the base.")


# the Instruct-Distil in W4A8 is the pack's default: the fastest of the three and the smallest file
DEFAULT_CHECKPOINT = "hunyuan_image_3_instruct_distil_w4a8.safetensors"


def _default_checkpoint():
    names = folder_paths.get_filename_list("diffusion_models")
    return DEFAULT_CHECKPOINT if DEFAULT_CHECKPOINT in names else next(
        (name for name in names if name.startswith("hunyuan_image_3_") and "cot_head" not in name), None)


class HunyuanImage3ModelLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HunyuanImage3ModelLoader",
            display_name="Load HunyuanImage 3.0 Model",
            category="model/loaders/hunyuan image 3",
            description="Loads any HunyuanImage 3.0 checkpoint (Instruct-Distil, Instruct or Base, any "
                        "format). Which one it is is read off the weights.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=folder_paths.get_filename_list("diffusion_models"),
                    default=_default_checkpoint(),
                    tooltip="The checkpoint converted for ComfyUI, from models/diffusion_models. Which one it "
                            "is (Instruct-Distil, Instruct or Base) is detected from the weights.",
                ),
            ],
            outputs=[IO.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model) -> IO.NodeOutput:
        path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        patcher, _ = load_hunyuan_image_3(path)
        return IO.NodeOutput(patcher)


class HunyuanImage3Resolutions(IO.ComfyNode):
    """A size from the model's own resolution table, as two numbers to wire into the other nodes.

    Presets only, and no `custom`: the table's rows are what the reference asks this checkpoint for, while
    an arbitrary size belongs on the encoder's and the empty latent's own width/height inputs, which are
    already free. Keeping the list here rather than in both of those is what stops them disagreeing about
    the grid, which is the one thing they must share.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HunyuanImage3Resolutions",
            display_name="HunyuanImage 3.0 Resolutions",
            category="hunyuan_image_3",
            inputs=[IO.Combo.Input("resolution", options=preset_options(BASE_SIZE),
                                   default=DEFAULT_RESOLUTION, tooltip=PRESET_TOOLTIP)],
            outputs=[IO.Int.Output(display_name="width"), IO.Int.Output(display_name="height")],
        )

    @classmethod
    def execute(cls, resolution) -> IO.NodeOutput:
        width, height = resolution.split("x")
        return IO.NodeOutput(int(width), int(height))


class HunyuanImage3PromptRewriting(IO.ComfyNode):
    """The settings for the prompt-rewriting stage, handed to an encoder.

    The stage itself runs inside the encoder, because it needs the prompt and the images, and because
    the image is conditioned on its whole output (the analysis too, in 'think + rewrite'), not only on
    the rewritten prompt.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HunyuanImage3PromptRewriting",
            display_name="HunyuanImage 3.0 Prompt Rewriting",
            category="conditioning/hunyuan image 3",
            description="Lets the model rewrite your prompt (and optionally reason about it first) before "
                        "it draws. Connect it to a HunyuanImage 3.0 Text Encode or Image Encode node. "
                        "Instruct and Instruct-Distil only; slow (about 1 s per token on a 4090).",
            inputs=[
                IO.Boolean.Input("enabled", default=True,
                                 tooltip="Off skips the stage entirely, as if the node were not connected."),
                IO.Combo.Input("mode", options=[REWRITE_ONLY, THINK_AND_REWRITE], default=REWRITE_ONLY,
                               tooltip="'rewrite only' asks for the rewritten prompt directly, typically a "
                                       "few hundred tokens. 'think + rewrite' first writes a full analysis "
                                       "and then the rewrite (the checkpoint's own default), often 1000+ "
                                       "tokens."),
                IO.Combo.Input("cot_head", options=[AUTO] + head_candidates(), default=AUTO,
                               tooltip="The text head the stage predicts with. 'auto' picks the file "
                                       "matching the loaded model: hunyuan_image_3_<model>_cot_head."
                                       "safetensors in models/diffusion_models (the Instruct and the "
                                       "Instruct-Distil heads are identical)."),
                IO.Int.Input("max_new_tokens", default=2048, min=1, max=8192,
                             tooltip="Upper bound on the stage; it stops by itself when the rewrite is "
                                     "finished, so this is a ceiling, not a target."),
                IO.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=IO.ControlAfterGenerate.fixed,
                             tooltip="Seed for the sampled text. Kept fixed so the slow stage is not rerun "
                                     "every time the image seed changes."),
                IO.Boolean.Input("do_sample", default=True, advanced=True,
                                 tooltip="The checkpoint's own default (generation_config.json): sampling at "
                                         "temperature 0.6, top_p 0.95, top_k 1024. Off picks the most likely "
                                         "token every time."),
                IO.Float.Input("temperature", default=0.6, min=0.0, max=10.0, step=0.01, advanced=True),
                IO.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01, advanced=True),
                IO.Int.Input("top_k", default=1024, min=0, max=16384, advanced=True),
                IO.Float.Input("repetition_penalty", default=1.0, min=0.0, max=10.0, step=0.01, advanced=True),
            ],
            outputs=[REWRITE_TYPE.Output(display_name="prompt_rewriting")],
        )

    @classmethod
    def execute(cls, enabled, mode, cot_head, max_new_tokens, seed, do_sample=True, temperature=0.6,
                top_p=0.95, top_k=1024, repetition_penalty=1.0) -> IO.NodeOutput:
        if not enabled:
            return IO.NodeOutput(None)
        return IO.NodeOutput(RewriteSettings(
            mode=mode, cot_head=cot_head, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
            seed=seed))


def _resize_and_crop(image, width, height):
    """The reference's `resize_and_crop` (image_processor.py:61), center crop: scale to cover
    `width`x`height` keeping the aspect ratio, with Lanczos, then crop the middle."""
    image_height, image_width = image.shape[1], image.shape[2]
    if image_height / image_width < height / width:
        resize_height, resize_width = height, int(round(height / image_height * image_width))
    else:
        resize_height, resize_width = int(round(width / image_width * image_height)), width
    resized = comfy.utils.common_upscale(image.movedim(-1, 1), resize_width, resize_height, "lanczos", "disabled")
    top, left = int(round((resize_height - height) / 2.0)), int(round((resize_width - width) / 2.0))
    return resized[:, :, top:top + height, left:left + width].movedim(1, -1)


def _condition_on_image(model, image, vae, clip_vision, config, vit_strength, latent_strength):
    """One conditioning image: its latent, its tower tokens, and the sizes its sequence block needs.

    The VAE sees the image resized and center-cropped to the nearest size in the model's resolution
    table (about 1 Mpx), as the reference's processor does (`get_image_with_size`). That is also what
    lets three images fit the model's positions. The tower sees the original image and resizes it
    itself to at most `COND_PATCH_LIMIT` patches.

    The latent goes through `LATENT_FORMAT.process_in`, because the model's own inputs are in model
    space — the reference's `vae_encode` multiplies by the VAE's `scaling_factor` (spec §46). The
    tower's preprocessing is done here rather than through `encode_image`, because the patch cap belongs
    to *this checkpoint* (`config.vit_processor.max_num_patches` is 1024) while the stock SigLIP2 config
    `load_clipvision_from_sd` selects says 256.
    """
    image = image[:1, :, :, :3]
    target_width, target_height = ResolutionGroup(config.image_base_size).get_target_size(image.shape[2], image.shape[1])
    cond_latent = vae.encode(_resize_and_crop(image, target_width, target_height))
    cond_latent = LATENT_FORMAT.process_in(cond_latent) * latent_strength

    comfy.model_management.load_model_gpu(clip_vision.patcher)
    patch_size = clip_vision.config.get("patch_size", 16)
    pixels = comfy.clip_model.siglip2_preprocess(
        image.to(clip_vision.load_device), size=-1, patch_size=patch_size, num_patches=COND_PATCH_LIMIT,
        mean=clip_vision.image_mean, std=clip_vision.image_std, crop=False).float()
    tokens = clip_vision.model(pixel_values=pixels, intermediate_output=-2)[0]
    tokens = tokens.to(comfy.model_management.intermediate_device())
    patch_height, patch_width = (size // patch_size for size in pixels.shape[-2:])
    assert tokens.shape[1] == patch_height * patch_width, \
        f"the tower returned {tokens.shape[1]} tokens for a {patch_height}x{patch_width} grid"
    cond_vit = model.model.diffusion_model.vision_aligner(tokens.to(cond_latent.dtype)) * vit_strength
    return {
        "latent": cond_latent,
        "vit": cond_vit,
        "grid": torch.tensor([patch_height, patch_width], dtype=torch.long),
        "size": ((target_height, target_width), (patch_height, patch_width)),
    }


def _encode(model, prompt, width, height, custom_system_prompt, rewriting, conds=()):
    """Positive and negative conditioning for one request, and the rewritten prompt if there is one.

    The negative is the reference's unconditional pass for the same request: the prompt (and any CoT
    text) replaced token for token by `<cfg>`, the images and everything else kept, so the two share one
    sequence length and one grid. The token ids go in the conditioning's metadata rather than its tensor
    slot, because `context` is cast to the compute dtype and bf16 cannot hold a token id; int tensors in
    the metadata are only moved between devices.
    """
    tokenizer = model.model.tokenizer
    # `model.model` is the BaseModel; the parsed checkpoint params live on its diffusion model
    config = model.model.diffusion_model.config
    system_prompt = custom_system_prompt if custom_system_prompt else SYSTEM_PROMPTS[config.model_type]
    cond_images = [cond["size"] for cond in conds]
    cond_meta = {}
    if conds:
        cond_meta = {"cond_latent": [cond["latent"] for cond in conds],
                     "cond_vit": [cond["vit"] for cond in conds],
                     "cond_vit_grid": [cond["grid"] for cond in conds]}

    cot_text, rewritten = None, ""
    if rewriting is not None:
        if config.model_type == "base":
            # never trained to rewrite: the reference's base runs bot_task 'image' only
            logging.warning("HunyuanImage3: the base checkpoint does not rewrite prompts; prompt rewriting "
                            "is skipped")
        else:
            cot_text, rewritten = run_rewrite(model, rewriting, prompt, system_prompt, cond_images,
                                              cond_meta.get("cond_latent"), cond_meta.get("cond_vit"))

    # Above the trained grid the rope base is rescaled by the grid's growth factor (see
    # build_rope_freqs). Measured, one prompt and seed: 1536x1536 (2.25x) renders the prompt with
    # it and drifts without it; 2048x2048 (4x) comes back to the right animal but not the prompt.
    area = width * height
    table_ceiling = max(entry.height * entry.width for entry in ResolutionGroup(BASE_SIZE).data)
    if area > table_ceiling:
        logging.info("HunyuanImage3: %dx%d is %.2fx the size this checkpoint's resolution table covers; "
                     "prompt adherence degrades well past the table", width, height, area / table_ceiling)

    def conditioning(uncond):
        sequence = build_sequence(tokenizer, prompt, f"{width}x{height}", system_prompt, cot_text=cot_text,
                                  base_size=config.image_base_size,
                                  max_position_embeddings=config.max_position_embeddings,
                                  cfg_distilled=config.cfg_distilled, use_meanflow=config.use_meanflow,
                                  sequence_template=config.sequence_template, cond_images=cond_images,
                                  uncond=uncond, extra_rows=config.model_type != "base")
        return [[None, dict(cond_meta, ids=sequence["ids"])]]

    return IO.NodeOutput(conditioning(False), conditioning(True), rewritten)


ENCODE_OUTPUTS = [
    IO.Conditioning.Output(display_name="positive"),
    IO.Conditioning.Output(display_name="negative",
                           tooltip="The model's own unconditional pass for this request (the prompt blanked "
                                   "out, everything else kept). Only used when the sampler's cfg is above 1."),
    IO.String.Output(display_name="rewritten_prompt",
                     tooltip="What the Prompt Rewriting stage wrote; empty without it."),
]


class HunyuanImage3TextEncode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HunyuanImage3TextEncode",
            display_name="HunyuanImage 3.0 Text Encode",
            category="conditioning/hunyuan image 3",
            description="Text to image: turns the prompt into positive and negative conditioning.",
            inputs=[
                IO.Model.Input("model"),
                IO.String.Input("prompt", multiline=True, dynamic_prompts=True),
                IO.Int.Input("width", default=BASE_SIZE, min=16, max=16384, step=16, tooltip=SIZE_TOOLTIP),
                IO.Int.Input("height", default=BASE_SIZE, min=16, max=16384, step=16, tooltip=SIZE_TOOLTIP),
                REWRITE_TYPE.Input("prompt_rewriting", optional=True, tooltip=REWRITING_TOOLTIP),
                IO.String.Input("custom_system_prompt", optional=True, force_input=True, multiline=True,
                                tooltip=CUSTOM_SYSTEM_PROMPT_TOOLTIP),
            ],
            outputs=ENCODE_OUTPUTS,
        )

    @classmethod
    def execute(cls, model, prompt, width, height, prompt_rewriting=None, custom_system_prompt=None) -> IO.NodeOutput:
        return _encode(model, prompt, width, height, custom_system_prompt, prompt_rewriting)


IMAGE_NAMES = [f"image_{index}" for index in range(1, MAX_COND_IMAGES + 1)]


class HunyuanImage3ImageEncode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HunyuanImage3ImageEncode",
            display_name="HunyuanImage 3.0 Image Encode",
            category="conditioning/hunyuan image 3",
            description="Image editing and multi-image fusion: the prompt plus one to three images. "
                        "Connecting an image adds a socket for the next one. Instruct and Instruct-Distil.",
            inputs=[
                IO.Model.Input("model"),
                IO.Vae.Input("vae"),
                IO.ClipVision.Input("clip_vision",
                                    tooltip="The checkpoint's own vision tower: "
                                            "hunyuan_image_3_<model>_siglip2_so400m_naflex.safetensors."),
                IO.Autogrow.Input(
                    "images",
                    template=IO.Autogrow.TemplateNames(IO.Image.Input("image"), names=IMAGE_NAMES, min=1),
                    tooltip="The images to edit or combine, in order: refer to them in the prompt as "
                            "'the first image', 'the second image'. Each is resized to about 1 Mpx."),
                IO.String.Input("prompt", multiline=True, dynamic_prompts=True),
                IO.Int.Input("width", default=BASE_SIZE, min=16, max=16384, step=16,
                             tooltip=SIZE_TOOLTIP + " The output size is independent of the input images."),
                IO.Int.Input("height", default=BASE_SIZE, min=16, max=16384, step=16, tooltip=SIZE_TOOLTIP),
                REWRITE_TYPE.Input("prompt_rewriting", optional=True, tooltip=REWRITING_TOOLTIP),
                IO.String.Input("custom_system_prompt", optional=True, force_input=True, multiline=True,
                                tooltip=CUSTOM_SYSTEM_PROMPT_TOOLTIP),
                IO.Float.Input("vit_strength", default=1.0, min=0.0, max=2.0, step=0.01, advanced=True,
                               tooltip="Scales the vision tower's contribution. 1.0 is the reference's "
                                       "behaviour."),
                IO.Float.Input("latent_strength", default=1.0, min=0.0, max=2.0, step=0.01, advanced=True,
                               tooltip="Scales the images' latent contribution. 1.0 is the reference's "
                                       "behaviour."),
            ],
            outputs=ENCODE_OUTPUTS,
        )

    @classmethod
    def execute(cls, model, vae, clip_vision, images, prompt, width, height, prompt_rewriting=None,
                custom_system_prompt=None, vit_strength=1.0, latent_strength=1.0) -> IO.NodeOutput:
        config = model.model.diffusion_model.config
        if config.model_type == "base":
            raise ValueError("the base checkpoint only does text to image; use the HunyuanImage 3.0 Text "
                             "Encode node, or load Instruct or Instruct-Distil for editing")
        ordered = [images[name] for name in IMAGE_NAMES if images.get(name) is not None]
        conds = [_condition_on_image(model, image, vae, clip_vision, config, vit_strength, latent_strength)
                 for image in ordered]
        return _encode(model, prompt, width, height, custom_system_prompt, prompt_rewriting, conds)


class HunyuanImage3Guidance(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HunyuanImage3Guidance",
            display_name="HunyuanImage 3.0 Guidance",
            category="conditioning/hunyuan image 3",
            description="Instruct-Distil only: its guidance is an input to the model rather than CFG. "
                        "2.5 is the checkpoint's own setting; keep the sampler's cfg at 1.0.",
            inputs=[
                IO.Conditioning.Input("conditioning"),
                IO.Float.Input("guidance_scale", default=DEFAULT_GUIDANCE_SCALE, min=0.0, max=100.0, step=0.01),
            ],
            outputs=[IO.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, conditioning, guidance_scale) -> IO.NodeOutput:
        # guidance here is a token embedding rather than CFG, so it belongs in the conditioning. It is
        # scaled by 1000 to match the reference's timestep units, the same units `KSampler`'s sigma
        # becomes on its way into the model.
        out = []
        for ids, meta in conditioning:
            meta = dict(meta)
            meta["guidance"] = torch.full((1,), 1000.0 * guidance_scale, dtype=torch.bfloat16)
            out.append([ids, meta])
        return IO.NodeOutput(out)


class HunyuanImage3EmptyLatent(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HunyuanImage3EmptyLatent",
            display_name="HunyuanImage 3.0 Empty Latent",
            category="latent/hunyuan image 3",
            inputs=[
                IO.Int.Input("width", default=BASE_SIZE, min=16, max=16384, step=16, tooltip=SIZE_TOOLTIP),
                IO.Int.Input("height", default=BASE_SIZE, min=16, max=16384, step=16),
                IO.Int.Input("batch_size", default=1, min=1, max=64),
            ],
            outputs=[IO.Latent.Output()],
        )

    @classmethod
    def execute(cls, width, height, batch_size) -> IO.NodeOutput:
        # VAE-space and unscaled: the scaling factor lives in the LatentFormat and is applied by the
        # sampler, so handing VAEDecode a scaled latent would divide twice
        samples = torch.zeros(
            (batch_size, LATENT_FORMAT.latent_channels, LATENT_T,
             height // LATENT_FORMAT.spacial_downscale_ratio, width // LATENT_FORMAT.spacial_downscale_ratio),
            dtype=torch.float32,
        )
        return IO.NodeOutput({"samples": samples})


@contextlib.contextmanager
def _without_log_message(prefix):
    """Drop root-logger records starting with `prefix` for the duration of the block."""
    def keep(record):
        return not str(record.msg).startswith(prefix)
    logging.root.addFilter(keep)
    try:
        yield
    finally:
        logging.root.removeFilter(keep)


class HunyuanImage3VAELoader(IO.ComfyNode):
    """Load hunyuan-image-vae-v1 through this package rather than comfy.sd's signature detection.

    `comfy.sd` picks a VAE implementation from key signatures. This checkpoint carries the same
    signature as the Hunyuan Video refiner VAE (`decoder.conv_in.weight`, 32 input channels, 5-D)
    while being a different architecture, so the greedy branch builds the wrong model: all 280 keys
    land, nothing is reported missing, and the decode is a washed-out copy of the same scene (13.35 dB
    against this implementation). The stock `comfy.sd.VAE` object is still what gets built and returned
    — it owns the device, dtype, size and patcher — but its architecture, ratios and fitted memory
    estimators are replaced here.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HunyuanImage3VAELoader",
            display_name="Load HunyuanImage-3.0 VAE",
            category="hunyuan_image_3",
            inputs=[IO.Combo.Input("vae_name", options=folder_paths.get_filename_list("vae"),
                                   tooltip="hunyuan_image_3_vae_fp16 or _fp32: one VAE serves all three checkpoints.")],
            outputs=[IO.Vae.Output()],
        )

    @classmethod
    def execute(cls, vae_name) -> IO.NodeOutput:
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd = comfy.utils.load_torch_file(vae_path)
        # the stock constructor builds the architecture its signature check picks, loads this state dict
        # into it and warns about the `temb_proj` keys that architecture has and this one lacks; that model
        # is replaced below and its warning describes nothing that runs, so it is not passed on
        with _without_log_message("Missing VAE keys"):
            vae = comfy.sd.VAE(sd=sd)
        vae.first_stage_model = HunyuanImage3VAE(
            in_channels=3, out_channels=3, latent_channels=32,
            block_out_channels=[128, 256, 512, 1024, 1024], layers_per_block=2,
            ffactor_spatial=16, ffactor_temporal=4, scaling_factor=1.0,
            downsample_match_channel=True, upsample_match_channel=True,
            operations=comfy.ops.disable_weight_init,
        )
        vae.latent_channels = 32
        vae.latent_dim = 3
        vae.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
        vae.upscale_index_formula = (4, 16, 16)
        vae.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
        vae.downscale_index_formula = (4, 16, 16)
        vae.not_video = True
        vae.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        # Fitted on this architecture, fp16, over 256x256..1536x1152 outputs: a fixed ~3 GiB of
        # mid-block temporaries plus streaming pool, and ~11 kB per output pixel. The refiner branch's
        # flat 2800 under-estimates this one by up to 2.5x (1536x1152 claims 8.2 GiB against 20.6 GiB
        # measured) — the direction that OOMs, because load_models_gpu declines an eviction it needs.
        # Both coefficients are >= the measurement at every size tested.
        vae.memory_used_decode = lambda shape, dtype: (1.5 * 2**30 + 5600 * shape[-3] * shape[-2] * shape[-1] * 16 * 16) * comfy.model_management.dtype_size(dtype)
        # measured through the stock wrapper (an IMAGE unsqueezed to (b, c, 1, h, w)): 2.99 GiB at
        # 512x512, 6.25 at 1024x1024, 7.21 at 1280x960 of activations above the resident weights
        vae.memory_used_encode = lambda shape, dtype: (3200 * shape[-2] * shape[-1]) * comfy.model_management.dtype_size(dtype)

        # the stock object's patcher wraps the model the detection built, so it is rebuilt for this one
        vae.patcher = comfy.model_patcher.CoreModelPatcher(
            vae.first_stage_model, load_device=vae.device,
            offload_device=vae.patcher.offload_device, fast_disk=vae.patcher.fast_disk)
        missing, unexpected = vae.first_stage_model.load_state_dict(
            sd, strict=False, assign=vae.patcher.is_dynamic())
        if not vae.patcher.is_dynamic():
            vae.first_stage_model.to(vae.vae_dtype)
        logging.info("HunyuanImage-3.0 VAE: {} keys loaded, {} missing, {} unexpected, dtype {}".format(
            len(sd) - len(missing), len(missing), len(unexpected), vae.vae_dtype))
        vae.model_size()
        return IO.NodeOutput(vae)


def _spectrum_wrapper(state):
    """Clear the forecast buffers around each sampling run, including a cancelled one."""
    def wrapper(executor, *args, **kwargs):
        state.reset()
        try:
            return executor(*args, **kwargs)
        finally:
            state.report()
            state.clear()
    return wrapper


class HunyuanImage3Spectrum(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HunyuanImage3Spectrum",
            display_name="HunyuanImage-3.0 Spectrum",
            category="model/patch/hunyuan image 3",
            inputs=[
                IO.Model.Input(
                    "model",
                    tooltip="The model from the loader, before the sampler.",
                ),
                IO.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip="Off passes the model through untouched, for an A/B in one workflow.",
                ),
                IO.Int.Input(
                    "warmup_steps",
                    default=5,
                    min=0,
                    max=20,
                    advanced=True,
                    tooltip="Full steps before any skipping. 5 is the setting behind the paper's "
                            "published 14 forwards at 50 steps — the usual 3 gives 12, which is more "
                            "aggressive than they report.",
                ),
                IO.Float.Input(
                    "window_size",
                    default=2.0,
                    min=1.0,
                    max=16.0,
                    step=0.5,
                    advanced=True,
                    tooltip="Steps the first gap covers. 1 never skips, which is how the no-side-"
                            "effects gate is run.",
                ),
                IO.Float.Input(
                    "flex_window",
                    default=0.75,
                    min=0.0,
                    max=4.0,
                    step=0.01,
                    advanced=True,
                    tooltip="How fast the gaps widen: alpha in the paper. 0.75 is their headline "
                            "setting (3.5x at 50 steps), 3.0 the aggressive one (5x).",
                ),
                IO.Float.Input(
                    "w",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    advanced=True,
                    tooltip="Chebyshev against Taylor in the blend, on the shortest forecast of a "
                            "window. 0 follows the last two steps only, 1 trusts the polynomial.",
                ),
                IO.Float.Input(
                    "max_w",
                    default=0.8,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    advanced=True,
                    tooltip="The same weight on the longest forecast of a window, where the "
                            "polynomial is more reliable than a local difference.",
                ),
                IO.Int.Input(
                    "M",
                    default=4,
                    min=1,
                    max=10,
                    advanced=True,
                    tooltip="Chebyshev degree. The paper's own default.",
                ),
                IO.Float.Input(
                    "lam",
                    default=0.1,
                    min=0.001,
                    max=10.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Ridge regularisation of the fit. The paper's own default.",
                ),
                IO.Int.Input(
                    "history",
                    default=12,
                    min=2,
                    max=64,
                    advanced=True,
                    tooltip="Stored steps the fit uses. Every stored step holds the whole image "
                            "token block (32 MiB at 1024, 72 MiB at 1536), so this is the node's "
                            "real memory knob.",
                ),
                IO.Combo.Input(
                    "time_axis",
                    options=["step", "sigma"],
                    default="step",
                    advanced=True,
                    tooltip="What the feature is fitted against. The flow schedule is non-uniform, "
                            "so sigma may prove the smoother axis; measure before preferring it.",
                ),
                IO.Boolean.Input(
                    "validate",
                    default=False,
                    advanced=True,
                    tooltip="Run the real forward on skip steps too, and report how far each forecast "
                            "was from it. Costs the speedup and does not change the image: this is the "
                            "instrument for choosing settings, not a sampling mode.",
                ),
                IO.Boolean.Input(
                    "verbose",
                    default=False,
                    advanced=True,
                    tooltip="Per-step log line: the step, the batch the model was handed, "
                            "cond_or_uncond, and whether the stack ran.",
                ),
            ],
            outputs=[IO.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, enabled, warmup_steps, window_size, flex_window, w, max_w, M, lam,
                history, time_axis, validate, verbose) -> IO.NodeOutput:
        model = model.clone()
        if not enabled:
            return IO.NodeOutput(model)

        state = SpectrumState(warmup_steps=warmup_steps, window_size=window_size,
                              flex_window=flex_window, w=w, max_w=max_w, M=M, lam=lam,
                              history=history, time_axis=time_axis, validate=validate, verbose=verbose)
        model.model_options.setdefault("transformer_options", {})["hy3_spectrum"] = state
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                   "hy3_spectrum", _spectrum_wrapper(state))
        return IO.NodeOutput(model)


class HunyuanImage3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [HunyuanImage3ModelLoader, HunyuanImage3Resolutions, HunyuanImage3TextEncode,
                HunyuanImage3ImageEncode, HunyuanImage3PromptRewriting, HunyuanImage3Guidance,
                HunyuanImage3EmptyLatent, HunyuanImage3VAELoader, HunyuanImage3Spectrum]


async def comfy_entrypoint() -> HunyuanImage3Extension:
    return HunyuanImage3Extension()
