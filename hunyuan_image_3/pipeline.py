"""HunyuanImage-3.0 text-to-image sampling for the Instruct-Distil checkpoint.

Ported from the reference pipeline's diffusion loop and its `FlowMatchDiscreteScheduler`
(`hunyuan_image_3_pipeline.py:152-450`), for the configuration the Instruct-Distil checkpoint actually uses:
`shift=flow_shift=3.0`, `reverse=True`, `solver="euler"`, `num_train_timesteps=1000`,
`use_flux_shift=False`, `diff_infer_steps=8`, `diff_guidance_scale=2.5`, `use_meanflow=True`.

The schedule is `sd3_time_shift(linspace(1, 0, steps + 1))` with no flip (reverse=True means "do not
flip"), i.e. `shift * t / (1 + (shift - 1) * t)`, so the steps concentrate near the high-noise end:
sigmas 1.0, 0.9545, 0.9, 0.8333, 0.75, 0.6429, 0.5, 0.3, 0.0, and `timesteps = sigmas[:-1] * 1000`.

The Euler update is fp32 and deliberately not cast back — the reference has the cast to the model
dtype commented out (`hunyuan_image_3_pipeline.py:410`). `get_timestep_r` is the *next* entry of the
full sigma schedule, which is what the `<timestep_r>` token carries for meanflow.

**No KV cache, deliberately.** The reference caches every layer's K/V at prefill and overwrites the
image rows in place on steps 1-7 (`position_ids = [<timestep>, <guidance>, <timestep_r>, <img>...]`),
which is the detail most likely to be got subtly wrong. This port recomputes the full sequence every
step instead, and the image rows' outputs are identical:

* the mask is causal for the text and for all six meta tokens (`<boi>`, `<img_size>`, `<img_ratio>`,
  `<timestep>`, `<guidance>`, `<timestep_r>`), all of which *precede* the image block, so nothing
  before the block attends to it — their K/V cannot depend on the latents, and recomputing them
  reproduces the cached values exactly;
* the image rows attend to the same key set either way, so their outputs match.

The step cost is also the same in practice, because the forward is weight-streaming bound: 6.3 s for
5348 tokens against 6.1-6.5 s for 4096 (Step 6b).
"""
import torch

# The reference's defaults for this checkpoint (`modeling:1799`, `distil_generation_config.json`).
NUM_TRAIN_TIMESTEPS = 1000
DEFAULT_STEPS = 8
DEFAULT_SHIFT = 3.0
DEFAULT_GUIDANCE_SCALE = 2.5


class FlowMatchScheduler:
    """`FlowMatchDiscreteScheduler` with the Instruct-Distil configuration, Euler solver."""

    def __init__(self, steps=DEFAULT_STEPS, shift=DEFAULT_SHIFT, num_train_timesteps=NUM_TRAIN_TIMESTEPS):
        self.num_train_timesteps = num_train_timesteps
        sigmas = torch.linspace(1, 0, steps + 1)
        self.sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas[:-1] * num_train_timesteps
        self.timesteps_full = self.sigmas * num_train_timesteps

    def timestep(self, step, device=None, dtype=torch.float32):
        # shape (1,): the embedders take a batch of timesteps, and a 0-dim tensor would fail there
        return self.timesteps[step].to(device=device, dtype=dtype).reshape(1)

    def timestep_r(self, step, device=None, dtype=torch.float32):
        """The meanflow r for this step: the *next* entry of the full schedule."""
        return self.timesteps_full[step + 1].to(device=device, dtype=dtype).reshape(1)

    def step(self, model_output, step, sample):
        """One Euler step. `sample` stays fp32: the reference does not cast the result back."""
        derivative = model_output.to(torch.float32)
        dt = self.sigmas[step + 1] - self.sigmas[step]
        return sample + derivative * dt


def build_input_embeddings(model, sequence, latents, timestep, guidance, timestep_r,
                           cond_latent=None, cond_vit=None):
    """The model's input for one step, as the reference constructs it.

    `wte` everywhere, then the `<img>` block overwritten with `patch_embed(latents, time_embed(t))`,
    and `<timestep>`/`<guidance>`/`<timestep_r>` overwritten with their own embedders. The other
    three meta tokens (`<boi>`, `<img_size>`, `<img_ratio>`) keep their `wte` embeddings.

    Each conditioning image's runs are filled the same way, at `t = 0`: the reference's `vae_encode`
    ends with "we always use t=0 to declare it is a clean conditional image" (distil_modeling:2447).
    Its latent arrives in model space — `vae_encode` multiplies by the VAE's `scaling_factor`, and this
    port puts that in its `LatentFormat` — unlike the sampler's own latent, which the sampler hands over
    unscaled. `cond_latent`/`cond_vit` are per-image lists in `sequence["cond_blocks"]` order.

    `latents=None` is the prompt-rewriting stage's prefill, which has conditioning images but no
    generated block.
    """
    from .model import _as_list

    ids = sequence["ids"]
    embeds = model.model.wte(ids.unsqueeze(0))
    dtype, device = embeds.dtype, ids.device

    if latents is not None:
        image_emb, token_h, token_w = model.patch_embed(latents.to(dtype), model.time_embed(timestep))
        assert (token_h, token_w) == (sequence["token_height"], sequence["token_width"]), \
            f"patch_embed grid {token_h}x{token_w} != sequence grid"
        embeds[0, sequence["image_slice"]] = image_emb[0].to(dtype)
        embeds[0, sequence["timestep_position"]] = model.timestep_emb(timestep)[0].to(dtype)
        # a checkpoint trained without the distilled schedule carries neither token, so these slots are
        # absent rather than filled from an embedder the model does not have
        if sequence["guidance_position"] is not None:
            embeds[0, sequence["guidance_position"]] = model.guidance_emb(guidance)[0].to(dtype)
        if sequence["timestep_r_position"] is not None:
            embeds[0, sequence["timestep_r_position"]] = model.timestep_r_emb(timestep_r)[0].to(dtype)

    blocks = sequence.get("cond_blocks", [])
    cond_latents, cond_vits = _as_list(cond_latent), _as_list(cond_vit)
    if len(cond_latents) != len(blocks) or len(cond_vits) != len(blocks):
        raise ValueError(f"this sequence conditions on {len(blocks)} images but {len(cond_latents)} "
                         f"latents and {len(cond_vits)} tower outputs arrived")
    if blocks:
        clean = torch.zeros(1, dtype=torch.float32, device=device)
        clean_time, clean_timestep = model.time_embed(clean), model.timestep_emb(clean)[0].to(dtype)
    for block, latent, vit in zip(blocks, cond_latents, cond_vits):
        # a `latent_dim == 3` VAE encodes with a one-frame time axis
        latent = latent[:, :, 0] if latent.dim() == 5 else latent
        cond_emb, cond_h, cond_w = model.patch_embed(latent.to(device=device, dtype=dtype), clean_time)
        assert (cond_h, cond_w) == (block["token_height"], block["token_width"]), \
            f"conditioning patch_embed grid {cond_h}x{cond_w} != sequence grid"
        embeds[0, block["vae_slice"]] = cond_emb[0].to(dtype)
        embeds[0, block["timestep_position"]] = clean_timestep
        embeds[0, block["vit_slice"]] = vit.reshape(-1, embeds.shape[-1]).to(device=device, dtype=dtype)
    return embeds


def build_attention_mask(sequence, seq_len, dtype, device):
    """The reference's bool mask: causal, plus fully bidirectional inside each image block, converted
    to the additive float form this port uses (`0` = keep, `-inf` = mask).

    Step 0 verified every query row has at least one visible key, so `-inf` cannot produce a NaN, and
    re-indexing this along the query axis — what the reference does for steps 1-7 — only selects a
    subset of rows, which cannot remove a row's visible keys.

    Each conditioning image is one more such block ahead of the generated one. Their spans matter for
    the cache argument: the mask is causal outside each block, so a conditioning row cannot attend to
    anything after it, its keys and values are functions of the conditioning inputs alone, and
    recomputing them per step reproduces them exactly.
    """
    keep = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril(diagonal=0)
    for full_attention in sequence["full_attention_slices"]:
        keep[full_attention, full_attention] = True
    return torch.zeros(1, 1, seq_len, seq_len, dtype=dtype, device=device).masked_fill(~keep, float("-inf"))


def sample_image(model, sequence, latents, scheduler, dtype, guidance_scale=DEFAULT_GUIDANCE_SCALE, verbose=True,
                 cond_latent=None, cond_vit=None):
    """Denoise `latents` (fp32, `(b, latent_channels, token_h, token_w)`) in place-free fashion.

    `model` is the `HunyuanImage3` nn.Module (whose `config` is the parsed `HunyuanImage3Params`);
    `dtype` is the compute dtype the caller resolved (`HunyuanImage3Generator.dtype`), since a
    quantized weight's own `.dtype` is the storage dtype, not the compute one.

    Returns `(latents, trace)`: the final latents in fp32 and one record per step.
    """
    from .model import build_rope_freqs

    config = model.config
    device = latents.device
    seq_len = sequence["ids"].shape[0]
    mask = build_attention_mask(sequence, seq_len, dtype, device)
    freqs = build_rope_freqs(seq_len, config.attention_head_dim, sequence["rope_image_info"],
                             config.rope_theta, device=device)
    guidance = torch.full((1,), 1000.0 * guidance_scale, dtype=torch.bfloat16, device=device)
    trace = []

    for step in range(len(scheduler.timesteps)):
        timestep = scheduler.timestep(step, device=device)
        timestep_r = scheduler.timestep_r(step, device=device)
        with torch.no_grad():
            embeds = build_input_embeddings(model, sequence, latents, timestep, guidance, timestep_r,
                                            cond_latent=cond_latent, cond_vit=cond_vit)
            hidden = model.model(embeds, freqs, mask)
            image_hidden = hidden[0, sequence["image_slice"]]
            # final_layer takes the token sequence and returns the velocity in latent space
            pred = model.final_layer(image_hidden.unsqueeze(0), model.time_embed_2(timestep),
                                     sequence["token_height"], sequence["token_width"])
        latents = scheduler.step(pred, step, latents)
        record = {
            "step": step, "timestep": timestep.item(), "timestep_r": timestep_r.item(),
            "pred_absmax": pred.float().abs().max().item(), "pred_mean": pred.float().mean().item(),
            "latent_absmax": latents.abs().max().item(), "latent_mean": latents.mean().item(),
            "finite": bool(torch.isfinite(latents).all() and torch.isfinite(pred.float()).all()),
        }
        trace.append(record)
        if verbose:
            print(f"  step {step}  t {record['timestep']:7.1f}  r {record['timestep_r']:7.1f}  "
                  f"pred absmax {record['pred_absmax']:9.4f}  latents absmax {record['latent_absmax']:9.4f}  "
                  f"mean {record['latent_mean']:+.5f}  finite {record['finite']}")
    return latents, trace
