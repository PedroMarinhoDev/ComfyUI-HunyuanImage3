"""HunyuanImage-3.0 as a native ComfyUI `BaseModel`.

This is the MODEL socket the stock sampler drives. `comfy.model_base.BaseModel` owns the denoising
contract — `apply_model(x, t, ...)` in, the velocity out — and a stock `KSampler` calls it once per
step. What this file adds is the three things our architecture needs on top of that contract:

* **the sampling class.** `ModelSamplingDiscreteFlow` carries the shifted sigma table and the
  `timestep(sigma) = sigma * 1000` mapping (which is exactly our `sigmas[:-1] * 1000`), but it is a
  bare `nn.Module` with no `calculate_input`/`calculate_denoised`. Those are the two methods
  `BaseModel._apply_model` calls, so a model using it must supply them. The flow convention is
  identity in and `x - v * sigma` out — the same update our pipeline's Euler step performs — verified
  against `calculate_sigmas(ModelSamplingDiscreteFlow(shift=3.0), "simple", 8)`, which reproduces our
  nine sigmas bit-for-bit. **The scheduler is part of this model's contract, not a user preference:**
  `simple` matches exactly while the frontend's usual `normal` default drifts by 2.9e-1.

* **`extra_conds`.** Guidance here is a token embedding, not CFG, so it arrives through the
  conditioning the way FluxGuidance does rather than through the sampler's `cfg` (which must stay 1.0:
  at 1.0 ComfyUI skips the unconditional pass, so negative conditioning is unused rather than harmful).
  The sequence geometry rides along with it, since the model rebuilds its input sequence per step.

* **the latent layout.** A `latent_dim == 3` LATENT is `(b, c, t, h, w)`; with a single frame the
  temporal dim is 1 and the model keeps it, so the velocity comes back in the shape the sampler's
  `calculate_denoised` expects to broadcast against.
"""
import comfy.conds
import comfy.model_base
import comfy.model_sampling


class HunyuanImage3Sampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    """Discrete flow, composed the way this repo composes it.

    `ModelSamplingDiscreteFlow` carries the shifted sigma table and the `timestep(sigma) = sigma * 1000`
    mapping; `CONST` is the mixin that supplies the three methods a sampler calls —
    `noise_scaling`, `calculate_input`, `calculate_denoised` — in the flow convention
    (`sigma * noise + (1 - sigma) * latent`, identity, and `x - v * sigma`). That combination is
    what `ModelSamplingSD3` builds for the in-tree flow models, verified here against the expected
    values rather than assumed: noise_scaling(0.9, 1, 2) = 1.1, calculate_input(2) = 2.0,
    calculate_denoised(1, 2) = 1.1.

    It is a bare `nn.Module` on its own, which is why a model that `model_sampling()` hands it to must
    compose the mixin in — and why `HunyuanImage3Model` replaces the dispatcher's pick below.
    """


class HunyuanImage3ModelConfig:
    """The subset of a supported-models config that `BaseModel` reads.

    Phase 1 builds the model from our own loader, so there is no `model_detection` entry and no
    `supported_models` signature; this carries the same fields those paths would fill in.
    """

    def __init__(self, params, latent_format, sampling_settings, quant_config=None, dtype=None, custom_operations=None):
        self.unet_config = {"config": params, "dtype": dtype}
        self.latent_format = latent_format
        self.sampling_settings = sampling_settings
        self.quant_config = quant_config
        self.manual_cast_dtype = None
        self.custom_operations = custom_operations
        self.optimizations = {}
        self.model_type = None
        # Scales the conditioning memory estimate, not the weight footprint: model_base.py:435/439
        # multiply it into an area-based figure. 2.0 is the base class's own default
        # (supported_models_base.py:49); an earlier guess of 1.15 here was accompanied by a claim that
        # nothing read it, which was wrong.
        self.memory_usage_factor = 2.0


class HunyuanImage3Model(comfy.model_base.BaseModel):
    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
        from .model import HunyuanImage3

        super().__init__(model_config, model_type, device=device, unet_model=HunyuanImage3)
        # the dispatcher's pick for this model_type does not implement the contract; ours does
        self.model_sampling = HunyuanImage3Sampling(model_config)

    def extra_conds(self, **kwargs):
        """Surface the token ids and the guidance embedding to the diffusion model.

        ComfyUI's conditioning format carries a token tensor plus a metadata dict, and the metadata's
        keys arrive here as kwargs. Whatever this returns becomes a `model_conds` entry and gets
        `process_cond` called on it, so the values have to be cond objects — which is why the base
        class wraps its own in `CONDCrossAttn`/`CONDRegular`/`CONDNoiseShape`.

        Both of these are tensors, which matters: the batching path sizes each cond and concatenates
        the ones that match, so a non-tensor value here would have nothing to size. The geometry the
        model needs is derived from the ids instead of travelling alongside them.
        """
        out = {}
        ids = kwargs.get("ids", None)
        if ids is not None:
            out["ids"] = comfy.conds.CONDConstant(ids)
        guidance = kwargs.get("guidance", None)
        if guidance is not None:
            out["guidance"] = comfy.conds.CONDConstant(guidance)
        # conditioning images, one list entry each in sequence order: the latent (already in model
        # space), the tower's aligned tokens, and that tower run's patch grid — the mask and the RoPE
        # need each run's own grid, and the run's length alone does not determine it
        for key in ("cond_latent", "cond_vit", "cond_vit_grid"):
            value = kwargs.get(key, None)
            if value is not None:
                out[key] = comfy.conds.CONDConstant(value)
        return out
