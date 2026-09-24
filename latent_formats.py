"""The latent format for HunyuanImage-3.0's hunyuan-image-vae-v1.

Upstream ComfyUI gained an identical class in `comfy/latent_formats.py` for the in-repo port; this
package carries its own so it runs against stock ComfyUI. The two are interchangeable, and the model
node hands this one to the sampler through the model's own config.

32 latent channels, spatial /16, temporal /4. The reference pipeline divides the transformer's latent
by this factor before its VAE and multiplies its encode by it, i.e. `model latent = vae latent *
scale_factor` — which is exactly what LatentFormat expresses, and it is applied at the sampler
boundary through `process_in`/`process_out` rather than inside the VAE (`comfy/sd.py` keeps every
VAE unscaled, so folding it in here would double-divide a LATENT arriving from elsewhere).

No `latent_rgb_factors`: no preview projection has been fitted for this VAE.
"""

import comfy.latent_formats


class HunyuanImage3(comfy.latent_formats.LatentFormat):
    latent_channels = 32
    latent_dimensions = 3
    spacial_downscale_ratio = 16
    temporal_downscale_ratio = 4
    scale_factor = 0.562679178327931
