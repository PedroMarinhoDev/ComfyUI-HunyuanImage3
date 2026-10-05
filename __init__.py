"""HunyuanImage-3.0-Instruct-Distil as a ComfyUI custom node package.

Everything lives in this package: the model, VAE, tokenizer and pipeline are implemented here and
never imported from `comfy/`. The quantized expert banks load through ComfyUI's own flat bank layout,
so this needs a ComfyUI from 2026-09-24 or later.
"""

from .latent_formats import HunyuanImage3
from .nodes import comfy_entrypoint

__all__ = ["HunyuanImage3", "comfy_entrypoint"]
