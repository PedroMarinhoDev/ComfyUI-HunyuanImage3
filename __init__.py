"""HunyuanImage-3.0-Instruct-Distil as a ComfyUI custom node package.

Everything lives in this package: the model, VAE, tokenizer and pipeline are implemented here and
never imported from `comfy/`, and the two behaviours this checkpoint needs from shared code that
stock ComfyUI does not have yet are applied by `compat` at import time.
"""

from . import compat  # noqa: F401  (must be imported before any model code touches comfy.ops)
from .latent_formats import HunyuanImage3
from .nodes import comfy_entrypoint

__all__ = ["HunyuanImage3", "comfy_entrypoint"]
