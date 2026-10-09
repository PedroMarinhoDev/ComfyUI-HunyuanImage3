"""Tiled refinement for a stock img2img KSampler pass (upscaled latent, denoise below 1).

`TiledDenoise` is a model function wrapper: every step runs the model on overlapping tiles of the latent and
blends the predictions with feathered weights (MultiDiffusion). Each call sees a trained size, so the canvas
can grow past the ~2048x2048 one sequence handles; the conditioning is encoded at the tile size.
"""
import math

import torch

from .model import tile_starts


def _feather(tile, overlap, device):
    """1 in the middle, a cosine fade over `overlap` cells at both ends; never 0."""
    index = torch.arange(tile, dtype=torch.float32, device=device)
    edge = (torch.minimum(index + 1, tile - index) / max(overlap, 1)).clamp(max=1.0)
    return 0.5 - 0.5 * torch.cos(math.pi * edge)


class TiledDenoise:
    def __init__(self, tile_height, tile_width, overlap):
        self.tile_height, self.tile_width, self.overlap = tile_height, tile_width, overlap

    def __call__(self, apply_model, args):
        x, timestep, c = args["input"], args["timestep"], args["c"]
        height, width = x.shape[-2:]
        th, tw = min(self.tile_height, height), min(self.tile_width, width)
        if (th, tw) == (height, width):
            return apply_model(x, timestep, **c)
        window = _feather(th, self.overlap, x.device)[:, None] * _feather(tw, self.overlap, x.device)[None]
        out = torch.zeros_like(x)
        total = torch.zeros(height, width, dtype=window.dtype, device=x.device)
        for top in tile_starts(height, th, th - self.overlap):
            for left in tile_starts(width, tw, tw - self.overlap):
                tile = apply_model(x[..., top:top + th, left:left + tw], timestep, **c)
                out[..., top:top + th, left:left + tw] += tile * window
                total[top:top + th, left:left + tw] += window
        return out / total
