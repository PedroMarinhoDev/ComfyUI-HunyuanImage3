"""Deliverable 1: the VAE's weights in native fp32, copied rather than converted.

The raw shards store all 280 `vae.*` tensors as F32, so this is a copy with the `vae.` prefix stripped —
the same naming the fp16 file uses — into a new file alongside it. Nothing is replaced.

Then it checks the two things that could silently go wrong: that `VAELoader` accepts it and that
`comfy/sd.py`'s branch still claims it as this port's VAE (the `temb_proj` discriminator is
dtype-independent, but the point is to check rather than assume), and it reports the `vae_dtype` ComfyUI
actually resolves for the file.

Usage: make_vae_fp32.py --weights ORIGINAL_SHARD_DIR --out models/vae/hunyuan_image_3_vae_fp32.safetensors
(run with PYTHONPATH at your ComfyUI checkout; add --verify-only to only re-check an existing file)
"""
import json
import struct
import sys

def _argument(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


RAW = _argument("--weights")
TARGET = _argument("--out")

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def extract():
    index = json.load(open(f"{RAW}/model.safetensors.index.json"))["weight_map"]
    shards = sorted({shard for key, shard in index.items() if key.startswith("vae.")})
    print(f"vae.* keys: {sum(1 for k in index if k.startswith('vae.'))} across shards {shards}")

    tensors, dtypes, source_meta = {}, {}, None
    for shard in shards:
        with safe_open(f"{RAW}/{shard}", framework="pt", device="cpu") as handle:
            if source_meta is None:
                source_meta = handle.metadata()
            for key in handle.keys():
                if not key.startswith("vae."):
                    continue
                tensor = handle.get_tensor(key)
                tensors[key[len("vae."):]] = tensor.contiguous()
                dtypes[str(tensor.dtype).replace("torch.", "")] = dtypes.get(
                    str(tensor.dtype).replace("torch.", ""), 0) + 1

    print(f"extracted {len(tensors)} tensors, dtypes {dtypes}")
    bytes_total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"total {bytes_total / 2**30:.3f} GiB")
    save_file(tensors, TARGET, metadata=source_meta or {"format": "pt"})
    print(f"wrote {TARGET}")
    return tensors


def verify():
    """Through the node a user would use, then the branch that has to claim it."""
    import folder_paths
    print(f"\nfolder_paths.base_path: {folder_paths.base_path}")
    listed = [n for n in folder_paths.get_filename_list("vae") if "hunyuan_image_3" in n]
    print(f"VAELoader sees: {listed}")

    from nodes import VAELoader
    node = VAELoader()
    # the stock VAELoader is a V1 node: its entry point is the `load_vae` function, not `execute`
    out = node.load_vae("hunyuan_image_3_instruct_distil_vae_fp32.safetensors")
    values = getattr(out, "result", None) or (out if isinstance(out, tuple) else (out,))
    vae = values[0]
    print(f"VAELoader -> {type(vae).__name__}")
    model = vae.first_stage_model
    print(f"  first_stage_model: {type(model).__name__}        <- must be HunyuanImage3VAE, not a stock VAE")
    print(f"  latent_channels {vae.latent_channels}  latent_dim {vae.latent_dim}  "
          f"downscale {vae.downscale_ratio}  not_video {getattr(vae, 'not_video', None)}")
    print(f"  vae_dtype ComfyUI resolved: {vae.vae_dtype}")
    weight_dtypes = {str(p.dtype).replace("torch.", "") for p in model.parameters()}
    print(f"  loaded parameter dtypes: {sorted(weight_dtypes)}")
    print(f"  temb_proj present in the decoder? "
          f"{any('temb_proj' in k for k in model.state_dict())}  (must be False: it is the discriminator)")
    print(f"  memory_used_decode(1024x1024 latent): "
          f"{vae.memory_used_decode((1, vae.latent_channels, 1, 64, 64), torch.float16) / 2**30:.2f} GiB (inherited, unvalidated)")

    # decode a small latent to prove the weights are real, not just present
    generator = torch.Generator().manual_seed(0)
    latent = torch.randn(1, vae.latent_channels, 1, 8, 8, generator=generator)
    with torch.no_grad():
        image = vae.decode(latent)
    print(f"  decode(1,{vae.latent_channels},1,8,8) -> {tuple(image.shape)}  "
          f"absmax {image.abs().max():.4f}  finite {bool(torch.isfinite(image).all())}")


if __name__ == "__main__":
    if "--verify-only" not in sys.argv:
        extract()
    verify()
