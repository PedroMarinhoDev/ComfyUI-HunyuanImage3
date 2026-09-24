"""2b: the vision tower and its aligner as their own bf16 file.

This is a copy, not a conversion. ComfyUI already carries SigLIP2 with naflex, and the checkpoint's
vision keys are HF SigLIP's names — which are also ComfyUI's — so `step10a_vision_reuse.py` measured a
passthrough: building `CLIPVisionModelProjection` from the shipped `clip_vision_siglip2_base_naflex.json`
and loading our tensors gives 0 missing and 0 unexpected outside `vision_model.head.*`, the 11-key
attention-pooling head the reference never reads because it takes `last_hidden_state`.

The file lives under `models/clip_vision/` so ComfyUI's `CLIPVisionLoader` can pick it up by signature:
`load_clipvision_from_sd` selects the naflex config when `vision_model.encoder.layers.22.layer_norm1.weight`
exists, `layers.0.layer_norm1.weight` is 1152 wide and `patch_embedding.weight` is 2-D — all true here.
The four `vision_aligner.*` tensors ride along for the port's own projector, which is a `LightProjector`
without `LlavaProjector`'s `x[:, 1:]` CLS slice (SigLIP has no CLS token, so that slice would eat the
first patch).

`layer_norm_eps` is deliberately not forced: measured at 1.27e-05 relative against a 1e-3 bf16 bar, and
identical at fp32, so the shipped config's implicit 1e-05 is a documented deviation rather than a fix.

Each checkpoint has its own tower — base, Instruct and Instruct-Distil hash differently — so each gets its own
file; the VAE, by contrast, is byte-identical across all three and ships once.

Usage: convert_vision.py --weights ORIGINAL_SHARD_DIR --out FILE
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file
from safetensors import safe_open

PREFIXES = ("vision_model.", "vision_aligner.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="the directory holding the original shards")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    WEIGHTS, out = args.weights, args.out
    index = json.load(open(os.path.join(WEIGHTS, "model.safetensors.index.json")))
    wanted = [k for k in index["weight_map"] if k.startswith(PREFIXES)]
    print(f"{len(wanted)} tensors to copy from {len(set(index['weight_map'].values()))} shards")

    tensors = {}
    for shard in sorted({index["weight_map"][k] for k in wanted}):
        with safe_open(os.path.join(WEIGHTS, shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in wanted:
                    tensors[key] = handle.get_tensor(key).to(torch.bfloat16).contiguous()

    tower = sum(1 for k in tensors if k.startswith("vision_model."))
    aligner = sum(1 for k in tensors if k.startswith("vision_aligner."))
    head = sum(1 for k in tensors if ".head." in k)
    assert tower + aligner == len(wanted), "key accounting"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_file(tensors, out, metadata={"format": "pt"})
    size = os.path.getsize(out) / 2**30
    print(f"wrote {out}")
    print(f"  {len(tensors)} tensors ({tower} vision_model including {head} head, {aligner} vision_aligner), "
          f"{size:.2f} GiB bf16")

    # read it back the way its consumer will
    with safe_open(out, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        dtypes = {handle.get_slice(k).get_dtype() for k in keys}
    print(f"  read back: {len(keys)} keys, dtypes {sorted(dtypes)}")
    for key in ("vision_model.embeddings.patch_embedding.weight",
                "vision_model.encoder.layers.26.layer_norm2.weight",
                "vision_aligner.layers.2.weight"):
        assert key in keys, key
    print(f"  spot-checked patch embedding, last layer and the aligner")


if __name__ == "__main__":
    main()
