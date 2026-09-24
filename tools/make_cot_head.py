"""The text head's two tensors in their own small file, for the Prompt Rewriting node.

`lm_head` and `ln_f` are read only by prompt rewriting, so they travel here rather than inside every
diffusion checkpoint: carrying them costs about 1.02 GiB per checkpoint for a stage most graphs never run.
The converters leave them out, and the loader ignores them in a checkpoint that still carries them.

Which checkpoint they belong to matters. The Instruct's and the Instruct-Distil's heads are byte-identical, so one
file serves both. The base was trained separately — same shapes, different values, max|diff| around 1e-1 —
so it needs its own file, and pointing the base at the Instruct's head would be silently wrong.

Usage: make_cot_head.py --out FILE [--weights DIR] [--verify-against CHECKPOINT]
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

KEYS = ("lm_head.weight", "model.ln_f.weight")

DEFAULT_WEIGHTS = os.environ.get("HUNYUAN_IMAGE_3_WEIGHTS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="the HuggingFace repo directory the head is read from")
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify-against", default=None,
                        help="a converted checkpoint that still carries these keys; asserts bit-identity")
    args = parser.parse_args()

    index = json.load(open(os.path.join(args.weights, "model.safetensors.index.json")))["weight_map"]
    tensors = {}
    for key in KEYS:
        with safe_open(os.path.join(args.weights, index[key]), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key).contiguous()
        tensors[key] = tensor
        print(f"  raw {key:22} {str(tensor.dtype).replace('torch.', ''):5} "
              f"{tuple(tensor.shape)}  {tensor.numel() * tensor.element_size() / 2**20:.1f} MiB")

    save_file(tensors, args.out, metadata={"format": "pt"})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"wrote {args.out}  ({total / 2**30:.3f} GiB)")

    if args.verify_against:
        print(f"\n=== bit-identity against {os.path.basename(args.verify_against)}")
        for key in KEYS:
            with safe_open(args.verify_against, framework="pt", device="cpu") as handle:
                other = handle.get_tensor(key)
            same = torch.equal(tensors[key], other)
            print(f"  {key:22} identical {same}  max|diff| "
                  f"{(tensors[key].float() - other.float()).abs().max():.3e}")
            assert same, f"{key} differs from the copy in {args.verify_against}"


if __name__ == "__main__":
    main()