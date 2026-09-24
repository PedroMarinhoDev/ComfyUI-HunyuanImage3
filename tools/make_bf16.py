"""Write a bf16 transformer file from the original shards, in the same key layout as the quantized ones.

Reuses the converter's checkpoint reader and streaming writer. The expert banks are assembled expert by
expert and written at the checkpoint's own dtype, with no `weight_scale` and no `comfy_quant`, so
`detect_layer_quantization` finds no quantization, the loader builds on the plain ops plus the package's
expert bank (`hunyuan_image_3/ops.py`), and the model, the load path and the offload gate stay identical
to the quantized case.

Memory stays at one bank: a gate/up bank is `64 x 2I x H` bf16, ~3.2 GiB, and it is written before the
next is read.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from convert_int8_convrot import CheckpointReader, MODEL_DIR, WEIGHTS, build_entries, dtype_bytes, SafetensorsWriter


def plan_bf16(reader, quantized, passthrough, num_experts):
    """Header plan in the order the writer will receive it: banks and dense first, then passthrough."""
    plan = {}
    offset = 0

    def add(name, dtype, shape):
        nonlocal offset
        length = int(math.prod(shape)) * dtype_bytes(dtype)
        plan[name] = (dtype, tuple(shape), offset, length)
        offset += length

    for module, source, is_bank in quantized:
        dtype, shape = reader.dtype_shape(source)
        add(f"{module}.weight", dtype, (num_experts,) + shape if is_bank else shape)
    for key in passthrough:
        dtype, shape = reader.dtype_shape(key)
        add(key, dtype, shape)
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=MODEL_DIR, help="the checkpoint directory holding config.json")
    parser.add_argument("--weights", default=WEIGHTS, help="the directory holding the original shards")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(WEIGHTS),
                                                      "hunyuan_image_3_instruct_distil_bf16.safetensors"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    reader = CheckpointReader(args.model_dir, args.weights)
    config = json.load(open(os.path.join(args.model_dir, "config.json")))
    num_experts = config["num_experts"]
    quantized, passthrough = build_entries(reader, 32)
    plan = plan_bf16(reader, quantized, passthrough, num_experts)
    total = sum(entry[3] for entry in plan.values())
    print(f"{num_experts} experts; {len(plan)} keys, {total / 2**30:.1f} GiB planned -> {args.out}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = SafetensorsWriter(args.out, plan, {"config": open(os.path.join(args.model_dir, "config.json")).read(),
                                               "format": "hunyuan_image_3_bf16"}, resume=args.resume)
    for module, source, is_bank in quantized:
        if is_bank:
            layer = int(source.split(".")[2])
            projection = source.rsplit(".", 2)[-2]
            experts = [reader.tensor(reader.expert_key(layer, expert, projection))
                       for expert in range(num_experts)]
            writer.add(f"{module}.weight", torch.stack(experts))
        else:
            writer.add(f"{module}.weight", reader.tensor(source))
    for key in passthrough:
        writer.add(key, reader.tensor(key))
    writer.close()
    reader.close()
    print(f"written: {os.path.getsize(args.out) / 2**30:.2f} GiB -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
