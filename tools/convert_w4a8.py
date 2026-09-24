"""W4A8 conversion: int4 weights, fp8 per-group scales, ConvRot rotation, Lloyd-Max codebook.

Quantizes the same modules as the int8 conversion (the routed expert banks and the dense projections;
everything else stays bf16), but through `AsymW4A8Int8Layout` instead of `TensorWiseINT8Layout`.

Two deliberate differences from the int8 converter:

* **Keys and `comfy_quant` come from the ops module's own `state_dict`.** The int8 path hand-writes
  `weight`/`weight_scale`, which is how `weight_correction` came to be written by the writer and dropped
  by the loader. Building a real `MoEExperts`/`Linear` and asking it for its state dict cannot drift.
* **Quantized in memory, then planned.** The writer needs the header before the data, and W4A8's keys and
  per-module shapes (fp8 `s_rel`, fp32 `s_channel`, optional `correction`, optional `codebook`) are only
  known after quantizing, so the file is planned from the finished tensors rather than predicted.

Quantization is per expert and stacked afterwards: the layout validates a 2-D matrix, so a stacked
`(64, out, in)` tensor cannot be quantized in one call.

Usage: convert_w4a8.py [--layer-limit 32] [--out PATH] [--device cuda:0] [--verify]
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# the package root, so the converter can pick up `compat` — the same shim the runtime applies, since the
# bank params class it needs is not in a stock checkout
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfy_kitchen.tensor.base import QuantizedTensor
from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout

from convert_int8_convrot import (CONVERTED, CONVROT_GROUP_SIZE, MODEL_DIR, WEIGHTS, CheckpointReader,
                                  SafetensorsWriter, _EXPERT_BANK, build_entries, dtype_bytes)

GROUP_SIZE = 16
FORMAT = "asym_w4a8_int8"
DEFAULT_OUT = os.path.join(CONVERTED, "hunyuan_image_3_instruct_distil_w4a8.safetensors")

# bank module name -> the checkpoint's projection name, from the converter's own map rather than by
# stripping a prefix (`experts_gate_up_proj` is `gate_and_up_proj` in the checkpoint)
BANK_TO_PROJECTION = {bank: projection for projection, bank in _EXPERT_BANK.items()}


def scratch_ops():
    import comfy.cli_args as cli_args
    cli_args.args.cpu = True
    import comfy.ops
    return comfy.ops.mixed_precision_ops({})


def bank_projection(module):
    """`model.layers.N.mlp.experts_gate_up_proj` -> (3, 'gate_and_up_proj')."""
    parts = module.split(".")
    return int(parts[2]), BANK_TO_PROJECTION[parts[-1]]


def file_tensor(tensor):
    """fp8 has to be stored as uint8; the loader views it back (comfy/ops.py, W4A8 branch)."""
    if tensor is not None and tensor.dtype == torch.float8_e4m3fn:
        return tensor.view(torch.uint8)
    return tensor


def quantize_module(reader, module, source, is_bank, device):
    """Returns (module, {key suffix: tensor}) with the layout already applied."""
    # where the upstream fix is present this is core's own class; where it is not, `compat` supplies it.
    # The converter has to go through the shim rather than reaching into comfy.ops directly, so it builds
    # the layout the loader will actually accept on the checkout the package is installed into
    import compat  # noqa: F401  (import for its side effect: it patches comfy.ops)
    from comfy.ops import _bank_params_class

    ops = scratch_ops()
    if is_bank:
        layer, projection = bank_projection(module)
        per_expert = [reader.tensor(reader.expert_key(layer, index, projection)).to(device)
                      for index in range(64)]
        quantized = [AsymW4A8Int8Layout.quantize(weight, group_size=GROUP_SIZE,
                                                 convrot_groupsize=CONVROT_GROUP_SIZE)
                     for weight in per_expert]
        fields = {}
        for name in ("scale", "s_channel", "correction", "codebook"):
            values = [getattr(params, name) for _, params in quantized]
            if values[0] is None:
                fields[name] = None
            else:
                fields[name] = torch.stack([value.cpu() for value in values])
        shape = per_expert[0].shape
        params_cls = _bank_params_class(AsymW4A8Int8Layout)
        params = params_cls(scale=fields["scale"], s_channel=fields["s_channel"],
                            correction=fields["correction"], codebook=fields["codebook"],
                            group_size=GROUP_SIZE, convrot_groupsize=CONVROT_GROUP_SIZE,
                            orig_dtype=torch.bfloat16, orig_shape=(64,) + tuple(shape))
        qdata = torch.stack([entry[0].cpu() for entry in quantized])
        module_op = ops.MoEExperts(64, shape[1], shape[0], bias=False, dtype=torch.bfloat16)
    else:
        weight = reader.tensor(source).to(device)
        qdata, params = AsymW4A8Int8Layout.quantize(weight, group_size=GROUP_SIZE,
                                                    convrot_groupsize=CONVROT_GROUP_SIZE)
        qdata = qdata.cpu()
        params = params.to_device(torch.device("cpu"))
        shape = weight.shape
        module_op = ops.Linear(shape[1], shape[0], bias=False, dtype=torch.bfloat16)

    module_op.quant_format = FORMAT
    module_op.layout_type = "AsymW4A8Int8Layout"
    module_op.weight = torch.nn.Parameter(
        QuantizedTensor(qdata, module_op.layout_type, params), requires_grad=False)

    entries = {}
    for key, value in module_op.state_dict().items():
        suffix = key.split(".", 1)[1] if "." in key else key
        entries[suffix] = file_tensor(value)
    return module, entries


def convert(args):
    from convert_int8_convrot import SafetensorsWriter as Writer

    reader = CheckpointReader(MODEL_DIR, WEIGHTS)
    quantized, passthrough = build_entries(reader, args.layer_limit)
    print(f"{len(quantized)} modules to quantize, {len(passthrough)} passthrough tensors", flush=True)

    tensors = {}
    codebook_stats = {"decided": 0, "absent": 0, "values": 0}
    start = time.perf_counter()
    for index, (module, source, is_bank) in enumerate(quantized):
        name, entries = quantize_module(reader, module, source, is_bank, args.device)
        for suffix, value in entries.items():
            tensors[f"{name}.{suffix}"] = value
        if "weight_codebook" in entries:
            codebook_stats["decided"] += 1
            codebook_stats["values"] += entries["weight_codebook"].numel()
        else:
            codebook_stats["absent"] += 1
        if (index + 1) % 16 == 0 or index + 1 == len(quantized):
            done = time.perf_counter() - start
            print(f"  {index + 1}/{len(quantized)} modules, {done:.0f}s, "
                  f"{sum(t.numel() * t.element_size() for t in tensors.values()) / 2**30:.1f} GiB held",
                  flush=True)
    for key in passthrough:
        tensors[key] = reader.tensor(key)

    plan = {}
    offset = 0
    for key, tensor in tensors.items():
        length = tensor.numel() * tensor.element_size()
        plan[key] = (tensor.dtype, tuple(tensor.shape), offset, length)
        offset += length
    print(f"planned {len(plan)} keys, {offset / 2**30:.2f} GiB", flush=True)

    writer = Writer(args.out, plan, {"format": "pt"}, resume=args.resume)
    for key in plan:
        writer.add(key, tensors[key])
    written = writer.close()
    print(f"wrote {written} keys to {args.out}")

    conf_counts = {}
    for key in plan:
        if key.endswith(".comfy_quant"):
            conf_counts[json.loads(tensors[key].numpy().tobytes())["format"]] = \
                conf_counts.get(json.loads(tensors[key].numpy().tobytes())["format"], 0) + 1
    print(f"comfy_quant formats: {conf_counts}")
    print(f"codebooks: decided {codebook_stats['decided']}, absent {codebook_stats['absent']}, "
          f"{codebook_stats['values']} values total")
    del tensors
    return args.out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-limit", type=int, default=32, help="convert only the first N layers")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    path = convert(args)
    print(f"done: {path} ({os.path.getsize(path) / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
