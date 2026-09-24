"""HunyuanImage-3.0 int8 conversion: bank-wise reader and the quantization error table.

Step 4.5's gate: measure dequantize(quantize(W)) vs W for the expert banks and dense projections,
with and without ConvRot, before converting the whole checkpoint.

Bank-wise, not shard-wise: 57 of the 64 expert banks have their expert tensors spread over two
shard files (Addendum 4 §C), so a shard loop cannot build them. Tensors are read one at a time
through safetensors, located by the index, which keeps peak memory at one expert tensor.

Usage:
    python convert_int8_convrot.py --report 2            # banks sourced from the first two shards
    python convert_int8_convrot.py --report 2 --layers 0,1,2
"""
import argparse
import json
import math
import os

import torch
from safetensors import safe_open

from comfy.cli_args import args as cli_args
cli_args.cpu = True
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

# convert_all.py sets these per model; the defaults are only for running a converter by hand
_COMFY_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models", "diffusion_models")
MODEL_DIR = os.environ.get("HUNYUAN_IMAGE_3_MODEL_DIR", os.path.normpath(_COMFY_MODELS))
WEIGHTS = os.environ.get("HUNYUAN_IMAGE_3_WEIGHTS", "")      # the original shard directory; convert_all.py sets it
CONVERTED = os.environ.get("HUNYUAN_IMAGE_3_CONVERTED", os.path.normpath(_COMFY_MODELS))
CONVROT_GROUP_SIZE = 256

# quantized: expert banks and the dense projections. Everything else stays bf16 (Addendum 2 §F2).
BANK_PROJECTIONS = ("gate_and_up_proj", "down_proj")
DENSE_SUFFIXES = (
    "self_attn.qkv_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.shared_mlp.gate_and_up_proj.weight",
    "mlp.shared_mlp.down_proj.weight",
)


def rel_error(a, b):
    a = a.float().to(b.device)
    b = b.float()
    return ((a - b).norm() / a.norm()).item(), (a - b).abs().max().item()


SAFETENSORS_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}
SAFETENSORS_NAMES = {dtype: name for name, dtype in SAFETENSORS_DTYPES.items()}


def read_header(path):
    """The safetensors JSON header of a shard: key -> dtype/shape/offsets, no tensor data read."""
    with open(path, "rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        return json.loads(handle.read(length))


class CheckpointReader:
    """Index-driven tensor reader. Handles are opened lazily and kept for the lifetime of the run."""

    def __init__(self, model_dir, weights_dir):
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as handle:
            self.index = json.load(handle)["weight_map"]
        self.weights_dir = weights_dir
        self.handles = {}
        self.headers = {}

    def tensor(self, key):
        shard = self.index[key]
        if shard not in self.handles:
            self.handles[shard] = safe_open(os.path.join(self.weights_dir, shard), framework="pt", device="cpu")
        return self.handles[shard].get_tensor(key)

    def dtype_shape(self, key):
        shard = self.index[key]
        if shard not in self.headers:
            self.headers[shard] = read_header(os.path.join(self.weights_dir, shard))
        entry = self.headers[shard][key]
        return SAFETENSORS_DTYPES[entry["dtype"]], tuple(entry["shape"])

    def complete(self, keys):
        """True when every key's shard file is on disk (the download may still be running)."""
        return all(os.path.exists(os.path.join(self.weights_dir, self.index[k])) for k in keys)

    def expert_key(self, layer, expert, projection):
        return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"

    def bank_shards(self, layer, projection):
        """Which shard files this bank's 64 expert tensors live in."""
        return {self.index[self.expert_key(layer, e, projection)] for e in range(64)}

    def close(self):
        for handle in self.handles.values():
            handle.__exit__(None, None, None)
        self.handles.clear()


def quantize_2d(weight, convrot, device):
    """One weight matrix -> (int8 qdata, scale). ConvRot is per-channel only (library constraint)."""
    w = weight.to(device)
    qdata, params = TensorWiseINT8Layout.quantize(w, per_channel=True, convrot=convrot, convrot_groupsize=CONVROT_GROUP_SIZE)
    back = TensorWiseINT8Layout.dequantize(qdata, params)
    return qdata, params.scale, back


def excess_kurtosis(weight):
    """Pearson excess kurtosis, the statistic comfy_kitchen gates its W4A8 codebook on.

    `_W4A8_GATE_KURTOSIS = -0.1` in comfy_kitchen's w4a8_int8 backend: below it the group-normalized
    rotated weight is Gaussian enough that fitting a codebook is not worth it. ConvRot is
    incoherence processing, so this is the statistic that says whether it can help at all.
    """
    x = weight.detach().flatten().float()
    return (((x - x.mean()) / (x.std() + 1e-9)).pow(4).mean() - 3.0).item()


def outlier_ratio(weight):
    """Mean per-row max|x| / std(x).

    Row-wise int8 error is driven by a row's range, so a row carrying one large outlier loses more
    precision than a row whose values are spread evenly — the property kurtosis only proxies.
    """
    x = weight.detach().float()
    return (x.abs().amax(dim=1) / (x.std(dim=1) + 1e-9)).mean().item()


def rotated(weight):
    """The portable ConvRot rotation (eager backend), for measuring the post-rotation statistic."""
    from comfy_kitchen.backends.eager.quantization import rotate_int8_convrot_weight
    return rotate_int8_convrot_weight(weight.float(), CONVROT_GROUP_SIZE)


def bank_function_error(reader, layer, projection, device, convrot):
    """End-to-end error of the real expert path: loaded bank -> expert_linear vs fp32 matmul.

    This exercises the loader, the per-expert view and the ConvRot matmul, not just the quantizer.
    """
    import comfy.ops as ops

    qdata, scales, weights = [], [], []
    for expert in range(64):
        weight = reader.tensor(reader.expert_key(layer, expert, projection))
        q, scale, _ = quantize_2d(weight, convrot, device)
        qdata.append(q)
        scales.append(scale)
        weights.append(weight.to(device))

    conf = {"format": "int8_tensorwise", "convrot": convrot, "convrot_groupsize": CONVROT_GROUP_SIZE, "num_experts": 64}
    in_features, out_features = weights[0].shape[1], weights[0].shape[0]

    class BankHost(torch.nn.Module):
        def __init__(self, bank):
            super().__init__()
            self.bank = bank

    host = BankHost(ops.mixed_precision_ops({}).MoEExperts(64, in_features, out_features, bias=False, device=device, dtype=torch.bfloat16))
    host.load_state_dict({
        "bank.weight": torch.stack(qdata),
        "bank.weight_scale": torch.stack(scales),
        "bank.comfy_quant": torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8),
    }, strict=False)
    module = host.bank

    x = torch.randn(64, in_features, dtype=torch.bfloat16, device=device)
    errors = []
    for expert in range(64):
        got = module.expert_linear(x, expert).float()
        reference = x.float() @ weights[expert].float().T
        errors.append(rel_error(reference, got)[0])
    return sum(errors) / len(errors), max(errors)


def kurtosis_report(args):
    reader = CheckpointReader(MODEL_DIR, WEIGHTS)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(32))
    device = args.device

    tensor_sets = [("bank gate_and_up", "bank", "gate_and_up_proj"), ("bank down", "bank", "down_proj")] + \
                  [(suffix.split(".")[-2], "dense", suffix) for suffix in DENSE_SUFFIXES]
    collected = {name: {"raw": [], "rotated": [], "outlier": [], "plain": [], "convrot": []} for name, _, _ in tensor_sets}

    print(f"excess kurtosis of the raw weight (higher = heavier tailed, ConvRot has more to work with)")
    print(f"{'layer':>5} {'bank gate_up':>13} {'bank down':>10} {'qkv_proj':>9} {'o_proj':>8} {'shared gu':>10} {'shared down':>12}")
    for layer in layers:
        values = []
        keys = [reader.expert_key(layer, 0, t) if kind == "bank" else f"model.layers.{layer}.{t}" for _, kind, t in tensor_sets]
        if not reader.complete(keys):
            print(f"{layer:>5} (not downloaded yet)")
            continue
        for name, kind, target in tensor_sets:
            weight = reader.tensor(reader.expert_key(layer, 0, target) if kind == "bank" else f"model.layers.{layer}.{target}")
            raw = excess_kurtosis(weight)
            collected[name]["raw"].append(raw)
            collected[name]["outlier"].append(outlier_ratio(weight))
            _, _, back_plain = quantize_2d(weight, False, device)
            _, _, back_convrot = quantize_2d(weight, True, device)
            collected[name]["plain"].append(rel_error(weight, back_plain)[0])
            collected[name]["convrot"].append(rel_error(weight, back_convrot)[0])
            if kind == "bank":
                collected[name]["rotated"].append(excess_kurtosis(rotated(weight.to(device))))
                values.append(raw)
        print(f"{layer:>5} {values[0]:>13.2f} {values[1]:>10.2f} "
              f"{collected['qkv_proj']['raw'][-1]:>9.2f} {collected['o_proj']['raw'][-1]:>8.2f} "
              f"{collected['gate_and_up_proj']['raw'][-1]:>10.2f} {collected['down_proj']['raw'][-1]:>12.2f}")

    print()
    print(f"{'tensor':<20} {'kurt mean':>9} {'kurt med':>8} {'kurt min':>8} {'kurt max':>8} {'rot kurt':>9} "
          f"{'outlier':>8} {'plain':>10} {'convrot':>10} {'delta':>8}")
    for name, kind, target in tensor_sets:
        raw = sorted(collected[name]["raw"])
        rot = collected[name]["rotated"]
        out = collected[name]["outlier"]
        plain = collected[name]["plain"]
        conv = collected[name]["convrot"]
        plain_mean = sum(plain) / len(plain)
        conv_mean = sum(conv) / len(conv)
        print(f"{name:<20} {sum(raw) / len(raw):>9.2f} {raw[len(raw) // 2]:>8.2f} {raw[0]:>8.2f} {raw[-1]:>8.2f} "
              f"{(sum(rot) / len(rot) if rot else float('nan')):>9.2f} {sum(out) / len(out):>8.2f} "
              f"{plain_mean:>10.4e} {conv_mean:>10.4e} {(conv_mean - plain_mean) / plain_mean * 100:>7.2f}%")
    reader.close()


def functional_report(args):
    """Re-measure the expert banks through the real path, with ConvRot working (Addendum 6 §C)."""
    reader = CheckpointReader(MODEL_DIR, WEIGHTS)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(32))
    prefix = {i: f"model-{i:04d}-of-0032.safetensors" for i in range(1, args.report + 1)}

    print(f"expert_linear on a loaded bank vs fp32 reference, {args.device}")
    print(f"{'bank':<28} {'convrot off':>16} {'convrot on':>16}")
    for layer in layers:
        for projection in BANK_PROJECTIONS:
            if not (reader.bank_shards(layer, projection) & set(prefix.values())):
                continue
            off = bank_function_error(reader, layer, projection, args.device, False)
            on = bank_function_error(reader, layer, projection, args.device, True)
            print(f"L{layer} {projection:<22} {off[0]:>10.4e}/{off[1]:>5.1e} {on[0]:>10.4e}/{on[1]:>5.1e}")
    reader.close()


def report(args):
    reader = CheckpointReader(MODEL_DIR, WEIGHTS)
    device = args.device

    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(32))
    targets = []
    for layer in layers:
        for projection in BANK_PROJECTIONS:
            if len(reader.bank_shards(layer, projection) & {f"model-{i:04d}-of-0032.safetensors" for i in range(1, args.report + 1)}) > 0:
                targets.append(("bank", layer, projection))
        for suffix in DENSE_SUFFIXES:
            key = f"model.layers.{layer}.{suffix}"
            if key in reader.index and int(reader.index[key][6:10]) <= args.report:
                targets.append(("dense", layer, suffix))

    print(f"device {device}, reading {WEIGHTS}")
    print(f"{'tensor':<62} {'shape':<20} {'plain':>11} {'convrot':>11} {'delta':>9}")
    print("-" * 120)

    summary = {"plain": [], "convrot": []}
    distributions = {}
    for kind, layer, name in targets:
        if kind == "bank":
            plain_errs, convrot_errs = [], []
            shape = None
            for expert in range(64):
                weight = reader.tensor(reader.expert_key(layer, expert, name))
                if shape is None:
                    shape = tuple(weight.shape)
                    scale = weight.shape[-1] % CONVROT_GROUP_SIZE == 0
                    if not scale:
                        raise ValueError(f"{name} in_features {weight.shape[-1]} is not a multiple of {CONVROT_GROUP_SIZE}")
                _, _, back_plain = quantize_2d(weight, False, device)
                _, _, back_convrot = quantize_2d(weight, True, device)
                plain_errs.append(rel_error(weight, back_plain)[0])
                convrot_errs.append(rel_error(weight, back_convrot)[0])
            label = f"L{layer} bank {name} (64 experts)"
            plain, convrot = sum(plain_errs) / 64, sum(convrot_errs) / 64
            label_for_shape = f"{shape[0]}x{shape[1]}"
            deltas = sorted((c - p) / p * 100 for p, c in zip(plain_errs, convrot_errs))
            distributions[label] = (deltas, sum(1 for d in deltas if d < 0))
        else:
            weight = reader.tensor(f"model.layers.{layer}.{name}")
            _, _, back_plain = quantize_2d(weight, False, device)
            _, _, back_convrot = quantize_2d(weight, True, device)
            plain = rel_error(weight, back_plain)[0]
            convrot = rel_error(weight, back_convrot)[0]
            label = f"L{layer} {name}"
            label_for_shape = "x".join(str(s) for s in weight.shape)

        summary["plain"].append(plain)
        summary["convrot"].append(convrot)
        print(f"{label:<62} {label_for_shape:<20} {plain:>11.4e} {convrot:>11.4e} {(convrot - plain) / plain * 100:>8.2f}%")

    reader.close()
    n = len(summary["plain"])
    if n:
        plain_mean = sum(summary["plain"]) / n
        convrot_mean = sum(summary["convrot"]) / n
        print("-" * 120)
        print(f"{'mean over ' + str(n) + ' tensors':<62} {'':<20} {plain_mean:>11.4e} {convrot_mean:>11.4e} "
              f"{(convrot_mean - plain_mean) / plain_mean * 100:>8.2f}%")

    for label, (deltas, improved) in distributions.items():
        print(f"{label}: convrot better on {improved}/64 experts | delta% min {deltas[0]:+.1f} "
              f"p25 {deltas[15]:+.1f} median {deltas[31]:+.1f} p75 {deltas[47]:+.1f} max {deltas[-1]:+.1f}")


_EXPERT_BANK = {"gate_and_up_proj": "experts_gate_up_proj", "down_proj": "experts_down_proj"}


def tobytes(tensor):
    """Raw safetensors bytes for a tensor, without materialising it again in float form."""
    t = tensor.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu):
        return t.view(torch.uint8).numpy().tobytes()
    return t.numpy().tobytes()


def quant_json(shape):
    conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": CONVROT_GROUP_SIZE}
    if len(shape) == 3:
        conf["num_experts"] = shape[0]
    return json.dumps(conf).encode("utf-8")


def dtype_bytes(dtype):
    return torch.empty(0, dtype=dtype).element_size()


class SafetensorsWriter:
    """Streaming safetensors writer with resume.

    The format puts the header before the data, and `safetensors.torch.save_file` needs every tensor
    in memory at once — 88 GB here. So the layout is planned from the checkpoint's own headers, the
    header is written first, then the tensors are appended one at a time, each checked against its
    planned byte length. A progress sidecar lets a killed run restart at the next unwritten key.
    """

    def __init__(self, path, plan, metadata, resume=False):
        self.path = path
        self.plan = plan
        self.entries = list(plan.items())
        header = {key: {"dtype": SAFETENSORS_NAMES[dtype], "shape": list(shape), "data_offsets": [start, start + length]}
                  for key, (dtype, shape, start, length) in plan.items()}
        header["__metadata__"] = metadata
        encoded = json.dumps(header).encode("utf-8")
        self.data_start = 8 + len(encoded)
        self.progress_path = path + ".progress"
        self.done = 0

        if resume and os.path.exists(self.progress_path):
            with open(self.progress_path) as handle:
                self.done = json.load(handle)["next_index"]
            self.file = open(path, "r+b")
            self.file.seek(self.data_start + plan[self.entries[self.done][0]][2])
            self.file.truncate()
            print(f"resuming at key {self.done}/{len(self.entries)} ({self.entries[self.done][0]})")
        else:
            self.file = open(path, "wb")
            self.file.write(len(encoded).to_bytes(8, "little"))
            self.file.write(encoded)

    def add(self, key, tensor):
        expected_index = self.entries[self.done][0]
        if key != expected_index:
            raise ValueError(f"out of order: expected {expected_index}, got {key}")
        _, shape, _, length = self.plan[key]
        data = tobytes(tensor)
        if len(data) != length:
            raise ValueError(f"{key}: planned {length} bytes, got {len(data)}")
        self.file.write(data)
        self.done += 1
        if self.done % 200 == 0:
            self.flush_progress()

    def flush_progress(self):
        with open(self.progress_path, "w") as handle:
            json.dump({"next_index": self.done, "keys": len(self.entries)}, handle)
        self.file.flush()

    def close(self):
        self.flush_progress()
        self.file.close()
        os.remove(self.progress_path)
        return self.done


def build_entries(reader, layer_limit=32):
    """(quantized, passthrough) output entries. Banks first, so the read pattern stays local."""
    quantized, passthrough = [], []
    for layer in range(layer_limit):
        for projection in BANK_PROJECTIONS:
            module = f"model.layers.{layer}.mlp.{_EXPERT_BANK[projection]}"
            quantized.append((module, reader.expert_key(layer, 0, projection), True))
    for layer in range(layer_limit):
        for suffix in DENSE_SUFFIXES:
            source = f"model.layers.{layer}.{suffix}"
            if source in reader.index:
                quantized.append((source[: -len(".weight")], source, False))

    quantized_sources = {source for _, source, _ in quantized}
    for key in reader.index:
        if key in quantized_sources or ".mlp.experts." in key:
            continue
        # `vae.` and `vision_model.` travel in their own files, and `lm_head.`/`model.ln_f.` in the text
        # head file (`make_cot_head.py`): only prompt rewriting reads them, so re-emitting them in every
        # checkpoint would cost 1.02 GiB per file. Everything else belongs in this one — the vision
        # aligner is what image-to-image runs — and passes through verbatim rather than being quantized.
        if key.startswith(("vae.", "vision_model.", "lm_head.", "model.ln_f.")):
            continue
        if key.startswith("model.layers.") and int(key.split(".")[2]) >= layer_limit:
            continue
        passthrough.append(key)
    return quantized, passthrough


def plan_output(reader, quantized, passthrough, scale_dtype):
    """Header plan: every output key with its dtype, shape, offset and byte length."""
    plan = {}
    offset = 0

    def add(name, dtype, shape):
        nonlocal offset
        length = int(math.prod(shape)) * dtype_bytes(dtype)
        plan[name] = (dtype, tuple(shape), offset, length)
        offset += length

    for module, source, is_bank in quantized:
        shape = reader.dtype_shape(source)[1]
        if shape[-1] % CONVROT_GROUP_SIZE != 0:
            raise ValueError(f"{source}: in_features {shape[-1]} is not a multiple of {CONVROT_GROUP_SIZE}")
        if is_bank:
            shape = (64,) + shape                      # the bank stacks all 64 experts
            scale_shape = shape[:-1] + (1,)
        else:
            scale_shape = (shape[0], 1)
        add(f"{module}.weight", torch.int8, shape)
        add(f"{module}.weight_scale", scale_dtype, scale_shape)
        add(f"{module}.comfy_quant", torch.uint8, (len(quant_json(shape)),))
    for key in passthrough:
        dtype, shape = reader.dtype_shape(key)
        add(key, dtype, shape)
    return plan


def convert(args):
    """Write the int8 ConvRot transformer file and the fp16 VAE file (Addendum 7 §D)."""
    reader = CheckpointReader(MODEL_DIR, WEIGHTS)
    with open(os.path.join(MODEL_DIR, "config.json")) as handle:
        config = handle.read()

    # the scale dtype the kernel produces: needed for the header, before any tensor exists
    _, probe_params = TensorWiseINT8Layout.quantize(torch.randn(16, CONVROT_GROUP_SIZE, device=args.device, dtype=torch.bfloat16),
                                                    per_channel=True, convrot=True, convrot_groupsize=CONVROT_GROUP_SIZE)
    quantized, passthrough = build_entries(reader, args.layer_limit)
    plan = plan_output(reader, quantized, passthrough, probe_params.scale.dtype)
    total = sum(length for _, _, _, length in plan.values())
    print(f"scale dtype {probe_params.scale.dtype}; {len(quantized)} quantized modules, "
          f"{len(passthrough)} passthrough; {len(plan)} keys, {total / 2**30:.1f} GiB planned -> {args.out}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = SafetensorsWriter(args.out, plan, {"config": config, "format": "hunyuan_image_3_quantized"}, resume=args.resume)
    for module, source, is_bank in quantized:
        if is_bank:
            layer = int(source.split(".")[2])
            projection = source.rsplit(".", 2)[-2]
            qdata, scales = [], []
            for expert in range(64):
                q, scale, _ = quantize_2d(reader.tensor(reader.expert_key(layer, expert, projection)), True, args.device)
                qdata.append(q.cpu())
                scales.append(scale.cpu())
            writer.add(f"{module}.weight", torch.stack(qdata))
            writer.add(f"{module}.weight_scale", torch.stack(scales))
        else:
            q, scale, _ = quantize_2d(reader.tensor(source), True, args.device)
            writer.add(f"{module}.weight", q.cpu())
            writer.add(f"{module}.weight_scale", scale.cpu())
        writer.add(f"{module}.comfy_quant", torch.tensor(list(quant_json(plan[f"{module}.weight"][1])), dtype=torch.uint8))

    for key in passthrough:
        writer.add(key, reader.tensor(key))
    writer.close()
    reader.close()
    print(f"transformer file written: {os.path.getsize(args.out) / 2**30:.2f} GiB -> {args.out}")

    if not args.no_vae:
        vae(args)


def vae(args):
    """vae.* in fp16, own file, with the `vae.` prefix stripped.

    ComfyUI's stock VAELoader keys off unprefixed names (`decoder.conv_in.weight`), so keeping the HF
    prefix makes the file match no VAE branch at all — a confusing failure unrelated to the
    architecture. The transformer file keeps its HF names, because this port's own loader reads those.
    """
    reader = CheckpointReader(MODEL_DIR, WEIGHTS)
    prefixed_keys = [key for key in reader.index if key.startswith("vae.")]
    keys = [(key[len("vae."):] if key.startswith("vae.") else key) for key in prefixed_keys]
    plan = {}
    offset = 0
    for key in keys:
        shape = reader.dtype_shape("vae." + key)[1]
        length = int(math.prod(shape)) * 2
        plan[key] = (torch.float16, shape, offset, length)
        offset += length
    print(f"vae file: {len(plan)} keys, prefix stripped, {offset / 2**30:.2f} GiB planned -> {args.vae_out}")
    writer = SafetensorsWriter(args.vae_out, plan, {"format": "hunyuan_image_3_vae_fp16"}, resume=args.resume)
    for prefixed_key, key in zip(prefixed_keys, keys):
        writer.add(key, reader.tensor(prefixed_key).to(torch.float16))
    writer.close()
    reader.close()
    print(f"vae file written: {os.path.getsize(args.vae_out) / 2**30:.2f} GiB -> {args.vae_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=int, default=2, help="include tensors sourced from the first N shards")
    parser.add_argument("--layers", default=None, help="comma separated layer indices, default all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kurtosis", action="store_true", help="excess kurtosis per tensor instead of the error table")
    parser.add_argument("--functional", action="store_true", help="bank error through expert_linear, not just the quantizer")
    parser.add_argument("--convert", action="store_true", help="write the int8 ConvRot transformer + fp16 VAE files")
    parser.add_argument("--vae", action="store_true", help="rewrite only the fp16 VAE file")
    parser.add_argument("--resume", action="store_true", help="continue a killed conversion at its progress sidecar")
    parser.add_argument("--layer-limit", type=int, default=32, help="convert only the first N layers (smoke tests)")
    parser.add_argument("--no-vae", action="store_true", help="with --convert, skip the fp16 VAE file")
    parser.add_argument("--out", default=os.path.join(CONVERTED, "hunyuan_image_3_instruct_distil_int8_convrot.safetensors"))
    parser.add_argument("--vae-out", default=os.path.join(CONVERTED, "hunyuan_image_3_instruct_distil_vae_fp16.safetensors"))
    args = parser.parse_args()
    if args.kurtosis:
        kurtosis_report(args)
    elif args.functional:
        functional_report(args)
    elif args.convert:
        convert(args)
    elif args.vae:
        vae(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
