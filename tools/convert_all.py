"""Build every HunyuanImage-3.0 file ComfyUI needs for one checkpoint, from its original shards.

Per model (`instruct_distil`, `instruct`, `base`) and per requested format:

  w4a8 | int8 | bf16   hunyuan_image_3_<model>_<format>.safetensors        the diffusion model
  head                 hunyuan_image_3_<model>_cot_head.safetensors         lm_head + ln_f, for prompt rewriting
  vision               hunyuan_image_3_<model>_siglip2_so400m_naflex.safetensors  (clip_vision)

The VAE is identical across the three checkpoints and is not rebuilt here.

Existing outputs are skipped, so an interrupted run can simply be started again.

Usage:
  convert_all.py --model instruct_distil --weights /path/to/HunyuanImage-3.0-Instruct-Distil \\
                 --out-dir models/diffusion_models --bf16-dir /hdd/models/diffusion_models \\
                 --clip-vision-dir models/clip_vision --formats w4a8,int8,head,vision --device cuda:1
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(TOOLS)
HEAD_KEYS = ("lm_head.weight", "model.ln_f.weight")
CONVERTERS = {"w4a8": "convert_w4a8.py", "int8": "convert_int8_convrot.py", "bf16": "make_bf16.py"}
SUFFIX = {"w4a8": "w4a8", "int8": "int8_convrot", "bf16": "bf16"}


def header_keys(path):
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return {key for key in json.loads(handle.read(length)) if key != "__metadata__"}


def run(command, env, log):
    print(f"$ {' '.join(command)}", flush=True)
    started = time.time()
    with open(log, "a") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        handle.flush()
        result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"failed ({result.returncode}) after {time.time() - started:.0f}s; see {log}")
    print(f"  done in {time.time() - started:.0f}s", flush=True)


def check_no_head(path):
    keys = header_keys(path)
    if any(key in keys for key in HEAD_KEYS):
        raise SystemExit(f"{path}: carries the text head, which belongs in its own file")
    return len(keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["instruct_distil", "instruct", "base"])
    parser.add_argument("--weights", required=True, help="the original shard directory (holds config.json)")
    parser.add_argument("--out-dir", required=True, help="where the w4a8/int8 files and the head go")
    parser.add_argument("--bf16-dir", default=None, help="where the bf16 files go (defaults to --out-dir)")
    parser.add_argument("--clip-vision-dir", default=None, help="where the vision tower goes")
    parser.add_argument("--formats", default="w4a8,int8,bf16,head,vision")
    parser.add_argument("--device", default="cuda:0", help="the GPU, by PCI bus order (cuda:1 = second card)")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    log = args.log or os.path.join(args.out_dir, f".convert_{args.model}.log")
    # The quantization kernels require the current CUDA device to be the tensors' device, so each run sees
    # exactly the requested GPU, addressed as cuda:0 inside ("Can't export tensors on a different CUDA
    # device index" otherwise).
    gpu = args.device.split(":")[1] if ":" in args.device else "0"
    env = dict(os.environ, HUNYUAN_IMAGE_3_WEIGHTS=args.weights, HUNYUAN_IMAGE_3_MODEL_DIR=args.weights,
               CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu,
               PYTHONPATH=os.pathsep.join(filter(None, [os.path.dirname(os.path.dirname(PACK)),
                                                        os.environ.get("PYTHONPATH")])))
    name = f"hunyuan_image_3_{args.model}"

    for fmt in formats:
        if fmt in CONVERTERS:
            directory = args.bf16_dir if (fmt == "bf16" and args.bf16_dir) else args.out_dir
            os.makedirs(directory, exist_ok=True)
            out = os.path.join(directory, f"{name}_{SUFFIX[fmt]}.safetensors")
            if not os.path.exists(out):
                command = [args.python, os.path.join(TOOLS, CONVERTERS[fmt]), "--out", out + ".part"]
                if fmt == "w4a8":
                    command += ["--device", "cuda:0"]
                elif fmt == "int8":
                    command += ["--convert", "--no-vae", "--device", "cuda:0"]
                else:
                    command += ["--model-dir", args.weights, "--weights", args.weights]
                run(command, env, log)
                os.replace(out + ".part", out)
            print(f"  {os.path.basename(out)}: {check_no_head(out)} keys", flush=True)
        elif fmt == "head":
            out = os.path.join(args.out_dir, f"{name}_cot_head.safetensors")
            if not os.path.exists(out):
                run([args.python, os.path.join(TOOLS, "make_cot_head.py"), "--weights", args.weights,
                     "--out", out + ".part"], env, log)
                os.replace(out + ".part", out)
            print(f"  {os.path.basename(out)}: {sorted(header_keys(out))}", flush=True)
        elif fmt == "vision":
            directory = args.clip_vision_dir or args.out_dir
            os.makedirs(directory, exist_ok=True)
            out = os.path.join(directory, f"{name}_siglip2_so400m_naflex.safetensors")
            if not os.path.exists(out):
                run([args.python, os.path.join(TOOLS, "convert_vision.py"), "--weights", args.weights,
                     "--out", out + ".part"], env, log)
                os.replace(out + ".part", out)
            print(f"  {os.path.basename(out)}: {len(header_keys(out))} keys", flush=True)
        else:
            raise SystemExit(f"unknown format {fmt!r}")


if __name__ == "__main__":
    main()
