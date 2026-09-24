"""Run the suite against the pack rather than against a copy of it inside ComfyUI.

The tests were written when this port lived in-tree, so they import `comfy.ldm.hunyuan_image_3.*`.
That in-tree copy still exists on the development branch and has already drifted from the pack — it
predates `full_attention_slices` and the per-checkpoint sequence template — so a suite that keeps
importing it tests code nobody installs. Rather than rewrite every import (and lose the ability to
diff the two files), the pack is loaded under its own name and aliased into `sys.modules` at the
names the tests already use. What runs is the pack; what is written is unchanged.

The pack directory is hyphenated and therefore not importable by name, and its modules use
parent-relative imports (`from ..latent_formats import ...`), so it has to be loaded from its path
under a parent package name.
"""
import importlib
import importlib.util
import os
import sys
import types

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# `--import-mode=importlib` does not put the test directory on the path, and the suite shares one
# helper module between files
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# `comfy` must import before the pack does: the pack's `compat` patches `comfy.ops` on import.
if importlib.util.find_spec("comfy") is None:                     # pragma: no cover - env guard
    raise RuntimeError(
        "ComfyUI is not importable. Run with PYTHONPATH pointing at your ComfyUI checkout, e.g.\n"
        "  PYTHONPATH=/path/to/ComfyUI pytest tests"
    )

# Default the data the suite reads to where the consolidated setup keeps it; an explicit environment
# variable still wins. The runtime files live in ComfyUI's model folders, the reference captures and
# the original checkpoint's config + shard index in this pack's dev/ (both local, not in git).
_COMFY = os.path.dirname(os.path.dirname(os.path.realpath(PACK)))   # ComfyUI root
for _variable, _path in (
        ("HUNYUAN_IMAGE_3_TOKENIZER", os.path.join(PACK, "hunyuan_image_3", "tencent", "tokenizer.json")),
        ("HUNYUAN_IMAGE_3_VAE", os.path.join(_COMFY, "models", "vae", "hunyuan_image_3_vae_fp16.safetensors")),
        ("HUNYUAN_IMAGE_3_CAPTURE", os.path.join(PACK, "dev", "instruct_path_sequences.pt")),
        ("HUNYUAN_IMAGE_3_MULTI_CAPTURE", os.path.join(PACK, "dev", "multi_image_reference_sequences.pt")),
        ("HUNYUAN_IMAGE_3_MODEL_DIR", os.path.join(PACK, "dev", "model_dir"))):
    if os.path.exists(os.path.normpath(_path)):
        os.environ.setdefault(_variable, os.path.normpath(_path))

import torch
from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True


def _load_pack():
    spec = importlib.util.spec_from_file_location(
        "hy3pack", os.path.join(PACK, "__init__.py"), submodule_search_locations=[PACK])
    module = importlib.util.module_from_spec(spec)
    sys.modules["hy3pack"] = module
    spec.loader.exec_module(module)
    return module


def _alias(name, module):
    """Publish `module` at `name`, creating the intermediate packages the name implies.

    `comfy.ldm` is real and `comfy_extras` may be; only the missing links are synthesised, and an
    alias is attached to its parent as an attribute too so `from a.b import c` resolves either way.
    """
    parts = name.split(".")
    for depth in range(1, len(parts)):
        parent = ".".join(parts[:depth])
        if parent not in sys.modules:
            package = types.ModuleType(parent)
            package.__path__ = []
            sys.modules[parent] = package
    sys.modules[name] = module
    setattr(sys.modules[".".join(parts[:-1])], parts[-1], module)


_load_pack()

_alias("comfy.ldm.hunyuan_image_3", importlib.import_module("hy3pack.hunyuan_image_3"))
for _name in ("loader", "model", "model_base", "ops", "pipeline", "rewrite", "system_prompt", "tokenizer", "vae"):
    _alias(f"comfy.ldm.hunyuan_image_3.{_name}",
           importlib.import_module(f"hy3pack.hunyuan_image_3.{_name}"))

# the nodes moved from `comfy_extras/nodes_hunyuan_image_3.py` to the pack's own `nodes.py`
_alias("comfy_extras.nodes_hunyuan_image_3", importlib.import_module("hy3pack.nodes"))
