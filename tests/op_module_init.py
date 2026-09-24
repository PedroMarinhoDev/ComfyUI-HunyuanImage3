"""Shared test helper: give `comfy.ops` modules the parameters a checkpoint would provide.

ComfyUI builds these classes for checkpoints, not for standalone construction:

* every ops norm/conv/embedding parameter is created but never initialised — `reset_parameters`
  is deliberately a no-op (`comfy/ops.py:682-685` for RMSNorm), so the parameter holds whatever
  the allocator left behind: zeros on fresh pages, garbage after churn;
* `mixed_precision_ops.Linear` and `MoEExperts` do not even create a `weight` attribute until a
  state dict arrives, and `MoEExperts` leaves it `None`.

A test or harness that constructs a model without loading a checkpoint therefore runs on
uninitialised memory and produces garbage or NaN that looks like a model bug. Call
`init_op_modules()` once after construction, before any forward.
"""
import torch
import torch.nn as nn

NORM_TYPES = (nn.RMSNorm, nn.GroupNorm, nn.LayerNorm)
CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def init_op_modules(module, dtype=None, seed=0, std=0.02):
    """Replace every parameter of an ops-built module tree with a sane, reproducible value.

    Norms get weight 1 / bias 0 (what a normalised checkpoint carries); parameter-bearing layers
    get small random values from a seeded generator, honouring each slot's dtype.
    """
    generator = torch.Generator().manual_seed(seed)

    def random_(shape, target_dtype, scale=std):
        tensor = torch.randn(shape, generator=generator, dtype=torch.float32)
        return nn.Parameter(tensor.to(target_dtype or torch.get_default_dtype()) * scale)

    def zeros_(shape, target_dtype):
        return nn.Parameter(torch.zeros(shape, dtype=target_dtype or torch.get_default_dtype()))

    for submodule in module.modules():
        weight = getattr(submodule, "weight", None)
        weight_dtype = weight.dtype if isinstance(weight, torch.Tensor) else dtype

        if isinstance(submodule, NORM_TYPES):
            submodule.weight = nn.Parameter(torch.ones_like(submodule.weight))
            if submodule.bias is not None:
                submodule.bias = nn.Parameter(torch.zeros_like(submodule.bias))
        elif hasattr(submodule, "num_experts") and hasattr(submodule, "out_features") and hasattr(submodule, "in_features"):
            submodule.weight = random_((submodule.num_experts, submodule.out_features, submodule.in_features), weight_dtype)
        elif isinstance(submodule, nn.Embedding):
            submodule.weight = random_((submodule.num_embeddings, submodule.embedding_dim), weight_dtype)
        elif isinstance(submodule, CONV_TYPES):
            submodule.weight = random_(submodule.weight.shape, weight_dtype)
            if submodule.bias is not None:
                submodule.bias = zeros_(submodule.bias.shape, weight_dtype)
        elif hasattr(submodule, "in_features") and hasattr(submodule, "out_features"):
            submodule.weight = random_((submodule.out_features, submodule.in_features), weight_dtype)
            if getattr(submodule, "bias", None) is not None:
                submodule.bias = zeros_(submodule.bias.shape, weight_dtype)

    return module
