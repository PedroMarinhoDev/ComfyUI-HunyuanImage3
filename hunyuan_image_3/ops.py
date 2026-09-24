"""Ops for a non-quantized checkpoint: the plain weight-dtype operations plus the routed-expert bank
the model needs.

`comfy.ops.mixed_precision_ops` supplies `MoEExperts` only when the checkpoint carries a `quant_config`,
so a bf16 checkpoint builds no bank and the MoE block raises `AttributeError: type object
'disable_weight_init' has no attribute 'MoEExperts'` at construction. The bank is a layout detail
rather than a quantization one — one `[num_experts, out, in]` tensor either way — so it is added here to
the plain operations, and the model, the load path and the offload gate stay identical between the two.
"""

import contextlib

import torch
import torch.nn.functional as F

import comfy.ops
from comfy.quant_ops import QuantizedTensor


class MoEExperts(comfy.ops.disable_weight_init.Linear):
    """One stacked expert bank, held at the checkpoint's own dtype.

    Same layout and the same `expert_linear`/`bank_resident` contract as the mixed-precision class the
    quantized checkpoints use, so the MoE block's forward does not branch on precision.
    """

    def __init__(self, num_experts, in_features, out_features, bias=False, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.num_experts = num_experts
        self.weight = torch.nn.Parameter(
            torch.empty((num_experts, out_features, in_features), dtype=dtype, device=device),
            requires_grad=False)
        if self.bias is not None:
            self.bias = torch.nn.Parameter(
                torch.empty((num_experts, out_features), dtype=dtype, device=device), requires_grad=False)
        self._resident_bank = None

    @contextlib.contextmanager
    def bank_resident(self, input):
        """Hold the whole bank on the input's device for the `expert_linear` calls inside, as the
        quantized class does. A bank at the compute dtype needs no casting, but it does need placing:
        the offload keeps it on the CPU between forwards."""
        weight, bias = self.weight, self.bias
        if weight.device != input.device:
            weight = weight.to(input.device)
            bias = None if bias is None else bias.to(input.device)
        self._resident_bank = (weight, bias)
        try:
            yield self
        finally:
            self._resident_bank = None

    def expert_linear(self, input, i):
        """Expert i's projection. Without a resident bank only that expert's slice is moved, so a
        forward routing to a few experts does not stream the whole layer."""
        resident = self._resident_bank
        if resident is not None:
            weight, bias = resident
            weight, bias = weight[i], None if bias is None else bias[i]
        else:
            # index first, then move: `.to()` on the bank would stream all of its experts to use one
            weight = self.weight[i].to(input.device)
            bias = None if self.bias is None else self.bias[i].to(input.device)
        return F.linear(input, weight, bias)


class QuantlessMoEOps(comfy.ops.disable_weight_init):
    MoEExperts = MoEExperts


# ------------------------------------------------------------------- per-expert fetch for decode steps

def _matmul(module, input, weight, bias):
    """The GEMM of `MoEExperts._expert_matmul`, for cores that predate that method."""
    if isinstance(weight, QuantizedTensor):
        use_fast = (not module._full_precision_mm
                    and weight.layout_cls.supports_fast_matmul()
                    and input.dim() == 2)
        if use_fast:
            return F.linear(QuantizedTensor.from_float(input, module.layout_type), weight, bias)
        out = input @ weight.dequantize().t()
        return out + bias if bias is not None else out
    return F.linear(input, weight, bias)


def expert_linear_sliced(module, input, i):
    """Expert `i`'s projection, moving only that expert's slice of the bank.

    This is the decode-step path. Stock ComfyUI's `MoEExperts.expert_linear`, outside a resident bank,
    casts the *whole* bank through `CastBiasWeightContext` and then indexes one expert out of it. On a
    card that cannot hold every bank, aimdo evicts and re-faults a ~0.5 GiB bank for each of the 8
    routed experts in each of 32 layers: measured 168 GiB and 17.2 s per CoT token on a 4090, against
    a 9 MiB slice per expert here.

    Anything that patches the weight (a LoRA, a lowvram patch) is applied by the core cast and not
    here, so a patched bank takes the core path: slower, but right.
    """
    if (getattr(module, "_resident_bank", None) is not None or not isinstance(module.weight, QuantizedTensor)
            or getattr(module, "weight_function", None) or getattr(module, "bias_function", None)
            or getattr(module, "weight_lowvram_function", None) is not None):
        return module.expert_linear(input, i)

    device = input.device
    # From the parameter itself, not from aimdo's pinned host copy of the bank: slicing the pin was
    # measured and bought nothing (806 vs 799 ms per token — at ~9 MiB per copy the per-copy overhead
    # dominates, not the link rate), and a pin can be handed to another module once evicted.
    weight = module._expert_qt_from(module.weight, i)
    if weight.device != device:
        weight = weight.to(device)
    bias = None
    if module.bias is not None:
        bias = comfy.ops.cast_to_input(module.bias[i].to(device), input, copy=False)
    matmul = getattr(module, "_expert_matmul", None)
    return matmul(input, weight, bias) if matmul is not None else _matmul(module, input, weight, bias)
