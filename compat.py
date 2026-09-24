"""The behaviours this checkpoint needs from shared code that stock ComfyUI lacks.

Both are already fixed on the port's own branch (`fix/moe-expert-quantized-views`); this module
vendors the same fixes so the package runs against a stock checkout. It is a no-op wherever the fixes
are present, so it removes itself as they land upstream.

Why a shim at all: `MoEExperts._expert_qt_from` rebuilds a per-expert quantization params object from
a hand-written kwarg list, so a field the layout gained later keeps its default in the view while the
data does not. For int8 ConvRot that means dequantize and matmul run on rotated data as if it were
unrotated (relative error around 1.4, no error and no warning); for W4A8 it drops `s_channel`,
`correction`, `codebook`, `group_size` and `transposed`. The bank load path needs the matching
`_bank_params_class` because a stacked bank carries an extra leading dimension that a matrix-shaped
layout refuses to validate.

`comfy.ops.mixed_precision_ops` builds the expert class at call time, so its methods cannot be patched
at module scope — the wrapper below patches the class it returns.
"""

import comfy.ops

if not hasattr(comfy.ops, "_bank_param"):

    _BANK_PARAMS_CLASSES = {}
    _BANK_PARAMS_BASE = {}

    def _bank_params_class(layout_cls):
        """A params class that skips the matrix-shape validation, for stacked bank tensors.

        A bank holds one matrix per expert, so every tensor field carries an extra leading dimension that a
        matrix-shaped layout cannot validate (W4A8 checks `scale == (n, k // group_size)`; the other layouts
        simply do not validate, which is why stacked params are the established representation here). It is a
        *subclass* rather than a one-off construction because `to_device`/`clone`/`copy_from` rebuild through
        `type(self)` and would re-raise on the expert dimension.

        The per-expert view that `_expert_qt_from` builds uses the layout's own `Params`, so an expert's
        parameter shapes are still validated — that is where a malformed checkpoint fails loudly.
        """
        if layout_cls not in _BANK_PARAMS_CLASSES:
            _BANK_PARAMS_CLASSES[layout_cls] = type(
                f"{layout_cls.__name__}BankParams",
                (layout_cls.Params,),
                {"_validate_tensor_fields": lambda self: None},
            )
            _BANK_PARAMS_BASE[_BANK_PARAMS_CLASSES[layout_cls]] = layout_cls.Params
        return _BANK_PARAMS_CLASSES[layout_cls]

    def _bank_param(tensor, index, per_expert_dim):
        """Index one expert out of a banked quantization parameter.

        A bank stacks one tensor per expert, so the per-expert form has exactly one more dimension than a
        shared one: `s_channel` is [E, n] per-expert or [n] shared, `correction` [E, groups, n] or
        [groups, n], `codebook` [E, 16] or one [16] table. Selecting on the expected rank rather than on the
        leading size avoids mistaking a shared table with exactly `num_experts` entries for a per-expert one.
        """
        if tensor is None or tensor.dim() != per_expert_dim:
            return tensor
        return tensor[index]

    def _load_quantized_module(module, super_load, state_dict, prefix, local_metadata, strict,
                                missing_keys, unexpected_keys, error_msgs, load_extra_params=False):
        """Shared _load_from_state_dict body for quantized-weight modules.

        Pops weight (+ scales, +/- extras), populates module.weight as a Parameter
        or Parameter-wrapped QuantizedTensor, then calls super_load and strips
        consumed keys from missing_keys. Reads compute_dtype from factory_kwargs
        and disabled formats from module._disabled_formats.
        """
        device = module.factory_kwargs["device"]
        compute_dtype = module.factory_kwargs["dtype"]
        disabled_formats = module._disabled_formats
        layer_name = prefix.rstrip('.')

        weight = state_dict.pop(f"{prefix}weight", None)
        if weight is None:
            logging.warning(f"Missing weight for layer {layer_name}")
            module.weight = None
            return
        manually_loaded_keys = [f"{prefix}weight"]

        def pop_scale(name, dtype=None):
            key = f"{prefix}{name}"
            v = state_dict.pop(key, None)
            if v is not None:
                v = v.to(device=device)
                if dtype is not None:
                    v = v.view(dtype=dtype)
                manually_loaded_keys.append(key)
            return v

        layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
        if layer_conf is not None:
            layer_conf = json.loads(layer_conf.numpy().tobytes())

        if layer_conf is None:
            module.weight = torch.nn.Parameter(weight.to(device=device, dtype=compute_dtype), requires_grad=False)
        else:
            module.quant_format = layer_conf.get("format", None)
            module._full_precision_mm_config = layer_conf.get("full_precision_matrix_mult", False)
            if not module._full_precision_mm:
                module._full_precision_mm = module._full_precision_mm_config
            if module.quant_format in disabled_formats:
                module._full_precision_mm = True
            if module.quant_format is None:
                raise ValueError(f"Unknown quantization format for layer {layer_name}")

            qconfig = QUANT_ALGOS[module.quant_format]
            module.layout_type = qconfig["comfy_tensor_layout"]
            layout_cls = get_layout_class(module.layout_type)

            # Per-format scales; fp8 dtype views handle both legacy uint8-on-disk and native fp8.
            if module.quant_format in ("float8_e4m3fn", "float8_e5m2"):
                scales = {"scale": pop_scale("weight_scale")}
            elif module.quant_format == "mxfp8":
                bs = pop_scale("weight_scale", torch.float8_e8m0fnu)
                if bs is None:
                    raise ValueError(f"Missing MXFP8 block scales for layer {layer_name}")
                scales = {"scale": bs}
            elif module.quant_format == "nvfp4":
                ts = pop_scale("weight_scale_2")
                bs = pop_scale("weight_scale", torch.float8_e4m3fn)
                if ts is None or bs is None:
                    raise ValueError(f"Missing NVFP4 scales for layer {layer_name}")
                scales = {"scale": ts, "block_scale": bs}
            elif module.quant_format == "int8_tensorwise":
                scale = pop_scale("weight_scale")
                if scale is None:
                    raise ValueError(f"Missing INT8 weight scale for layer {layer_name}")
                scales = {"scale": scale}
                params_conf = layer_conf.get("params", {})
                if not isinstance(params_conf, dict):
                    params_conf = {}
                if layer_conf.get("convrot", params_conf.get("convrot", False)):
                    scales["convrot"] = True
                    scales["convrot_groupsize"] = int(
                        layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))
                    )
            elif module.quant_format == "convrot_w4a4":
                scale = pop_scale("weight_scale")
                if scale is None:
                    raise ValueError(f"Missing ConvRot W4A4 weight scale for layer {layer_name}")
                params_conf = layer_conf.get("params", {})
                if not isinstance(params_conf, dict):
                    params_conf = {}
                scales = {
                    "scale": scale,
                    "convrot_groupsize": int(
                        layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))
                    ),
                    "quant_group_size": 64,
                    "linear_dtype": layer_conf.get("linear_dtype", params_conf.get("linear_dtype", "int4")),
                }
            elif module.quant_format == "asym_w4a8_int8":
                # int4 weight (packed int8 [N,K/2]) + fp8 per-group scale (weight_s_rel),
                # fp32 per-channel scale (weight_s_channel) + optional Lloyd-Max codebook.
                scale = pop_scale("weight_s_rel")
                if scale is None:
                    raise ValueError(f"Missing W4A8 group scale (weight_s_rel) for layer {layer_name}")
                if scale.dtype == torch.uint8:
                    scale = scale.view(torch.float8_e4m3fn)
                params_conf = layer_conf.get("params", {})
                if not isinstance(params_conf, dict):
                    params_conf = {}
                scales = {
                    "scale": scale,
                    "s_channel": pop_scale("weight_s_channel"),
                    "codebook": pop_scale("weight_codebook"),
                    "correction": pop_scale("weight_correction"),
                    "group_size": int(layer_conf.get("group_size", params_conf.get("group_size", 16))),
                    "convrot_groupsize": int(
                        layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))
                    ),
                }
            else:
                raise ValueError(f"Unsupported quantization format: {module.quant_format}")

            params_cls = _bank_params_class(layout_cls) if len(module._orig_shape) == 3 else layout_cls.Params
            params = params_cls(**scales, orig_dtype=compute_dtype, orig_shape=module._orig_shape)
            module.weight = torch.nn.Parameter(
                QuantizedTensor(weight.to(device=device, dtype=qconfig["storage_t"]), module.layout_type, params),
                requires_grad=False,
            )

            if load_extra_params:
                for param_name in qconfig["parameters"]:
                    if param_name in {"weight_scale", "weight_scale_2"}:
                        continue
                    param_key = f"{prefix}{param_name}"
                    _v = state_dict.pop(param_key, None)
                    if _v is None:
                        continue
                    module.register_parameter(param_name, torch.nn.Parameter(_v.to(device=device), requires_grad=False))
                    manually_loaded_keys.append(param_key)

        super_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        for key in manually_loaded_keys:
            if key in missing_keys:
                missing_keys.remove(key)

    _ORIGINAL_MIXED_PRECISION_OPS = comfy.ops.mixed_precision_ops

    # the vendored _load_quantized_module resolves these as module globals, so they must exist in
    # comfy.ops too; the patched expert view resolves them in this module
    for _name, _value in (("_bank_params_class", _bank_params_class),
                          ("_bank_param", _bank_param),
                          ("_load_quantized_module", _load_quantized_module)):
        setattr(comfy.ops, _name, _value)
    comfy.ops._BANK_PARAMS_BASE = _BANK_PARAMS_BASE

    def _patched_expert_qt_from(self, weight, i):
        """Build a per-expert QuantizedTensor by indexing into a resident bank."""
        params = weight._params
        kwargs = {
            "scale": params.scale[i] if params.scale.dim() else params.scale,
            "orig_dtype": params.orig_dtype,
            "orig_shape": (self.out_features, self.in_features),
        }
        if hasattr(params, "block_scale"): # NVFP4
            kwargs["block_scale"] = params.block_scale[i]
        if hasattr(params, "quant_group_size"):
            kwargs["quant_group_size"] = params.quant_group_size
        if hasattr(params, "convrot"):
            kwargs["convrot"] = params.convrot
        if hasattr(params, "convrot_groupsize"):
            kwargs["convrot_groupsize"] = params.convrot_groupsize
        if hasattr(params, "linear_dtype"):
            kwargs["linear_dtype"] = params.linear_dtype
        if hasattr(params, "s_channel"): # W4A8, per-expert [E, n]
            kwargs["s_channel"] = _bank_param(params.s_channel, i, 2)
        if hasattr(params, "correction"): # W4A8, per-expert [E, groups, n]
            kwargs["correction"] = _bank_param(params.correction, i, 3)
        if hasattr(params, "codebook"): # W4A8, per-expert [E, 16] or one shared table [16]
            kwargs["codebook"] = _bank_param(params.codebook, i, 2)
        if hasattr(params, "group_size"):
            kwargs["group_size"] = params.group_size
        if hasattr(params, "transposed"):
            kwargs["transposed"] = params.transposed
        # the layout's own Params, not the bank's subclass: a bank skips the matrix validation,
        # and an expert view *must* be validated so a malformed file fails on access. `_layout_cls`
        # is a layout *name*, so the base class comes from the map built alongside the subclass.
        params_cls = _BANK_PARAMS_BASE.get(type(params), type(params))
        return QuantizedTensor(weight._qdata[i], weight._layout_cls, params_cls(**kwargs))

    # A vendored function keeps the globals of the module it is written into, so every core-module
    # name the copied bodies reach for (json, logging, torch, QuantizedTensor, ...) has to exist here
    # too — mirroring them is more reliable than discovering one missing name per run.
    _NEEDED_GLOBALS = set()
    for _fn in (_load_quantized_module, _bank_params_class, _bank_param, _patched_expert_qt_from):
        _NEEDED_GLOBALS |= set(_fn.__code__.co_names)
    globals().update({_name: comfy.ops.__dict__[_name] for _name in _NEEDED_GLOBALS
                      if _name in comfy.ops.__dict__})
    del _NEEDED_GLOBALS, _name, _fn

    def _mixed_precision_ops(quant_config={}, *args, **kwargs):
        operations = _ORIGINAL_MIXED_PRECISION_OPS(quant_config, *args, **kwargs)
        expert_cls = getattr(operations, "MoEExperts", None)
        if expert_cls is not None:
            expert_cls._expert_qt_from = _patched_expert_qt_from
        return operations

    comfy.ops.mixed_precision_ops = _mixed_precision_ops
