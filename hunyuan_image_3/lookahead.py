"""Layer lookahead for the diffusion step: fault the next layers in while the current one computes.

Core's prefetch queue (`comfy.model_prefetch`) batches a block's weights into one fault, but faults
the block it is *about to run* and makes compute wait for it, so no transfer overlaps any compute.
That costs a model which fits in VRAM nothing. This one does not fit: an image step streams ~36 GiB of
expert banks through a 24 GiB card, and every layer's transfer sat in series with its own compute.

How a streamed weight arrives, and why the lookahead has streams of its own
---------------------------------------------------------------------------
A weight outside the card's resident budget is copied by `cast_modules_with_vbar` into the *offload
stream's cast buffer*: one scratch buffer per stream, rewritten from offset 0 by the next cast on that
stream (816 of 896 faults in an image step take this path). Faulting a layer early therefore means
holding its weights in a scratch buffer until it runs — and on the core's two round-robin streams,
any other cast in between rewrites it. Measured: with the CoT's prefill casting its expert banks
through the core mid-layer, the lookahead's staged weights were overwritten and the tokens came out as
garbage, while the core's stream counter — two calls on two streams — ended where it began.

So the lookahead faults onto a private pool of streams, one per layer in flight, which the core never
hands out and whose cast buffers nothing else writes. The core's own casts keep the core's streams.
Each pool stream waits for all compute enqueued so far before its next fault, so a buffer is only
rewritten after the layer that read it has run.

Cost: one cast buffer per pool stream, about one layer's streamed weights (~1.2 GiB) each, held for
the life of the process like the core's own. Set HUNYUAN_IMAGE_3_NO_LOOKAHEAD=1 to disable it.
"""
import contextlib
import os

import torch

import comfy.memory_management
import comfy.model_management
import comfy.model_prefetch
import comfy.ops

DISABLED = bool(os.environ.get("HUNYUAN_IMAGE_3_NO_LOOKAHEAD"))
# Layers faulted ahead of the one computing. Measured at 1024^2 on a 4090, ms per step: depth 0 4084,
# 1 2889, 2 2894, 3 2972, with staging buffers of 0 / 2.6 / 3.9 / 5.2 GiB. One layer ahead already
# keeps the link busy; deeper only costs VRAM.
DEPTH = 1
_POOLS = {}        # device -> private offload streams


def _pool(device, size):
    pool = _POOLS.setdefault(device, [])
    while len(pool) < size:
        stream = torch.cuda.Stream(device=device, priority=0)
        stream.as_context = torch.cuda.stream       # what `cast_to_gathered` expects of an offload stream
        pool.append(stream)
    return pool


@contextlib.contextmanager
def _offload_stream(stream):
    """Hand `stream` to the core cast issued inside, instead of the core's round-robin.

    `cast_modules_with_vbar` asks `comfy.model_management.get_offload_stream` for its stream at call
    time; this swaps that lookup for the duration of one synchronous call on this thread.
    """
    real = comfy.model_management.get_offload_stream
    comfy.model_management.get_offload_stream = lambda device: stream
    try:
        yield
    finally:
        comfy.model_management.get_offload_stream = real


class LayerLookahead:
    def __init__(self, layers, device, enabled, depth=None):
        self.layers = layers
        self.device = device
        self.depth = DEPTH if depth is None else depth
        self.pending = {}                 # layer index -> (stream, modules)
        self.enabled = bool(
            enabled and not DISABLED and self.depth > 0
            and comfy.model_management.NUM_STREAMS > 0          # the core streams asynchronously at all
            and comfy.model_management.is_device_cuda(device)
            and comfy.model_management.device_supports_non_blocking(device))
        self.streams = _pool(device, self.depth + 1) if self.enabled else []

    def _issue(self, index):
        if index >= len(self.layers) or index in self.pending:
            return
        modules = [m for m in self.layers[index].modules() if hasattr(m, "_v")]
        if not modules:
            self.pending[index] = (None, [])
            return
        stream = self.streams[index % len(self.streams)]
        # nothing in this stream's buffer is overwritten before the compute that read it has run
        stream.wait_stream(comfy.model_management.current_stream(self.device))
        with _offload_stream(stream):
            comfy.ops.cast_modules_with_vbar(modules, None, self.device, None, True)
        # as `model_prefetch.pin_modules` does: keep the pinned-registration budget honest
        if not modules[0]._pin_state["fast_disk"]:
            size = sum(comfy.memory_management.vram_aligned_size([m.weight, m.bias]) for m in modules)
            comfy.model_management.ensure_pin_registerable(size)
        self.pending[index] = (stream, modules)

    def before(self, index):
        if not self.enabled:
            return
        for ahead in range(self.depth + 1):
            self._issue(index + ahead)     # index itself is already in flight, except for layer 0
        stream, _ = self.pending[index]
        comfy.model_management.sync_stream(self.device, stream)

    def after(self, index):
        if not self.enabled:
            return
        self._release(index)

    def _release(self, index):
        stream, modules = self.pending.pop(index)
        if stream is not None:
            stream.wait_stream(comfy.model_management.current_stream(self.device))
        comfy.model_prefetch.cleanup_prefetched_modules(self.layers[index], modules)

    def abort(self):
        """Release whatever is still faulted, e.g. when a layer raised (an interrupted prompt)."""
        for index in list(self.pending):
            self._release(index)
