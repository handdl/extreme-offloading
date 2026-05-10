"""Layer-wise weight offloading. No CPU fallback logic."""

from threading import Thread

import torch
from torch import nn

LORA_MARKER = "lora_"


class OffloadedCheckpoint(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, wrapper, *args):
        wrapper.load()
        # if we are transfer bound - the earlier we dispatch prefetch the better
        # if we are comppute bound - there's no difference
        wrapper.schedule_prefetch_forward()
        out = wrapper.layer(*args, **wrapper._fwd_kwargs)
        out = out[0] if isinstance(out, tuple) else out
        ctx.save_for_backward(*args)
        ctx.wrapper = wrapper
        wrapper.offload()
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grad_outputs):
        args = ctx.saved_tensors
        wrapper = ctx.wrapper
        wrapper.load()
        # if we are transfer bound - the earlier we dispatch prefetch the better
        # if we are compute bound - there's no difference
        wrapper.schedule_prefetch_backward()
        args = [a.detach().requires_grad_(a.requires_grad) for a in args]
        with torch.enable_grad():
            out = wrapper.layer(*args, **wrapper._fwd_kwargs)
        out = out[0] if isinstance(out, tuple) else out
        torch.autograd.backward(out, grad_outputs)
        wrapper.offload()
        return (None,) + tuple(a.grad for a in args)


class OffloadedLayer(nn.Module):
    def __init__(self, layer, load_fn, device, prefetch_stream):
        super().__init__()
        self.layer = layer
        self.load_fn = load_fn
        self.device = device
        self._stream = prefetch_stream
        self._cache = None
        self._thread = None
        self._offload_event = None
        object.__setattr__(self, "next_layer", None)
        object.__setattr__(self, "prev_layer", None)

    def forward(self, *args, **kwargs):
        self._fwd_kwargs = kwargs
        return OffloadedCheckpoint.apply(self, *args)

    def prefetch(self, wait_event=None):
        if self._cache is not None or self._thread is not None:
            return
        self._thread = Thread(target=self._prefetch_worker, args=(wait_event,), daemon=True)
        self._thread.start()

    def _prefetch_worker(self, wait_event):
        self._cache = self.load_fn()
        # pin while waiting for previous layer's offload to finish
        if wait_event:
            for k, v in self._cache.items():
                self._cache[k] = v.pin_memory()
                if wait_event.query():
                    break
            wait_event.synchronize()
        with torch.cuda.stream(self._stream):
            # with pin_memory and non_blocking next tensor pins while current transfers
            self._cache = {k: v.pin_memory().to(self.device, non_blocking=True) for k, v in self._cache.items()}

    def load(self):
        if self._thread:
            self._thread.join()
            self._thread = None

        if self._cache is None:
            self._cache = self.load_fn()
            # pin + transfer from pinned is faster than direct pageable transfer (on this hardware)
            self._cache = {k: v.pin_memory().to(self.device, non_blocking=True) for k, v in self._cache.items()}
            torch.cuda.current_stream(self.device).synchronize()

        if self._stream is not None:
            self._stream.synchronize()

        self.layer.load_state_dict(self._cache, assign=True, strict=False)

        if self._stream is not None:
            current = torch.cuda.current_stream(self.device)
            # the caching allocator is aware of only the stream where a tensor was allocated
            for v in self._cache.values():
                v.record_stream(current)

        self._cache = None

    def offload(self):
        for name, module in self.layer.named_modules():
            if LORA_MARKER in name:
                continue
            for pname, param in list(module.named_parameters(recurse=False)):
                if LORA_MARKER in pname:
                    continue
                setattr(
                    module,
                    pname,
                    nn.Parameter(
                        torch.empty(param.shape, device="meta", dtype=param.dtype),
                        requires_grad=False,
                    ),
                )
        self._offload_event = torch.cuda.Event()
        self._offload_event.record()

    def schedule_prefetch_forward(self):
        if self.next_layer is not None:
            wait = self.prev_layer._offload_event if self.prev_layer else None
            self.next_layer.prefetch(wait_event=wait)

    def schedule_prefetch_backward(self):
        if self.prev_layer is not None:
            wait = self.next_layer._offload_event if self.next_layer else None
            self.prev_layer.prefetch(wait_event=wait)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)


def link_layers(wrappers):
    for i in range(len(wrappers) - 1):
        object.__setattr__(wrappers[i], "next_layer", wrappers[i + 1])
        object.__setattr__(wrappers[i + 1], "prev_layer", wrappers[i])
