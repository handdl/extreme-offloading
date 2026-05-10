# Extreme Offloading

Training a 7B model in 7GB VRAM. LoRA adapters live on GPU permanently; base weights stream from disk one layer at a time with prefetch. The code is ~200 lines. Most of the work was profiling and debugging async correctness.

## Results

| Scenario | Environment | dtype | Baseline | + Pinning | + Prefetch | Notes |
|----------|------------|-------|----------|-----------|------------|-------|
| Compute-bound | Kaggle T4 | bf16 | 55.0s | 50.8s | 49.5s | No tensor cores, fp32 emulation |
| Memory-bound | Kaggle T4 | fp16 | 24.3s | 21.7s | 21.9s | Prefetch *hurts* - sync overhead |
| Disk-bound | Colab | bf16 | — | 1m 38.7s | 1m 33.5s | Slower disk, no OS cache benefit unlike Kaggle |
| Disk + cast | Colab | fp16 | — | 1m 45.8s | 1m 40.0s | Slower than bf16 - see below |

**Key observations:**
- High GPU utilization != high performance. 
- Extreme bottlenecks resist micro-optimizations.
- The OS is part of your system (and you may not like it)
- Whether pinning helps depends on the hardware.
- One must be careful with custom CUDA streams and the caching allocator.
- A bug that's invisible at one speed becomes catastrophic at another.

**Some of the questions I investigated:**
- How to maximize memory transfer? Should I always pin?
- Why might prefetch be slower than no prefetch?
- Why might fp16 end up slower than bf16 despite using tensor cores?
- Why does training loss diverge only in bf16, and only with prefetch enabled?

## Building Blocks

### Safetensors

The whole approach requires loading individual layers from disk. With pickle-based checkpoints you'd typically have to load the entire file to extract one tensor, or pre-split into per-layer files. Safetensors gives random access out of the box:

```python
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    tensor = f.get_tensor("model.layers.15.self_attn.q_proj.weight")
```

### Meta device

We need the model's architecture without allocating memory. `meta` device creates tensors that know their shape and dtype but occupy zero bytes:

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
```

Later, `load_state_dict(..., assign=True)` replaces meta parameters with real tensors. Without `assign=True` PyTorch tries to copy into existing storage — which doesn't exist on meta.

### LoRA stays on GPU

LoRA adapters are a few MB vs multiple GB of base weights. They stay materialized on GPU for the entire run — no offloading. This is the key asymmetry: gradients flow only through LoRA, base weights are read-only context that we stream in and out!

## The Offload Cycle

Each layer wrapper is linked into a chain and triggers prefetch of the next one (or previous, in backward) - **one layer ahead only**, to keep memory usage minimal.

## Optimizations

### Pinned memory and overlapping transfers

`tensor.to(device)` from pageable CPU memory forces CUDA to internally copy to a pinned staging buffer first, then DMA to GPU. Two copies, no overlap possible.

Whether pinning helps depends on the situation. If the GPU is ready and waiting for data right now, the overhead of pinning first might not pay off — you'd need to profile. On our Kaggle T4 setup, pin + transfer was consistently cheaper than direct pageable transfer, so we always pin: 

```bash
   Size MB   Pageable GB/s     Pinned GB/s     Pin+To GB/s  P vs Page  P+T vs Page
--------------------------------------------------------------------------------
         1            4.03           11.65            6.17      2.89x       1.53x
        10            6.39           12.30            7.62      1.92x       1.19x
        50            6.86           12.36            6.58      1.80x       0.96x
       100            6.82           12.38            7.23      1.82x       1.06x
       500            6.85           12.38            7.18      1.81x       1.05x
      1000            7.02           12.38            7.30      1.76x       1.04x
      2000            7.02           12.38            7.20      1.76x       1.03x
```

On Colab it was actually slower in some configurations — so the tradeoff isn't universal.

The real win is overlapping pinning with transfers across tensors. Not "pin everything, then send everything" — pin tensor N, start its transfer with `non_blocking=True`, immediately pin tensor N+1 while the GPU is still pulling N:

```python
cache = {k: v.pin_memory().to(device, non_blocking=True) for k, v in cache.items()}
torch.cuda.synchronize()
```

`pin_memory()` blocks CPU but `.to(non_blocking=True)` returns immediately, so while GPU pulls tensor N across PCIe, CPU is already pinning N+1. On our setup this roughly halved transfer time compared to the naive sequential approach.

### Prefetch in a separate CUDA stream

Pinning helps but loading is still synchronous — the GPU waits for weights before computing. Prefetching loads the next layer while the current one computes, using a separate CUDA stream so the transfer doesn't block compute. The pinning work runs in a separate CPU thread since `pin_memory()` is blocking.

```python
with torch.cuda.stream(self._stream):
    self._cache = {k: v.pin_memory().to(self.device, non_blocking=True) for k, v in self._cache.items()}
```

---

### Debugging: the `record_stream` story

This cost four hours and is worth telling in detail.

When you allocate a tensor in CUDA stream B and use it in stream A, PyTorch's caching allocator **only knows about stream B**. Once stream B moves on, the allocator considers that memory free. If stream A is still reading it — silent data corruption.

How did I find it? My training loss was noticeably different, but only in bf16, and only with prefetch. And what is more dangerous - values were plausible-looking because of bf16's lower precision and its emulation on T4! Yet I didn't believe in this explanation and decided to go deeper.

**Root cause:** prefetch stream allocated weight tensors -> main compute stream started using them while allocater used it for something else.

Why bf16 specifically? On T4 bf16 compute is slow enough that the prefetch stream completes and its memory gets reclaimed *while* the compute stream is still reading it. In fp16, tensor cores are so fast that the weights get consumed before the allocator reclaims anything — same buggy code, no visible corruption. A timing-dependent correctness bug that only manifests under specific compute/memory ratios.

**Fix:**

```python
current = torch.cuda.current_stream(self.device)
for v in self._cache.values():
    v.record_stream(current)
```

Tell the allocator these tensors are alive in the compute stream too. **Lesson:** a bug that's invisible at one speed becomes catastrophic at another.

---

## Scenario Analysis

### Compute-bound (Kaggle T4, bf16, pin vs no pin).

T4 has no native bf16 tensor cores — all bf16 ops emulate through fp32 on CUDA cores. Compute dominates, so there's room to hide transfers behind it.

Pinning shaves ~4s by speeding up the transfer itself. Prefetch hides most of the remaining transfer time behind compute. But the floor is ~49.5s of pure compute — no offloading trick gets past that. The only way forward is fp16 + tensor cores.

### Utilization != performance (Kaggle T4, bf16 vs fp16)

bf16, 49.5s — GPU is fully utilized, but it's CUDA cores emulating fp32:

![bf16 full utilization](traces/bf16_full_utilization_but_long.jpg)

fp16, 21.7s — GPU looks idle most of the time, but tensor cores finish the actual work 2.5x faster:

![fp16 low utilization but fast](traces/fp16_no_prefetch_but_fast.jpg)

### Prefetch != faster (Kaggle T4, fp16, prefetch vs no prefetch).

Tensor cores make compute near-instant. The bottleneck flips to PCIe bandwidth. Pinning helps (~24.3s -> 21.7s) by making transfers faster. But prefetch actually makes it slightly *worse* (21.9s). Why? Probably synchronization overhead (stream sync, event queries) costs more than the ~6ms of **dispatch time** (not compute!) it saves. When compute is that fast, there's no window to hide anything in.

Without prefetch, 21.7s:

![fp16 no prefetch](traces/fp16_no_prefetch_but_fast.jpg)

With prefetch, 21.9s — extra streams and sync overhead visible at the bottom:

![fp16 with prefetch](traces/fp16_prefetch_but_longer.jpg)

**The 17-second anomaly:** some runs without prefetch clocked ~17s — faster than any optimized configuration. My best explanation: without concurrent CUDA dispatch and background threads, the OS sees a cleaner sequential read pattern on the safetensors file and its page cache prefetches more aggressively. Adding concurrency fragments the access pattern and increases syscall overhead. I'm not fully certain — it wasn't consistent enough to isolate definitively, but the correlation with simpler execution patterns was clear.

### Disk-bound (Colab, bf16, prefetch vs no prefetch)

On Colab disk read dominates almost all the time. But it was interesting to find that the backward pass (on bf16!) benefits *more* — it has *maximum* compute per layer, so prefetch can hide pin+transfer behind compute (~5s improvement).

### Tensor cores != faster (Colab, bf16 vs fp16)


bf16, 1m 33.5s:
![colab bf16](traces/bf16_but_faster.jpg)

fp16, 1m 45.8s — 12 seconds slower despite tensor cores:

![colab fp16](traces/fp16_but_long.jpg)


So, fp16 is 7 seconds slower than bf16 despite faster compute. **Wow!** This is the result I'm least certain about, but here's what the profiler traces suggest.

The safetensors checkpoint stores weights in bf16. Loading as bf16 is essentially a no-op — data goes straight from mmap into the pinning buffer. Loading as fp16 triggers an actual dtype conversion on CPU: new allocation, element-wise copy.

**bf16:** load time is mostly `pin_memory()`, typically ~1.8s per layer. But a significant number of layers pin in ~0.5s — frequent enough to add up to a ~7s advantage over the full model.

**fp16:** the dtype cast to fp16 dominates, also landing around ~1.8s total per layer. Pinning after the cast is near-instant (~20ms) — the freshly allocated memory is trivial to pin. But the fast ~0.5s loads that bf16 benefits from almost never occur in fp16.

bf16 load — time spent in `pin_memory()`, with a fast 24ms pin visible:

![bf16 load structure](traces/bf16_long_cast.jpg)

fp16 load — time spent in cast, followed by `cudaStreamSynchronize`:

![fp16 load structure](traces/fp16_fast_cast_and_pin.jpg)

Why bf16 occasionally pins so quickly — whether it's reusing previously pinned memory, better OS page cache behavior, or something else — *I didn't establish*. The cast in fp16 likely prevents whatever mechanism is responsible, possibly by forcing a fresh allocation every time.
