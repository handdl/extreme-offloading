import torch
import time

sizes_mb = [1, 10, 50, 100, 500, 1000, 2000]
warmup = 5
runs = 20
device = torch.device("cuda:0")

print(
    f"{'Size MB':>10} {'Pageable GB/s':>15} {'Pinned GB/s':>15} {'Pin+To GB/s':>15} {'P vs Page':>10} {'P+T vs Page':>12}"
)
print("-" * 80)


for mb in sizes_mb:
    torch.cuda.empty_cache()
    nbytes = mb * 1024 * 1024
    pageable = torch.empty(nbytes, dtype=torch.uint8)
    pinned = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)

    for _ in range(warmup):
        pageable.to(device)
        pinned.to(device)
        pageable.pin_memory().to(device)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        pageable.to(device)
    torch.cuda.synchronize()
    t_pageable = (time.perf_counter() - t0) / runs

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        pinned.to(device, non_blocking=True)
    torch.cuda.synchronize()
    t_pinned = (time.perf_counter() - t0) / runs

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        pageable.pin_memory().to(device, non_blocking=False)
    torch.cuda.synchronize()
    t_pin_and_to = (time.perf_counter() - t0) / runs

    bw = lambda t: nbytes / t / 1e9
    bw_p, bw_pin, bw_pt = bw(t_pageable), bw(t_pinned), bw(t_pin_and_to)

    print(f"{mb:>10} {bw_p:>15.2f} {bw_pin:>15.2f} {bw_pt:>15.2f} {bw_pin/bw_p:>9.2f}x {bw_pt/bw_p:>10.2f}x")
    del pageable, pinned
