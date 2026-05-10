import torch
import gc, psutil, os


def limit_gpu_mem(device, limit_gb):
    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    frac = min(1.0, limit_gb / total)
    torch.cuda.set_per_process_memory_fraction(frac, device)
    print(f"GPU memory capped at {limit_gb}G ({frac*100:.0f}% of {total:.1f}G)")


def mem_report(label=""):
    gc.collect()
    torch.cuda.empty_cache()
    process = psutil.Process(os.getpid())
    real_ram = process.memory_info().rss
    virtual_mem = process.memory_info().vms
    print(
        f"[{label}]\n"
        f"RAM: {real_ram/1e9:.1f}G rss & ({virtual_mem / (1024**3):.1f}G vms) | "
        f"GPU: {torch.cuda.memory_allocated()/1e9:.2f}G now & "
        f"{torch.cuda.max_memory_allocated()/1e9:.2f}G peak"
    )