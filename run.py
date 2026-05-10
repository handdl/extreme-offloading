"""
Usage:
    python run.py                           # fp16, no prefetch
    python run.py --prefetch                # fp16, prefetch
    python run.py --dtype bf16              # bf16, no prefetch
    python run.py --dtype bf16 --prefetch   # bf16, prefetch
"""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile, schedule

parser = argparse.ArgumentParser()
parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
parser.add_argument("--prefetch", action="store_true")
args = parser.parse_args()

from utils import limit_gpu_mem, mem_report
import config as C

limit_gpu_mem(C.DEVICE, C.MAX_MEMORY_GIB)

from huggingface_hub import login, snapshot_download

login(token="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")
ckpt_dir = snapshot_download(C.MODEL_ID, allow_patterns=["*.safetensors", "*.json", "*.model"])

from model_qwen import build_model

dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
prefetch = args.prefetch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
peft_model, tokenizer, model_config = build_model(ckpt_dir, dtype, prefetch)
mem_report("after build")

input_ids = torch.randint(0, tokenizer.vocab_size, (1, C.SEQ_LEN), device=C.DEVICE)
print(f"hash-check on inputs: {input_ids.sum()}")

batch = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
peft_model.enable_adapter_layers()
peft_model.eval()
peft_model.enable_input_require_grads()
optimizer = torch.optim.Adam([p for p in peft_model.parameters() if p.requires_grad], lr=C.LR)

tag = f"{C.MODEL_ID.split('/')[-1]}_{args.dtype}_pref{'ON' if args.prefetch else 'OFF'}"

losses = []
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=0, warmup=1, active=2, repeat=1),
    with_stack=True,
    acc_events=True,
) as prof:
    for step in range(3):
        optimizer.zero_grad()
        loss = peft_model(**batch, labels=batch["input_ids"], use_cache=False).loss
        loss.backward()
        optimizer.step()
        prof.step()
        losses.append(loss.item())
print(f"losses {losses}")

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
trace_path = f"{tag}_trace.json"
prof.export_chrome_trace(trace_path)
print(f"Trace: {trace_path}")

peak = torch.cuda.max_memory_allocated() / 1e9
lora_mb = sum(p.numel() * p.element_size() for p in peft_model.parameters() if p.requires_grad) / 1e6

assert peak < C.MAX_MEMORY_GIB, f"Peak {peak:.2f}G exceeds {C.MAX_MEMORY_GIB}G"

logits_weight = (model_config.vocab_size * C.SEQ_LEN * 4) / (10**9)
print(f"Model:      {C.MODEL_ID}")
print(f"GPU peak:   {peak:.2f} GiB (limit {C.MAX_MEMORY_GIB}) [only logits weight {logits_weight:0.2f} GB]")
print(f"LoRA:       {lora_mb:.1f} MB")
