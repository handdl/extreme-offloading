import torch

MODEL_ID = "Qwen/Qwen2.5-7B"  # for quick debug: "Qwen/Qwen2.5-0.5B"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_MEMORY_GIB = 7.0


LORA_R = 8
LORA_TARGETS = "all-linear"

LR = 1e-4
TRAIN_STEPS = 5
SEQ_LEN = 2048
