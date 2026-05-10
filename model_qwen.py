"""Qwen2.5-specific model setup.

Pay attention to "<-- architecture-specific"
"""

import json
import os

import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding  # <-- architecture-specific

from config import DEVICE, LORA_R, LORA_TARGETS, MODEL_ID
from offload import OffloadedLayer, link_layers


def _build_weight_index(ckpt_dir):
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        return {k: os.path.join(ckpt_dir, v) for k, v in weight_map.items()}
    shard = os.path.join(ckpt_dir, "model.safetensors")
    assert os.path.exists(shard), f"No safetensors found in {ckpt_dir}"
    with safe_open(shard, framework="pt") as f:
        return {k: shard for k in f.keys()}


def _make_layer_loader(param_names, weight_index, dtype):
    def load_fn():
        tensors = {}
        by_file = {}
        for model_name, ckpt_key in param_names:
            path = weight_index[ckpt_key]
            by_file.setdefault(path, []).append((model_name, ckpt_key))
        for path, keys in by_file.items():
            with safe_open(path, framework="pt", device="cpu") as f:
                for model_name, ckpt_key in keys:
                    tensors[model_name] = f.get_tensor(ckpt_key).to(dtype)
        return tensors

    return load_fn


def build_model(ckpt_dir, dtype, prefetch):
    model_config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config, dtype=dtype)

    peft_model = get_peft_model(model, LoraConfig(r=LORA_R, target_modules=LORA_TARGETS))

    torch.manual_seed(42)
    _materialize_lora(peft_model, DEVICE, dtype)

    weight_index = _build_weight_index(ckpt_dir)
    base = peft_model.model.model  # <-- architecture-specific
    layers = base.layers  # <-- architecture-specific

    stream = torch.cuda.Stream(device=DEVICE) if prefetch else None

    wrappers = []
    for i in range(len(layers)):
        param_names = []
        for name, _ in layers[i].named_parameters():
            if "lora_" in name:
                continue
            clean = name.replace(".base_layer", "")
            ckpt_key = f"model.layers.{i}.{clean}"  # <-- architecture-specific
            assert ckpt_key in weight_index, f"Key {ckpt_key} not in checkpoint"
            param_names.append((name, ckpt_key))

        w = OffloadedLayer(layers[i], _make_layer_loader(param_names, weight_index, dtype), DEVICE, stream)
        wrappers.append(w)

    if prefetch:
        link_layers(wrappers)

    for i, w in enumerate(wrappers):
        w.offload()
        for name, param in w.layer.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(DEVICE)
        layers[i] = w

    _load_non_layer_weights(peft_model, base, weight_index, dtype)
    _reinit_rope(peft_model, model_config, DEVICE)

    lora_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    tied = peft_model.config.tie_word_embeddings
    print(f"Model: {MODEL_ID}")
    print(f"Layers: {len(wrappers)}, LoRA params: {lora_params/1e6:.1f}M, device: {DEVICE}")
    print(f"Weights: {ckpt_dir}, tied embeddings: {tied}")
    print(f"Prefetch: {prefetch}")
    print(f"embed dtype: {base.embed_tokens.weight.dtype}")
    print(f"layer0 first lora dtype: {[p.dtype for n, p in list(peft_model.named_parameters()) if 'lora_' in n][0]}")
    print(f"model config dtype: {model_config.dtype}")

    _verify_model(peft_model, wrappers)
    return peft_model, tokenizer, model_config


def _materialize_lora(peft_model, device, dtype):
    for name, module in peft_model.named_modules():
        if "lora_" not in name:
            continue
        for pname, param in list(module.named_parameters(recurse=False)):
            new = nn.Parameter(
                torch.empty(param.shape, device=device, dtype=torch.float32)
            )  # grad must be in FP32 for scaling?
            if "lora_A" in name:
                nn.init.kaiming_uniform_(new.data)
            else:
                new.data.zero_()
            setattr(module, pname, new)


def _load_non_layer_weights(peft_model, base, weight_index, dtype):
    # <-- architecture-specific keys
    non_layer_keys = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm.weight",
    }
    sd = {}
    for ckpt_key, model_key in non_layer_keys.items():
        assert ckpt_key in weight_index, f"Missing {ckpt_key}"
        with safe_open(weight_index[ckpt_key], framework="pt", device="cpu") as f:
            sd[model_key] = f.get_tensor(ckpt_key).to(dtype)
    base.load_state_dict(sd, strict=False, assign=True)
    base.embed_tokens.weight = nn.Parameter(base.embed_tokens.weight.to(DEVICE), requires_grad=False)
    base.norm.weight = nn.Parameter(base.norm.weight.to(DEVICE), requires_grad=False)

    if peft_model.config.tie_word_embeddings:  # <-- architecture-specific (Qwen2.5-0.5B has tied weights)
        peft_model.model.lm_head.weight = base.embed_tokens.weight
    else:
        lm_key = "lm_head.weight"
        assert lm_key in weight_index, f"Missing {lm_key}"
        with safe_open(weight_index[lm_key], framework="pt", device="cpu") as f:
            lm_weight = f.get_tensor(lm_key).to(dtype)
        peft_model.model.lm_head.weight = nn.Parameter(lm_weight.to(DEVICE), requires_grad=False)
    del sd


def _reinit_rope(peft_model, model_config, device):
    for module in peft_model.model.modules():
        if isinstance(module, Qwen2RotaryEmbedding):  # <-- architecture-specific
            module.__init__(config=model_config, device=device)


def _verify_model(peft_model, wrappers):
    base = peft_model.model.model  # <-- architecture-specific

    for n, p in peft_model.named_parameters():
        if "lora_" in n and p.requires_grad:
            assert p.device == DEVICE, f"LoRA param {n} on {p.device}"

    for i, w in enumerate(wrappers):
        for n, p in w.layer.named_parameters():
            if "lora_" not in n:
                assert p.device == torch.device("meta"), f"Layer {i} {n} on {p.device}"

    assert base.embed_tokens.weight.device == DEVICE, "embed_tokens not on device"
    assert base.norm.weight.device == DEVICE, "norm not on device"
    assert peft_model.model.lm_head.weight.device == DEVICE, "lm_head not on device"

    print("Verification passed")
