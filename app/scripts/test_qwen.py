# test_qwen.py
import os
import sys
import types
import importlib.util
import traceback
import torch

print("Testing Qwen2 loading...")

# ================================================================
# FIX: Mock flash_attn
# ================================================================
def create_mock_module(name):
    mock = types.ModuleType(name)
    mock.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    mock.__spec__.submodule_search_locations = []
    mock.flash_attn_func = None
    mock.flash_attn_varlen_func = None
    mock.flash_attn_with_kvcache = None
    return mock

for mod_name in [
    "flash_attn",
    "flash_attn.flash_attn_interface",
    "flash_attn.bert_padding",
    "flash_attn.flash_attn_utils"
]:
    sys.modules[mod_name] = create_mock_module(mod_name)

print("✅ flash_attn mock installed")

import transformers
print(f"Transformers version: {transformers.__version__}")

from transformers import AutoTokenizer, AutoModel

model_id = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_fast=True
)
print(f"✅ Tokenizer: {tokenizer.__class__.__name__}")

print("Loading model...")
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    attn_implementation="eager"
    # low_cpu_mem_usage RETIRÉ (nécessite accelerate)
)
model.eval()
print(f"✅ Model: {model.__class__.__name__}")

# Test embedding
print("\nTesting embedding...")
encoded = tokenizer(
    ["Hello world"],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors='pt'
)

with torch.no_grad():
    output = model(**encoded)

dim = output.last_hidden_state.shape[-1]
print(f"✅ Embedding dimension: {dim}")
print("\n🎉 SUCCESS!")