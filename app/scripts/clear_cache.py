# clear_cache.py
import shutil
from pathlib import Path

print("🧹 Cleaning HuggingFace cache...")

caches = [
    Path.home() / ".cache" / "huggingface",
    Path("./embeddings_cache"),
    # Nettoie aussi le cache dans ton dossier modèle
    Path("./app/scripts/models/gte-qwen2-1.5B/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct"),
]

for cache in caches:
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
        print(f"   ✅ Cleared: {cache}")
    else:
        print(f"   ℹ️  Not found: {cache}")

print("✅ Done!")