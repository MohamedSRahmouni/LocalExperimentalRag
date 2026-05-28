"""
Script to download embedding models
Run: python download_model.py
"""

import os
import sys
from pathlib import Path

def download_embedder():
    """Download embedding model locally"""
    
    print("="*60)
    print("📦 Downloading Embedding Models")
    print("="*60)
    
    # Model options
    models = {
        "1": {
            "name": "BAAI/bge-m3",
            "local_dir": "./models/bge-m3",
            "size": "~2.2GB",
            "dim": 1024,
            "description": "BGE-M3 - Multilingual ✅ RECOMMENDED",
            "ram": "~4GB (float32), ~2GB (bfloat16)",
            "features": "Multi-lingual, dense+sparse+colbert retrieval"
        },
        "2": {
            "name": "BAAI/bge-large-en-v1.5",
            "local_dir": "./models/bge-large-en-v1.5",
            "size": "~1.3GB",
            "dim": 1024,
            "description": "BGE-Large - English only, high quality",
            "ram": "~3GB (float32), ~1.5GB (bfloat16)",
            "features": "English-optimized, state-of-the-art"
        },
        "3": {
            "name": "BAAI/bge-small-en-v1.5",
            "local_dir": "./models/bge-small-en-v1.5",
            "size": "~130MB",
            "dim": 384,
            "description": "BGE-Small - Fast & lightweight",
            "ram": "~500MB (float32), ~250MB (bfloat16)",
            "features": "Very fast, low RAM, good quality"
        },
        "4": {
            "name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "local_dir": "./models/gte-qwen2-1.5B",
            "size": "~3GB",
            "dim": 1536,
            "description": "Qwen2-1.5B - Alternative option",
            "ram": "~6GB (float32), ~3GB (bfloat16)",
            "features": "High dimension, instruction-following"
        },
        "5": {
            "name": "intfloat/multilingual-e5-small",
            "local_dir": "./models/multilingual-e5-small",
            "size": "~470MB",
            "dim": 384,
            "description": "multilingual-e5-small ✅ LIGHTWEIGHT",
            "ram": "~500MB (float32)",
            "features": (
                "Multilingual, very fast, "
                "query/passage prefixes required"
            )
        }
    }
    
    # Show options
    print("\nAvailable models:")
    print()
    for key, model in models.items():
        print(f"  {key}. {model['description']}")
        print(f"     Repo ID   : {model['name']}")
        print(f"     Download  : {model['size']}")
        print(f"     RAM Usage : {model['ram']}")
        print(f"     Dimension : {model['dim']}")
        print(f"     Features  : {model['features']}")
        print()
    
    print("💡 TIP: BGE-M3 (option 1) is best for multilingual + CPU")
    print()
    
    # Default to BGE-M3
    choice = input("Choose model (1-4) [default: 1]: ").strip() or "1"
    
    if choice not in models:
        print("❌ Invalid choice, using default (1 - BGE-M3)")
        choice = "1"
    
    selected = models[choice]
    
    print()
    print("="*60)
    print("📥 DOWNLOAD SETTINGS")
    print("="*60)
    print(f"✅ Model        : {selected['name']}")
    print(f"📁 Local path   : {selected['local_dir']}")
    print(f"📦 Download size: {selected['size']}")
    print(f"💾 RAM usage    : {selected['ram']}")
    print(f"📐 Embedding dim: {selected['dim']}")
    print(f"🎯 Features     : {selected['features']}")
    print("="*60)
    
    # Confirm
    confirm = input("\nProceed with download? [Y/n]: ").strip().lower()
    if confirm and confirm not in ['y', 'yes']:
        print("❌ Download cancelled")
        sys.exit(0)
    
    # Create directory
    Path(selected['local_dir']).mkdir(parents=True, exist_ok=True)
    
    try:
        from huggingface_hub import snapshot_download
        
        print(f"\n⏳ Downloading {selected['size']}...")
        print("   Please wait...\n")
        
        snapshot_download(
            repo_id=selected['name'],
            local_dir=selected['local_dir'],
            local_dir_use_symlinks=False,
            ignore_patterns=[
                "*.msgpack",
                "*.h5",
                "flax_model*",
                "tf_model*",
                "*.ot",
                "rust_model*",
                ".gitattributes",
                "*.onnx",
                "*.onnx_data"
            ],
            resume_download=True
        )
        
        print("\n" + "="*60)
        print("✅ DOWNLOAD COMPLETE!")
        print("="*60)
        print(f"\n📁 Model saved to: {selected['local_dir']}")
        print(f"📐 Vector dimension: {selected['dim']}")
        print(f"💾 Expected RAM: {selected['ram']}")
        
        # Next steps
        print("\n📝 NEXT STEPS:")
        print("="*60)
        print("\n1️⃣  Update .env file:")
        print(f"   EMBEDDING_MODEL={selected['local_dir']}")
        print(f"   WEAVIATE_VECTOR_DIMS={selected['dim']}")
        
        # Batch size recommendations
        if choice == "1":  # BGE-M3
            print(f"   EMBEDDING_BATCH_SIZE=8")
        elif choice == "2":  # BGE-Large
            print(f"   EMBEDDING_BATCH_SIZE=16")
        elif choice == "3":  # BGE-Small
            print(f"   EMBEDDING_BATCH_SIZE=32")
        else:
            print(f"   EMBEDDING_BATCH_SIZE=4")
        
        print("\n2️⃣  Recreate Weaviate collection (dimension changed):")
        print("   python app/scripts/create_collection.py")
        print("   Choose option 2 (Recreate)")
        
        print("\n3️⃣  Restart application:")
        print("   python main.py")
        
        if choice == "1":
            print("\n" + "="*60)
            print("🎯 BGE-M3 Special Features:")
            print("   • Supports 100+ languages")
            print("   • Dense + Sparse + ColBERT retrieval")
            print("   • Max length: 8192 tokens")
            print("   • Best for multilingual RAG")
            print("="*60)
        
        return selected
        
    except ImportError:
        print("\n❌ ERROR: huggingface_hub not installed!")
        print("   pip install huggingface_hub")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("="*60)
        print("  1. Check internet connection")
        print("  2. Verify disk space")
        print("  3. Try with VPN")
        print(f"\n  Manual download:")
        print(f"  https://huggingface.co/{selected['name']}/tree/main")
        print(f"  Save to: {os.path.abspath(selected['local_dir'])}")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    try:
        result = download_embedder()
        print("\n✅ Script completed!")
        print(f"📂 Model ready at: {result['local_dir']}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(1)