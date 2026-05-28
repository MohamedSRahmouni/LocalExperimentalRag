"""
Script to create/recreate Qdrant collection
Run from project root: python app/scripts/create_collection.py
"""

import sys
import os
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================
script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

print(f"📂 Project root: {project_root}")
print(f"📂 Script dir  : {script_dir}")
print()

# ============================================================
# IMPORTS
# ============================================================
from app.pipeline.vectorstore import LangChainQdrantStore
from app.pipeline.embeddings import LangChainEmbeddingManager
from dotenv import load_dotenv

load_dotenv(project_root / '.env')

print("=" * 70)
print("🔄 QDRANT COLLECTION MANAGER")
print("=" * 70)


# ============================================================
# CONFIG
# ============================================================

def resolve_model_path(model_path: str) -> str:
    p = Path(model_path)
    if p.is_absolute():
        return str(p)

    candidate = project_root / "models" / p.name
    if candidate.exists():
        return str(candidate)

    return str(project_root / model_path)


def get_env_config() -> dict:
    raw_model_path = os.getenv(
        'EMBEDDING_MODEL',
        './models/multilingual-e5-small'
    )

    config = {
        'embedding_model': resolve_model_path(raw_model_path),
        'batch_size': int(os.getenv('EMBEDDING_BATCH_SIZE', 32)),
        'use_gpu': os.getenv('USE_GPU', 'false').lower() == 'true',
        'qdrant_url': os.getenv('QDRANT_URL', 'http://localhost:6333'),
        'qdrant_api_key': os.getenv('QDRANT_API_KEY') or None,
        'collection_name': os.getenv('QDRANT_COLLECTION_NAME', 'rag_documents'),
        'vector_size': int(os.getenv('QDRANT_VECTOR_SIZE', 384)),
        'distance': os.getenv('QDRANT_DISTANCE', 'Cosine'),
    }

    print(f"🧠 Resolved embedding model path: {config['embedding_model']}")
    return config


# ============================================================
# OPERATIONS
# ============================================================

def op_status(vectorstore, config):
    stats = vectorstore.get_stats()
    print("\n📊 Collection Status:")
    print(f"   Name     : {config['collection_name']}")
    print(f"   Points   : {stats.get('document_count', 0)}")
    print(f"   Vec size : {stats.get('vector_size', config['vector_size'])}")
    print(f"   Distance : {stats.get('distance', config['distance'])}")


def op_create(vectorstore, config):
    print(f"\n📦 Creating {config['collection_name']}...")
    if vectorstore._ensure_collection():
        print("✅ Created")
    else:
        print("❌ Creation failed")


def op_recreate(vectorstore, config):
    print("\n⚠️  This will DELETE ALL DATA")
    confirm = input("Type 'yes': ").strip().lower()
    if confirm != "yes":
        print("❌ Cancelled")
        return

    if vectorstore.recreate_collection():
        print("✅ Recreated successfully")
    else:
        print("❌ Recreate failed")


def op_delete(vectorstore, config):
    confirm = input("Type 'yes': ").strip().lower()
    if confirm != "yes":
        print("❌ Cancelled")
        return

    if vectorstore.delete_collection():
        print("🗑️  Deleted")
    else:
        print("❌ Delete failed")


# ============================================================
# MAIN
# ============================================================

def main():
    config = get_env_config()

    print("\n📋 CONFIG:")
    print(f"   Model        : {config['embedding_model']}")
    print(f"   Vector size  : {config['vector_size']}")
    print(f"   Distance     : {config['distance']}")
    print(f"   Batch size   : {config['batch_size']}")
    print(f"   Use GPU      : {config['use_gpu']}")
    print(f"   Qdrant       : {config['qdrant_url']}")
    print(f"   Collection   : {config['collection_name']}")

    print("\n1 Create")
    print("2 Recreate (recommended)")
    print("3 Delete")
    print("4 Status")

    choice = input("\nChoice: ").strip()

    print("\n🔄 Init embeddings...")

    embeddings = LangChainEmbeddingManager(
        model_name=config['embedding_model'],
        batch_size=config['batch_size'],
        use_gpu=config['use_gpu'],
    )

    if not embeddings.is_ready():
        print("❌ Embeddings not ready")
        return

    print(f"✅ Embedding dim: {embeddings.embedding_dim}")

    if embeddings.embedding_dim != config['vector_size']:
        print("\n" + "="*70)
        print("⚠️  DIMENSION MISMATCH DETECTED")
        print("="*70)
        print(f"   Model dimension    : {embeddings.embedding_dim}")
        print(f"   .env VECTOR_SIZE   : {config['vector_size']}")
        print("\n   Fix your .env file:")
        print(f"   QDRANT_VECTOR_SIZE={embeddings.embedding_dim}")
        print("="*70)

        cont = input("\nContinue anyway? [y/N]: ").strip().lower()
        if cont != 'y':
            print("❌ Cancelled")
            return

    print("\n🔄 Connect Qdrant...")

    vectorstore = LangChainQdrantStore(
        url=config['qdrant_url'],
        api_key=config['qdrant_api_key'],
        collection_name=config['collection_name'],
        embeddings=embeddings.embeddings,
        vector_size=config['vector_size'],
        distance=config['distance'],
    )

    if not vectorstore.is_connected():
        print("❌ Qdrant connection failed")
        return

    print("✅ Connected")
    print("=" * 70)

    try:
        if choice == "1":
            op_create(vectorstore, config)
        elif choice == "2":
            op_recreate(vectorstore, config)
        elif choice == "3":
            op_delete(vectorstore, config)
        elif choice == "4":
            op_status(vectorstore, config)
        else:
            print("❌ Invalid choice")

    finally:
        vectorstore.close()
        print("✅ Done")


if __name__ == "__main__":
    main()