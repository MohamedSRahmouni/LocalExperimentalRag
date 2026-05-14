"""
Script to create/recreate Weaviate collection with new dimensions
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
from app.pipeline.vectorstore import LangChainVectorStore
from app.pipeline.embeddings import LangChainEmbeddingManager
from dotenv import load_dotenv

load_dotenv(project_root / '.env')

print("=" * 70)
print("🔄 WEAVIATE COLLECTION MANAGER")
print("=" * 70)


# ============================================================
# CONFIG
# ============================================================

def resolve_model_path(model_path: str) -> str:
    """
    Convert relative model path to absolute project-based path
    """
    p = Path(model_path)

    # already absolute
    if p.is_absolute():
        return str(p)

    # try project root / models / xxx
    candidate = project_root / "models" / p.name
    if candidate.exists():
        return str(candidate)

    # fallback: original relative to project root
    return str(project_root / model_path)


def get_env_config() -> dict:
    """Load configuration from .env"""

    raw_model_path = os.getenv(
        'EMBEDDING_MODEL',
        './models/bge-m3'   # FIXED default
    )

    config = {
        'embedding_model': resolve_model_path(raw_model_path),
        'batch_size': int(os.getenv('EMBEDDING_BATCH_SIZE', 8)),
        'weaviate_url': os.getenv('WEAVIATE_URL'),
        'weaviate_api_key': os.getenv('WEAVIATE_API_KEY'),
        'class_name': os.getenv('WEAVIATE_CLASS_NAME', 'ragdocument'),
        'vector_dims': int(os.getenv('WEAVIATE_VECTOR_DIMS', 1024)),  # safer default for bge-m3
    }

    if not config['weaviate_url']:
        print("❌ WEAVIATE_URL not found in .env")
        sys.exit(1)

    if not config['weaviate_api_key']:
        print("❌ WEAVIATE_API_KEY not found in .env")
        sys.exit(1)

    print(f"🧠 Resolved embedding model path: {config['embedding_model']}")

    return config


# ============================================================
# SCHEMA
# ============================================================

NEW_PROPERTIES = [
    ("chunk_id", "TEXT", "Unique chunk identifier"),
    ("text", "TEXT", "Chunk text content"),
    ("text_length", "INT", "Text length in characters"),
    ("embedding_model", "TEXT", "Embedding model name"),
    ("model_type", "TEXT", "Model type"),
    ("embedded_at", "DATE", "Embedding timestamp"),
    ("indexed_at", "DATE", "Indexing timestamp"),
    ("filename", "TEXT", "Source filename"),
    ("file_type", "TEXT", "File extension"),
    ("chunk_index", "INT", "Chunk position in document"),
    ("total_chunks", "INT", "Total chunks in document"),
    ("similarity_score", "NUMBER", "Similarity score"),
    ("sentence_count", "INT", "Sentence count"),

    # TABLE SUPPORT
    ("type", "TEXT", "Chunk type: text | table"),
    ("is_table", "BOOL", "True when chunk is a table"),
    ("table_index", "INT", "Table index"),
    ("page_no", "INT", "Page number"),
    ("caption", "TEXT", "Table caption"),
    ("row_count", "INT", "Rows"),
    ("col_count", "INT", "Columns"),
]


def print_schema_preview():
    print("\n📋 NEW SCHEMA PROPERTIES:")
    print("   " + "-" * 50)

    new_fields = {
        "type", "is_table", "table_index",
        "page_no", "caption", "row_count", "col_count",
    }

    for name, dtype, desc in NEW_PROPERTIES:
        tag = " ← NEW 🆕" if name in new_fields else ""
        print(f"   {name:<20} {dtype:<8} {desc}{tag}")

    print("   " + "-" * 50)
    print(f"   Total properties: {len(NEW_PROPERTIES)}")
    print(f"   New table fields: {len(new_fields)}")


# ============================================================
# OPERATIONS
# ============================================================

def op_status(vectorstore, config):
    exists = vectorstore.weaviate_client.collections.exists(config['class_name'])

    print("\n📊 Collection Status:")
    print(f"   Name   : {config['class_name']}")
    print(f"   Exists : {exists}")

    if exists:
        stats = vectorstore.get_stats()
        print(f"   Chunks : {stats.get('document_count', 0)}")
        print(f"   Dims   : {config['vector_dims']}")


def op_create(vectorstore, config):
    if vectorstore.weaviate_client.collections.exists(config['class_name']):
        print("⚠️ Already exists. Use recreate.")
        return

    print(f"📦 Creating {config['class_name']}...")
    vectorstore._ensure_collection()
    print("✅ Created")


def op_recreate(vectorstore, config):
    print_schema_preview()

    print("\n⚠️ This will DELETE ALL DATA")
    confirm = input("Type 'yes': ").strip().lower()
    if confirm != "yes":
        print("❌ Cancelled")
        return

    if vectorstore.weaviate_client.collections.exists(config['class_name']):
        vectorstore.delete_collection()
        print("🗑️ Deleted old collection")

    vectorstore._ensure_collection()
    print("✅ Recreated successfully")


def op_delete(vectorstore, config):
    if not vectorstore.weaviate_client.collections.exists(config['class_name']):
        print("ℹ️ Does not exist")
        return

    confirm = input("Type 'yes': ").strip().lower()
    if confirm != "yes":
        print("❌ Cancelled")
        return

    vectorstore.delete_collection()
    print("🗑️ Deleted")


# ============================================================
# MAIN
# ============================================================

def main():
    config = get_env_config()

    print("\n📋 CONFIG:")
    print(f"   Model        : {config['embedding_model']}")
    print(f"   Vector dims  : {config['vector_dims']}")
    print(f"   Weaviate     : {config['weaviate_url']}")
    print(f"   Class        : {config['class_name']}")

    print("\n1 Create")
    print("2 Recreate (recommended)")
    print("3 Delete")
    print("4 Status")

    choice = input("\nChoice: ").strip()

    print("\n🔄 Init embeddings...")

    embeddings = LangChainEmbeddingManager(
        model_name=config['embedding_model'],
        batch_size=config['batch_size'],
    )

    if not embeddings.is_ready():
        print("❌ Embeddings not ready")
        return

    print(f"✅ Embedding dim: {embeddings.embedding_dim}")

    print("\n🔄 Connect Weaviate...")

    vectorstore = LangChainVectorStore(
        url=config['weaviate_url'],
        api_key=config['weaviate_api_key'],
        class_name=config['class_name'],
        embeddings=embeddings.embeddings,
        vector_dims=config['vector_dims'],
    )

    if not vectorstore.is_connected():
        print("❌ Weaviate connection failed")
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