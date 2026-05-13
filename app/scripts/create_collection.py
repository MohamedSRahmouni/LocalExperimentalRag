"""
Script to create/recreate Weaviate collection with new dimensions
Run from project root: python app/scripts/create_collection.py
"""

import sys
import os
from pathlib import Path

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================
# Get script directory
script_dir = Path(__file__).resolve().parent

# Get project root (2 levels up from app/scripts/)
project_root = script_dir.parent.parent

# Add to Python path
sys.path.insert(0, str(project_root))

print(f"📂 Project root: {project_root}")
print(f"📂 Script dir  : {script_dir}")
print()

# ============================================================
# NOW IMPORTS WILL WORK
# ============================================================
from app.pipeline.vectorstore import LangChainVectorStore
from app.pipeline.embeddings import LangChainEmbeddingManager
from dotenv import load_dotenv

# Load .env from project root
env_path = project_root / '.env'
load_dotenv(env_path)

print("="*70)
print("🔄 WEAVIATE COLLECTION MANAGER")
print("="*70)

def get_env_config():
    """Get configuration from .env"""
    config = {
        'embedding_model': os.getenv('EMBEDDING_MODEL', './models/qwen2.5-embedding-0.6B'),
        'batch_size': int(os.getenv('EMBEDDING_BATCH_SIZE', 8)),
        'weaviate_url': os.getenv('WEAVIATE_URL'),
        'weaviate_api_key': os.getenv('WEAVIATE_API_KEY'),
        'class_name': os.getenv('WEAVIATE_CLASS_NAME', 'ragdocument'),
        'vector_dims': int(os.getenv('WEAVIATE_VECTOR_DIMS', 896))
    }
    
    # Validate
    if not config['weaviate_url']:
        print("❌ WEAVIATE_URL not found in .env")
        sys.exit(1)
    if not config['weaviate_api_key']:
        print("❌ WEAVIATE_API_KEY not found in .env")
        sys.exit(1)
    
    return config

def main():
    """Main script"""
    
    # Get config
    config = get_env_config()
    
    print("\n📋 Configuration:")
    print(f"   Embedding Model : {config['embedding_model']}")
    print(f"   Vector Dims     : {config['vector_dims']}")
    print(f"   Weaviate URL    : {config['weaviate_url'][:50]}...")
    print(f"   Collection Name : {config['class_name']}")
    print()
    
    # Ask what to do
    print("What would you like to do?")
    print("  1. Create new collection (if doesn't exist)")
    print("  2. Recreate collection (delete + create)")
    print("  3. Delete collection only")
    print("  4. Check collection status")
    
    choice = input("\nChoice [1-4]: ").strip()
    
    if choice not in ['1', '2', '3', '4']:
        print("❌ Invalid choice")
        return
    
    # Initialize embeddings
    print("\n🔄 Initializing embeddings...")
    try:
        embeddings = LangChainEmbeddingManager(
            model_name=config['embedding_model'],
            batch_size=config['batch_size']
        )
        
        if not embeddings.is_ready():
            print("❌ Embeddings not ready")
            return
            
        print(f"✅ Embeddings ready ({embeddings.embedding_dim}D)")
        
    except Exception as e:
        print(f"❌ Failed to initialize embeddings: {e}")
        return
    
    # Initialize vector store
    print("\n🔄 Connecting to Weaviate...")
    try:
        vectorstore = LangChainVectorStore(
            url=config['weaviate_url'],
            api_key=config['weaviate_api_key'],
            class_name=config['class_name'],
            embeddings=embeddings.embeddings,
            vector_dims=config['vector_dims']
        )
        
        if not vectorstore.is_connected():
            print("❌ Failed to connect to Weaviate")
            return
            
        print("✅ Connected to Weaviate")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Execute choice
    print()
    print("="*70)
    
    try:
        if choice == '1':
            # Create only if doesn't exist
            if vectorstore.weaviate_client.collections.exists(config['class_name']):
                print(f"⚠️  Collection '{config['class_name']}' already exists")
                print("   Use option 2 to recreate")
            else:
                print(f"📦 Creating collection '{config['class_name']}'...")
                vectorstore._ensure_collection()
                print(f"✅ Collection created ({config['vector_dims']}D)")
        
        elif choice == '2':
            # Recreate
            print("⚠️  WARNING: This will DELETE all existing data!")
            confirm = input("   Type 'yes' to confirm: ").strip().lower()
            
            if confirm == 'yes':
                print(f"\n🗑️  Deleting collection '{config['class_name']}'...")
                vectorstore.delete_collection()
                
                print(f"📦 Creating new collection ({config['vector_dims']}D)...")
                vectorstore._ensure_collection()
                
                print("\n✅ Collection recreated successfully!")
                print(f"   Dimension: {config['vector_dims']}")
                print("   Re-upload your documents to populate")
            else:
                print("❌ Cancelled")
        
        elif choice == '3':
            # Delete only
            print("⚠️  WARNING: This will DELETE the collection!")
            confirm = input("   Type 'yes' to confirm: ").strip().lower()
            
            if confirm == 'yes':
                print(f"\n🗑️  Deleting collection '{config['class_name']}'...")
                vectorstore.delete_collection()
                print("✅ Collection deleted")
            else:
                print("❌ Cancelled")
        
        elif choice == '4':
            # Status
            exists = vectorstore.weaviate_client.collections.exists(config['class_name'])
            print(f"📊 Collection Status:")
            print(f"   Name   : {config['class_name']}")
            print(f"   Exists : {'Yes ✅' if exists else 'No ❌'}")
            
            if exists:
                stats = vectorstore.get_stats()
                print(f"   Chunks : {stats.get('document_count', 0)}")
                print(f"   Dims   : {config['vector_dims']}")
    
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        vectorstore.close()
        print()
        print("="*70)
        print("✅ Done")
        print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)