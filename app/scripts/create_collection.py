import os
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, Property, DataType

load_dotenv()

# Configuration
url = os.getenv("WEAVIATE_URL")
api_key = os.getenv("WEAVIATE_API_KEY")
class_name = os.getenv("WEAVIATE_CLASS_NAME", "Ragdocument")

if not url.startswith('http'):
    url = f"https://{url}"

print("="*70)
print("CREATING WEAVIATE COLLECTION")
print("="*70)
print(f"URL: {url}")
print(f"Collection name: '{class_name}'")
print()

# Connect
client = weaviate.connect_to_wcs(
    cluster_url=url,
    auth_credentials=AuthApiKey(api_key)
)

try:
    # Check if already exists
    if client.collections.exists(class_name):
        print(f"⚠️  Collection '{class_name}' already exists!")
        
        response = input("Do you want to delete and recreate? (yes/no): ")
        if response.lower() == 'yes':
            print(f"Deleting '{class_name}'...")
            client.collections.delete(class_name)
            print("✓ Deleted")
        else:
            print("Keeping existing collection")
            client.close()
            exit()
    
    # Create collection
    print(f"\nCreating collection '{class_name}'...")
    
    client.collections.create(
        name=class_name,
        description="RAG document chunks with embeddings",
        
        # No vectorizer - we provide our own embeddings
        vectorizer_config=Configure.Vectorizer.none(),
        
        # Define properties/schema
        properties=[
            Property(
                name="chunk_id",
                data_type=DataType.TEXT,
                description="Unique identifier for the chunk"
            ),
            Property(
                name="text",
                data_type=DataType.TEXT,
                description="The actual text content"
            ),
            Property(
                name="text_length",
                data_type=DataType.INT,
                description="Length of the text in characters"
            ),
            Property(
                name="embedding_model",
                data_type=DataType.TEXT,
                description="Name of the embedding model used"
            ),
            Property(
                name="model_type",
                data_type=DataType.TEXT,
                description="Type of model (e.g., sentence-transformer)"
            ),
            Property(
                name="embedded_at",
                data_type=DataType.DATE,
                description="When the embedding was created"
            ),
            Property(
                name="indexed_at",
                data_type=DataType.DATE,
                description="When the chunk was indexed"
            ),
            Property(
                name="filename",
                data_type=DataType.TEXT,
                description="Source document filename"
            ),
            Property(
                name="file_type",
                data_type=DataType.TEXT,
                description="File extension/type"
            ),
            Property(
                name="chunk_index",
                data_type=DataType.INT,
                description="Position of chunk in document"
            ),
            Property(
                name="total_chunks",
                data_type=DataType.INT,
                description="Total number of chunks in document"
            ),
            Property(
                name="similarity_score",
                data_type=DataType.NUMBER,
                description="Similarity score if applicable"
            ),
            Property(
                name="sentence_count",
                data_type=DataType.INT,
                description="Number of sentences in chunk"
            )
        ]
    )
    
    print(f"✓ Collection '{class_name}' created successfully!")
    
    # Verify
    print("\nVerifying collection...")
    collections = client.collections.list_all()
    
    if class_name in collections:
        print(f"✓ Collection '{class_name}' found in cluster")
        
        col = client.collections.get(class_name)
        schema = collections[class_name]
        
        print(f"\nCollection details:")
        print(f"  Name: {class_name}")
        print(f"  Properties: {len(schema.properties)}")
        print(f"  Current objects: 0")
        
        print(f"\nProperty names:")
        for prop in schema.properties:
            print(f"  - {prop.name} ({prop.data_type})")
    else:
        print(f"❌ Collection '{class_name}' NOT found!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    client.close()
    print("\n" + "="*70)
    print("Done!")
    print("="*70)