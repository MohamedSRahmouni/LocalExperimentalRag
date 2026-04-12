import os
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey

load_dotenv(override=True)

url = os.getenv("WEAVIATE_URL")
api_key = os.getenv("WEAVIATE_API_KEY")
class_name = os.getenv("WEAVIATE_CLASS_NAME")

if not url.startswith('http'):
    url = f"https://{url}"

print(f"URL: {url}")
print(f"Class name from .env: '{class_name}'")
print(f"Class name repr: {repr(class_name)}")
print("="*70)

client = weaviate.connect_to_wcs(
    cluster_url=url,
    auth_credentials=AuthApiKey(api_key)
)

# List all collections
collections = client.collections.list_all()
print("\nAll collections:")
for name in collections.keys():
    print(f"  - '{name}' (repr: {repr(name)})")

# Check existence
print(f"\nChecking existence of '{class_name}':")
exists = client.collections.exists(class_name)
print(f"  Exists: {exists}")

# Try to get it
if exists:
    print(f"\nTrying to get collection '{class_name}':")
    try:
        col = client.collections.get(class_name)
        print(f"  ✓ Successfully retrieved")
        print(f"  Type: {type(col)}")
        
        # Try to count
        count = col.aggregate.over_all(total_count=True)
        print(f"  Count: {count.total_count}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

client.close()