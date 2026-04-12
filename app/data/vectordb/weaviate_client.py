"""
Weaviate Client - Manages connection to Weaviate vector database
"""

import logging
from typing import Optional, Dict, Any, List
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
import os

logger = logging.getLogger(__name__)


class WeaviateClient:
    """
    Weaviate connection and basic operations
    """
    
    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize Weaviate client
        
        Args:
            url: Weaviate URL
            api_key: Optional API key for Weaviate Cloud
            timeout: Request timeout in seconds
        """
        self.url = url
        self.api_key = api_key
        self.client = None
        
        self._connect(timeout)
    
    def _connect(self, timeout: int):
        try:
            logger.info("="*70)
            logger.info("Connecting to Weaviate Cloud")
            logger.info("="*70)
            
            # Add https:// if not present
            cluster_url = self.url
            if not cluster_url.startswith(('http://', 'https://')):
                cluster_url = f"https://{cluster_url}"
            
            logger.info(f"URL: {cluster_url}")
            
            # Connect to Weaviate Cloud
            self.client = weaviate.connect_to_wcs(
                cluster_url=cluster_url,
                auth_credentials=weaviate.auth.AuthApiKey(self.api_key),
                skip_init_checks=False  # Important: verify connection
            )
            
            # Verify connection
            if self.client.is_ready():
                logger.info("✓ Successfully connected to Weaviate Cloud")
            else:
                logger.error("❌ Connected but Weaviate is not ready")
            
        except Exception as e:
            logger.error("="*70)
            logger.error("❌ Failed to connect to Weaviate")
            logger.error("="*70)
            logger.error(f"Error: {str(e)}")
            logger.error("\nTroubleshooting:")
            logger.error("  1. Check your WEAVIATE_URL is correct")
            logger.error("  2. Verify your WEAVIATE_API_KEY")
            logger.error("  3. Ensure your cluster is running in Weaviate Cloud Console")
            logger.error("="*70)
            import traceback
            traceback.print_exc()
            self.client = None
        
    def is_connected(self) -> bool:
        """Check if connected to Weaviate"""
        return self.client is not None and self.client.is_ready()
    
    def create_collection(
        self,
        class_name: str,
        description: str = "RAG document chunks",
        vector_dims: int = 768
    ) -> bool:
        """Create a collection (class) in Weaviate"""
        if not self.is_connected():
            logger.error("Not connected to Weaviate")
            return False
        
        try:
            # Check if collection exists
            if self.client.collections.exists(class_name):
                logger.info(f"Collection '{class_name}' already exists")
                return True
            
            # Create collection
            logger.info(f"Creating collection '{class_name}'...")
            
            self.client.collections.create(
                name=class_name,
                description=description,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="text_length", data_type=DataType.INT),
                    Property(name="embedding_model", data_type=DataType.TEXT),
                    Property(name="model_type", data_type=DataType.TEXT),
                    Property(name="embedded_at", data_type=DataType.DATE),
                    Property(name="indexed_at", data_type=DataType.DATE),
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="file_type", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="total_chunks", data_type=DataType.INT),
                    # ✅ Make these optional by not marking as required
                    Property(
                        name="similarity_score", 
                        data_type=DataType.NUMBER,
                        skip_vectorization=True
                    ),
                    Property(
                        name="sentence_count", 
                        data_type=DataType.INT,
                        skip_vectorization=True
                    )
                ]
            )
            
            logger.info(f"✓ Created collection: {class_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def delete_collection(self, class_name: str) -> bool:
        """Delete a collection"""
        if not self.is_connected():
            return False
        
        try:
            if self.client.collections.exists(class_name):
                self.client.collections.delete(class_name)
                logger.info(f"✓ Deleted collection: {class_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
    
    def get_collection(self, class_name: str):
        """Get a collection object"""
        if not self.is_connected():
            logger.error("Not connected to Weaviate")
            return None
        
        try:
            # Check if collection exists first
            if not self.client.collections.exists(class_name):
                logger.error(f"Collection '{class_name}' does not exist")
                
                # List available collections for debugging
                available = list(self.client.collections.list_all().keys())
                logger.error(f"Available collections: {available}")
                
                return None
            
            # Get the collection - DON'T LOG SUCCESS YET
            collection = self.client.collections.get(class_name)
            
            # Verify we actually got an object
            if collection is None:
                logger.error(f"client.collections.get('{class_name}') returned None!")
                return None
            
            # NOW log success
            logger.info(f"✓ Successfully retrieved collection: '{class_name}'")
            
            # Return the collection object
            return collection
            
        except Exception as e:
            logger.error(f"Error getting collection '{class_name}': {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def get_stats(self, class_name: str) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.is_connected():
            return {}
        
        try:
            collection = self.get_collection(class_name)
            if not collection:
                return {}
            
            # Get object count
            aggregate = collection.aggregate.over_all(total_count=True)
            
            return {
                "document_count": aggregate.total_count,
                "collection_name": class_name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
    
    def close(self):
        """Close the connection"""
        if self.client:
            self.client.close()
            logger.info("✓ Weaviate connection closed")