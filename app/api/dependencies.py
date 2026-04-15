"""
Shared Dependencies
Singleton service instances accessible across the application
"""

import logging
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL SERVICE INSTANCES
# ============================================================================
_doc_processor: Optional[object] = None
_embedding_manager: Optional[object] = None
_weaviate_client: Optional[object] = None
_vector_store: Optional[object] = None
_search_engine: Optional[object] = None


# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

def initialize_document_processor():
    """Initialize the document processor"""
    global _doc_processor
    
    from app.services.preprocessing import DocumentProcessor
    
    logger.info("Initializing Document Processor...")
    _doc_processor = DocumentProcessor(
        lang=settings.OCR_LANGUAGE,
        use_gpu=settings.USE_GPU,
        embedding_model=settings.EMBEDDING_MODEL,
        chunking_method=settings.CHUNKING_METHOD
    )
    logger.info("✅ Document processor initialized")


def initialize_embedding_manager():
    """Initialize the embedding manager"""
    global _embedding_manager
    
    from app.services.embedding import EmbeddingManager
    
    logger.info("Initializing Embedding Manager...")
    _embedding_manager = EmbeddingManager(
        model_name=settings.EMBEDDING_MODEL,
        use_gpu=settings.USE_GPU,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        max_workers=2 if settings.ENABLE_PARALLEL_EMBEDDING else 1,
        save_intermediate=True,
        output_dir=str(settings.EMBEDDINGS_FOLDER)
    )
    
    if _embedding_manager.is_ready():
        logger.info("✅ Embedding manager initialized successfully")
        info = _embedding_manager.get_system_info()
        logger.info(f"   Model: {info.get('model_name', 'unknown')}")
        logger.info(f"   Dimension: {info.get('embedding_dim', 'unknown')}")
        logger.info(f"   Device: {info.get('device', 'cpu')}")
    else:
        logger.warning("⚠️  Embedding manager initialized but model not ready")


def initialize_weaviate():
    """Initialize Weaviate client, vector store, and search engine"""
    global _weaviate_client, _vector_store, _search_engine
    
    # ✅ FIXED: Correct import path
    from app.data.vectordb import WeaviateClient, VectorStore, SearchEngine
    
    logger.info("Initializing Weaviate Vector Database...")
    
    _weaviate_client = WeaviateClient(
        url=settings.WEAVIATE_URL,
        api_key=settings.WEAVIATE_API_KEY
    )
    
    if _weaviate_client.is_connected():
        logger.info("✅ Weaviate connected successfully")
        
        # List collections
        try:
            collections = _weaviate_client.client.collections.list_all()
            logger.info(f"📋 Found {len(collections)} collection(s) in Weaviate")
            
            for name in list(collections.keys())[:5]:  # Show first 5
                try:
                    col = _weaviate_client.client.collections.get(name)
                    count = col.aggregate.over_all(total_count=True)
                    logger.info(f"   📦 '{name}': {count.total_count} objects")
                except:
                    logger.info(f"   📦 '{name}'")
        except Exception as e:
            logger.warning(f"Could not list collections: {e}")
        
        # Initialize vector store
        _vector_store = VectorStore(
            weaviate_client=_weaviate_client,
            class_name=settings.WEAVIATE_CLASS_NAME,
            vector_dims=settings.WEAVIATE_VECTOR_DIMS
        )
        logger.info("✅ Vector store initialized")
        
        # Initialize search engine
        _search_engine = SearchEngine(
            weaviate_client=_weaviate_client,
            class_name=settings.WEAVIATE_CLASS_NAME
        )
        logger.info("✅ Search engine initialized")
        
        # Get stats
        stats = _vector_store.get_stats()
        logger.info(f"   Current documents in vector DB: {stats.get('document_count', 0)}")
    else:
        logger.warning("⚠️  Weaviate not connected - vector storage disabled")


# ============================================================================
# DEPENDENCY GETTERS (for FastAPI dependency injection)
# ============================================================================

def get_doc_processor():
    """Get document processor instance"""
    return _doc_processor


def get_embedding_manager():
    """Get embedding manager instance"""
    return _embedding_manager


def get_weaviate_client():
    """Get Weaviate client instance"""
    return _weaviate_client


def get_vector_store():
    """Get vector store instance"""
    return _vector_store


def get_search_engine():
    """Get search engine instance"""
    return _search_engine