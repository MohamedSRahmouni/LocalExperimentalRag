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
_lm_studio_service: Optional[object] = None
_retrieval_service: Optional[object] = None
_rag_service: Optional[object] = None


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
    
    from app.data.vectordb import WeaviateClient, VectorStore, SearchEngine
    
    logger.info("Initializing Weaviate Vector Database...")
    
    _weaviate_client = WeaviateClient(
        url=settings.WEAVIATE_URL,
        api_key=settings.WEAVIATE_API_KEY
    )
    
    if _weaviate_client.is_connected():
        logger.info("✅ Weaviate connected successfully")
        
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


def initialize_lm_studio():
    """Initialize LM Studio service"""
    global _lm_studio_service
    
    from app.services.retrieval.lm_studio_service import LMStudioService, LMStudioConfig
    
    logger.info("Initializing LM Studio...")
    
    config = LMStudioConfig(
        base_url=settings.LM_STUDIO_URL,
        model=settings.LM_STUDIO_MODEL,
        temperature=settings.LM_STUDIO_TEMPERATURE,
        max_tokens=settings.LM_STUDIO_MAX_TOKENS,
        timeout=settings.LM_STUDIO_TIMEOUT
    )
    
    _lm_studio_service = LMStudioService(config=config)
    
    if _lm_studio_service.is_available():
        logger.info("✅ LM Studio initialized successfully")
    else:
        logger.warning("⚠️  LM Studio not available - generation features disabled")


def initialize_retrieval_service():
    """Initialize retrieval service"""
    global _retrieval_service
    
    from app.services.retrieval.retrieval_service import RetrievalService, RetrievalConfig
    
    search_engine = get_search_engine()
    doc_processor = get_doc_processor()
    
    if not search_engine:
        logger.error("❌ Cannot initialize retrieval service - search engine not available")
        return
    
    # Get embedder
    embedder = None
    if doc_processor and hasattr(doc_processor, 'semantic_chunker'):
        if doc_processor.semantic_chunker.is_model_available():
            embedder = doc_processor.semantic_chunker.embedder
    
    if not embedder:
        logger.error("❌ Cannot initialize retrieval service - embedder not available")
        return
    
    logger.info("Initializing Retrieval Service...")
    
    _retrieval_service = RetrievalService(
        search_engine=search_engine,
        embedder=embedder,
        config=RetrievalConfig(
            top_k=5,
            min_score=0.5,
            max_context_length=2000,
            enable_reranking=True
        )
    )
    
    logger.info("✅ Retrieval service initialized")


def initialize_rag_service():
    """Initialize RAG service"""
    global _rag_service
    
    from app.services.retrieval.rag_service import RAGService, RAGConfig
    
    retrieval_service = get_retrieval_service()
    lm_studio_service = get_lm_studio_service()
    
    if not retrieval_service:
        logger.error("❌ Cannot initialize RAG service - retrieval service not available")
        return
    
    if not lm_studio_service or not lm_studio_service.is_available():
        logger.error("❌ Cannot initialize RAG service - LM Studio not available")
        return
    
    logger.info("Initializing RAG Service...")
    
    _rag_service = RAGService(
        retrieval_service=retrieval_service,
        lm_studio_service=lm_studio_service,
        config=RAGConfig(
            top_k=3,
            min_score=0.55,
            enable_reranking=True,
            temperature=0.4,
            max_tokens=200
        )
    )
    
    logger.info("✅ RAG service initialized")


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


def get_lm_studio_service():
    """Get LM Studio service instance"""
    return _lm_studio_service


def get_retrieval_service():
    """Get retrieval service instance"""
    return _retrieval_service


def get_rag_service():
    """Get RAG service instance"""
    return _rag_service