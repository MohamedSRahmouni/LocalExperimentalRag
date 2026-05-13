"""
API Dependencies
Singleton service instances for dependency injection

Replaces:
    - Original dependencies.py with 6+ initialize_* functions
    
New:
    - Single initialize_all() function
    - LangChain pipeline singletons
"""

import logging
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# GLOBAL SINGLETON INSTANCES
# ============================================================

_document_loader: Optional[object] = None
_embedding_manager: Optional[object] = None
_vector_store: Optional[object] = None
_rag_service: Optional[object] = None
_memory_manager: Optional[object] = None


# ============================================================
# INITIALIZATION (called once at startup)
# ============================================================

def initialize_all():
    """
    Initialize all pipeline services
    
    Replaces:
        - initialize_document_processor()
        - initialize_embedding_manager()
        - initialize_weaviate()
        - initialize_lm_studio()
        - initialize_retrieval_service()
        - initialize_rag_service()
    
    New:
        - Single function for all initialization
        - Proper dependency order
        - Better error handling
    """
    global _document_loader
    global _embedding_manager
    global _vector_store
    global _rag_service
    global _memory_manager
    
    logger.info("="*80)
    logger.info("🚀 Initializing Pipeline Services")
    logger.info("="*80)
    
    try:
        # ================================================================
        # STEP 1: Document Loader
        # ================================================================
        from app.pipeline.document_loader import LangChainDocumentLoader
        
        logger.info("\n📄 Step 1: Initializing Document Loader...")
        
        _document_loader = LangChainDocumentLoader(
            lang=settings.OCR_LANGUAGE,
            use_gpu=settings.USE_GPU,
            chunk_size=settings.MAX_CHUNK_SIZE,
            chunk_overlap=200,
            min_chunk_size=settings.MIN_CHUNK_SIZE,
            similarity_threshold=settings.SIMILARITY_THRESHOLD,
            chunking_method=settings.CHUNKING_METHOD,
            embedding_model=settings.EMBEDDING_MODEL
        )
        
        logger.info("✅ Document Loader initialized")
        
        # ================================================================
        # STEP 2: Embedding Manager
        # ================================================================
        from app.pipeline.embeddings import LangChainEmbeddingManager
        
        logger.info("\n🔢 Step 2: Initializing Embedding Manager...")
        
        _embedding_manager = LangChainEmbeddingManager(
            model_name=settings.EMBEDDING_MODEL,
            use_gpu=settings.USE_GPU,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            cache_dir="./embeddings_cache",
            output_dir=str(settings.EMBEDDINGS_FOLDER),
            save_intermediate=True,
            max_workers=2 if settings.ENABLE_PARALLEL_EMBEDDING else 1
        )
        
        if not _embedding_manager.is_ready():
            logger.error("❌ Embedding manager not ready!")
            raise RuntimeError("Embedding manager initialization failed")
        
        logger.info("✅ Embedding Manager initialized")
        
        # ================================================================
        # STEP 3: Vector Store (Weaviate)
        # ================================================================
        from app.pipeline.vectorstore import LangChainVectorStore
        
        logger.info("\n🗄️  Step 3: Initializing Vector Store...")
        
        _vector_store = LangChainVectorStore(
            url=settings.WEAVIATE_URL,
            api_key=settings.WEAVIATE_API_KEY,
            class_name=settings.WEAVIATE_CLASS_NAME,
            embeddings=_embedding_manager.base_embeddings,  # Share embeddings
            vector_dims=settings.WEAVIATE_VECTOR_DIMS
        )
        
        if not _vector_store.is_connected():
            logger.error("❌ Weaviate not connected!")
            raise RuntimeError("Weaviate connection failed")
        
        logger.info("✅ Vector Store initialized")
        
        # ================================================================
        # STEP 4: Memory Manager
        # ================================================================
        from app.pipeline.memory import ConversationMemoryManager
        
        logger.info("\n🧠 Step 4: Initializing Memory Manager...")
        
        _memory_manager = ConversationMemoryManager(
            memory_type="buffer_window",
            window_size=6,  # Same as conversation_history[-6:]
            return_messages=True
        )
        
        logger.info("✅ Memory Manager initialized")
        
        # ================================================================
        # STEP 5: RAG Service
        # ================================================================
        from app.pipeline.rag_chain import LangChainRAGService, RAGConfig
        
        logger.info("\n🤖 Step 5: Initializing RAG Service...")
        
        _rag_service = LangChainRAGService(
            vectorstore=_vector_store,
            config=RAGConfig(
                top_k=3,
                min_score=0.55,
                enable_reranking=True,
                temperature=settings.LM_STUDIO_TEMPERATURE,
                max_tokens=settings.LM_STUDIO_MAX_TOKENS,
                language="français"
            ),
            lm_studio_url=settings.LM_STUDIO_URL,
            lm_studio_model=settings.LM_STUDIO_MODEL,
            memory_manager=_memory_manager  # Share memory manager
        )
        
        # Check LM Studio availability
        if _rag_service.is_available():
            logger.info("✅ RAG Service initialized (LM Studio available)")
        else:
            logger.warning("⚠️  RAG Service initialized but LM Studio not available")
            logger.warning("   Start LM Studio to enable full RAG features")
        
        # ================================================================
        # SUMMARY
        # ================================================================
        logger.info("\n" + "="*80)
        logger.info("✅ ALL SERVICES INITIALIZED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"✓ Document Loader  : Ready")
        logger.info(f"✓ Embedding Manager: {_embedding_manager.embedding_dim}D embeddings")
        logger.info(f"✓ Vector Store     : {_vector_store.get_stats()['document_count']} chunks")
        logger.info(f"✓ Memory Manager   : {_memory_manager.get_session_count()} sessions")
        logger.info(f"✓ RAG Service      : {'Active' if _rag_service.is_available() else 'LM Studio offline'}")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error("="*80)
        logger.error("❌ PIPELINE INITIALIZATION FAILED")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error("="*80)
        raise


# ============================================================
# DEPENDENCY GETTERS (for FastAPI routes)
# ============================================================

def get_doc_processor():
    """
    Get document loader instance
    
    Replaces: get_doc_processor() but returns DocumentLoader
    Used by: upload.py
    """
    return _document_loader


def get_document_loader():
    """Alias for clarity"""
    return _document_loader


def get_embedding_manager():
    """
    Get embedding manager instance
    
    Same interface as before
    Used by: upload.py
    """
    return _embedding_manager


def get_weaviate_client():
    """
    Get Weaviate client
    
    Same interface as before
    Used by: weaviate.py routes
    """
    if _vector_store:
        return _vector_store.weaviate_client
    return None


def get_vector_store():
    """
    Get vector store instance
    
    Same interface as before
    Used by: upload.py, stats.py
    """
    return _vector_store


def get_search_engine():
    """
    Get search engine
    
    Note: Now part of VectorStore (semantic_search method)
    Used by: retrieve.py
    
    Backward compatibility: Returns VectorStore (it has semantic_search)
    """
    return _vector_store


def get_rag_service():
    """
    Get RAG service instance
    
    Same interface as before
    Used by: chat.py, rag.py
    """
    return _rag_service


def get_retrieval_service():
    """
    Get retrieval service
    
    Note: Now embedded in RAGService
    Used by: retrieve.py
    
    Backward compatibility: Returns RAGService.retrieval_service
    """
    if _rag_service:
        return _rag_service.retrieval_service
    return None


def get_lm_studio_service():
    """
    Get LM Studio service
    
    Note: Now wrapped in RAGService as ChatOpenAI
    Used by: rag.py
    
    Backward compatibility: Returns RAGService (has is_available)
    """
    return _rag_service


def get_memory_manager():
    """
    Get memory manager instance
    
    New dependency (not in original code)
    Used by: chat.py (optional)
    """
    return _memory_manager


# ============================================================
# CLEANUP (for testing/reload)
# ============================================================

def reset_all_services():
    """
    Reset all services (for testing)
    
    Warning: Only use in development/testing
    """
    global _document_loader
    global _embedding_manager
    global _vector_store
    global _rag_service
    global _memory_manager
    
    logger.warning("🔄 Resetting all services...")
    
    # Close connections
    if _vector_store:
        _vector_store.close()
    
    # Clear memory
    if _memory_manager:
        _memory_manager.clear_all()
    
    # Reset to None
    _document_loader = None
    _embedding_manager = None
    _vector_store = None
    _rag_service = None
    _memory_manager = None
    
    logger.info("✅ All services reset")


# ============================================================
# HEALTH CHECK HELPERS
# ============================================================

def get_service_health() -> dict:
    """
    Get health status of all services
    
    Returns:
        Dictionary with health status
    """
    return {
        "document_loader": _document_loader is not None,
        "embedding_manager": _embedding_manager is not None and _embedding_manager.is_ready(),
        "vector_store": _vector_store is not None and _vector_store.is_connected(),
        "rag_service": _rag_service is not None,
        "lm_studio": _rag_service is not None and _rag_service.is_available(),
        "memory_manager": _memory_manager is not None,
        "all_ready": all([
            _document_loader is not None,
            _embedding_manager is not None and _embedding_manager.is_ready(),
            _vector_store is not None and _vector_store.is_connected(),
            _rag_service is not None
        ])
    }


def get_service_info() -> dict:
    """
    Get detailed service information
    
    Returns:
        Dictionary with service details
    """
    info = {
        "services": {},
        "configuration": {
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunking_method": settings.CHUNKING_METHOD,
            "lm_studio_url": settings.LM_STUDIO_URL,
            "weaviate_class": settings.WEAVIATE_CLASS_NAME
        }
    }
    
    if _embedding_manager:
        info["services"]["embedding"] = _embedding_manager.get_system_info()
    
    if _vector_store:
        info["services"]["vector_store"] = _vector_store.get_stats()
    
    if _memory_manager:
        info["services"]["memory"] = _memory_manager.get_stats()
    
    if _rag_service:
        info["services"]["rag"] = {
            "available": _rag_service.is_available(),
            "model": _rag_service.lm_studio_model,
            "temperature": _rag_service.config.temperature,
            "max_tokens": _rag_service.config.max_tokens
        }
    
    return info