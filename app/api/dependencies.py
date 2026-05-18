"""
API Dependencies
Singleton service instances for dependency injection

Replaces:
    - Original dependencies.py with 6+ initialize_* functions
    
New:
    - Single initialize_all() function
    - LangChain pipeline singletons
    - Prometheus metrics integration
"""

import logging
import time
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# GLOBAL SINGLETON INSTANCES
# ============================================================

_document_loader:  Optional[object] = None
_embedding_manager: Optional[object] = None
_vector_store:     Optional[object] = None
_rag_service:      Optional[object] = None
_memory_manager:   Optional[object] = None
_langsmith_config: Optional[object] = None


# ============================================================
# INITIALIZATION (called once at startup)
# ============================================================

def initialize_all():
    """
    Initialize all pipeline services + Prometheus metrics.
    """
    global _document_loader
    global _embedding_manager
    global _vector_store
    global _rag_service
    global _memory_manager
    global _langsmith_config

    logger.info("=" * 80)
    logger.info("🚀 Initializing Pipeline Services")
    logger.info("=" * 80)

    # ── Track startup time ─────────────────────────────────────
    startup_start = time.time()

    # ── Import metrics ─────────────────────────────────────────
    try:
        from app.core.metrics import (
            system_ready,
            system_startup_time,
            weaviate_connection_status,
            weaviate_total_chunks,
            init_metrics,
        )
        init_metrics()
        _metrics_available = True
        logger.info("✅ Prometheus metrics initialized")
    except Exception as e:
        logger.warning(f"⚠️  Prometheus metrics not available: {e}")
        _metrics_available = False

    try:
        # ================================================================
        # STEP 0: LangSmith
        # ================================================================
        logger.info("\n📊 Step 0: Initializing LangSmith...")

        try:
            from app.core.langsmith_config import get_langsmith_config
            _langsmith_config = get_langsmith_config()

            if _langsmith_config.is_enabled():
                logger.info("✅ LangSmith enabled")
            else:
                logger.warning("⚠️  LangSmith disabled (check API key)")
        except Exception as e:
            logger.warning(f"⚠️  LangSmith init failed: {e}")
            _langsmith_config = None

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
            embedding_model=settings.EMBEDDING_MODEL,
            preserve_tables=True,
            max_table_size=5000,
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
            max_workers=2 if settings.ENABLE_PARALLEL_EMBEDDING else 1,
        )

        if not _embedding_manager.is_ready():
            logger.error("❌ Embedding manager not ready!")
            raise RuntimeError("Embedding manager initialization failed")

        logger.info("✅ Embedding Manager initialized")
        logger.info(f"   Dimension: {_embedding_manager.embedding_dim}D")

        # ================================================================
        # STEP 3: Vector Store (Weaviate)
        # ================================================================
        from app.pipeline.vectorstore import LangChainVectorStore

        logger.info("\n🗄️  Step 3: Initializing Vector Store...")

        _vector_store = LangChainVectorStore(
            url=settings.WEAVIATE_URL,
            api_key=settings.WEAVIATE_API_KEY,
            class_name=settings.WEAVIATE_CLASS_NAME,
            embeddings=_embedding_manager.base_embeddings,
            vector_dims=settings.WEAVIATE_VECTOR_DIMS,
        )

        if not _vector_store.is_connected():
            logger.error("❌ Weaviate not connected!")
            raise RuntimeError("Weaviate connection failed")

        logger.info("✅ Vector Store initialized")

        # ── Prometheus: Weaviate connected ─────────────────────
        if _metrics_available:
            try:
                weaviate_connection_status.set(1)
                stats = _vector_store.get_stats()
                chunk_count = stats.get('document_count', 0)
                weaviate_total_chunks.set(chunk_count)
                logger.info(
                    f"   📊 Prometheus: weaviate_connection_status=1 | "
                    f"chunks={chunk_count}"
                )
            except Exception as e:
                logger.warning(f"⚠️  Could not update Weaviate metrics: {e}")

        # ================================================================
        # STEP 4: Memory Manager
        # ================================================================
        from app.pipeline.memory import ConversationMemoryManager

        logger.info("\n🧠 Step 4: Initializing Memory Manager...")

        _memory_manager = ConversationMemoryManager(
            memory_type="buffer_window",
            window_size=6,
            return_messages=True,
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
                top_k=5,
                min_score=0.1,
                enable_reranking=True,
                use_hybrid=True,
                bm25_weight=0.4,
                vector_weight=0.6,
                temperature=settings.LM_STUDIO_TEMPERATURE,
                max_tokens=settings.LM_STUDIO_MAX_TOKENS,
                language="français",
            ),
            lm_studio_url=settings.LM_STUDIO_URL,
            lm_studio_model=settings.LM_STUDIO_MODEL,
            memory_manager=_memory_manager,
        )

        if _rag_service.is_available():
            logger.info("✅ RAG Service initialized (LM Studio available)")
        else:
            logger.warning("⚠️  RAG Service initialized but LM Studio not available")
            logger.warning("   Start LM Studio to enable full RAG features")

        # ================================================================
        # STEP 6: Prometheus — Mark system ready
        # ================================================================
        startup_duration = time.time() - startup_start

        if _metrics_available:
            try:
                system_ready.set(1)
                system_startup_time.set(startup_duration)
                logger.info(
                    f"   📊 Prometheus: system_ready=1 | "
                    f"startup_time={startup_duration:.2f}s"
                )
            except Exception as e:
                logger.warning(f"⚠️  Could not update system metrics: {e}")

        # ================================================================
        # SUMMARY
        # ================================================================
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL SERVICES INITIALIZED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"✓ Document Loader  : Ready")
        logger.info(
            f"✓ Embedding Manager: "
            f"{_embedding_manager.embedding_dim}D embeddings"
        )
        logger.info(
            f"✓ Vector Store     : "
            f"{_vector_store.get_stats()['document_count']} chunks"
        )
        logger.info(
            f"✓ Memory Manager   : "
            f"{_memory_manager.get_session_count()} sessions"
        )
        logger.info(
            f"✓ RAG Service      : "
            f"{'Active' if _rag_service.is_available() else 'LM Studio offline'}"
        )
        logger.info(
            f"✓ LangSmith        : "
            f"{'Enabled' if _langsmith_config and _langsmith_config.is_enabled() else 'Disabled'}"
        )
        logger.info(
            f"✓ Prometheus       : "
            f"{'Enabled' if _metrics_available else 'Disabled'}"
        )
        logger.info(f"✓ Startup time     : {startup_duration:.2f}s")
        logger.info("=" * 80 + "\n")

    except Exception as e:
        # ── Prometheus: Mark system NOT ready on failure ────────
        try:
            from app.core.metrics import system_ready, weaviate_connection_status
            system_ready.set(0)
            weaviate_connection_status.set(0)
        except Exception:
            pass

        logger.error("=" * 80)
        logger.error("❌ PIPELINE INITIALIZATION FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error("=" * 80)
        raise


# ============================================================
# DEPENDENCY GETTERS (for FastAPI routes)
# ============================================================

def get_doc_processor():
    """Get document loader instance."""
    return _document_loader


def get_document_loader():
    """Alias for clarity."""
    return _document_loader


def get_embedding_manager():
    """Get embedding manager instance."""
    return _embedding_manager


def get_weaviate_client():
    """
    Get Weaviate client.
    Returns the native weaviate client from the vector store.
    """
    if _vector_store:
        return _vector_store.weaviate_client
    return None


def get_vector_store():
    """Get vector store instance."""
    return _vector_store


def get_search_engine():
    """
    Get search engine.
    Backward compatibility: Returns VectorStore (has semantic_search).
    """
    return _vector_store


def get_rag_service():
    """Get RAG service instance."""
    return _rag_service


def get_retrieval_service():
    """
    Get retrieval service.
    Backward compatibility: Returns RAGService.retrieval_service
    """
    if _rag_service:
        return _rag_service.retrieval_service
    return None


def get_lm_studio_service():
    """
    Get LM Studio service.
    Backward compatibility: Returns RAGService (has is_available).
    """
    return _rag_service


def get_memory_manager():
    """Get memory manager instance."""
    return _memory_manager


def get_langsmith_config():
    """Get LangSmith config instance."""
    return _langsmith_config


# ============================================================
# CLEANUP (for testing/reload)
# ============================================================

def reset_all_services():
    """Reset all services (for testing only)."""
    global _document_loader
    global _embedding_manager
    global _vector_store
    global _rag_service
    global _memory_manager
    global _langsmith_config

    logger.warning("🔄 Resetting all services...")

    # ── Prometheus: Mark not ready ──────────────────────────────
    try:
        from app.core.metrics import system_ready, weaviate_connection_status
        system_ready.set(0)
        weaviate_connection_status.set(0)
    except Exception:
        pass

    # Close connections
    if _vector_store:
        try:
            _vector_store.close()
        except Exception:
            pass

    # Clear memory
    if _memory_manager:
        try:
            _memory_manager.clear_all()
        except Exception:
            pass

    # Reset to None
    _document_loader   = None
    _embedding_manager = None
    _vector_store      = None
    _rag_service       = None
    _memory_manager    = None
    _langsmith_config  = None

    logger.info("✅ All services reset")


# ============================================================
# HEALTH CHECK HELPERS
# ============================================================

def get_service_health() -> dict:
    """Get health status of all services."""
    
    # ── Update Prometheus gauges on health check ────────────────
    try:
        from app.core.metrics import (
            weaviate_connection_status,
            weaviate_total_chunks,
            memory_active_sessions,
            system_ready,
        )

        is_weaviate_connected = (
            _vector_store is not None and
            _vector_store.is_connected()
        )
        weaviate_connection_status.set(1 if is_weaviate_connected else 0)

        if is_weaviate_connected:
            stats = _vector_store.get_stats()
            weaviate_total_chunks.set(stats.get('document_count', 0))

        if _memory_manager:
            memory_active_sessions.set(_memory_manager.get_session_count())

        all_ready = all([
            _document_loader  is not None,
            _embedding_manager is not None and _embedding_manager.is_ready(),
            _vector_store     is not None and _vector_store.is_connected(),
            _rag_service      is not None,
        ])
        system_ready.set(1 if all_ready else 0)

    except Exception:
        pass

    return {
        "document_loader":   _document_loader  is not None,
        "embedding_manager": (
            _embedding_manager is not None and
            _embedding_manager.is_ready()
        ),
        "vector_store":      (
            _vector_store is not None and
            _vector_store.is_connected()
        ),
        "rag_service":       _rag_service is not None,
        "lm_studio":         (
            _rag_service is not None and
            _rag_service.is_available()
        ),
        "memory_manager":    _memory_manager is not None,
        "langsmith":         (
            _langsmith_config is not None and
            _langsmith_config.is_enabled()
        ),
        "all_ready": all([
            _document_loader  is not None,
            _embedding_manager is not None and _embedding_manager.is_ready(),
            _vector_store     is not None and _vector_store.is_connected(),
            _rag_service      is not None,
        ]),
    }


def get_service_info() -> dict:
    """Get detailed service information."""
    info = {
        "services":      {},
        "configuration": {
            "embedding_model":  settings.EMBEDDING_MODEL,
            "chunking_method":  settings.CHUNKING_METHOD,
            "lm_studio_url":    settings.LM_STUDIO_URL,
            "weaviate_class":   settings.WEAVIATE_CLASS_NAME,
        },
    }

    if _embedding_manager:
        info["services"]["embedding"] = _embedding_manager.get_system_info()

    if _vector_store:
        info["services"]["vector_store"] = _vector_store.get_stats()

    if _memory_manager:
        info["services"]["memory"] = _memory_manager.get_stats()

    if _rag_service:
        info["services"]["rag"] = {
            "available":   _rag_service.is_available(),
            "model":       _rag_service.lm_studio_model,
            "temperature": _rag_service.config.temperature,
            "max_tokens":  _rag_service.config.max_tokens,
            "top_k":       _rag_service.config.top_k,
            "min_score":   _rag_service.config.min_score,
        }

    if _langsmith_config:
        info["services"]["langsmith"] = {
            "enabled": _langsmith_config.is_enabled(),
            "project": _langsmith_config.project,
        }

    return info


# ============================================================
# PROMETHEUS HELPER
# ============================================================

def update_weaviate_metrics():
    """
    Manually refresh Weaviate metrics.
    Call this after bulk inserts.
    """
    try:
        from app.core.metrics import (
            weaviate_connection_status,
            weaviate_total_chunks,
        )

        if _vector_store and _vector_store.is_connected():
            weaviate_connection_status.set(1)
            stats = _vector_store.get_stats()
            weaviate_total_chunks.set(stats.get('document_count', 0))
            logger.info(
                f"📊 Weaviate metrics updated: "
                f"{stats.get('document_count', 0)} chunks"
            )
        else:
            weaviate_connection_status.set(0)

    except Exception as e:
        logger.warning(f"⚠️  Could not update Weaviate metrics: {e}")