"""
API Dependencies
Singleton service instances for dependency injection
"""

import logging
import os
import time
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# GLOBAL SINGLETON INSTANCES
# ============================================================

_document_loader:   Optional[object] = None
_embedding_manager: Optional[object] = None
_vector_store:      Optional[object] = None
_rag_service:       Optional[object] = None
_memory_manager:    Optional[object] = None
_langsmith_config:  Optional[object] = None
_mcp_client: Optional[object] = None

# ============================================================
# INITIALIZATION
# ============================================================

def initialize_all():
    global _document_loader, _embedding_manager, _vector_store
    global _rag_service, _memory_manager, _langsmith_config
    global _mcp_client
    
    logger.info("=" * 80)
    logger.info("🚀 Initializing Pipeline Services")
    logger.info("=" * 80)

    startup_start = time.time()

    # ── Prometheus metrics ──────────────────────────────────────
    try:
        from app.core.metrics import (
            system_ready,
            system_startup_time,
            weaviate_connection_status,   # kept name for Prometheus compat
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
        logger.info("\n📄 Step 1: Initializing Document Loader...")
        from app.pipeline.document_loader import LangChainDocumentLoader

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
        logger.info("\n🔢 Step 2: Initializing Embedding Manager...")
        from app.pipeline.embeddings import LangChainEmbeddingManager

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
            raise RuntimeError("Embedding manager initialization failed")

        logger.info("✅ Embedding Manager initialized")
        logger.info(f"   Dimension: {_embedding_manager.embedding_dim}D")

        # ================================================================
        # STEP 3: Vector Store (Qdrant) ← CHANGED
        # ================================================================
        logger.info("\n🗄️  Step 3: Initializing Vector Store (Qdrant)...")
        from app.pipeline.vectorstore import LangChainQdrantStore  # ← CHANGED

        _vector_store = LangChainQdrantStore(
            url=settings.QDRANT_URL,                          # ← CHANGED
            api_key=settings.QDRANT_API_KEY,                  # ← CHANGED
            collection_name=settings.QDRANT_COLLECTION_NAME,  # ← CHANGED
            embeddings=_embedding_manager.base_embeddings,
            vector_size=settings.QDRANT_VECTOR_SIZE,          # ← CHANGED
            distance=settings.QDRANT_DISTANCE,                # ← CHANGED
        )

        if not _vector_store.is_connected():
            raise RuntimeError("Qdrant connection failed")

        logger.info("✅ Vector Store (Qdrant) initialized")

        # ── Prometheus: mark connected ──────────────────────────
        if _metrics_available:
            try:
                weaviate_connection_status.set(1)
                stats = _vector_store.get_stats()
                chunk_count = stats.get('document_count', 0)
                weaviate_total_chunks.set(chunk_count)
                logger.info(
                    f"   📊 Prometheus: connected=1 | chunks={chunk_count}"
                )
            except Exception as e:
                logger.warning(f"⚠️  Could not update metrics: {e}")

        # ================================================================
        # STEP 4: Memory Manager
        # ================================================================
        logger.info("\n🧠 Step 4: Initializing Memory Manager...")
        from app.pipeline.memory import ConversationMemoryManager

        _memory_manager = ConversationMemoryManager(
            memory_type="buffer_window",
            window_size=6,
            return_messages=True,
        )
        logger.info("✅ Memory Manager initialized")



        # ================================================================
        # STEP 5: MCP Client (connects to external servers)
        # ================================================================
        logger.info("\n🔧 Step 5: Initializing MCP Client...")

        try:
            from app.services.mcp import MCPClientManager

            _mcp_client = MCPClientManager(
                filesystem_url=os.getenv(
                    "MCP_FILESYSTEM_URL",
                    "http://localhost:8001/sse"
                ),
                websearch_url=os.getenv(
                    "MCP_WEBSEARCH_URL",
                    "http://localhost:8002/sse"
                ),
            )

            # Check which servers are online
            health = _mcp_client.check_all_health()

            # Discover tools from online servers
            _mcp_client.discover_all_tools()
            _mcp_client.set_ready(True)

            logger.info("✅ MCP Client initialized")
            logger.info(
                f"   FileSystem server : "
                f"{'✅ online' if health.get('filesystem') else '❌ offline'}"
            )
            logger.info(
                f"   WebSearch server  : "
                f"{'✅ online' if health.get('websearch') else '❌ offline'}"
            )

        except Exception as e:
            logger.warning(f"⚠️  MCP Client failed (non-fatal): {e}")
            _mcp_client = None

        # ================================================================
        # STEP 6: RAG Service
        # ================================================================
        logger.info("\n🤖 Step 6: Initializing RAG Service...")
        from app.pipeline.rag_chain import LangChainRAGService, RAGConfig

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
            mcp_client=_mcp_client,
        )

        if _rag_service.is_available():
            logger.info("✅ RAG Service initialized (LM Studio available)")
        else:
            logger.warning("⚠️  RAG Service initialized but LM Studio offline")

        # ================================================================
        # STEP 6: Prometheus — Mark system ready
        # ================================================================
        startup_duration = time.time() - startup_start

        if _metrics_available:
            try:
                system_ready.set(1)
                system_startup_time.set(startup_duration)
            except Exception as e:
                logger.warning(f"⚠️  Could not update system metrics: {e}")

        # ================================================================
        # SUMMARY
        # ================================================================
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL SERVICES INITIALIZED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"✓ Document Loader   : Ready")
        logger.info(
            f"✓ Embedding Manager : "
            f"{_embedding_manager.embedding_dim}D | "
            f"multilingual-e5-small"
        )
        logger.info(
            f"✓ Vector Store      : "
            f"Qdrant | "
            f"{_vector_store.get_stats()['document_count']} chunks"
        )
        logger.info(
            f"✓ Memory Manager    : "
            f"{_memory_manager.get_session_count()} sessions"
        )
        logger.info(
            f"✓ RAG Service       : "
            f"{'Active' if _rag_service.is_available() else 'LM Studio offline'}"
        )
        logger.info(
            f"✓ LangSmith         : "
            f"{'Enabled' if _langsmith_config and _langsmith_config.is_enabled() else 'Disabled'}"
        )
        logger.info(
            f"✓ Prometheus        : "
            f"{'Enabled' if _metrics_available else 'Disabled'}"
        )
        logger.info(f"✓ Startup time      : {startup_duration:.2f}s")
        logger.info("=" * 80 + "\n")

    except Exception as e:
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
# DEPENDENCY GETTERS
# ============================================================

def get_doc_processor():
    return _document_loader

def get_document_loader():
    return _document_loader

def get_embedding_manager():
    return _embedding_manager

def get_mcp_client():
    return _mcp_client

def get_vector_store():
    return _vector_store

def get_search_engine():
    """Backward compatibility — returns vector store."""
    return _vector_store

def get_rag_service():
    return _rag_service

def get_retrieval_service():
    if _rag_service:
        return _rag_service.retrieval_service
    return None

def get_lm_studio_service():
    """Backward compatibility — returns RAG service."""
    return _rag_service

def get_memory_manager():
    return _memory_manager

def get_langsmith_config():
    return _langsmith_config

# ── REMOVED: get_weaviate_client() ← no longer applicable ──
# Replace any callers with get_vector_store().client
def get_qdrant_client():
    """Get native Qdrant client."""
    if _vector_store:
        return _vector_store.client
    return None


# ============================================================
# CLEANUP
# ============================================================

def reset_all_services():
    global _document_loader, _embedding_manager, _vector_store
    global _rag_service, _memory_manager, _langsmith_config

    logger.warning("🔄 Resetting all services...")

    try:
        from app.core.metrics import system_ready, weaviate_connection_status
        system_ready.set(0)
        weaviate_connection_status.set(0)
    except Exception:
        pass

    if _vector_store:
        try:
            _vector_store.close()
        except Exception:
            pass

    if _memory_manager:
        try:
            _memory_manager.clear_all()
        except Exception:
            pass

    _document_loader   = None
    _embedding_manager = None
    _vector_store      = None
    _rag_service       = None
    _memory_manager    = None
    _langsmith_config  = None

    logger.info("✅ All services reset")


# ============================================================
# HEALTH CHECK
# ============================================================

def get_service_health() -> dict:
    try:
        from app.core.metrics import (
            weaviate_connection_status,
            weaviate_total_chunks,
            memory_active_sessions,
            system_ready,
        )

        is_store_connected = (
            _vector_store is not None and
            _vector_store.is_connected()
        )
        weaviate_connection_status.set(1 if is_store_connected else 0)

        if is_store_connected:
            stats = _vector_store.get_stats()
            weaviate_total_chunks.set(stats.get('document_count', 0))

        if _memory_manager:
            memory_active_sessions.set(_memory_manager.get_session_count())

        all_ready = all([
            _document_loader   is not None,
            _embedding_manager is not None and _embedding_manager.is_ready(),
            _vector_store      is not None and _vector_store.is_connected(),
            _rag_service       is not None,
        ])
        system_ready.set(1 if all_ready else 0)

    except Exception:
        pass

    return {
        "document_loader":   _document_loader is not None,
        "embedding_manager": (
            _embedding_manager is not None and
            _embedding_manager.is_ready()
        ),
        "vector_store": (
            _vector_store is not None and
            _vector_store.is_connected()
        ),
        "rag_service":   _rag_service is not None,
        "lm_studio": (
            _rag_service is not None and
            _rag_service.is_available()
        ),
        "memory_manager":  _memory_manager is not None,
        "langsmith": (
            _langsmith_config is not None and
            _langsmith_config.is_enabled()
        ),
        "all_ready": all([
            _document_loader   is not None,
            _embedding_manager is not None and _embedding_manager.is_ready(),
            _vector_store      is not None and _vector_store.is_connected(),
            _rag_service       is not None,
        ]),
    }


def get_service_info() -> dict:
    info = {
        "services": {},
        "configuration": {
            "embedding_model":    settings.EMBEDDING_MODEL,
            "chunking_method":    settings.CHUNKING_METHOD,
            "lm_studio_url":      settings.LM_STUDIO_URL,
            "qdrant_collection":  settings.QDRANT_COLLECTION_NAME,  # ← CHANGED
            "qdrant_url":         settings.QDRANT_URL,              # ← CHANGED
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

def update_vector_store_metrics():
    """Refresh vector store metrics after bulk inserts."""
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
                f"📊 Qdrant metrics updated: "
                f"{stats.get('document_count', 0)} chunks"
            )
        else:
            weaviate_connection_status.set(0)
    except Exception as e:
        logger.warning(f"⚠️  Could not update metrics: {e}")


# ── Backward compat alias ──────────────────────────────────
update_weaviate_metrics = update_vector_store_metrics