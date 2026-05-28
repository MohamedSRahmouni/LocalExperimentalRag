"""
Application Startup and Shutdown Logic
Handles service initialization and cleanup
"""

import logging
from app.core.config import settings
from app.api.dependencies import (
    initialize_all,
    get_doc_processor,
    get_embedding_manager,
    get_vector_store,        # ← CHANGED: was get_weaviate_client
    get_qdrant_client,       # ← NEW
)

logger = logging.getLogger(__name__)


async def startup_event():
    """Initialize all services on application startup"""

    logger.info("=" * 80)
    logger.info("🚀 Initializing Enterprise RAG System")
    logger.info("=" * 80)

    # ── Configuration log ──────────────────────────────────────
    logger.info(f"📝 OCR Language       : {settings.OCR_LANGUAGE}")
    logger.info(f"💻 Device             : {'GPU' if settings.USE_GPU else 'CPU'}")
    logger.info(f"🤖 Embedding Model    : {settings.EMBEDDING_MODEL}")
    logger.info(f"✂️  Chunking Method    : {settings.CHUNKING_METHOD}")
    logger.info(f"📊 Similarity Thresh  : {settings.SIMILARITY_THRESHOLD}")
    logger.info(
        f"📏 Chunk Size Range   : "
        f"{settings.MIN_CHUNK_SIZE} - {settings.MAX_CHUNK_SIZE}"
    )
    logger.info(f"📦 Embedding Batch    : {settings.EMBEDDING_BATCH_SIZE}")
    logger.info(f"⚡ Parallel Embedding : {settings.ENABLE_PARALLEL_EMBEDDING}")

    # ── Qdrant config log ──────────────────────────────────────
    logger.info(f"🗄️  Vector DB          : Qdrant (Docker)")
    logger.info(f"🔗 Qdrant URL         : {settings.QDRANT_URL}")
    logger.info(f"📁 Collection         : {settings.QDRANT_COLLECTION_NAME}")
    logger.info(f"📐 Vector Size        : {settings.QDRANT_VECTOR_SIZE}D")
    logger.info(f"📏 Distance Metric    : {settings.QDRANT_DISTANCE}")
    logger.info("=" * 80)

    try:
        initialize_all()

        logger.info("=" * 80)
        logger.info("✅ All services initialized successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        logger.error("Application will start but functionality may be limited")
        logger.error("=" * 80)
        import traceback
        traceback.print_exc()


async def shutdown_event():
    """Cleanup on application shutdown"""

    logger.info("=" * 80)
    logger.info("🛑 Shutting down gracefully...")

    # ── Clear document processor cache ────────────────────────
    doc_processor = get_doc_processor()
    if doc_processor and hasattr(doc_processor, 'semantic_chunker'):
        if hasattr(doc_processor.semantic_chunker, 'clear_cache'):
            doc_processor.semantic_chunker.clear_cache()
            logger.info("✓ Document processor cache cleared")

    # ── Clear embedding manager cache ──────────────────────────
    embedding_manager = get_embedding_manager()
    if embedding_manager:
        embedding_manager.clear_cache()
        logger.info("✓ Embedding manager cache cleared")

    # ── Close Qdrant connection ────────────────────────────────
    # ← CHANGED: was get_weaviate_client() + weaviate_client.close()
    vector_store = get_vector_store()
    if vector_store:
        vector_store.close()
        logger.info("✓ Qdrant connection closed")

    logger.info("✓ Shutdown complete")
    logger.info("=" * 80)