"""
Application Startup and Shutdown Logic
Handles service initialization and cleanup
"""

import logging
from app.core.config import settings
from app.api.dependencies import (
    initialize_document_processor,
    initialize_embedding_manager,
    initialize_weaviate,
    get_doc_processor,
    get_embedding_manager,
    get_weaviate_client
)

logger = logging.getLogger(__name__)


async def startup_event():
    """Initialize all services on application startup"""
    
    logger.info("="*80)
    logger.info("🚀 Initializing Enterprise RAG System")
    logger.info("="*80)
    
    # Log configuration
    logger.info(f"📝 OCR Language: {settings.OCR_LANGUAGE}")
    logger.info(f"💻 Device: CPU (forced)")
    logger.info(f"🤖 Embedding Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"✂️  Chunking Method: {settings.CHUNKING_METHOD}")
    logger.info(f"📊 Similarity Threshold: {settings.SIMILARITY_THRESHOLD}")
    logger.info(f"📏 Chunk Size Range: {settings.MIN_CHUNK_SIZE} - {settings.MAX_CHUNK_SIZE}")
    logger.info(f"📦 Embedding Batch Size: {settings.EMBEDDING_BATCH_SIZE}")
    logger.info(f"⚡ Parallel Embedding: {settings.ENABLE_PARALLEL_EMBEDDING}")
    logger.info("="*80)
    
    # Initialize services
    try:
        # Document processor
        initialize_document_processor()
        
        # Embedding manager
        initialize_embedding_manager()
        
        # Weaviate vector database
        initialize_weaviate()
        
        logger.info("="*80)
        logger.info("✅ All services initialized successfully!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error("="*80)
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        logger.error("Application will start but functionality may be limited")
        logger.error("="*80)
        import traceback
        traceback.print_exc()


async def shutdown_event():
    """Cleanup on application shutdown"""
    
    logger.info("="*80)
    logger.info("🛑 Shutting down gracefully...")
    
    # Clear document processor cache
    doc_processor = get_doc_processor()
    if doc_processor and hasattr(doc_processor, 'semantic_chunker'):
        if hasattr(doc_processor.semantic_chunker, 'clear_cache'):
            doc_processor.semantic_chunker.clear_cache()
            logger.info("✓ Document processor cache cleared")
    
    # Clear embedding manager cache
    embedding_manager = get_embedding_manager()
    if embedding_manager:
        embedding_manager.clear_cache()
        logger.info("✓ Embedding manager cache cleared")
    
    # Close Weaviate connection
    weaviate_client = get_weaviate_client()
    if weaviate_client:
        weaviate_client.close()
        logger.info("✓ Weaviate connection closed")
    
    logger.info("✓ Shutdown complete")
    logger.info("="*80)