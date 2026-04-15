"""
Statistics Routes
Provides system statistics and information
"""

import os
import logging
import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.dependencies import (
    get_doc_processor,
    get_embedding_manager,
    get_vector_store,
    get_weaviate_client
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/stats")
async def get_stats():
    """Get processing and embedding statistics"""
    
    try:
        # Count uploaded files
        upload_count = len(list(Path(settings.UPLOAD_FOLDER).glob("*")))
        
        # Get services
        doc_processor = get_doc_processor()
        embedding_manager = get_embedding_manager()
        weaviate_client = get_weaviate_client()
        vector_store = get_vector_store()
        
        # Base stats
        stats = {
            "uploaded_files": upload_count,
            "processor_available": doc_processor is not None,
            "embedding_manager_available": embedding_manager is not None,
            "weaviate_connected": weaviate_client.is_connected() if weaviate_client else False,
            "embedding_model": settings.EMBEDDING_MODEL,
            "device": "cpu",
            "chunking_method": settings.CHUNKING_METHOD
        }
        
        # Document processor stats
        if doc_processor:
            processor_stats = doc_processor.get_processing_stats()
            stats.update({
                "processed_documents": processor_stats.get('unique_documents', 0),
                "scanned_documents": processor_stats.get('scanned_documents', 0),
                "ocr_operations": processor_stats.get('ocr_operations', 0),
                "semantic_model_available": processor_stats.get('semantic_model_available', False)
            })
        
        # Embedding manager stats
        if embedding_manager:
            embedding_info = embedding_manager.get_system_info()
            stats.update({
                "embedding_dimension": embedding_info.get('embedding_dim'),
                "embedding_ready": embedding_info.get('is_ready', False),
                "embedding_model_type": embedding_info.get('model_type')
            })
        
        # Weaviate stats
        if vector_store and weaviate_client and weaviate_client.is_connected():
            vector_stats = vector_store.get_stats()
            stats.update({
                "vector_db_chunks": vector_stats.get('document_count', 0),
                "vector_db_collection": settings.WEAVIATE_CLASS_NAME
            })
        
        # Load embedded data if available
        processed_data_path = os.path.join(
            settings.PROCESSED_FOLDER,
            "processed_chunks_with_embeddings.json"
        )
        if os.path.exists(processed_data_path):
            with open(processed_data_path, 'r') as f:
                data = json.load(f)
                stats['total_embedded_chunks'] = data.get('session_info', {}).get('total_chunks', 0)
        
        return JSONResponse(stats)
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/model-info")
async def model_info():
    """Get detailed model information"""
    
    try:
        doc_processor = get_doc_processor()
        embedding_manager = get_embedding_manager()
        weaviate_client = get_weaviate_client()
        
        info = {
            "embedding_model": settings.EMBEDDING_MODEL,
            "model_type": "sentence-transformer",
            "device": "cpu",
            "chunking_method": settings.CHUNKING_METHOD,
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
            "min_chunk_size": settings.MIN_CHUNK_SIZE,
            "max_chunk_size": settings.MAX_CHUNK_SIZE,
            "embedding_batch_size": settings.EMBEDDING_BATCH_SIZE,
            "parallel_embedding": settings.ENABLE_PARALLEL_EMBEDDING,
            "ocr_language": settings.OCR_LANGUAGE,
            "vector_db": "weaviate",
            "weaviate_class": settings.WEAVIATE_CLASS_NAME,
            "weaviate_connected": weaviate_client.is_connected() if weaviate_client else False
        }
        
        # Chunker info
        if doc_processor and hasattr(doc_processor, 'semantic_chunker'):
            chunker = doc_processor.semantic_chunker
            info.update({
                "chunker_available": chunker.is_model_available(),
                "chunker_embedding_dimension": getattr(chunker, 'embedding_dim', None),
                "chunker_model": getattr(chunker, 'model_name', 'unknown')
            })
        
        # Embedding manager info
        if embedding_manager:
            emb_info = embedding_manager.get_system_info()
            info.update({
                "embedding_manager_ready": emb_info.get('is_ready', False),
                "embedding_dimension": emb_info.get('embedding_dim'),
                "embedding_model_type": emb_info.get('model_type')
            })
        
        return JSONResponse(info)
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)