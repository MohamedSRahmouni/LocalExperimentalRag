"""
Retrieval Routes
Advanced RAG retrieval endpoints
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.api.dependencies import (
    get_search_engine,
    get_doc_processor
)
from app.services.retrieval.retrieval_service import RetrievalService, RetrievalConfig

router = APIRouter()
logger = logging.getLogger(__name__)

# Global retrieval service (initialized lazily)
_retrieval_service: Optional[RetrievalService] = None


def get_retrieval_service() -> Optional[RetrievalService]:
    """Get or create retrieval service"""
    global _retrieval_service
    
    if _retrieval_service is not None:
        return _retrieval_service
    
    # Initialize
    search_engine = get_search_engine()
    doc_processor = get_doc_processor()
    
    if not search_engine:
        logger.error("Search engine not available")
        return None
    
    # Get embedder
    embedder = None
    if doc_processor and hasattr(doc_processor, 'semantic_chunker'):
        if doc_processor.semantic_chunker.is_model_available():
            embedder = doc_processor.semantic_chunker.embedder
    
    if not embedder:
        logger.error("Embedder not available")
        return None
    
    # Create service
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
    
    return _retrieval_service


@router.post("/retrieve")
async def retrieve(
    query: str = Query(..., description="User query"),
    top_k: Optional[int] = Query(5, ge=1, le=20, description="Number of chunks to retrieve"),
    min_score: Optional[float] = Query(0.5, ge=0.0, le=1.0, description="Minimum similarity score"),
    enable_reranking: Optional[bool] = Query(True, description="Enable re-ranking")
):
    """
    Retrieve relevant chunks for a query
    
    Returns context ready for LLM consumption
    
    **Response format:**
    ```json
    {
        "success": true,
        "query": "user question",
        "chunks": [
            {
                "chunk_id": "abc123",
                "text": "chunk content",
                "score": 0.85,
                "source": "document.pdf",
                "rank": 1
            }
        ],
        "context": "formatted context for LLM",
        "metadata": {
            "total_found": 10,
            "retrieval_time": 0.123,
            "top_k": 5
        }
    }
    ```
    """
    
    retrieval_service = get_retrieval_service()
    
    if not retrieval_service:
        return JSONResponse({
            "success": False,
            "error": "Retrieval service not available"
        }, status_code=503)
    
    try:
        # Override config if needed
        if not enable_reranking:
            retrieval_service.config.enable_reranking = False
        
        # Retrieve
        result = retrieval_service.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score
        )
        
        # Format response
        return JSONResponse({
            "success": True,
            "query": result.query,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "score": chunk.score,
                    "source": chunk.source,
                    "rank": chunk.rank,
                    "metadata": chunk.metadata
                }
                for chunk in result.chunks
            ],
            "context": result.context,
            "metadata": {
                "total_found": result.total_found,
                "retrieval_time": result.retrieval_time,
                "chunks_returned": len(result.chunks),
                **result.metadata
            }
        })
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.get("/retrieve/config")
async def get_retrieval_config():
    """Get current retrieval configuration"""
    
    retrieval_service = get_retrieval_service()
    
    if not retrieval_service:
        return JSONResponse({
            "success": False,
            "error": "Retrieval service not available"
        }, status_code=503)
    
    return JSONResponse(retrieval_service.get_retrieval_stats())


@router.post("/retrieve/test")
async def test_retrieval():
    """Test retrieval with a sample query"""
    
    return await retrieve(
        query="What is machine learning?",
        top_k=3,
        min_score=0.5,
        enable_reranking=True
    )