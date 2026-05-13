"""
Retrieval Routes
Advanced RAG retrieval endpoints

Updated to use new pipeline architecture
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.api.dependencies import (
    get_rag_service,
    get_vector_store
)

router = APIRouter()
logger = logging.getLogger(__name__)


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
    
    # ================================================================
    # Get services from new pipeline
    # ================================================================
    rag_service = get_rag_service()
    vector_store = get_vector_store()
    
    if not rag_service or not vector_store:
        return JSONResponse({
            "success": False,
            "error": "Retrieval service not available"
        }, status_code=503)
    
    try:
        logger.info(f"🔍 Retrieve query: '{query[:100]}...'")
        logger.info(f"   top_k: {top_k}, min_score: {min_score}")
        
        # ================================================================
        # Use RAG service's embedded RetrievalService
        # ================================================================
        retrieval_service = rag_service.retrieval_service
        
        # Perform retrieval
        result = retrieval_service.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score
        )
        
        # ================================================================
        # Format response
        # ================================================================
        return JSONResponse({
            "success": True,
            "query": result.query,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "score": chunk.score,
                    "similarity": chunk.score,
                    "source": chunk.source,
                    "rank": chunk.rank,
                    "metadata": chunk.metadata,
                    "preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                }
                for chunk in result.chunks
            ],
            "context": result.context,
            "metadata": {
                "total_found": result.total_found,
                "retrieval_time": result.retrieval_time,
                "chunks_returned": len(result.chunks),
                "top_k": top_k,
                "min_score": min_score,
                "enable_reranking": result.metadata.get('reranked', False),
                **result.metadata
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Retrieval error: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.get("/retrieve/config")
async def get_retrieval_config():
    """
    Get current retrieval configuration
    
    **Response:**
    ```json
    {
        "top_k": 5,
        "min_score": 0.5,
        "enable_reranking": true,
        "diversity_weight": 0.3,
        "recency_weight": 0.1
    }
    ```
    """
    
    rag_service = get_rag_service()
    
    if not rag_service:
        return JSONResponse({
            "success": False,
            "error": "RAG service not available"
        }, status_code=503)
    
    retrieval_config = rag_service.retrieval_service.config
    
    return JSONResponse({
        "success": True,
        "config": {
            "top_k": retrieval_config.top_k,
            "min_score": retrieval_config.min_score,
            "max_context_length": retrieval_config.max_context_length,
            "enable_reranking": retrieval_config.enable_reranking,
            "diversity_weight": retrieval_config.diversity_weight,
            "recency_weight": retrieval_config.recency_weight
        }
    })


@router.post("/retrieve/test")
async def test_retrieval(
    query: str = Query("What is machine learning?", description="Test query")
):
    """
    Test retrieval with a sample query
    
    Returns retrieval results to verify the system is working
    """
    
    logger.info(f"🧪 Testing retrieval with: '{query}'")
    
    return await retrieve(
        query=query,
        top_k=3,
        min_score=0.5,
        enable_reranking=True
    )


@router.post("/retrieve/by-file")
async def retrieve_by_file(
    filename: str = Query(..., description="Filename to retrieve from"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum chunks to return")
):
    """
    Retrieve all chunks from a specific file
    
    **Response:**
    ```json
    {
        "success": true,
        "filename": "document.pdf",
        "chunks": [
            {
                "chunk_id": "...",
                "text": "...",
                "chunk_index": 0,
                "total_chunks": 10
            }
        ]
    }
    ```
    """
    
    vector_store = get_vector_store()
    
    if not vector_store:
        return JSONResponse({
            "success": False,
            "error": "Vector store not available"
        }, status_code=503)
    
    try:
        logger.info(f"📄 Retrieving chunks from file: {filename}")
        
        # Use VectorStore's search_by_filename method
        chunks = vector_store.search_by_filename(
            filename=filename,
            limit=limit
        )
        
        return JSONResponse({
            "success": True,
            "filename": filename,
            "chunks": [
                {
                    "chunk_id": chunk['chunk_id'],
                    "text": chunk['text'],
                    "chunk_index": chunk.get('chunk_index', 0),
                    "total_chunks": chunk.get('metadata', {}).get('total_chunks', 0),
                    "metadata": chunk.get('metadata', {})
                }
                for chunk in chunks
            ],
            "total": len(chunks)
        })
        
    except Exception as e:
        logger.error(f"❌ Error retrieving by file: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.get("/retrieve/random-samples")
async def get_random_samples(
    count: int = Query(5, ge=1, le=50, description="Number of samples to return")
):
    """
    Get random sample documents from vector store
    
    Useful for testing and debugging
    
    **Response:**
    ```json
    {
        "success": true,
        "samples": [
            {
                "chunk_id": "...",
                "text": "...",
                "filename": "...",
                "preview": "..."
            }
        ]
    }
    ```
    """
    
    vector_store = get_vector_store()
    
    if not vector_store:
        return JSONResponse({
            "success": False,
            "error": "Vector store not available"
        }, status_code=503)
    
    try:
        logger.info(f"🎲 Getting {count} random samples")
        
        samples = vector_store.get_random_samples(count=count)
        
        return JSONResponse({
            "success": True,
            "samples": samples,
            "total": len(samples)
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting samples: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)