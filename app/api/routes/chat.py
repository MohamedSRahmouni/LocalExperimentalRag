"""
Chat Routes
Handles chat/search queries using semantic search
"""

import os
import logging
import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.dependencies import (
    get_doc_processor,
    get_search_engine,
    get_weaviate_client
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat")
async def chat(request: Request):
    """
    Chat endpoint with Weaviate semantic search
    
    Workflow:
    1. Try Weaviate semantic search (if connected)
    2. Fallback to local JSON search
    """
    
    try:
        data = await request.json()
        user_message = data.get('message', '')
        
        if not user_message:
            return JSONResponse({
                "success": False,
                "message": "Empty message"
            }, status_code=400)
        
        logger.info(f"💬 Chat query: {user_message[:100]}...")
        
        # Get services
        search_engine = get_search_engine()
        weaviate_client = get_weaviate_client()
        doc_processor = get_doc_processor()
        
        # ====================================================================
        # OPTION 1: Use Weaviate for search (preferred)
        # ====================================================================
        if search_engine and weaviate_client and weaviate_client.is_connected():
            logger.info("🔍 Using Weaviate for semantic search...")
            
            # Get embedder
            embedder = None
            if hasattr(doc_processor, 'semantic_chunker') and doc_processor.semantic_chunker.is_model_available():
                embedder = doc_processor.semantic_chunker.embedder
            
            if embedder:
                try:
                    # Perform semantic search
                    similar_chunks = search_engine.semantic_search(
                        query_text=user_message,
                        embedder=embedder,
                        top_k=5,
                        min_score=0.5
                    )
                    
                    if similar_chunks:
                        response_text = _format_search_results(user_message, similar_chunks)
                        
                        logger.info(f"✅ Found {len(similar_chunks)} relevant chunks from Weaviate")
                        
                        return JSONResponse({
                            "success": True,
                            "message": response_text,
                            "relevant_chunks": similar_chunks,
                            "model": settings.EMBEDDING_MODEL,
                            "device": "cpu",
                            "source": "weaviate",
                            "total_searched": "vector_database"
                        })
                    else:
                        return JSONResponse({
                            "success": True,
                            "message": "No relevant chunks found in the vector database. Try rephrasing your question or upload more documents.",
                            "relevant_chunks": []
                        })
                        
                except Exception as e:
                    logger.error(f"❌ Weaviate search error: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.warning("Embedder not available for search")
        
        # ====================================================================
        # OPTION 2: Fallback to local JSON search
        # ====================================================================
        logger.info("Using local JSON fallback for search...")
        processed_data_path = os.path.join(
            settings.PROCESSED_FOLDER,
            "processed_chunks_with_embeddings.json"
        )
        
        if not os.path.exists(processed_data_path):
            return JSONResponse({
                "success": True,
                "message": "No documents have been processed yet. Please upload documents first.",
                "relevant_chunks": []
            })
        
        # Load chunks from local storage
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        
        chunks = []
        for doc in data_json.get('documents', []):
            if doc.get('embedding_complete'):
                for chunk_data in doc.get('embedded_chunks', []):
                    chunks.append(chunk_data['text'])
        
        logger.info(f"📚 Loaded {len(chunks)} chunks from local storage")
        
        # Use semantic search with local chunks
        if doc_processor and hasattr(doc_processor, 'semantic_chunker'):
            if doc_processor.semantic_chunker.is_model_available():
                similar_chunks = doc_processor.semantic_chunker.get_most_similar_chunks(
                    query=user_message,
                    chunks=chunks[:200],  # Limit to first 200 chunks
                    top_k=5
                )
                
                if similar_chunks:
                    response_text = _format_search_results(user_message, similar_chunks, source="local")
                    
                    return JSONResponse({
                        "success": True,
                        "message": response_text,
                        "relevant_chunks": similar_chunks,
                        "source": "local_storage"
                    })
        
        # Default response
        return JSONResponse({
            'success': True,
            'message': f'Query received: {user_message}. Processing...',
            'relevant_chunks': []
        })
    
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)


def _format_search_results(query: str, chunks: list, source: str = "vector database") -> str:
    """Format search results for display"""
    
    response_text = f"Based on your query **'{query}'** (from {source}):\n\n"
    
    for i, chunk_info in enumerate(chunks, 1):
        similarity = chunk_info.get('similarity', 0)
        response_text += f"**Result {i}** (Similarity: {similarity:.2%})\n"
        
        # Add source filename if available
        metadata = chunk_info.get('metadata', {})
        if metadata.get('filename'):
            response_text += f"*Source: {metadata['filename']}*\n"
        
        # Add preview
        preview = chunk_info.get('preview', chunk_info.get('text', ''))
        response_text += f"{preview}\n\n"
        response_text += "---\n\n"
    
    return response_text