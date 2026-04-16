"""
RAG Routes
Complete RAG endpoints with streaming support
"""

import logging
from typing import Optional, List, Dict

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
import json

from app.api.dependencies import get_rag_service, get_lm_studio_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ask")
async def ask_question(request: Request):
    """
    Ask a question using RAG (Retrieval-Augmented Generation)
    
    **Request Body:**
    ```json
    {
        "question": "What is machine learning?",
        "conversation_history": [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
    }
    ```
    
    **Response:**
    ```json
    {
        "success": true,
        "answer": "Generated answer...",
        "sources": [
            {
                "source": "document.pdf",
                "relevance": 0.89,
                "text_preview": "..."
            }
        ],
        "metadata": {
            "retrieval_time": 0.245,
            "chunks_found": 5,
            "model": "gemma-2b"
        }
    }
    ```
    """
    
    rag_service = get_rag_service()
    
    if not rag_service:
        return JSONResponse({
            "success": False,
            "error": "RAG service not available. Make sure LM Studio is running."
        }, status_code=503)
    
    try:
        # Parse request
        data = await request.json()
        question = data.get('question', '')
        conversation_history = data.get('conversation_history', [])
        
        if not question:
            return JSONResponse({
                "success": False,
                "error": "Question is required"
            }, status_code=400)
        
        logger.info(f"💬 Question: '{question[:100]}...'")
        
        # Ask using RAG
        result = rag_service.ask(
            question=question,
            conversation_history=conversation_history
        )
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"❌ Error in ask endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.post("/ask/stream")
async def ask_question_stream(request: Request):
    """
    Ask a question with streaming response
    
    Returns Server-Sent Events (SSE) stream
    
    **Request Body:**
    ```json
    {
        "question": "What is machine learning?",
        "conversation_history": []
    }
    ```
    
    **Response Stream:**
    ```
    data: {"type": "sources", "content": [...]}
    
    data: {"type": "text", "content": "Machine"}
    
    data: {"type": "text", "content": " learning"}
    
    data: {"type": "metadata", "content": {...}}
    
    data: [DONE]
    ```
    """
    
    rag_service = get_rag_service()
    
    if not rag_service:
        return JSONResponse({
            "success": False,
            "error": "RAG service not available"
        }, status_code=503)
    
    try:
        # Parse request
        data = await request.json()
        question = data.get('question', '')
        conversation_history = data.get('conversation_history', [])
        
        if not question:
            return JSONResponse({
                "success": False,
                "error": "Question is required"
            }, status_code=400)
        
        logger.info(f"💬 Streaming question: '{question[:100]}...'")
        
        # Generator function for SSE
        async def event_generator():
            try:
                for chunk in rag_service.ask_stream(
                    question=question,
                    conversation_history=conversation_history
                ):
                    # Format as SSE
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                
                # Send completion signal
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Error in stream: {e}")
                error_chunk = {
                    "type": "error",
                    "content": str(e)
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error in streaming endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.get("/rag/status")
async def get_rag_status():
    """
    Get RAG system status
    
    **Response:**
    ```json
    {
        "rag_available": true,
        "lm_studio_available": true,
        "retrieval_available": true,
        "model": "gemma-2b",
        "lm_studio_url": "http://localhost:1234"
    }
    ```
    """
    
    rag_service = get_rag_service()
    lm_studio_service = get_lm_studio_service()
    
    status = {
        "rag_available": rag_service is not None,
        "lm_studio_available": lm_studio_service is not None and lm_studio_service.is_available() if lm_studio_service else False,
        "retrieval_available": rag_service is not None and rag_service.retrieval_service is not None
    }
    
    if lm_studio_service:
        status["model"] = lm_studio_service.config.model
        status["lm_studio_url"] = lm_studio_service.config.base_url
        status["temperature"] = lm_studio_service.config.temperature
        status["max_tokens"] = lm_studio_service.config.max_tokens
    
    return JSONResponse(status)


@router.post("/rag/test")
async def test_rag():
    """
    Test RAG system with a sample question
    
    **Response:**
    Complete RAG response with answer and sources
    """
    
    test_question = "Qu'est-ce que le machine learning?"
    
    return await ask_question(Request(
        scope={
            "type": "http",
            "method": "POST",
            "headers": [],
        },
        receive=lambda: {"body": json.dumps({"question": test_question}).encode()}
    ))