"""
Chat Routes
Enhanced chat with full RAG support
"""

import os
import logging
import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.dependencies import get_rag_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat")
async def chat(request: Request):
    """
    Chat endpoint with full RAG support
    
    Uses LM Studio if available, falls back to retrieval-only
    
    **Request Body:**
    ```json
    {
        "message": "User question",
        "history": []
    }
    ```
    """
    
    try:
        data = await request.json()
        user_message = data.get('message', '')
        conversation_history = data.get('history', [])
        
        if not user_message:
            return JSONResponse({
                "success": False,
                "message": "Empty message"
            }, status_code=400)
        
        logger.info(f"💬 Chat query: {user_message[:100]}...")
        
        # ================================================================
        # OPTION 1: Use full RAG with LM Studio
        # ================================================================
        rag_service = get_rag_service()
        
        if rag_service:
            logger.info("🤖 Using full RAG with LM Studio...")
            
            try:
                result = rag_service.ask(
                    question=user_message,
                    conversation_history=conversation_history
                )
                
                if result['success']:
                    # Format response for chat UI
                    response_text = result['answer']
                    
                    # Add sources at the end
                    if result.get('sources'):
                        response_text += "\n\n---\n**Sources:**\n"
                        for source in result['sources']:
                            response_text += f"- {source['source']} (pertinence: {source['relevance']*100:.0f}%)\n"
                    
                    return JSONResponse({
                        "success": True,
                        "message": response_text,
                        "answer": result['answer'],
                        "sources": result['sources'],
                        "metadata": result['metadata'],
                        "mode": "rag"
                    })
                else:
                    # RAG failed, fall through to retrieval-only
                    logger.warning("RAG failed, falling back to retrieval-only")
            
            except Exception as e:
                logger.error(f"RAG error: {e}")
                # Fall through to retrieval-only
        
        # ================================================================
        # OPTION 2: Fallback - Retrieval-only (no generation)
        # ================================================================
        logger.info("📚 Using retrieval-only mode (LM Studio not available)...")
        
        # Import fallback dependencies
        from app.api.dependencies import get_search_engine, get_doc_processor
        
        search_engine = get_search_engine()
        doc_processor = get_doc_processor()
        
        if not search_engine:
            return JSONResponse({
                "success": False,
                "message": "Search engine not available"
            }, status_code=503)
        
        # Get embedder
        embedder = None
        if doc_processor and hasattr(doc_processor, 'semantic_chunker'):
            if doc_processor.semantic_chunker.is_model_available():
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
                    response_text = f"Voici les informations pertinentes que j'ai trouvées (mode retrieval-only - LM Studio non disponible):\n\n"
                    
                    for i, chunk_info in enumerate(similar_chunks, 1):
                        response_text += f"**Résultat {i}** (Similarité: {chunk_info['similarity']:.2%})\n"
                        if chunk_info.get('metadata', {}).get('filename'):
                            response_text += f"*Source: {chunk_info['metadata']['filename']}*\n"
                        response_text += f"{chunk_info['preview']}\n\n"
                        response_text += "---\n\n"
                    
                    response_text += "\n💡 *Note: Pour obtenir une réponse générée, démarrez LM Studio.*"
                    
                    return JSONResponse({
                        "success": True,
                        "message": response_text,
                        "relevant_chunks": similar_chunks,
                        "mode": "retrieval-only",
                        "warning": "LM Studio not available"
                    })
                else:
                    return JSONResponse({
                        "success": True,
                        "message": "Aucune information pertinente trouvée dans les documents.",
                        "relevant_chunks": []
                    })
                    
            except Exception as e:
                logger.error(f"Search error: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # ================================================================
        # OPTION 3: Last resort - Local JSON fallback
        # ================================================================
        logger.info("📄 Using local JSON fallback...")
        processed_data_path = os.path.join(
            settings.PROCESSED_FOLDER,
            "processed_chunks_with_embeddings.json"
        )
        
        if not os.path.exists(processed_data_path):
            return JSONResponse({
                "success": True,
                "message": "Aucun document n'a été traité. Veuillez d'abord uploader des documents.",
                "relevant_chunks": []
            })
        
        return JSONResponse({
            'success': True,
            'message': f'Question reçue: {user_message}. Système en mode dégradé.',
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