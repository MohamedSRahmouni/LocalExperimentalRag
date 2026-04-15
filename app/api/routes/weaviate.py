"""
Weaviate Admin Routes
Provides administrative endpoints for Weaviate management
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.dependencies import (
    get_weaviate_client,
    get_vector_store
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/collections")
async def list_collections():
    """List all Weaviate collections"""
    
    weaviate_client = get_weaviate_client()
    
    if not weaviate_client or not weaviate_client.is_connected():
        raise HTTPException(status_code=503, detail="Weaviate not connected")
    
    try:
        collections = weaviate_client.client.collections.list_all()
        
        result = []
        for name in collections.keys():
            try:
                col = weaviate_client.client.collections.get(name)
                count = col.aggregate.over_all(total_count=True)
                result.append({
                    "name": name,
                    "count": count.total_count,
                    "is_target": name == settings.WEAVIATE_CLASS_NAME
                })
            except:
                result.append({
                    "name": name,
                    "count": None,
                    "is_target": name == settings.WEAVIATE_CLASS_NAME
                })
        
        return {
            "collections": result,
            "total": len(result),
            "target_collection": settings.WEAVIATE_CLASS_NAME
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections/recreate")
async def recreate_collection():
    """Delete and recreate the target collection"""
    
    weaviate_client = get_weaviate_client()
    
    if not weaviate_client or not weaviate_client.is_connected():
        raise HTTPException(status_code=503, detail="Weaviate not connected")
    
    try:
        # Delete if exists
        if weaviate_client.client.collections.exists(settings.WEAVIATE_CLASS_NAME):
            weaviate_client.client.collections.delete(settings.WEAVIATE_CLASS_NAME)
            logger.info(f"Deleted old collection: {settings.WEAVIATE_CLASS_NAME}")
        
        # Recreate
        success = weaviate_client.create_collection(
            class_name=settings.WEAVIATE_CLASS_NAME,
            description="RAG document chunks with embeddings",
            vector_dims=settings.WEAVIATE_VECTOR_DIMS
        )
        
        return {
            "success": success,
            "message": f"Collection '{settings.WEAVIATE_CLASS_NAME}' recreated"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def vector_db_stats():
    """Get Weaviate vector database statistics"""
    
    vector_store = get_vector_store()
    weaviate_client = get_weaviate_client()
    
    if not vector_store or not weaviate_client or not weaviate_client.is_connected():
        return JSONResponse({
            "success": False,
            "message": "Weaviate not connected"
        }, status_code=503)
    
    try:
        stats = vector_store.get_stats()
        
        return JSONResponse({
            "success": True,
            "collection": settings.WEAVIATE_CLASS_NAME,
            "total_chunks": stats.get('document_count', 0),
            "status": "connected"
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/test-insert")
async def test_insert():
    """Test inserting a minimal object to diagnose issues"""
    
    weaviate_client = get_weaviate_client()
    
    if not weaviate_client or not weaviate_client.is_connected():
        raise HTTPException(status_code=503, detail="Weaviate not connected")
    
    try:
        collection = weaviate_client.get_collection(settings.WEAVIATE_CLASS_NAME)
        
        if not collection:
            return {"error": "Collection not found"}
        
        # Create minimal test object
        test_vector = [0.1] * 768  # Simple test vector
        
        test_properties = {
            "chunk_id": "test_123",
            "text": "This is a test",
            "text_length": 14,
            "embedding_model": "test-model",
            "model_type": "test",
            "embedded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "indexed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "filename": "test.txt",
            "file_type": ".txt",
            "chunk_index": 0,
            "total_chunks": 1,
            "similarity_score": 0.5,
            "sentence_count": 1
        }
        
        logger.info(f"Test properties: {test_properties}")
        logger.info(f"Test vector length: {len(test_vector)}")
        
        # Try to insert
        try:
            uuid_result = collection.data.insert(
                properties=test_properties,
                vector=test_vector
            )
            
            logger.info(f"✓ Insert successful! UUID: {uuid_result}")
            
            # Verify
            count = collection.aggregate.over_all(total_count=True)
            
            return {
                "success": True,
                "uuid": str(uuid_result),
                "total_count": count.total_count,
                "message": "Test object inserted successfully"
            }
            
        except Exception as insert_error:
            logger.error(f"Insert failed: {insert_error}")
            import traceback
            error_trace = traceback.format_exc()
            logger.error(error_trace)
            
            return {
                "success": False,
                "error": str(insert_error),
                "traceback": error_trace,
                "properties_sent": test_properties
            }
        
    except Exception as e:
        logger.error(f"Test error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))