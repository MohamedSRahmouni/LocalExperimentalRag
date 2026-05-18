"""
Upload Routes
Handles file upload, processing, embedding, and storage
"""

import os
import logging
import shutil
import json
from pathlib import Path
from typing import List
from datetime import datetime

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from app.api.dependencies import update_weaviate_metrics

from app.core.config import settings
from app.api.dependencies import (
    get_doc_processor,
    get_embedding_manager,
    get_vector_store,
    get_weaviate_client
)
from app.utils.file_utils import allowed_file

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload, process, embed, and store files in Weaviate
    
    Pipeline:
    1. Save uploaded files
    2. Process documents (extract text, clean, chunk)
    3. Embed chunks
    4. Store in Weaviate vector database
    5. Save local backup
    """
    
    # Get services
    doc_processor = get_doc_processor()
    embedding_manager = get_embedding_manager()
    vector_store = get_vector_store()
    weaviate_client = get_weaviate_client()
    
    # Check if services are available
    if doc_processor is None or embedding_manager is None:
        logger.error("Services not initialized")
        return JSONResponse({
            "success": False,
            "message": "Services not initialized. Please check server logs."
        }, status_code=503)
    
    uploaded_files = []
    file_paths = []
    
    logger.info(f"📤 Received {len(files)} file(s) for upload")
    
    # ========================================================================
    # STEP 0: Save uploaded files
    # ========================================================================
    for file in files:
        if file.filename and allowed_file(file.filename):
            filename = Path(file.filename).name
            file_path = os.path.join(settings.UPLOAD_FOLDER, filename)
            
            try:
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                uploaded_files.append(filename)
                file_paths.append(file_path)
                logger.info(f"✅ Saved file: {filename}")
            except Exception as e:
                logger.error(f"❌ Failed to save file {filename}: {str(e)}")
                continue
        else:
            logger.warning(f"⚠️  Skipped invalid file: {file.filename}")
    
    if not uploaded_files:
        return JSONResponse({
            "success": False,
            "message": "No valid files uploaded"
        }, status_code=400)
    
    # ========================================================================
    # PIPELINE: Process → Embed → Store
    # ========================================================================
    try:
        logger.info("="*80)
        logger.info(f"🔄 Starting document processing pipeline")
        logger.info(f"📁 Files to process: {len(file_paths)}")
        logger.info(f"🤖 Model: {settings.EMBEDDING_MODEL}")
        logger.info("="*80)
        
        # ====================================================================
        # STEP 1: Process documents (extract text, clean, chunk)
        # ====================================================================
        logger.info("STEP 1: Processing documents...")
        processing_results = doc_processor.process_multiple_files(
            file_paths,
            similarity_threshold=settings.SIMILARITY_THRESHOLD,
            min_chunk_size=settings.MIN_CHUNK_SIZE,
            max_chunk_size=settings.MAX_CHUNK_SIZE,
            dynamic_threshold=True
        )
        
        logger.info(f"✅ Processing complete: {processing_results['successful']}/{len(file_paths)} files")
        
        # ====================================================================
        # STEP 2: Embed processed documents
        # ====================================================================
        logger.info("="*80)
        logger.info("STEP 2: Embedding documents...")
        logger.info("="*80)
        
        # Get successfully processed documents
        successful_docs = [
            doc for doc in processing_results['files']
            if doc.get('success', False)
        ]
        
        embedding_results = {
            "successful": 0,
            "failed": 0,
            "total_chunks_embedded": 0,
            "statistics": {},
            "documents": []
        }
        
        if successful_docs:
            embedding_results = embedding_manager.embed_multiple_documents(
                successful_docs,
                parallel=settings.ENABLE_PARALLEL_EMBEDDING,
                show_progress=True
            )
            
            logger.info(f"✅ Embedding complete: {embedding_results['successful']}/{len(successful_docs)} documents")
            logger.info(f"📦 Total chunks embedded: {embedding_results['total_chunks_embedded']}")
            
            # ================================================================
            # STEP 3: Store in Weaviate Vector Database
            # ================================================================
            logger.info("="*80)
            logger.info("STEP 3: Storing embeddings in Weaviate...")
            logger.info("="*80)
            
            vector_stats = {
                "documents_processed": 0,
                "chunks_stored": 0,
                "chunks_failed": 0
            }
            
            if vector_store and weaviate_client and weaviate_client.is_connected():
                logger.info(f"✓ Vector store available: True")
                logger.info(f"✓ Weaviate connected: True")
                
                # Get embedded documents
                embedded_docs = [
                    doc for doc in embedding_results['documents']
                    if doc.get('embedding_complete', False)
                ]
                
                logger.info(f"📊 Documents with embedding_complete=True: {len(embedded_docs)}")
                
                if embedded_docs:
                    logger.info(f"📦 Storing {len(embedded_docs)} documents in Weaviate")
                    
                    vector_stats = vector_store.store_batch(embedded_docs)
                    update_weaviate_metrics()

                    logger.info(f"✅ Storage complete - Stored: {vector_stats['chunks_stored']} chunks")
                else:
                    logger.warning("❌ No embedded documents to store in Weaviate")
            else:
                logger.error("❌ Vector store not available!")
            
            # ================================================================
            # STEP 4: Save local backup
            # ================================================================
            _save_local_backup(embedding_results, uploaded_files, vector_stats)
            
        else:
            logger.warning("No successful documents to embed")
            vector_stats = {
                "documents_processed": 0,
                "chunks_stored": 0,
                "chunks_failed": 0
            }
        
        # ====================================================================
        # Cleanup uploaded files
        # ====================================================================
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {file_path}: {e}")
        
        # ====================================================================
        # Final summary
        # ====================================================================
        logger.info("="*80)
        logger.info("✅ PIPELINE COMPLETE!")
        logger.info(f"  📄 Documents processed: {processing_results['successful']}")
        logger.info(f"  🔢 Documents embedded: {embedding_results['successful']}")
        logger.info(f"  📦 Total chunks embedded: {embedding_results['total_chunks_embedded']}")
        logger.info(f"  🗄️  Chunks stored in Weaviate: {vector_stats['chunks_stored']}")
        logger.info(f"  📄 Scanned docs: {processing_results.get('scanned_count', 0)}")
        logger.info(f"  ⏱️  Total time: {processing_results.get('processing_time', 0):.2f}s")
        logger.info("="*80)
        
        return JSONResponse({
            "success": True,
            "message": f"Successfully processed, embedded, and stored {embedding_results['successful']} file(s)",
            "details": {
                "uploaded": len(uploaded_files),
                "processed_successful": processing_results['successful'],
                "processed_failed": processing_results['failed'],
                "duplicates": processing_results['duplicate'],
                "embedded_successful": embedding_results['successful'],
                "embedded_failed": embedding_results.get('failed', 0),
                "total_chunks": processing_results['total_unique_chunks'],
                "total_chunks_embedded": embedding_results['total_chunks_embedded'],
                "vector_db_stored": vector_stats['chunks_stored'],
                "vector_db_failed": vector_stats['chunks_failed'],
                "scanned_docs": processing_results.get('scanned_count', 0),
                "processing_time": f"{processing_results.get('processing_time', 0):.2f}s",
                "embedding_time": f"{embedding_results.get('processing_time', 0):.2f}s",
                "model": settings.EMBEDDING_MODEL,
                "device": "cpu",
                "chunking_method": settings.CHUNKING_METHOD,
                "processing_statistics": processing_results.get('statistics', {}),
                "embedding_statistics": embedding_results.get('statistics', {}),
                "vector_statistics": vector_stats
            },
            "files": uploaded_files
        })
    
    except Exception as e:
        logger.error("="*80)
        logger.error(f"❌ Error in processing pipeline: {str(e)}")
        logger.error("="*80)
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except:
                pass
        
        return JSONResponse({
            "success": False,
            "message": f"Error in processing pipeline: {str(e)}"
        }, status_code=500)


def _save_local_backup(embedding_results, uploaded_files, vector_stats):
    """Save processed data as local backup"""
    
    # Save JSON backup
    processed_data_path = os.path.join(
        settings.PROCESSED_FOLDER,
        "processed_chunks_with_embeddings.json"
    )
    
    save_data = {
        "session_info": {
            "timestamp": embedding_results.get('processing_time'),
            "files": uploaded_files,
            "model": settings.EMBEDDING_MODEL,
            "device": "cpu",
            "chunking_method": settings.CHUNKING_METHOD,
            "total_documents": len([
                doc for doc in embedding_results['documents']
                if doc.get('embedding_complete', False)
            ]),
            "total_chunks": embedding_results['total_chunks_embedded'],
            "vector_db_stored": vector_stats['chunks_stored']
        },
        "documents": embedding_results['documents'],
        "statistics": embedding_results['statistics'],
        "vector_stats": vector_stats
    }
    
    with open(processed_data_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved JSON backup to: {processed_data_path}")
    
    # Save text version
    text_output_path = os.path.join(settings.PROCESSED_FOLDER, "processed_chunks.txt")
    
    with open(text_output_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Processing Session\n")
        f.write(f"{'='*80}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Files: {', '.join(uploaded_files)}\n")
        f.write(f"Model: {settings.EMBEDDING_MODEL}\n")
        f.write(f"Device: CPU\n")
        f.write(f"Chunking Method: {settings.CHUNKING_METHOD}\n")
        f.write(f"Total Chunks: {embedding_results['total_chunks_embedded']}\n")
        f.write(f"Stored in Weaviate: {vector_stats['chunks_stored']}\n")
        f.write(f"{'='*80}\n\n")
        
        for doc in embedding_results['documents']:
            if doc.get('embedding_complete'):
                for idx, chunk_data in enumerate(doc.get('embedded_chunks', []), 1):
                    f.write(f"Chunk #{idx}\n")
                    f.write(f"{'-'*80}\n")
                    f.write(f"Document: {doc.get('filename', 'unknown')}\n")
                    f.write(f"Chunk ID: {chunk_data.get('chunk_id', 'N/A')}\n")
                    f.write(f"Length: {chunk_data.get('text_length', 'N/A')} characters\n")
                    f.write(f"Embedding Dim: {chunk_data.get('embedding_dim', 'N/A')}\n")
                    f.write(f"Model: {chunk_data.get('model_type', 'N/A')}\n")
                    f.write(f"{'-'*80}\n")
                    f.write(chunk_data.get('text', '') + "\n")
                    f.write(f"{'='*80}\n\n")
    
    logger.info(f"💾 Saved text backup to: {text_output_path}")