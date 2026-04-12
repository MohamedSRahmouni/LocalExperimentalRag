from http.client import HTTPException
import os
import sys
import warnings
from datetime import datetime, timezone

# ============================================================================
# WINDOWS MULTIPROCESSING FIX - MUST BE AT THE TOP
# ============================================================================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Force single-threaded numpy/BLAS operations to prevent conflicts
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')

# Suppress dependency warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables before imports
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = os.getenv('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
os.environ['KMP_DUPLICATE_LIB_OK'] = os.getenv('KMP_DUPLICATE_LIB_OK', 'TRUE')

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import shutil
from pathlib import Path
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
logging.getLogger("multipart").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

try:
    from app.services.preprocessing import DocumentProcessor
    from app.services.embedding import EmbeddingManager
    from app.data.vectordb import WeaviateClient, VectorStore, SearchEngine
    logger.info("✓ Document processor module loaded")
    logger.info("✓ Embedding manager module loaded")
    logger.info("✓ Weaviate modules loaded")
except Exception as e:
    logger.error(f"Failed to load modules: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

app = FastAPI(
    title="Enterprise RAG",
    description="Document processing and RAG system with Weaviate Vector Database",
    version="1.0.0"
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Static files directory not found: {e}")

# Templates
templates = Jinja2Templates(directory="app/components")

# Folders
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
EMBEDDINGS_FOLDER = "embeddings_output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)

# Configuration from environment variables
OCR_LANGUAGE = os.getenv('OCR_LANGUAGE', 'en')
USE_GPU = False  # Force CPU for embeddings
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
CHUNKING_METHOD = os.getenv('CHUNKING_METHOD', 'semantic')
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.5'))
MIN_CHUNK_SIZE = int(os.getenv('MIN_CHUNK_SIZE', '100'))
MAX_CHUNK_SIZE = int(os.getenv('MAX_CHUNK_SIZE', '1000'))
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '8'))
ENABLE_PARALLEL_EMBEDDING = os.getenv('ENABLE_PARALLEL_EMBEDDING', 'false').lower() == 'true'

# Weaviate Configuration
WEAVIATE_URL = os.getenv('WEAVIATE_URL')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')
WEAVIATE_CLASS_NAME = os.getenv('WEAVIATE_CLASS_NAME', 'Ragdocument')
WEAVIATE_VECTOR_DIMS = int(os.getenv('WEAVIATE_VECTOR_DIMS', '768'))
USE_EMBEDDED_WEAVIATE = os.getenv('USE_EMBEDDED_WEAVIATE', 'false').lower() == 'true'

# Global services
doc_processor = None
embedding_manager = None
weaviate_client = None
vector_store = None
search_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global doc_processor, embedding_manager, weaviate_client, vector_store, search_engine
    
    try:
        logger.info("="*80)
        logger.info("🚀 Initializing Enterprise RAG System")
        logger.info("="*80)
        logger.info(f"📝 OCR Language: {OCR_LANGUAGE}")
        logger.info(f"💻 Device: CPU (forced)")
        logger.info(f"🤖 Embedding Model: {EMBEDDING_MODEL}")
        logger.info(f"✂️  Chunking Method: {CHUNKING_METHOD}")
        logger.info(f"📊 Similarity Threshold: {SIMILARITY_THRESHOLD}")
        logger.info(f"📏 Chunk Size Range: {MIN_CHUNK_SIZE} - {MAX_CHUNK_SIZE}")
        logger.info(f"📦 Embedding Batch Size: {EMBEDDING_BATCH_SIZE}")
        logger.info(f"⚡ Parallel Embedding: {ENABLE_PARALLEL_EMBEDDING}")
        logger.info("="*80)
        
        # Initialize document processor
        logger.info("Initializing Document Processor...")
        doc_processor = DocumentProcessor(
            lang=OCR_LANGUAGE,
            use_gpu=USE_GPU,
            embedding_model=EMBEDDING_MODEL,
            chunking_method=CHUNKING_METHOD
        )
        logger.info("✅ Document processor initialized")
        
        # Initialize embedding manager
        logger.info("Initializing Embedding Manager...")
        embedding_manager = EmbeddingManager(
            model_name=EMBEDDING_MODEL,
            use_gpu=USE_GPU,
            batch_size=EMBEDDING_BATCH_SIZE,
            max_workers=2 if ENABLE_PARALLEL_EMBEDDING else 1,
            save_intermediate=True,
            output_dir=EMBEDDINGS_FOLDER
        )
        
        if embedding_manager.is_ready():
            logger.info("✅ Embedding manager initialized successfully")
            info = embedding_manager.get_system_info()
            logger.info(f"   Model: {info.get('model_name', 'unknown')}")
            logger.info(f"   Type: {info.get('model_type', 'sentence-transformer')}")
            logger.info(f"   Dimension: {info.get('embedding_dim', 'unknown')}")
            logger.info(f"   Device: {info.get('device', 'cpu')}")
        else:
            logger.warning("⚠️  Embedding manager initialized but model not ready")
        
        # Initialize Weaviate
        logger.info("Initializing Weaviate Vector Database...")
        if USE_EMBEDDED_WEAVIATE:
            logger.info("Using Weaviate Embedded (Python)")
            weaviate_client = WeaviateClient(use_embedded=True)
        else:
            logger.info(f"Connecting to Weaviate at {WEAVIATE_URL}")
            weaviate_client = WeaviateClient(
                url=WEAVIATE_URL,
                api_key=WEAVIATE_API_KEY
            )
        
        if weaviate_client.is_connected():
            logger.info("✅ Weaviate connected successfully")
            
            # List all collections
            logger.info("="*70)
            logger.info("📋 WEAVIATE COLLECTIONS")
            logger.info("="*70)
            
            try:
                collections = weaviate_client.client.collections.list_all()
                
                if collections:
                    logger.info(f"Found {len(collections)} collection(s):")
                    for name in collections.keys():
                        try:
                            col = weaviate_client.client.collections.get(name)
                            count = col.aggregate.over_all(total_count=True)
                            logger.info(f"  📦 '{name}': {count.total_count} objects")
                        except Exception as e:
                            logger.info(f"  📦 '{name}': (unable to count)")
                else:
                    logger.warning("  ⚠️  No collections found in cluster")
                
                logger.info(f"\n  🎯 Target collection: '{WEAVIATE_CLASS_NAME}'")
                if WEAVIATE_CLASS_NAME in collections:
                    logger.info(f"  ✅ Target collection exists")
                else:
                    logger.warning(f"  ⚠️  Target collection will be created")
                    
            except Exception as e:
                logger.error(f"Error listing collections: {e}")
            
            logger.info("="*70)
            
            # Initialize Vector Store
            vector_store = VectorStore(
                weaviate_client=weaviate_client,
                class_name=WEAVIATE_CLASS_NAME,
                vector_dims=WEAVIATE_VECTOR_DIMS
            )
            logger.info("✅ Vector store initialized")
            
            # Initialize Search Engine
            search_engine = SearchEngine(
                weaviate_client=weaviate_client,
                class_name=WEAVIATE_CLASS_NAME
            )
            logger.info("✅ Search engine initialized")
            
            # Get current stats
            stats = vector_store.get_stats()
            logger.info(f"   Current documents in vector DB: {stats.get('document_count', 0)}")
        else:
            logger.warning("⚠️  Weaviate not connected - vector storage disabled")
        
        # Check semantic chunker
        if hasattr(doc_processor, 'semantic_chunker') and doc_processor.semantic_chunker.is_model_available():
            logger.info(f"✅ Semantic chunking enabled")
            logger.info(f"   Model: {doc_processor.semantic_chunker.model_name}")
            logger.info(f"   Embedding dimension: {doc_processor.semantic_chunker.embedding_dim}")
        else:
            logger.warning("⚠️  Semantic chunker not available")
        
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

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("="*80)
    logger.info("Shutting down gracefully...")
    
    # Clear caches
    if doc_processor and hasattr(doc_processor, 'semantic_chunker'):
        if hasattr(doc_processor.semantic_chunker, 'clear_cache'):
            doc_processor.semantic_chunker.clear_cache()
            logger.info("✓ Document processor cache cleared")
    
    if embedding_manager:
        embedding_manager.clear_cache()
        logger.info("✓ Embedding manager cache cleared")
    
    # Close Weaviate connection
    if weaviate_client:
        weaviate_client.close()
        logger.info("✓ Weaviate connection closed")
    
    logger.info("✓ Shutdown complete")
    logger.info("="*80)

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.csv', '.txt', '.eml', '.msg',
    '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'
}

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Home page"""
    return templates.TemplateResponse("homePage/home.html", {"request": request})

@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("homePage/home.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Upload page"""
    return templates.TemplateResponse("uploadPage/upload.html", {"request": request})

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload, process, embed, and store files in Weaviate"""
    if doc_processor is None or embedding_manager is None:
        logger.error("Services not initialized")
        return JSONResponse({
            "success": False,
            "message": "Services not initialized. Please check server logs."
        }, status_code=503)
    
    uploaded_files = []
    file_paths = []
    
    logger.info(f"📤 Received {len(files)} file(s) for upload")
    
    # Save uploaded files
    for file in files:
        if file.filename and allowed_file(file.filename):
            filename = Path(file.filename).name
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
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
    
    # Process, embed, and store files
    try:
        logger.info("="*80)
        logger.info(f"🔄 Starting document processing pipeline")
        logger.info(f"📁 Files to process: {len(file_paths)}")
        logger.info(f"🤖 Model: {EMBEDDING_MODEL}")
        logger.info("="*80)
        
        # Step 1: Process documents (extract text, clean, chunk)
        logger.info("STEP 1: Processing documents...")
        processing_results = doc_processor.process_multiple_files(
            file_paths,
            similarity_threshold=SIMILARITY_THRESHOLD,
            min_chunk_size=MIN_CHUNK_SIZE,
            max_chunk_size=MAX_CHUNK_SIZE,
            dynamic_threshold=True
        )
        
        logger.info(f"✅ Processing complete: {processing_results['successful']}/{len(file_paths)} files")
        
        # Step 2: Embed processed documents
        logger.info("="*80)
        logger.info("STEP 2: Embedding documents...")
        logger.info("="*80)
        
        # Get successfully processed documents
        successful_docs = [
            doc for doc in processing_results['files']
            if doc.get('success', False)
        ]
        
        embedding_results = {"successful": 0, "failed": 0, "total_chunks_embedded": 0, "statistics": {}, "documents": []}
        
        if successful_docs:
            embedding_results = embedding_manager.embed_multiple_documents(
                successful_docs,
                parallel=ENABLE_PARALLEL_EMBEDDING,
                show_progress=True
            )
            
            logger.info(f"✅ Embedding complete: {embedding_results['successful']}/{len(successful_docs)} documents")
            logger.info(f"📦 Total chunks embedded: {embedding_results['total_chunks_embedded']}")
            
            # Step 3: Store in Weaviate Vector Database
            logger.info("="*80)
            logger.info("STEP 3: Storing embeddings in Weaviate...")
            logger.info("="*80)
            
            vector_stats = {"documents_processed": 0, "chunks_stored": 0, "chunks_failed": 0}
            
            if vector_store and weaviate_client and weaviate_client.is_connected():
                logger.info(f"✓ Vector store available: True")
                logger.info(f"✓ Weaviate connected: True")
                
                # Get embedded documents
                embedded_docs = [
                    doc for doc in embedding_results['documents']
                    if doc.get('embedding_complete', False)
                ]
                
                logger.info(f"📊 Total documents from embedding: {len(embedding_results['documents'])}")
                logger.info(f"📊 Documents with embedding_complete=True: {len(embedded_docs)}")
                
                if embedded_docs:
                    logger.info(f"📦 About to store {len(embedded_docs)} documents in Weaviate")
                    
                    # Debug: Show first document structure
                    logger.info(f"First document keys: {list(embedded_docs[0].keys())}")
                    logger.info(f"First document has 'embedded_chunks': {'embedded_chunks' in embedded_docs[0]}")
                    if 'embedded_chunks' in embedded_docs[0]:
                        logger.info(f"First document chunk count: {len(embedded_docs[0]['embedded_chunks'])}")
                        if len(embedded_docs[0]['embedded_chunks']) > 0:
                            first_chunk = embedded_docs[0]['embedded_chunks'][0]
                            logger.info(f"First chunk keys: {list(first_chunk.keys())}")
                            logger.info(f"First chunk has 'embedding': {'embedding' in first_chunk}")
                    
                    vector_stats = vector_store.store_batch(embedded_docs)
                    
                    logger.info(f"✅ Storage complete - Stats: {vector_stats}")
                    logger.info(f"✅ Stored in Weaviate: {vector_stats['chunks_stored']} chunks")
                else:
                    logger.warning("❌ No embedded documents to store in Weaviate")
                    logger.warning(f"   Total documents: {len(embedding_results['documents'])}")
                    logger.warning("   Checking why embedding_complete is False...")
                    
                    for i, doc in enumerate(embedding_results['documents']):
                        logger.warning(f"   Doc {i}: filename={doc.get('filename')}")
                        logger.warning(f"   Doc {i}: embedding_complete={doc.get('embedding_complete', 'KEY_MISSING')}")
                        logger.warning(f"   Doc {i}: has embedded_chunks={'embedded_chunks' in doc}")
                        if 'embedded_chunks' in doc:
                            logger.warning(f"   Doc {i}: chunk count={len(doc.get('embedded_chunks', []))}")
            else:
                logger.error("❌ Vector store not available!")
                logger.error(f"   vector_store is None: {vector_store is None}")
                logger.error(f"   weaviate_client is None: {weaviate_client is None}")
                if weaviate_client:
                    logger.error(f"   weaviate_client.is_connected(): {weaviate_client.is_connected()}")
            
            # Save processed chunks with embeddings (local backup)
            processed_data_path = os.path.join(PROCESSED_FOLDER, "processed_chunks_with_embeddings.json")
            
            save_data = {
                "session_info": {
                    "timestamp": embedding_results.get('processing_time'),
                    "files": uploaded_files,
                    "model": EMBEDDING_MODEL,
                    "device": "cpu",
                    "chunking_method": CHUNKING_METHOD,
                    "total_documents": len(successful_docs),
                    "total_chunks": embedding_results['total_chunks_embedded'],
                    "vector_db_stored": vector_stats['chunks_stored']
                },
                "documents": embedding_results['documents'],
                "statistics": embedding_results['statistics'],
                "vector_stats": vector_stats
            }
            
            with open(processed_data_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved local backup to: {processed_data_path}")
            
            # Also save plain text version
            text_output_path = os.path.join(PROCESSED_FOLDER, "processed_chunks.txt")
            with open(text_output_path, "a", encoding="utf-8") as f:
                from datetime import datetime
                
                f.write(f"\n{'='*80}\n")
                f.write(f"Processing Session\n")
                f.write(f"{'='*80}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Files: {', '.join(uploaded_files)}\n")
                f.write(f"Model: {EMBEDDING_MODEL}\n")
                f.write(f"Device: CPU\n")
                f.write(f"Chunking Method: {CHUNKING_METHOD}\n")
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
        else:
            logger.warning("No successful documents to embed")
            vector_stats = {"documents_processed": 0, "chunks_stored": 0, "chunks_failed": 0}
        
        # Cleanup uploaded files
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {file_path}: {e}")
        
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
                "model": EMBEDDING_MODEL,
                "device": "cpu",
                "chunking_method": CHUNKING_METHOD,
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

@app.get("/prompt", response_class=HTMLResponse)
async def prompt_page(request: Request):
    """Chat/Prompt page"""
    return templates.TemplateResponse("promptPage/prompt.html", {"request": request})

@app.post("/api/chat")
async def chat(request: Request):
    """Chat endpoint with Weaviate semantic search"""
    try:
        data = await request.json()
        user_message = data.get('message', '')
        
        if not user_message:
            return JSONResponse({
                "success": False,
                "message": "Empty message"
            }, status_code=400)
        
        logger.info(f"💬 Chat query: {user_message[:100]}...")
        
        # Use Weaviate for search if available
        if search_engine and weaviate_client.is_connected():
            logger.info("🔍 Using Weaviate for semantic search...")
            
            # Get embedder from document processor
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
                        response_text = f"Based on your query **'{user_message}'**, here are the most relevant sections from the vector database:\n\n"
                        for i, chunk_info in enumerate(similar_chunks, 1):
                            response_text += f"**Result {i}** (Similarity: {chunk_info['similarity']:.2%})\n"
                            if chunk_info.get('metadata', {}).get('filename'):
                                response_text += f"*Source: {chunk_info['metadata']['filename']}*\n"
                            response_text += f"{chunk_info['preview']}\n\n"
                            response_text += "---\n\n"
                        
                        logger.info(f"✅ Found {len(similar_chunks)} relevant chunks from Weaviate")
                        
                        return JSONResponse({
                            "success": True,
                            "message": response_text,
                            "relevant_chunks": similar_chunks,
                            "model": EMBEDDING_MODEL,
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
        
        # Fallback: Load from local JSON
        logger.info("Using local JSON fallback for search...")
        processed_data_path = os.path.join(PROCESSED_FOLDER, "processed_chunks_with_embeddings.json")
        
        if not os.path.exists(processed_data_path):
            return JSONResponse({
                "success": True,
                "message": "No documents have been processed yet. Please upload documents first.",
                "relevant_chunks": []
            })
        
        # Load and search local chunks
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
                    chunks=chunks[:200],
                    top_k=5
                )
                
                if similar_chunks:
                    response_text = f"Based on your query **'{user_message}'** (from local storage):\n\n"
                    for i, chunk_info in enumerate(similar_chunks, 1):
                        response_text += f"**Result {i}** (Similarity: {chunk_info['similarity']:.2%})\n"
                        response_text += f"{chunk_info['preview']}\n\n---\n\n"
                    
                    return JSONResponse({
                        "success": True,
                        "message": response_text,
                        "relevant_chunks": similar_chunks,
                        "source": "local_storage"
                    })
        
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

@app.get("/api/stats")
async def get_stats():
    """Get processing and embedding statistics"""
    try:
        upload_count = len(list(Path(UPLOAD_FOLDER).glob("*")))
        
        stats = {
            "uploaded_files": upload_count,
            "processor_available": doc_processor is not None,
            "embedding_manager_available": embedding_manager is not None,
            "weaviate_connected": weaviate_client.is_connected() if weaviate_client else False,
            "embedding_model": EMBEDDING_MODEL,
            "device": "cpu",
            "chunking_method": CHUNKING_METHOD
        }
        
        if doc_processor:
            processor_stats = doc_processor.get_processing_stats()
            stats.update({
                "processed_documents": processor_stats.get('unique_documents', 0),
                "scanned_documents": processor_stats.get('scanned_documents', 0),
                "ocr_operations": processor_stats.get('ocr_operations', 0),
                "semantic_model_available": processor_stats.get('semantic_model_available', False)
            })
        
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
                "vector_db_collection": WEAVIATE_CLASS_NAME
            })
        
        # Load embedded data if available
        processed_data_path = os.path.join(PROCESSED_FOLDER, "processed_chunks_with_embeddings.json")
        if os.path.exists(processed_data_path):
            with open(processed_data_path, 'r') as f:
                data = json.load(f)
                stats['total_embedded_chunks'] = data.get('session_info', {}).get('total_chunks', 0)
        
        return JSONResponse(stats)
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "processor_status": "ready" if doc_processor else "unavailable",
        "embedding_manager_status": "ready" if embedding_manager and embedding_manager.is_ready() else "unavailable",
        "weaviate_status": "connected" if weaviate_client and weaviate_client.is_connected() else "disconnected",
        "embedding_model": EMBEDDING_MODEL,
        "model_type": "sentence-transformer",
        "device": "cpu"
    }
    
    if doc_processor and hasattr(doc_processor, 'semantic_chunker'):
        health_status["embedder_status"] = "ready" if doc_processor.semantic_chunker.is_model_available() else "unavailable"
    
    return JSONResponse(health_status)

@app.get("/api/model-info")
async def model_info():
    """Get detailed model information"""
    try:
        info = {
            "embedding_model": EMBEDDING_MODEL,
            "model_type": "sentence-transformer",
            "device": "cpu",
            "chunking_method": CHUNKING_METHOD,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "min_chunk_size": MIN_CHUNK_SIZE,
            "max_chunk_size": MAX_CHUNK_SIZE,
            "embedding_batch_size": EMBEDDING_BATCH_SIZE,
            "parallel_embedding": ENABLE_PARALLEL_EMBEDDING,
            "ocr_language": OCR_LANGUAGE,
            "vector_db": "weaviate",
            "weaviate_class": WEAVIATE_CLASS_NAME,
            "weaviate_connected": weaviate_client.is_connected() if weaviate_client else False
        }
        
        if doc_processor and hasattr(doc_processor, 'semantic_chunker'):
            chunker = doc_processor.semantic_chunker
            info.update({
                "chunker_available": chunker.is_model_available(),
                "chunker_embedding_dimension": getattr(chunker, 'embedding_dim', None),
                "chunker_model": getattr(chunker, 'model_name', 'unknown')
            })
        
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

@app.get("/api/vector-db/stats")
async def vector_db_stats():
    """Get Weaviate vector database statistics"""
    try:
        if not vector_store or not weaviate_client or not weaviate_client.is_connected():
            return JSONResponse({
                "success": False,
                "message": "Weaviate not connected"
            }, status_code=503)
        
        stats = vector_store.get_stats()
        
        return JSONResponse({
            "success": True,
            "collection": WEAVIATE_CLASS_NAME,
            "total_chunks": stats.get('document_count', 0),
            "status": "connected"
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/weaviate/collections")
async def list_collections():
    """List all Weaviate collections"""
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
                    "is_target": name == WEAVIATE_CLASS_NAME
                })
            except:
                result.append({
                    "name": name,
                    "count": None,
                    "is_target": name == WEAVIATE_CLASS_NAME
                })
        
        return {
            "collections": result,
            "total": len(result),
            "target_collection": WEAVIATE_CLASS_NAME
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/weaviate/collections/recreate")
async def recreate_collection():
    """Delete and recreate the collection"""
    if not weaviate_client or not weaviate_client.is_connected():
        raise HTTPException(status_code=503, detail="Weaviate not connected")
    
    try:
        # Delete if exists
        if weaviate_client.client.collections.exists(WEAVIATE_CLASS_NAME):
            weaviate_client.client.collections.delete(WEAVIATE_CLASS_NAME)
            logger.info(f"Deleted old collection: {WEAVIATE_CLASS_NAME}")
        
        # Recreate
        success = weaviate_client.create_collection(
            class_name=WEAVIATE_CLASS_NAME,
            description="RAG document chunks with embeddings",
            vector_dims=WEAVIATE_VECTOR_DIMS
        )
        
        return {"success": success, "message": f"Collection '{WEAVIATE_CLASS_NAME}' recreated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/weaviate/test-insert")
async def test_insert():
    """Test inserting a minimal object to see exact error"""
    if not weaviate_client or not weaviate_client.is_connected():
        raise HTTPException(status_code=503, detail="Weaviate not connected")
    
    try:
        collection = weaviate_client.get_collection(WEAVIATE_CLASS_NAME)
        
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
        
def main():
    """Run the application"""
    import uvicorn
    
    # Configuration
    port = int(os.getenv('PORT', '8000'))
    host = os.getenv('HOST', '127.0.0.1')
    
    config = {
        "app": "main:app",
        "host": host,
        "port": port,
        "reload": True,
        "log_level": "info",
        "access_log": False
    }
    
    logger.info("="*80)
    logger.info("🚀 Starting Enterprise RAG Server with Weaviate")
    logger.info("="*80)
    logger.info(f"🌐 Server URL: http://{config['host']}:{config['port']}")
    logger.info(f"🤖 Embedding Model: {EMBEDDING_MODEL}")
    logger.info(f"💻 Device: CPU")
    logger.info(f"✂️  Chunking: {CHUNKING_METHOD}")
    logger.info(f"📦 Batch Size: {EMBEDDING_BATCH_SIZE}")
    logger.info(f"🗄️  Vector DB: Weaviate ({'Embedded' if USE_EMBEDDED_WEAVIATE else 'Remote'})")
    logger.info("="*80)
    logger.info("Press Ctrl+C to stop")
    logger.info("="*80)
    
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        logger.info("\n" + "="*80)
        logger.info("Server stopped by user")
        logger.info("="*80)
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()