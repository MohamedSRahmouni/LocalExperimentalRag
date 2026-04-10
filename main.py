import os
import sys
import warnings

# Suppress dependency warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables before imports
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = os.getenv('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
os.environ['KMP_DUPLICATE_LIB_OK'] = os.getenv('KMP_DUPLICATE_LIB_OK', 'TRUE')

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
logging.getLogger("multipart").setLevel(logging.ERROR)

try:
    from app.services.preprocessing import DocumentProcessor
    logger.info("✓ Document processor module loaded")
except Exception as e:
    logger.error(f"Failed to load document processor: {str(e)}")
    sys.exit(1)

app = FastAPI(
    title="Enterprise RAG",
    description="Document processing and RAG system",
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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize document processor
OCR_LANGUAGE = os.getenv('OCR_LANGUAGE', 'en')
USE_GPU = os.getenv('USE_GPU', 'False').lower() == 'true'

doc_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global doc_processor
    try:
        logger.info("Initializing document processor...")
        doc_processor = DocumentProcessor(lang=OCR_LANGUAGE, use_gpu=USE_GPU)
        logger.info("✓ Document processor ready")
    except Exception as e:
        logger.error(f"Failed to initialize document processor: {str(e)}")
        logger.warning("Application will start but document processing may not work")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down gracefully...")

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.csv', '.txt', '.eml', '.msg',
    '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'
}

def allowed_file(filename: str) -> bool:
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
    """Upload and process files"""
    if doc_processor is None:
        return JSONResponse({
            "success": False,
            "message": "Document processor not initialized. Please check server logs."
        }, status_code=503)
    
    uploaded_files = []
    file_paths = []
    
    # Save uploaded files
    for file in files:
        if file.filename and allowed_file(file.filename):
            filename = Path(file.filename).name
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append(filename)
            file_paths.append(file_path)
            logger.info(f"Saved file: {filename}")
    
    if not uploaded_files:
        return JSONResponse({
            "success": False,
            "message": "No valid files uploaded"
        }, status_code=400)
    
    # Process files
    try:
        processing_results = doc_processor.process_multiple_files(file_paths)
        
        # Save processed chunks
        processed_data_path = os.path.join(PROCESSED_FOLDER, "processed_chunks.txt")
        with open(processed_data_path, "a", encoding="utf-8") as f:
            for chunk in processing_results['all_chunks']:
                f.write(chunk + "\n" + "="*80 + "\n\n")
        
        logger.info(f"Processing complete: {processing_results['successful']} successful, "
                   f"{processing_results['failed']} failed, "
                   f"{processing_results['total_unique_chunks']} unique chunks created")
        
        return JSONResponse({
            "success": True,
            "message": f"Successfully processed {processing_results['successful']} file(s)",
            "details": {
                "uploaded": len(uploaded_files),
                "successful": processing_results['successful'],
                "failed": processing_results['failed'],
                "duplicates": processing_results['duplicate'],
                "total_chunks": processing_results['total_unique_chunks'],
                "scanned_docs": processing_results.get('scanned_count', 0)
            },
            "files": uploaded_files
        })
    
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}", exc_info=True)
        return JSONResponse({
            "success": False,
            "message": f"Error processing files: {str(e)}"
        }, status_code=500)

@app.get("/prompt", response_class=HTMLResponse)
async def prompt_page(request: Request):
    """Chat/Prompt page"""
    return templates.TemplateResponse("promptPage/prompt.html", {"request": request})

@app.post("/api/chat")
async def chat(request: Request):
    """Chat endpoint"""
    data = await request.json()
    user_message = data.get('message', '')
    
    response = {
        'message': f'You asked: {user_message}. RAG processing will search through processed document chunks.'
    }
    
    return JSONResponse(response)

@app.get("/api/stats")
async def get_stats():
    """Get processing statistics"""
    upload_count = len(list(Path(UPLOAD_FOLDER).glob("*")))
    
    stats = {
        "uploaded_files": upload_count,
        "processor_available": doc_processor is not None
    }
    
    if doc_processor:
        stats["processed_chunks"] = len(doc_processor.processed_hashes)
    
    return JSONResponse(stats)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "processor_status": "ready" if doc_processor else "unavailable"
    })

def main():
    """Run the application"""
    import uvicorn
    
    # Configuration
    config = {
        "app": "main:app",
        "host": "127.0.0.1",
        "port": 8000,
        "reload": True,
        "log_level": "info",
        "access_log": False  # Disable access logs for cleaner output
    }
    
    logger.info("="*60)
    logger.info("Starting Enterprise RAG Server")
    logger.info(f"Server will be available at: http://{config['host']}:{config['port']}")
    logger.info("Press Ctrl+C to stop")
    logger.info("="*60)
    
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()