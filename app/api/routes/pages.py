"""
HTML Page Routes
Serves the frontend templates
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/components")


@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Home page"""
    return templates.TemplateResponse("homePage/home.html", {"request": request})


@router.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("homePage/home.html", {"request": request})


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Upload page"""
    return templates.TemplateResponse("uploadPage/upload.html", {"request": request})


@router.get("/prompt", response_class=HTMLResponse)
async def prompt_page(request: Request):
    """Chat/Prompt page"""
    return templates.TemplateResponse("promptPage/prompt.html", {"request": request})


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    from app.api.dependencies import (
        get_doc_processor,
        get_embedding_manager,
        get_vector_store        # ← CHANGED: was get_weaviate_client
    )
    from app.core.config import settings
    
    doc_processor = get_doc_processor()
    embedding_manager = get_embedding_manager()
    vector_store = get_vector_store()           # ← CHANGED
    
    health_status = {
        "status": "healthy",
        "processor_status": "ready" if doc_processor else "unavailable",
        "embedding_manager_status": (
            "ready" if embedding_manager and embedding_manager.is_ready() 
            else "unavailable"
        ),
        "qdrant_status": (                      # ← CHANGED: was weaviate_status
            "connected" if vector_store and vector_store.is_connected() 
            else "disconnected"
        ),
        "embedding_model": settings.EMBEDDING_MODEL,
        "model_type": "multilingual-e5-small",  # ← UPDATED
        "device": "gpu" if settings.USE_GPU else "cpu"  # ← UPDATED
    }
    
    if doc_processor and hasattr(doc_processor, 'semantic_chunker'):
        health_status["embedder_status"] = (
            "ready" if doc_processor.semantic_chunker.is_model_available() 
            else "unavailable"
        )
    
    return health_status