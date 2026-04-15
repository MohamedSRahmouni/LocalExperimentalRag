"""
Enterprise RAG System - Main Entry Point
Modular FastAPI application with clean architecture
"""

import sys
import os
import warnings
import multiprocessing

# ============================================================================
# WINDOWS MULTIPROCESSING FIX - MUST BE AT THE TOP
# ============================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # Force single-threaded operations
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = os.getenv('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
os.environ['KMP_DUPLICATE_LIB_OK'] = os.getenv('KMP_DUPLICATE_LIB_OK', 'TRUE')

# ============================================================================
# IMPORTS
# ============================================================================
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.startup import startup_event, shutdown_event

# Import routers
from app.api.routes import pages, upload, chat, stats, weaviate, retrieve 

# ============================================================================
# SETUP
# ============================================================================

# Setup logging
setup_logging(level=settings.LOG_LEVEL)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass  # Static files not critical

# ============================================================================
# EVENT HANDLERS
# ============================================================================

app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

# ============================================================================
# ROUTERS
# ============================================================================

app.include_router(pages.router, tags=["Pages"])
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(stats.router, prefix="/api", tags=["Statistics"])
app.include_router(weaviate.router, prefix="/api/weaviate", tags=["Weaviate Admin"])
app.include_router(retrieve.router, prefix="/api", tags=["Retrieval"])

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the application"""
    import uvicorn
    import logging
    
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("🚀 Starting Enterprise RAG Server")
    logger.info("="*80)
    logger.info(f"🌐 Server URL: http://{settings.HOST}:{settings.PORT}")
    logger.info(f"🤖 Embedding Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"💻 Device: CPU")
    logger.info(f"✂️  Chunking: {settings.CHUNKING_METHOD}")
    logger.info(f"📦 Batch Size: {settings.EMBEDDING_BATCH_SIZE}")
    logger.info(f"🗄️  Vector DB: Weaviate")
    logger.info("="*80)
    logger.info("Press Ctrl+C to stop")
    logger.info("="*80)
    
    try:
        uvicorn.run(
            "main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.RELOAD,
            log_level=settings.LOG_LEVEL.lower(),
            access_log=False
        )
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