"""
Application Configuration
Centralized settings management using Pydantic
"""

import os
from pathlib import Path
from typing import Optional, Set
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # ============================================================================
    # APPLICATION INFO
    # ============================================================================
    APP_TITLE: str = "Enterprise RAG"
    APP_DESCRIPTION: str = "Document processing and RAG system with Weaviate Vector Database"
    APP_VERSION: str = "1.0.0"
    
    # ============================================================================
    # ENVIRONMENT & DEBUG
    # ============================================================================
    ENV: str = os.getenv('ENV', 'development')
    DEBUG: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    
    # ============================================================================
    # SERVER CONFIGURATION
    # ============================================================================
    HOST: str = os.getenv('HOST', '127.0.0.1')
    PORT: int = int(os.getenv('PORT', '8000'))
    RELOAD: bool = True
    LOG_LEVEL: str = "INFO"
    WORKERS: int = int(os.getenv('WORKERS', '4'))
    
    # ============================================================================
    # LOGGING
    # ============================================================================
    LOG_FILE: Optional[str] = os.getenv('LOG_FILE', './logs/app.log')
    
    # ============================================================================
    # DIRECTORIES
    # ============================================================================
    UPLOAD_FOLDER: Path = Path("uploads")
    PROCESSED_FOLDER: Path = Path("processed")
    EMBEDDINGS_FOLDER: Path = Path("embeddings_output")
    
    # ============================================================================
    # FILE UPLOAD
    # ============================================================================
    MAX_FILE_SIZE: int = int(os.getenv('MAX_FILE_SIZE', '52428800'))  # 50MB default
    ALLOWED_EXTENSIONS: Set[str] = {
        '.pdf', '.doc', '.docx', '.csv', '.txt', '.eml', '.msg',
        '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'
    }
    
    # ============================================================================
    # OCR & DOCUMENT PROCESSING
    # ============================================================================
    OCR_LANGUAGE: str = os.getenv('OCR_LANGUAGE', 'en')
    USE_GPU: bool = False  # Force CPU
    
    # ============================================================================
    # EMBEDDING CONFIGURATION
    # ============================================================================
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
    EMBEDDING_BATCH_SIZE: int = int(os.getenv('EMBEDDING_BATCH_SIZE', '8'))
    ENABLE_PARALLEL_EMBEDDING: bool = os.getenv('ENABLE_PARALLEL_EMBEDDING', 'false').lower() == 'true'
    
    # ============================================================================
    # CHUNKING CONFIGURATION
    # ============================================================================
    CHUNKING_METHOD: str = os.getenv('CHUNKING_METHOD', 'semantic')
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.5'))
    MIN_CHUNK_SIZE: int = int(os.getenv('MIN_CHUNK_SIZE', '10'))
    MAX_CHUNK_SIZE: int = int(os.getenv('MAX_CHUNK_SIZE', '1000'))
    DYNAMIC_THRESHOLD: bool = os.getenv('DYNAMIC_THRESHOLD', 'true').lower() == 'true'
    
    # ============================================================================
    # WEAVIATE CONFIGURATION
    # ============================================================================
    WEAVIATE_URL: Optional[str] = os.getenv('WEAVIATE_URL')
    WEAVIATE_API_KEY: Optional[str] = os.getenv('WEAVIATE_API_KEY')
    WEAVIATE_CLASS_NAME: str = os.getenv('WEAVIATE_CLASS_NAME', 'Ragdocument')
    WEAVIATE_VECTOR_DIMS: int = int(os.getenv('WEAVIATE_VECTOR_DIMS', '768'))
    USE_EMBEDDED_WEAVIATE: bool = os.getenv('USE_EMBEDDED_WEAVIATE', 'false').lower() == 'true'
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories on initialization
        self.UPLOAD_FOLDER.mkdir(exist_ok=True)
        self.PROCESSED_FOLDER.mkdir(exist_ok=True)
        self.EMBEDDINGS_FOLDER.mkdir(exist_ok=True)
        
        # Create logs directory if log file is specified
        if self.LOG_FILE:
            log_path = Path(self.LOG_FILE)
            log_path.parent.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        # IMPORTANT: Allow extra fields from .env that aren't defined
        extra = "ignore"  # This fixes the validation error


# Singleton instance
settings = Settings()