"""
Document Loader Pipeline
LangChain-powered document loading, cleaning, and chunking

Replaces:
    - app/services/preprocessing/document_processor.py
    - app/services/preprocessing/file_handler.py
    - app/services/preprocessing/text_cleaner.py
    - app/services/preprocessing/semantic_chunker.py
    - app/services/preprocessing/transformer_enhanced_chunker.py
"""

import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)

logger = logging.getLogger(__name__)


# ============================================================
# SUPPORTED FORMATS (same as your FileHandler)
# ============================================================
SUPPORTED_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.csv',
    '.txt', '.eml', '.msg',
    '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'
}

IMAGE_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'
}


class LangChainDocumentLoader:
    """
    Unified document loader using LangChain
    
    Replaces:
        - DocumentProcessor
        - FileHandler
        - TextCleaner
        - SemanticChunker
        
    Keeps:
        - Same method interfaces
        - Same return formats
        - Same chunking strategies
        - OCRHandler (reused for images/scanned PDFs)
    """
    
    def __init__(
        self,
        lang: str = 'en',
        use_gpu: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 10,
        similarity_threshold: float = 0.5,
        chunking_method: str = "semantic",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize Document Loader
        
        Args:
            lang: OCR language (same as DocumentProcessor)
            use_gpu: GPU flag (same as DocumentProcessor)
            chunk_size: Max chunk size (same as MAX_CHUNK_SIZE)
            chunk_overlap: Overlap between chunks
            min_chunk_size: Min chunk size (same as MIN_CHUNK_SIZE)
            similarity_threshold: Semantic similarity threshold
            chunking_method: 'semantic' or 'simple'
            embedding_model: Model for semantic chunking
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.chunking_method = chunking_method
        self.embedding_model = embedding_model
        
        # Duplicate tracking (same as DocumentProcessor)
        self.processed_hashes = set()
        self.scanned_count = 0
        self.ocr_count = 0
        
        # Initialize OCR (reused from your existing code)
        self._init_ocr()
        
        # Initialize text splitter
        self.splitter = self._create_splitter()
        
        # Loader dispatch map (same extensions as FileHandler)
        self._loaders = {
            '.pdf':  self._load_pdf,
            '.docx': self._load_docx,
            '.doc':  self._load_doc,
            '.txt':  self._load_txt,
            '.csv':  self._load_csv,
            '.eml':  self._load_email,
            '.msg':  self._load_msg,
            '.png':  self._load_image,
            '.jpg':  self._load_image,
            '.jpeg': self._load_image,
            '.tiff': self._load_image,
            '.bmp':  self._load_image,
            '.gif':  self._load_image,
        }
        
        logger.info("="*70)
        logger.info("✅ LangChainDocumentLoader initialized")
        logger.info(f"   Chunking method : {chunking_method}")
        logger.info(f"   Chunk size      : {chunk_size}")
        logger.info(f"   Min chunk size  : {min_chunk_size}")
        logger.info(f"   OCR language    : {lang}")
        logger.info("="*70)
    
    # ============================================================
    # INITIALIZATION
    # ============================================================
    
    def _init_ocr(self):
        """Initialize OCR handler (reuses your OCRHandler)"""
        try:
            from app.services.preprocessing.ocr_handler import OCRHandler
            self.ocr_handler = OCRHandler(
                lang=self.lang,
                use_gpu=self.use_gpu
            )
            self.ocr_available = True
            logger.info("✅ OCR handler initialized")
        except Exception as e:
            logger.warning(f"⚠️  OCR not available: {e}")
            self.ocr_handler = None
            self.ocr_available = False
    
    def _create_splitter(self):
        """
        Create text splitter based on chunking method
        """
        if self.chunking_method == "semantic":
            try:
                from pathlib import Path
                import os
                
                # ================================================================
                # FIX: Résoudre le chemin du modèle
                # ================================================================
                model_path = self.embedding_model
                
                if model_path.startswith('./') or model_path.startswith('.\\'):
                    # Essaie chemin direct
                    absolute_path = Path(model_path).resolve()
                    
                    if absolute_path.exists():
                        model_path = str(absolute_path)
                        logger.info(f"✅ Splitter model found at: {model_path}")
                    else:
                        # Essaie dans scripts/
                        scripts_path = Path("scripts") / model_path.lstrip('./').lstrip('.\\')
                        if scripts_path.resolve().exists():
                            model_path = str(scripts_path.resolve())
                            logger.info(f"✅ Splitter model found in scripts/: {model_path}")
                        else:
                            logger.warning(
                                f"⚠️  Model path not found: {model_path}, "
                                f"using simple splitter"
                            )
                            return self._create_simple_splitter()
                
                # Mode offline
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                
                # Semantic splitter
                splitter = SentenceTransformersTokenTextSplitter(
                    model_name=model_path,
                    chunk_overlap=50,
                    tokens_per_chunk=256
                )
                logger.info("✅ Semantic splitter initialized")
                return splitter
                
            except Exception as e:
                logger.warning(f"⚠️  Semantic splitter failed: {e}, using simple")
                return self._create_simple_splitter()
        else:
            return self._create_simple_splitter()
    
    def _create_simple_splitter(self):
        """Simple fallback splitter (replaces _simple_chunk_text)"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    # ============================================================
    # PUBLIC INTERFACE (same as DocumentProcessor)
    # ============================================================
    
    def process_file(
        self,
        file_path: str,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        dynamic_threshold: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single file
        
        Same interface as DocumentProcessor.process_file()
        Same return format
        
        Args:
            file_path: Path to file
            similarity_threshold: Semantic threshold
            min_chunk_size: Min chunk size
            max_chunk_size: Max chunk size
            dynamic_threshold: Dynamic threshold flag
            
        Returns:
            Same dict format as DocumentProcessor
        """
        file_path = Path(file_path)
        
        # Same result structure as DocumentProcessor
        result = {
            'filename': file_path.name,
            'filepath': str(file_path),
            'success': False,
            'processed_text': '',
            'original_length': 0,
            'processed_length': 0,
            'chunks': [],
            'chunks_with_metadata': [],
            'chunk_count': 0,
            'metadata': {},
            'errors': [],
            'warnings': []
        }
        
        # Validate file
        if not file_path.exists():
            result['errors'].append('File not found')
            logger.error(f"❌ File not found: {file_path}")
            return result
        
        extension = file_path.suffix.lower()
        
        if extension not in SUPPORTED_EXTENSIONS:
            result['errors'].append(f'Unsupported format: {extension}')
            logger.error(f"❌ Unsupported format: {extension}")
            return result
        
        logger.info(f"📄 Processing: {file_path.name}")
        
        try:
            # ================================================
            # STEP 1: Load documents
            # ================================================
            logger.info("Step 1: Loading document...")
            loader_func = self._loaders.get(extension)
            documents = loader_func(file_path)
            
            if not documents:
                result['errors'].append('No text extracted')
                logger.warning(f"⚠️  No text extracted from {file_path.name}")
                return result
            
            # ================================================
            # STEP 2: Get full text
            # ================================================
            full_text = "\n\n".join([
                d.page_content for d in documents
                if d.page_content.strip()
            ])
            
            result['original_length'] = len(full_text)
            
            if len(full_text.strip()) < 10:
                result['errors'].append('Extracted text too short')
                return result
            
            # ================================================
            # STEP 3: Duplicate detection (same as DocumentProcessor)
            # ================================================
            logger.info("Step 2: Checking duplicates...")
            text_hash = self._generate_hash(full_text)
            
            if text_hash in self.processed_hashes:
                result['errors'].append('Duplicate content detected')
                result['warnings'].append('Already processed')
                logger.warning(f"⚠️  Duplicate: {file_path.name}")
                return result
            
            # ================================================
            # STEP 4: Validate text quality (same as TextCleaner)
            # ================================================
            logger.info("Step 3: Validating text quality...")
            if not self._is_valid_text(full_text):
                result['errors'].append('Text validation failed')
                return result
            
            # ================================================
            # STEP 5: Adaptive chunking (same strategy as DocumentProcessor)
            # ================================================
            logger.info("Step 4: Chunking...")
            chunks_with_metadata = self._adaptive_chunk(
                documents=documents,
                full_text=full_text,
                file_path=file_path,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size
            )
            
            if not chunks_with_metadata:
                result['errors'].append('Chunking produced no results')
                return result
            
            # ================================================
            # Mark as processed
            # ================================================
            self.processed_hashes.add(text_hash)
            
            # Extract plain chunks (backward compatibility)
            plain_chunks = [c['text'] for c in chunks_with_metadata]
            
            # ================================================
            # Build result (same format as DocumentProcessor)
            # ================================================
            result.update({
                'success': True,
                'processed_text': full_text,
                'processed_length': len(full_text),
                'original_length': len(full_text),
                'chunks': plain_chunks,
                'chunks_with_metadata': chunks_with_metadata,
                'chunk_count': len(chunks_with_metadata),
                'content_hash': text_hash,
                'compression_ratio': 1.0,
                'metadata': {
                    'file_size': file_path.stat().st_size,
                    'file_type': extension,
                    'processed_at': datetime.now().isoformat(),
                    'is_scanned': self._is_scanned(extension),
                    'is_image': extension in IMAGE_EXTENSIONS,
                    'chunking_method': self.chunking_method,
                    'similarity_threshold': similarity_threshold,
                    'dynamic_threshold': dynamic_threshold,
                    'total_pages': len(documents)
                }
            })
            
            logger.info(
                f"✅ {file_path.name}: "
                f"{len(chunks_with_metadata)} chunks, "
                f"{len(full_text)} chars"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            result['errors'].append(str(e))
            return result
    
    def process_multiple_files(
        self,
        file_paths: List[str],
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        dynamic_threshold: bool = False
    ) -> Dict[str, Any]:
        """
        Process multiple files
        
        Same interface as DocumentProcessor.process_multiple_files()
        Same return format
        """
        logger.info("="*70)
        logger.info(f"📁 Batch processing: {len(file_paths)} files")
        logger.info("="*70)
        
        # Same result structure as DocumentProcessor
        results = {
            'total_files': len(file_paths),
            'successful': 0,
            'failed': 0,
            'duplicate': 0,
            'total_chunks': 0,
            'total_unique_chunks': 0,
            'scanned_count': 0,
            'ocr_count': 0,
            'total_size': 0,
            'total_original_length': 0,
            'total_processed_length': 0,
            'files': [],
            'all_chunks': [],
            'all_chunks_with_metadata': [],
            'errors': [],
            'processing_time': None,
            'statistics': {}
        }
        
        start_time = datetime.now()
        
        for idx, file_path in enumerate(file_paths, 1):
            logger.info(
                f"Processing {idx}/{len(file_paths)}: "
                f"{Path(file_path).name}"
            )
            
            result = self.process_file(
                file_path,
                similarity_threshold=similarity_threshold,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                dynamic_threshold=dynamic_threshold
            )
            
            results['files'].append(result)
            
            if result['success']:
                results['successful'] += 1
                results['total_chunks'] += result['chunk_count']
                results['all_chunks'].extend(result['chunks'])
                results['all_chunks_with_metadata'].extend(
                    result['chunks_with_metadata']
                )
                results['total_original_length'] += result['original_length']
                results['total_processed_length'] += result['processed_length']
                results['total_size'] += result.get(
                    'metadata', {}
                ).get('file_size', 0)
                
                if result.get('metadata', {}).get('is_scanned'):
                    results['scanned_count'] += 1
                    
            elif 'Duplicate' in str(result.get('errors', [])):
                results['duplicate'] += 1
            else:
                results['failed'] += 1
                results['errors'].extend(result.get('errors', []))
        
        results['ocr_count'] = self.ocr_count
        
        # Remove duplicate chunks (same as DocumentProcessor)
        logger.info("Removing duplicate chunks...")
        before = len(results['all_chunks'])
        results['all_chunks'] = self._remove_duplicates(results['all_chunks'])
        results['total_unique_chunks'] = len(results['all_chunks'])
        duplicates_removed = before - results['total_unique_chunks']
        
        # Processing time
        duration = (datetime.now() - start_time).total_seconds()
        results['processing_time'] = duration
        
        # Statistics (same as DocumentProcessor)
        results['statistics'] = {
            'average_chunks_per_file': (
                results['total_unique_chunks'] /
                results['successful']
                if results['successful'] > 0 else 0
            ),
            'success_rate': (
                results['successful'] /
                results['total_files'] * 100
                if results['total_files'] > 0 else 0
            ),
            'duplicate_rate': (
                results['duplicate'] /
                results['total_files'] * 100
                if results['total_files'] > 0 else 0
            ),
            'scanned_rate': (
                results['scanned_count'] /
                results['total_files'] * 100
                if results['total_files'] > 0 else 0
            ),
            'average_processing_time': (
                duration / results['total_files']
                if results['total_files'] > 0 else 0
            ),
            'chunks_removed_as_duplicates': duplicates_removed,
            'chunking_method': self.chunking_method
        }
        
        logger.info("="*70)
        logger.info("✅ Batch Processing Complete!")
        logger.info(f"   Total    : {results['total_files']}")
        logger.info(f"   Success  : {results['successful']}")
        logger.info(f"   Failed   : {results['failed']}")
        logger.info(f"   Duplicate: {results['duplicate']}")
        logger.info(f"   Chunks   : {results['total_unique_chunks']}")
        logger.info(f"   Time     : {duration:.2f}s")
        logger.info("="*70)
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Same interface as DocumentProcessor.get_processing_stats()"""
        return {
            'unique_documents': len(self.processed_hashes),
            'scanned_documents': self.scanned_count,
            'ocr_operations': self.ocr_count,
            'chunking_method': self.chunking_method,
            'semantic_model_available': self.chunking_method == "semantic"
        }
    
    def is_model_available(self) -> bool:
        """Same interface as SemanticChunker.is_model_available()"""
        return self.splitter is not None
    
    def reset_duplicate_tracking(self):
        """Same interface as DocumentProcessor.reset_duplicate_tracking()"""
        self.processed_hashes.clear()
        self.scanned_count = 0
        self.ocr_count = 0
        logger.info("✅ Duplicate tracking reset")
    
    def _simple_chunk(
        self,
        text: str,
        file_path: Path,
        chunk_size: int = 1000,
        min_size: int = 20
    ) -> List[Dict[str, Any]]:
        """Simple chunking fallback"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=min(100, chunk_size // 5),
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        
        if not chunks and text.strip():
            chunks = [text.strip()]
        
        return [
            {
                'text': chunk,
                'similarity_score': 0.0,        # ← Changed None → 0.0
                'sentence_count': (
                    chunk.count('.') +
                    chunk.count('!') +
                    chunk.count('?')
                ),
                'length': len(chunk),
                'model_type': 'simple',
                'source': str(file_path),
                'chunk_index': i
            }
            for i, chunk in enumerate(chunks)
            if chunk.strip() and len(chunk.strip()) >= min_size
        ]

    def _adaptive_chunk(
        self,
        documents: List[Document],
        full_text: str,
        file_path: Path,
        min_chunk_size: int,
        max_chunk_size: int
    ) -> List[Dict[str, Any]]:
        """Adaptive chunking strategy"""
        text_length = len(full_text.strip())
        
        # Strategy 1: Very short text
        if text_length < 50:
            return [{
                'text': full_text.strip(),
                'similarity_score': 0.0,        # ← Changed None → 0.0
                'sentence_count': 1,
                'length': text_length,
                'model_type': 'single',
                'source': str(file_path),
                'chunk_index': 0
            }]
        
        # Strategy 2: Short text
        if text_length < 200:
            return self._simple_chunk(
                full_text, file_path,
                chunk_size=text_length,
                min_size=max(20, text_length // 3)
            )
        
        # Strategy 3: Medium text
        if text_length < 500:
            return self._simple_chunk(
                full_text, file_path,
                chunk_size=max(200, text_length // 2),
                min_size=max(50, text_length // 5)
            )
        
        # Strategy 4: Long text
        try:
            chunks = self.splitter.split_documents(documents)
            
            if chunks:
                return [
                    {
                        'text': c.page_content,
                        'similarity_score': 0.0,    # ← Changed None → 0.0
                        'sentence_count': (
                            c.page_content.count('.') +
                            c.page_content.count('!') +
                            c.page_content.count('?')
                        ),
                        'length': len(c.page_content),
                        'model_type': self.chunking_method,
                        'source': str(file_path),
                        'chunk_index': i,
                        'metadata': c.metadata
                    }
                    for i, c in enumerate(chunks)
                    if len(c.page_content.strip()) >= min_chunk_size
                ]
        except Exception as e:
            logger.warning(f"⚠️  Splitter failed: {e}, fallback to simple")
        
        # Fallback
        return self._simple_chunk(
            full_text, file_path,
            chunk_size=max_chunk_size,
            min_size=min_chunk_size
        )
    # ============================================================
    # PRIVATE: FILE LOADERS (replaces FileHandler methods)
    # ============================================================
    
    def _load_pdf(self, path: Path) -> List[Document]:
        """Replaces FileHandler.extract_from_pdf()"""
        try:
            from langchain.document_loaders import PyPDFLoader
            
            loader = PyPDFLoader(str(path))
            documents = loader.load()
            
            # Check if scanned (same logic as OCRHandler.is_scanned_pdf_page)
            full_text = " ".join([d.page_content for d in documents])
            meaningful_chars = sum(1 for c in full_text if c.isalnum())
            
            if meaningful_chars < 50:
                logger.info(f"🔍 Scanned PDF detected: {path.name}")
                self.scanned_count += 1
                return self._ocr_pdf(path)
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ PDF error: {e}")
            return []
    
    def _load_docx(self, path: Path) -> List[Document]:
        """Replaces FileHandler.extract_from_docx()"""
        try:
            from langchain.document_loaders import Docx2txtLoader
            
            loader = Docx2txtLoader(str(path))
            return loader.load()
            
        except Exception as e:
            logger.error(f"❌ DOCX error: {e}")
            return []
    
    def _load_doc(self, path: Path) -> List[Document]:
        """Replaces FileHandler.extract_from_doc()"""
        try:
            from langchain.document_loaders import UnstructuredWordDocumentLoader
            
            loader = UnstructuredWordDocumentLoader(str(path))
            return loader.load()
            
        except Exception as e:
            logger.error(f"❌ DOC error: {e}")
            return []
    
    def _load_txt(self, path: Path) -> List[Document]:
        """Replaces FileHandler.extract_from_txt() with multi-encoding"""
        
        for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                from langchain.document_loaders import TextLoader
                
                loader = TextLoader(str(path), encoding=encoding)
                return loader.load()
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"❌ TXT error: {e}")
                return []
        
        return []
    
    def _load_csv(self, path: Path) -> List[Document]:
        """Replaces FileHandler.extract_from_csv()"""
        try:
            from langchain.document_loaders import CSVLoader
            
            loader = CSVLoader(str(path))
            return loader.load()
            
        except Exception as e:
            logger.error(f"❌ CSV error: {e}")
            return []
    
    def _load_email(self, path: Path) -> List[Document]:
        """Replaces FileHandler.extract_from_eml()"""
        try:
            from langchain.document_loaders import UnstructuredEmailLoader
            
            loader = UnstructuredEmailLoader(str(path))
            return loader.load()
            
        except Exception as e:
            logger.error(f"❌ EML error: {e}")
            return []
    
    def _load_msg(self, path: Path) -> List[Document]:
        """Replaces FileHandler.extract_from_msg()"""
        try:
            from langchain.document_loaders import OutlookMessageLoader
            
            loader = OutlookMessageLoader(str(path))
            return loader.load()
            
        except Exception as e:
            logger.error(f"❌ MSG error: {e}")
            return []
    
    def _load_image(self, path: Path) -> List[Document]:
        """
        Replaces FileHandler.extract_from_image() + OCRHandler
        Reuses your existing OCRHandler
        """
        try:
            self.ocr_count += 1
            
            if self.ocr_available and self.ocr_handler:
                # Reuse your existing OCRHandler
                result = self.ocr_handler.extract_text_from_image(str(path))
                
                if result['success'] and result['text']:
                    return [Document(
                        page_content=result['text'],
                        metadata={
                            'source': str(path),
                            'is_image': True,
                            'ocr_confidence': result.get('confidence', 0),
                            'word_count': result.get('word_count', 0)
                        }
                    )]
            
            # LangChain fallback
            from langchain.document_loaders import UnstructuredImageLoader
            loader = UnstructuredImageLoader(str(path))
            return loader.load()
            
        except Exception as e:
            logger.error(f"❌ Image OCR error: {e}")
            return []
    
    def _ocr_pdf(self, path: Path) -> List[Document]:
        """
        OCR for scanned PDFs
        Reuses your existing OCRHandler
        """
        if not self.ocr_available:
            logger.warning("⚠️  OCR not available for scanned PDF")
            return []
        
        documents = []
        
        try:
            from pdf2image import convert_from_path
            import tempfile
            import os
            
            images = convert_from_path(str(path), dpi=300)
            
            for i, image in enumerate(images):
                # Save temp file
                temp_path = f"temp_ocr_page_{i}_{os.getpid()}.png"
                image.save(temp_path, 'PNG')
                
                try:
                    result = self.ocr_handler.extract_text_from_image(
                        temp_path,
                        preprocess=True
                    )
                    
                    if result['success'] and result['text']:
                        documents.append(Document(
                            page_content=result['text'],
                            metadata={
                                'source': str(path),
                                'page': i,
                                'is_scanned': True,
                                'ocr_confidence': result.get('confidence', 0)
                            }
                        ))
                        self.ocr_count += 1
                        
                finally:
                    # Cleanup temp file
                    if Path(temp_path).exists():
                        Path(temp_path).unlink()
            
            logger.info(
                f"✅ OCR complete: {len(documents)} pages "
                f"from {path.name}"
            )
            
        except Exception as e:
            logger.error(f"❌ OCR PDF error: {e}")
        
        return documents
    
    # ============================================================
    # PRIVATE: UTILS (replaces TextCleaner)
    # ============================================================
    
    def _generate_hash(self, text: str) -> str:
        """Same as DocumentProcessor._generate_hash()"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _is_valid_text(self, text: str, min_length: int = 5) -> bool:
        """
        Same logic as TextCleaner.is_valid_text()
        Minimum length lowered to 5 (same fix)
        """
        if not text or not isinstance(text, str):
            return False
        
        cleaned = text.strip()
        
        if len(cleaned) < min_length:
            return False
        
        if not any(c.isalnum() for c in cleaned):
            return False
        
        # Only check repetition for longer texts (same as TextCleaner)
        if len(cleaned) >= 50:
            words = cleaned.split()
            if len(words) >= 10:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:
                    return False
        
        return True
    
    def _remove_duplicates(self, texts: List[str]) -> List[str]:
        """Same logic as TextCleaner.remove_duplicates()"""
        seen = set()
        unique = []
        
        for text in texts:
            normalized = text.lower().strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(text)
        
        removed = len(texts) - len(unique)
        if removed > 0:
            logger.info(f"✅ Removed {removed} duplicate chunks")
        
        return unique
    
    def _is_scanned(self, extension: str) -> bool:
        """Check if file is likely scanned"""
        return extension in IMAGE_EXTENSIONS