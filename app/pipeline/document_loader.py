"""
Document Loader Pipeline
LangChain-powered document loading, cleaning, and chunking

Replaces:
    - app/services/preprocessing/document_processor.py
    - app/services/preprocessing/file_handler.py
    - app/services/preprocessing/text_cleaner.py
    - app/services/preprocessing/semantic_chunker.py
    - app/services/preprocessing/transformer_enhanced_chunker.py

Features:
    - Table detection & preservation
    - Tables kept in single chunks
    - Markdown table extraction
"""

import logging
import hashlib
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)

logger = logging.getLogger(__name__)


# ============================================================
# SUPPORTED FORMATS
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
    
    Features:
        - Table detection and preservation
        - Tables kept in single chunks
        - Markdown/HTML table extraction
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
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        preserve_tables: bool = True,         # ← NEW
        max_table_size: int = 5000            # ← NEW: max chars for table chunk
    ):
        """
        Initialize Document Loader
        
        Args:
            preserve_tables: Keep tables in single chunks
            max_table_size: Maximum size for table chunks
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.chunking_method = chunking_method
        self.embedding_model = embedding_model
        self.preserve_tables = preserve_tables      # ← NEW
        self.max_table_size = max_table_size        # ← NEW
        
        # Duplicate tracking
        self.processed_hashes = set()
        self.scanned_count = 0
        self.ocr_count = 0
        
        # Initialize OCR
        self._init_ocr()
        
        # Initialize text splitter
        self.splitter = self._create_splitter()
        
        # Loader dispatch map
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

        from app.services.preprocessing.docling_table_extractor import (
            DoclingTableExtractor
        )
        self.docling_extractor = DoclingTableExtractor(preserve_caption=True)
        
        if self.docling_extractor.available:
            logger.info("✅ Docling table extractor ready")
        else:
            logger.warning("⚠️  Docling unavailable — fallback to pdfplumber")
        
        logger.info("="*70)
        logger.info("✅ LangChainDocumentLoader initialized")
        logger.info(f"   Chunking method : {chunking_method}")
        logger.info(f"   Chunk size      : {chunk_size}")
        logger.info(f"   Min chunk size  : {min_chunk_size}")
        logger.info(f"   Preserve tables : {preserve_tables}")  # ← NEW
        logger.info(f"   OCR language    : {lang}")
        logger.info("="*70)
    
    # ============================================================
    # INITIALIZATION
    # ============================================================
    
    def _init_ocr(self):
        """Initialize OCR handler"""
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
        """Create text splitter"""
        if self.chunking_method != "semantic":
            return self._create_simple_splitter()

        try:
            model_path_str = self.embedding_model  # e.g., "./models/bge-m3"
            resolved_path = None

            # 1. Check relative to current working directory (e.g., executing from root)
            cwd_path = Path(model_path_str)
            if cwd_path.exists():
                resolved_path = cwd_path.resolve()
                logger.info(f"✅ Splitter model found at: {resolved_path}")
            
            # 2. Check up one level if executing from inside the 'app' directory
            else:
                parent_context_path = Path("..") / model_path_str.lstrip('./').lstrip('.\\')
                if parent_context_path.exists():
                    resolved_path = parent_context_path.resolve()
                    logger.info(f"✅ Splitter model found via parent directory: {resolved_path}")
                
                # 3. Fallback: explicitly look for 'models/bge-m3' relative to the workspace root
                else:
                    # If running deep inside app/pipeline/, look for a known workspace root anchor
                    # or check if 'models' directory exists adjacent to 'app'
                    root_fallback = Path(__file__).parents[2] / "models" / "bge-m3"
                    if root_fallback.exists():
                        resolved_path = root_fallback.resolve()
                        logger.info(f"✅ Splitter model found via absolute file anchoring: {resolved_path}")

            # If it wasn't found anywhere, fall back to simple splitter
            if not resolved_path:
                logger.warning(
                    f"⚠️  Model path not found: {model_path_str}, "
                    f"falling back to simple splitter"
                )
                return self._create_simple_splitter()

            # Enforce offline loading for SentenceTransformers
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            # Initialize splitter with the verified absolute string path
            splitter = SentenceTransformersTokenTextSplitter(
                model_name=str(resolved_path),
                chunk_overlap=50,
                tokens_per_chunk=256
            )
            logger.info(f"✅ Semantic splitter initialized using local model at {resolved_path}")
            return splitter
            
        except Exception as e:
            logger.warning(f"⚠️  Semantic splitter failed: {e}, using simple")
            return self._create_simple_splitter()
        
    def _create_simple_splitter(self):
        """Simple fallback splitter"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    # ============================================================
    # TABLE DETECTION & EXTRACTION (NEW)
    # ============================================================
    
    def _detect_tables(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Detect tables in text
        
        Returns:
            List of (start_pos, end_pos, table_text) tuples
        """
        tables = []
        
        # Pattern 1: Markdown tables
        markdown_pattern = r'(\|[^\n]+\|\n)+(\|[-: ]+\|[-: |\n]+)'
        for match in re.finditer(markdown_pattern, text):
            tables.append((
                match.start(),
                match.end(),
                match.group(0)
            ))
        
        # Pattern 2: Grid tables (ASCII art)
        grid_pattern = r'(\+[-+]+\+\n(\|[^\n]+\|\n)+)+'
        for match in re.finditer(grid_pattern, text):
            tables.append((
                match.start(),
                match.end(),
                match.group(0)
            ))
        
        # Pattern 3: Tab-separated values (TSV)
        tsv_pattern = r'(^[^\t\n]+\t[^\t\n]+(\t[^\t\n]+)*$\n){3,}'
        for match in re.finditer(tsv_pattern, text, re.MULTILINE):
            tables.append((
                match.start(),
                match.end(),
                match.group(0)
            ))
        
        # Pattern 4: CSV-like patterns
        csv_pattern = r'(^[^,\n]+,[^,\n]+(,[^,\n]+)*$\n){3,}'
        for match in re.finditer(csv_pattern, text, re.MULTILINE):
            tables.append((
                match.start(),
                match.end(),
                match.group(0)
            ))
        
        # Pattern 5: Multiple consecutive lines with pipe separators
        pipe_pattern = r'(^[^\n]*\|[^\n]*\|[^\n]*$\n){3,}'
        for match in re.finditer(pipe_pattern, text, re.MULTILINE):
            # Avoid duplicates with markdown tables
            start, end = match.start(), match.end()
            if not any(t[0] <= start < t[1] or t[0] < end <= t[1] for t in tables):
                tables.append((start, end, match.group(0)))
        
        # Sort by position
        tables.sort(key=lambda x: x[0])
        
        # Merge overlapping tables
        merged = []
        for start, end, content in tables:
            if merged and start <= merged[-1][1]:
                # Extend previous table
                prev_start, prev_end, prev_content = merged[-1]
                merged[-1] = (
                    prev_start,
                    max(end, prev_end),
                    text[prev_start:max(end, prev_end)]
                )
            else:
                merged.append((start, end, content))
        
        if merged:
            logger.info(f"📊 Detected {len(merged)} tables")
        
        return merged
    
    def _extract_table_context(
        self,
        text: str,
        table_start: int,
        table_end: int,
        context_lines: int = 2
    ) -> str:
        """
        Extract table with surrounding context
        
        Args:
            text: Full text
            table_start: Table start position
            table_end: Table end position
            context_lines: Number of lines before/after to include
            
        Returns:
            Table with context
        """
        lines = text[:table_start].split('\n')
        before = '\n'.join(lines[-context_lines:]) if len(lines) > context_lines else ''
        
        lines_after = text[table_end:].split('\n')
        after = '\n'.join(lines_after[:context_lines]) if len(lines_after) > context_lines else ''
        
        table_text = text[table_start:table_end]
        
        parts = []
        if before.strip():
            parts.append(before.strip())
        parts.append(table_text.strip())
        if after.strip():
            parts.append(after.strip())
        
        return '\n\n'.join(parts)
    
    def _split_with_tables(
        self,
        text: str,
        file_path: Path,
        min_chunk_size: int,
        max_chunk_size: int
    ) -> List[Dict[str, Any]]:
        """
        Split text while preserving tables in single chunks
        
        Strategy:
            1. Detect all tables
            2. Extract text between tables
            3. Chunk non-table text normally
            4. Keep each table + context as single chunk
        """
        if not self.preserve_tables:
            # Standard chunking
            return self._simple_chunk(text, file_path, max_chunk_size, min_chunk_size)
        
        tables = self._detect_tables(text)
        
        if not tables:
            # No tables, use standard chunking
            return self._simple_chunk(text, file_path, max_chunk_size, min_chunk_size)
        
        chunks = []
        current_pos = 0
        chunk_index = 0
        
        for table_start, table_end, table_content in tables:
            # ── Chunk 1: Text BEFORE table ────────────────────
            before_text = text[current_pos:table_start].strip()
            
            if before_text and len(before_text) >= min_chunk_size:
                # Chunk the non-table text
                before_chunks = self._simple_chunk(
                    before_text,
                    file_path,
                    max_chunk_size,
                    min_chunk_size
                )
                for chunk_data in before_chunks:
                    chunk_data['chunk_index'] = chunk_index
                    chunks.append(chunk_data)
                    chunk_index += 1
            
            # ── Chunk 2: Table ITSELF (with context) ──────────
            table_with_context = self._extract_table_context(
                text, table_start, table_end, context_lines=2
            )
            
            # Check size limit
            if len(table_with_context) > self.max_table_size:
                logger.warning(
                    f"⚠️  Table too large ({len(table_with_context)} chars), "
                    f"splitting..."
                )
                # Split large table
                table_chunks = self._simple_chunk(
                    table_with_context,
                    file_path,
                    self.max_table_size,
                    min_chunk_size
                )
                for chunk_data in table_chunks:
                    chunk_data['chunk_index'] = chunk_index
                    chunk_data['is_table'] = True
                    chunks.append(chunk_data)
                    chunk_index += 1
            else:
                # Keep table as single chunk
                chunks.append({
                    'text': table_with_context,
                    'similarity_score': 0.0,
                    'sentence_count': table_with_context.count('\n'),
                    'length': len(table_with_context),
                    'model_type': 'table_preserved',
                    'source': str(file_path),
                    'chunk_index': chunk_index,
                    'is_table': True  # ← Flag for table chunks
                })
                chunk_index += 1
            
            # Move position
            current_pos = table_end
        
        # ── Chunk 3: Text AFTER last table ────────────────────
        after_text = text[current_pos:].strip()
        
        if after_text and len(after_text) >= min_chunk_size:
            after_chunks = self._simple_chunk(
                after_text,
                file_path,
                max_chunk_size,
                min_chunk_size
            )
            for chunk_data in after_chunks:
                chunk_data['chunk_index'] = chunk_index
                chunks.append(chunk_data)
                chunk_index += 1
        
        logger.info(
            f"📊 Table-aware chunking: {len(chunks)} chunks "
            f"({len([c for c in chunks if c.get('is_table')])} tables)"
        )
        
        return chunks
    
    # ============================================================
    # PUBLIC INTERFACE
    # ============================================================
    
    def process_file(
        self,
        file_path: str,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        dynamic_threshold: bool = False
    ) -> Dict[str, Any]:
        """Process a single file"""
        file_path = Path(file_path)
        
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
            # ── STEP 1: Load documents ─────────────────────────
            logger.info("Step 1: Loading document...")
            loader_func = self._loaders.get(extension)
            documents = loader_func(file_path)
            
            if not documents:
                result['errors'].append('No text extracted')
                logger.warning(f"⚠️  No text extracted from {file_path.name}")
                return result
            
            # ── STEP 2: Get full text ──────────────────────────
            full_text = "\n\n".join([
                d.page_content for d in documents
                if d.page_content.strip()
            ])
            
            result['original_length'] = len(full_text)
            
            if len(full_text.strip()) < 10:
                result['errors'].append('Extracted text too short')
                return result
            
            # ── STEP 3: Duplicate detection ────────────────────
            logger.info("Step 2: Checking duplicates...")
            text_hash = self._generate_hash(full_text)
            
            if text_hash in self.processed_hashes:
                result['errors'].append('Duplicate content detected')
                result['warnings'].append('Already processed')
                logger.warning(f"⚠️  Duplicate: {file_path.name}")
                return result
            
            # ── STEP 4: Validate text quality ──────────────────
            logger.info("Step 3: Validating text quality...")
            if not self._is_valid_text(full_text):
                result['errors'].append('Text validation failed')
                return result
            
            # ── STEP 5: Adaptive chunking (TABLE-AWARE) ────────
            logger.info("Step 4: Chunking (table-aware)...")
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
            
            # Mark as processed
            self.processed_hashes.add(text_hash)
            
            # Extract plain chunks
            plain_chunks = [c['text'] for c in chunks_with_metadata]
            
            # Count tables
            table_chunk_count = sum(
                1 for c in chunks_with_metadata if c.get('is_table', False)
            )
            
            # Build result
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
                    'total_pages': len(documents),
                    'table_chunks': table_chunk_count,      # ← NEW
                    'preserve_tables': self.preserve_tables  # ← NEW
                }
            })
            
            logger.info(
                f"✅ {file_path.name}: "
                f"{len(chunks_with_metadata)} chunks "
                f"({table_chunk_count} tables), "
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
        """Process multiple files"""
        logger.info("="*70)
        logger.info(f"📁 Batch processing: {len(file_paths)} files")
        logger.info("="*70)
        
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
        
        # Remove duplicate chunks
        logger.info("Removing duplicate chunks...")
        before = len(results['all_chunks'])
        results['all_chunks'] = self._remove_duplicates(results['all_chunks'])
        results['total_unique_chunks'] = len(results['all_chunks'])
        duplicates_removed = before - results['total_unique_chunks']
        
        # Processing time
        duration = (datetime.now() - start_time).total_seconds()
        results['processing_time'] = duration
        
        # Count table chunks
        table_chunks = sum(
            1 for c in results['all_chunks_with_metadata']
            if c.get('is_table', False)
        )
        
        # Statistics
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
            'chunking_method': self.chunking_method,
            'table_chunks': table_chunks,                    # ← NEW
            'table_preservation_enabled': self.preserve_tables  # ← NEW
        }
        
        logger.info("="*70)
        logger.info("✅ Batch Processing Complete!")
        logger.info(f"   Total      : {results['total_files']}")
        logger.info(f"   Success    : {results['successful']}")
        logger.info(f"   Failed     : {results['failed']}")
        logger.info(f"   Duplicate  : {results['duplicate']}")
        logger.info(f"   Chunks     : {results['total_unique_chunks']}")
        logger.info(f"   Table chunks: {table_chunks}")  # ← NEW
        logger.info(f"   Time       : {duration:.2f}s")
        logger.info("="*70)
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'unique_documents': len(self.processed_hashes),
            'scanned_documents': self.scanned_count,
            'ocr_operations': self.ocr_count,
            'chunking_method': self.chunking_method,
            'semantic_model_available': self.splitter is not None,
            'table_preservation': self.preserve_tables  # ← NEW
        }
    
    def is_model_available(self) -> bool:
        """Check if model is available"""
        return self.splitter is not None
    
    def reset_duplicate_tracking(self):
        """Reset duplicate tracking"""
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
                'similarity_score': 0.0,
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
        max_chunk_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Adaptive chunking — tables (type='table') become single chunks.
        """
        text_length = len(full_text.strip())

        # ── Separate table docs from text docs ─────────────────────
        table_docs = [d for d in documents if d.metadata.get("type") == "table"]
        text_docs  = [d for d in documents if d.metadata.get("type") != "table"]

        # ── Re-build full_text without table content ────────────────
        text_only = "\n\n".join(
            d.page_content for d in text_docs if d.page_content.strip()
        )

        chunks: List[Dict[str, Any]] = []
        chunk_index = 0

        # ── Chunk 1: normal text ────────────────────────────────────
        if text_only.strip():
            if len(text_only) < 50:
                chunks.append(self._single_chunk_dict(
                    text_only.strip(), file_path, chunk_index
                ))
                chunk_index += 1
            elif len(text_only) < 200:
                for c in self._simple_chunk(
                    text_only, file_path,
                    chunk_size=len(text_only),
                    min_size=max(20, len(text_only) // 3),
                ):
                    c["chunk_index"] = chunk_index
                    chunks.append(c)
                    chunk_index += 1
            else:
                # table-aware splitting on remaining text
                for c in self._split_with_tables(
                    text_only, file_path, min_chunk_size, max_chunk_size
                ):
                    c["chunk_index"] = chunk_index
                    chunks.append(c)
                    chunk_index += 1

        # ── Chunk 2: one chunk per Docling table ────────────────────
        for doc in table_docs:
            table_text = doc.page_content.strip()
            if not table_text or len(table_text) < min_chunk_size:
                continue

            chunk = {
                # ── content ──────────────────────────────────
                "text":        table_text,
                "length":      len(table_text),

                # ── scoring ──────────────────────────────────
                "similarity_score": 0.0,
                "sentence_count":   table_text.count("\n"),

                # ── type markers ─────────────────────────────
                "type":        "table",           # ← key field
                "is_table":    True,

                # ── docling metadata ─────────────────────────
                "table_index": doc.metadata.get("table_index", 0),
                "page_no":     doc.metadata.get("page_no"),
                "caption":     doc.metadata.get("caption"),
                "row_count":   doc.metadata.get("row_count"),
                "col_count":   doc.metadata.get("col_count"),

                # ── provenance ───────────────────────────────
                "model_type":  "docling_table",
                "source":      str(file_path),
                "chunk_index": chunk_index,
            }

            chunks.append(chunk)
            chunk_index += 1

            logger.info(
                f"📊 Table chunk #{chunk_index - 1}: "
                f"{chunk['row_count']}×{chunk['col_count']} "
                f"| {len(table_text)} chars"
            )

        return chunks


    def _single_chunk_dict(
        self, text: str, file_path: Path, index: int
    ) -> Dict[str, Any]:
        """Helper: single-text chunk dict."""
        return {
            "text":             text,
            "similarity_score": 0.0,
            "sentence_count":   1,
            "length":           len(text),
            "model_type":       "single",
            "source":           str(file_path),
            "chunk_index":      index,
            "is_table":         False,
            "type":             "text",
        }    
    # ============================================================
    # FILE LOADERS
    # ============================================================
    
    def _load_pdf(self, path: Path) -> List[Document]:
        """
        Load PDF.
        
        Table extraction priority:
            1. Docling (DataFrame → to_markdown())
            2. pdfplumber (fallback)
            3. PyPDFLoader (text only)
            4. OCR (scanned)
        """
        try:
            from langchain_community.document_loaders import PyPDFLoader

            # ── Pass 1: fast text check ────────────────────────
            loader    = PyPDFLoader(str(path))
            documents = loader.load()
            full_text = " ".join(d.page_content for d in documents)
            meaningful = sum(1 for c in full_text if c.isalnum())

            if meaningful < 50:
                logger.info(f"🔍 Scanned PDF: {path.name}")
                self.scanned_count += 1
                return self._ocr_pdf(path)

            # ── Pass 2: Docling table extraction ───────────────
            if self.docling_extractor.available and self.preserve_tables:
                return self._load_pdf_docling(path, documents)

            # ── Pass 3: pdfplumber fallback ─────────────────────
            logger.info("📄 Using Camelot for table extraction")
            structured = self._load_pdf_with_tables(path)
            if structured:
                return structured

            return documents

        except Exception as e:
            logger.error(f"❌ PDF load error: {e}")
            return []


 
    def _load_pdf_with_tables(self, path: Path) -> List[Document]:
        """
        Extract tables using Camelot.
        Each table becomes a separate Document (one chunk per table).
        """
        try:
            import camelot
            
            documents = []

            # 1️⃣ Extract normal text using PyPDFLoader
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(path))
            page_docs = loader.load()

            documents.extend(page_docs)

            # 2️⃣ Extract tables using Camelot
            logger.info("📊 Extracting tables with Camelot...")
            
            tables = camelot.read_pdf(
                str(path),
                pages="all",
                flavor="lattice",   # try "stream" if lattice fails
                strip_text="\n"
            )

            logger.info(f"✅ Camelot found {len(tables)} tables")

            for idx, table in enumerate(tables):
                df = table.df  # pandas DataFrame

                if df.empty:
                    continue

                markdown = df.to_markdown(index=False)

                documents.append(Document(
                    page_content=markdown,
                    metadata={
                        "source": str(path),
                        "type": "table",
                        "is_table": True,
                        "table_index": idx,
                        "page_no": table.page,
                        "row_count": df.shape[0],
                        "col_count": df.shape[1],
                    }
                ))

            return documents

        except Exception as e:
            logger.error(f"❌ Camelot error: {e}")
            return []
    
    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """
        Convert table (list of lists) to markdown format
        
        Args:
            table: [[cell1, cell2], [cell3, cell4], ...]
            
        Returns:
            Markdown table string
        """
        if not table or len(table) < 2:
            return ""
        
        # Clean cells
        def clean_cell(cell):
            if cell is None:
                return ""
            return str(cell).strip().replace('\n', ' ')
        
        cleaned_table = [
            [clean_cell(cell) for cell in row]
            for row in table
        ]
        
        # Get max columns
        max_cols = max(len(row) for row in cleaned_table)
        
        # Pad rows to same length
        for row in cleaned_table:
            while len(row) < max_cols:
                row.append("")
        
        # Build markdown
        lines = []
        
        # Header row
        header = cleaned_table[0]
        lines.append("| " + " | ".join(header) + " |")
        
        # Separator
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        
        # Data rows
        for row in cleaned_table[1:]:
            lines.append("| " + " | ".join(row) + " |")
        
        markdown = "\n".join(lines)
        
        logger.debug(f"📊 Converted table with {len(cleaned_table)} rows, {max_cols} cols")
        
        return markdown   
    
    def _load_docx(self, path: Path) -> List[Document]:
        """Load DOCX"""
        try:
            from langchain.document_loaders import Docx2txtLoader
            
            loader = Docx2txtLoader(str(path))
            return loader.load()
            
        except Exception as e:
            logger.error(f"❌ DOCX error: {e}")
            return []
    
    def _load_doc(self, path: Path) -> List[Document]:
        """Load DOC"""
        try:
            from langchain.document_loaders import UnstructuredWordDocumentLoader
            
            loader = UnstructuredWordDocumentLoader(str(path))
            return loader.load()
            
        except Exception as e:
            logger.error(f"❌ DOC error: {e}")
            return []
    
    def _load_txt(self, path: Path) -> List[Document]:
        """Load TXT with multi-encoding"""
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
        """Load CSV"""
        try:
            from langchain.document_loaders import CSVLoader
            
            loader = CSVLoader(str(path))
            return loader.load()
            
        except Exception as e:
            logger.error(f"❌ CSV error: {e}")
            return []
    
    def _load_email(self, path: Path) -> List[Document]:
        """Load EML"""
        try:
            from langchain.document_loaders import UnstructuredEmailLoader
            
            loader = UnstructuredEmailLoader(str(path))
            return loader.load()
            
        except Exception as e:
            logger.error(f"❌ EML error: {e}")
            return []
    
    def _load_msg(self, path: Path) -> List[Document]:
        """Load MSG"""
        try:
            from langchain.document_loaders import OutlookMessageLoader
            
            loader = OutlookMessageLoader(str(path))
            return loader.load()
            
        except Exception as e:
            logger.error(f"❌ MSG error: {e}")
            return []
    
    def _load_image(self, path: Path) -> List[Document]:
        """Load image with OCR"""
        try:
            self.ocr_count += 1
            
            if self.ocr_available and self.ocr_handler:
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
            
            from langchain.document_loaders import UnstructuredImageLoader
            loader = UnstructuredImageLoader(str(path))
            return loader.load()
            
        except Exception as e:
            logger.error(f"❌ Image OCR error: {e}")
            return []
    
    def _ocr_pdf(self, path: Path) -> List[Document]:
        """OCR for scanned PDFs"""
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
    # UTILS
    # ============================================================
    
    def _generate_hash(self, text: str) -> str:
        """Generate MD5 hash"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _is_valid_text(self, text: str, min_length: int = 5) -> bool:
        """Validate text quality"""
        if not text or not isinstance(text, str):
            return False
        
        cleaned = text.strip()
        
        if len(cleaned) < min_length:
            return False
        
        if not any(c.isalnum() for c in cleaned):
            return False
        
        if len(cleaned) >= 50:
            words = cleaned.split()
            if len(words) >= 10:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:
                    return False
        
        return True
    
    def _remove_duplicates(self, texts: List[str]) -> List[str]:
        """Remove duplicate texts"""
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
    