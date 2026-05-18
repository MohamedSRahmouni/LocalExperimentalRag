"""
Document Loader Pipeline
LangChain-powered document loading, cleaning, and chunking

Features:
    - Camelot table extraction (PDF only)
    - Semantic chunking (max 700 tokens)
    - Table preservation in single chunks
    - OCR support for scanned PDFs and images
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
        - Camelot table extraction (PDF only)
        - Semantic chunking (max 700 tokens)
        - Table preservation in single chunks
        - OCR for scanned PDFs and images
    """

    def __init__(
        self,
        lang:                 str   = 'en',
        use_gpu:              bool  = False,
        chunk_size:           int   = 700,       # ✅ max 700 tokens
        chunk_overlap:        int   = 100,
        min_chunk_size:       int   = 10,
        similarity_threshold: float = 0.5,
        chunking_method:      str   = "semantic",
        embedding_model:      str   = "sentence-transformers/all-MiniLM-L6-v2",
        preserve_tables:      bool  = True,
        max_table_size:       int   = 5000,
    ):
        self.lang                 = lang
        self.use_gpu              = use_gpu
        self.chunk_size           = chunk_size
        self.chunk_overlap        = chunk_overlap
        self.min_chunk_size       = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.chunking_method      = chunking_method
        self.embedding_model      = embedding_model
        self.preserve_tables      = preserve_tables
        self.max_table_size       = max_table_size

        # Duplicate tracking
        self.processed_hashes = set()
        self.scanned_count    = 0
        self.ocr_count        = 0

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

        logger.info("=" * 70)
        logger.info("✅ LangChainDocumentLoader initialized")
        logger.info(f"   Chunking method : {chunking_method}")
        logger.info(f"   Chunk size      : {chunk_size} tokens")
        logger.info(f"   Min chunk size  : {min_chunk_size}")
        logger.info(f"   Preserve tables : {preserve_tables}")
        logger.info(f"   OCR language    : {lang}")
        logger.info("=" * 70)

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
            self.ocr_handler   = None
            self.ocr_available = False

    def _create_splitter(self):
        """Create text splitter — semantic (700 tokens) or simple fallback"""
        if self.chunking_method != "semantic":
            return self._create_simple_splitter()

        try:
            model_path_str = self.embedding_model
            resolved_path  = None

            # 1. Check relative to current working directory
            cwd_path = Path(model_path_str)
            if cwd_path.exists():
                resolved_path = cwd_path.resolve()
                logger.info(f"✅ Splitter model found at: {resolved_path}")

            # 2. Check one level up
            else:
                parent_path = Path("..") / model_path_str.lstrip('./').lstrip('.\\')
                if parent_path.exists():
                    resolved_path = parent_path.resolve()
                    logger.info(
                        f"✅ Splitter model found via parent: {resolved_path}"
                    )

                # 3. Fallback: anchor from this file
                else:
                    root_fallback = Path(__file__).parents[2] / "models" / "bge-m3"
                    if root_fallback.exists():
                        resolved_path = root_fallback.resolve()
                        logger.info(
                            f"✅ Splitter model found via anchoring: {resolved_path}"
                        )

            if not resolved_path:
                logger.warning(
                    f"⚠️  Model path not found: {model_path_str}, "
                    f"falling back to simple splitter"
                )
                return self._create_simple_splitter()

            # Enforce offline loading
            os.environ['HF_HUB_OFFLINE']       = '1'
            os.environ['TRANSFORMERS_OFFLINE']  = '1'

            # ✅ 700 tokens max
            splitter = SentenceTransformersTokenTextSplitter(
                model_name=str(resolved_path),
                chunk_overlap=50,
                tokens_per_chunk=700        # ✅ changed from 256 to 700
            )
            logger.info(
                f"✅ Semantic splitter initialized "
                f"(700 tokens) at {resolved_path}"
            )
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
    # TABLE DETECTION (text-based, for non-PDF files)
    # ============================================================

    def _detect_tables(self, text: str) -> List[Tuple[int, int, str]]:
        """Detect markdown/ASCII tables in plain text"""
        tables = []

        # Pattern 1: Markdown tables
        markdown_pattern = r'(\|[^\n]+\|\n)+(\|[-: ]+\|[-: |\n]+)'
        for match in re.finditer(markdown_pattern, text):
            tables.append((match.start(), match.end(), match.group(0)))

        # Pattern 2: Grid tables (ASCII art)
        grid_pattern = r'(\+[-+]+\+\n(\|[^\n]+\|\n)+)+'
        for match in re.finditer(grid_pattern, text):
            tables.append((match.start(), match.end(), match.group(0)))

        # Pattern 3: TSV
        tsv_pattern = r'(^[^\t\n]+\t[^\t\n]+(\t[^\t\n]+)*$\n){3,}'
        for match in re.finditer(tsv_pattern, text, re.MULTILINE):
            tables.append((match.start(), match.end(), match.group(0)))

        # Pattern 4: Pipe lines
        pipe_pattern = r'(^[^\n]*\|[^\n]*\|[^\n]*$\n){3,}'
        for match in re.finditer(pipe_pattern, text, re.MULTILINE):
            start, end = match.start(), match.end()
            if not any(
                t[0] <= start < t[1] or t[0] < end <= t[1]
                for t in tables
            ):
                tables.append((start, end, match.group(0)))

        # Sort by position
        tables.sort(key=lambda x: x[0])

        # Merge overlapping
        merged = []
        for start, end, content in tables:
            if merged and start <= merged[-1][1]:
                prev_start, prev_end, _ = merged[-1]
                merged[-1] = (
                    prev_start,
                    max(end, prev_end),
                    text[prev_start:max(end, prev_end)]
                )
            else:
                merged.append((start, end, content))

        if merged:
            logger.info(f"📊 Detected {len(merged)} text tables")

        return merged

    def _extract_table_context(
        self,
        text:          str,
        table_start:   int,
        table_end:     int,
        context_lines: int = 2
    ) -> str:
        """Extract table with surrounding context lines"""
        lines  = text[:table_start].split('\n')
        before = '\n'.join(lines[-context_lines:]) if len(lines) > context_lines else ''

        lines_after = text[table_end:].split('\n')
        after = '\n'.join(lines_after[:context_lines]) if lines_after else ''

        table_text = text[table_start:table_end]

        parts = []
        if before.strip():
            parts.append(before.strip())
        parts.append(table_text.strip())
        if after.strip():
            parts.append(after.strip())

        return '\n\n'.join(parts)

    # ============================================================
    # CHUNKING
    # ============================================================

    def _split_with_tables(
        self,
        text:           str,
        file_path:      Path,
        min_chunk_size: int,
        max_chunk_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Split text preserving detected tables.
        If no tables detected → use semantic chunking directly.
        """
        if not self.preserve_tables:
            return self._semantic_chunk(text, file_path, min_chunk_size)

        tables = self._detect_tables(text)

        if not tables:
            # ✅ No tables → pure semantic chunking
            return self._semantic_chunk(text, file_path, min_chunk_size)

        chunks      = []
        current_pos = 0
        chunk_index = 0

        for table_start, table_end, _ in tables:

            # ── Text BEFORE table ──────────────────────────────
            before_text = text[current_pos:table_start].strip()
            if before_text and len(before_text) >= min_chunk_size:
                for c in self._semantic_chunk(before_text, file_path, min_chunk_size):
                    c['chunk_index'] = chunk_index
                    chunks.append(c)
                    chunk_index += 1

            # ── Table itself ───────────────────────────────────
            table_with_context = self._extract_table_context(
                text, table_start, table_end, context_lines=2
            )

            if len(table_with_context) > self.max_table_size:
                logger.warning(
                    f"⚠️  Table too large ({len(table_with_context)} chars), splitting..."
                )
                for c in self._semantic_chunk(
                    table_with_context, file_path, min_chunk_size
                ):
                    c['chunk_index'] = chunk_index
                    c['is_table']    = True
                    chunks.append(c)
                    chunk_index += 1
            else:
                chunks.append({
                    'text':             table_with_context,
                    'similarity_score': 0.0,
                    'sentence_count':   table_with_context.count('\n'),
                    'length':           len(table_with_context),
                    'model_type':       'table_preserved',
                    'source':           str(file_path),
                    'chunk_index':      chunk_index,
                    'is_table':         True,
                    'type':             'table',
                })
                chunk_index += 1

            current_pos = table_end

        # ── Text AFTER last table ──────────────────────────────
        after_text = text[current_pos:].strip()
        if after_text and len(after_text) >= min_chunk_size:
            for c in self._semantic_chunk(after_text, file_path, min_chunk_size):
                c['chunk_index'] = chunk_index
                chunks.append(c)
                chunk_index += 1

        logger.info(
            f"📊 Table-aware chunking: {len(chunks)} chunks "
            f"({sum(1 for c in chunks if c.get('is_table'))} tables)"
        )
        return chunks

    def _semantic_chunk(
        self,
        text:           str,
        file_path:      Path,
        min_size:       int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Semantic chunking (700 tokens via SentenceTransformers).
        Falls back to simple splitter if model unavailable.
        """
        try:
            if self.splitter and isinstance(
                self.splitter, SentenceTransformersTokenTextSplitter
            ):
                chunks = self.splitter.split_text(text)
                model_type = 'semantic_700'
            else:
                raise ValueError("No semantic splitter")

        except Exception as e:
            logger.warning(f"⚠️  Semantic split failed: {e}, using simple")
            return self._simple_chunk(text, file_path, self.chunk_size, min_size)

        if not chunks and text.strip():
            chunks = [text.strip()]

        return [
            {
                'text':             chunk,
                'similarity_score': 0.0,
                'sentence_count':   (
                    chunk.count('.') + chunk.count('!') + chunk.count('?')
                ),
                'length':           len(chunk),
                'model_type':       model_type,
                'source':           str(file_path),
                'chunk_index':      i,
                'is_table':         False,
                'type':             'text',
            }
            for i, chunk in enumerate(chunks)
            if chunk.strip() and len(chunk.strip()) >= min_size
        ]

    def _simple_chunk(
        self,
        text:       str,
        file_path:  Path,
        chunk_size: int = 700,
        min_size:   int = 20,
    ) -> List[Dict[str, Any]]:
        """Simple character-based fallback splitter"""
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
                'text':             chunk,
                'similarity_score': 0.0,
                'sentence_count':   (
                    chunk.count('.') + chunk.count('!') + chunk.count('?')
                ),
                'length':           len(chunk),
                'model_type':       'simple',
                'source':           str(file_path),
                'chunk_index':      i,
                'is_table':         False,
                'type':             'text',
            }
            for i, chunk in enumerate(chunks)
            if chunk.strip() and len(chunk.strip()) >= min_size
        ]

    def _single_chunk_dict(
        self,
        text:      str,
        file_path: Path,
        index:     int
    ) -> Dict[str, Any]:
        """Single-text chunk dict"""
        return {
            'text':             text,
            'similarity_score': 0.0,
            'sentence_count':   1,
            'length':           len(text),
            'model_type':       'single',
            'source':           str(file_path),
            'chunk_index':      index,
            'is_table':         False,
            'type':             'text',
        }

    def _adaptive_chunk(
        self,
        documents:      List[Document],
        full_text:      str,
        file_path:      Path,
        min_chunk_size: int,
        max_chunk_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Adaptive chunking:
        - Table docs (from Camelot) → single chunk each
        - Text docs → semantic chunking (700 tokens)
          with inline table detection
        """
        # ── Separate table docs from text docs ─────────────────
        table_docs = [d for d in documents if d.metadata.get("type") == "table"]
        text_docs  = [d for d in documents if d.metadata.get("type") != "table"]

        # ── Rebuild text-only content ───────────────────────────
        text_only = "\n\n".join(
            d.page_content for d in text_docs if d.page_content.strip()
        )

        chunks:      List[Dict[str, Any]] = []
        chunk_index: int                  = 0

        # ── Chunk text content ──────────────────────────────────
        if text_only.strip():
            if len(text_only) < 50:
                chunks.append(
                    self._single_chunk_dict(text_only.strip(), file_path, chunk_index)
                )
                chunk_index += 1

            elif len(text_only) < 200:
                for c in self._simple_chunk(
                    text_only, file_path,
                    chunk_size=len(text_only),
                    min_size=max(20, len(text_only) // 3),
                ):
                    c['chunk_index'] = chunk_index
                    chunks.append(c)
                    chunk_index += 1

            else:
                # ✅ Semantic chunking with inline table detection
                for c in self._split_with_tables(
                    text_only, file_path, min_chunk_size, max_chunk_size
                ):
                    c['chunk_index'] = chunk_index
                    chunks.append(c)
                    chunk_index += 1

        # ── One chunk per Camelot table ─────────────────────────
        for doc in table_docs:
            table_text = doc.page_content.strip()
            if not table_text or len(table_text) < min_chunk_size:
                continue

            chunk = {
                'text':             table_text,
                'length':           len(table_text),
                'similarity_score': 0.0,
                'sentence_count':   table_text.count('\n'),
                'type':             'table',
                'is_table':         True,
                'table_index':      doc.metadata.get('table_index', 0),
                'page_no':          doc.metadata.get('page_no'),
                'caption':          doc.metadata.get('caption'),
                'row_count':        doc.metadata.get('row_count'),
                'col_count':        doc.metadata.get('col_count'),
                'model_type':       'camelot_table',
                'source':           str(file_path),
                'chunk_index':      chunk_index,
            }

            chunks.append(chunk)
            chunk_index += 1

            logger.info(
                f"📊 Camelot table chunk #{chunk_index - 1}: "
                f"{chunk['row_count']}×{chunk['col_count']} "
                f"| {len(table_text)} chars"
            )

        return chunks

    # ============================================================
    # PUBLIC INTERFACE
    # ============================================================

    def process_file(
        self,
        file_path:            str,
        similarity_threshold: float = 0.5,
        min_chunk_size:       int   = 100,
        max_chunk_size:       int   = 700,    # ✅ aligned with 700 token limit
        dynamic_threshold:    bool  = False
    ) -> Dict[str, Any]:
        """Process a single file"""
        file_path = Path(file_path)

        result = {
            'filename':             file_path.name,
            'filepath':             str(file_path),
            'success':              False,
            'processed_text':       '',
            'original_length':      0,
            'processed_length':     0,
            'chunks':               [],
            'chunks_with_metadata': [],
            'chunk_count':          0,
            'metadata':             {},
            'errors':               [],
            'warnings':             []
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
            # ── STEP 1: Load ───────────────────────────────────
            logger.info("Step 1: Loading document...")
            loader_func = self._loaders.get(extension)
            documents   = loader_func(file_path)

            if not documents:
                result['errors'].append('No text extracted')
                logger.warning(f"⚠️  No text extracted from {file_path.name}")
                return result

            # ── STEP 2: Full text ──────────────────────────────
            full_text = "\n\n".join([
                d.page_content for d in documents
                if d.page_content.strip()
            ])

            result['original_length'] = len(full_text)

            if len(full_text.strip()) < 10:
                result['errors'].append('Extracted text too short')
                return result

            # ── STEP 3: Duplicate check ────────────────────────
            logger.info("Step 2: Checking duplicates...")
            text_hash = self._generate_hash(full_text)

            if text_hash in self.processed_hashes:
                result['errors'].append('Duplicate content detected')
                result['warnings'].append('Already processed')
                logger.warning(f"⚠️  Duplicate: {file_path.name}")
                return result

            # ── STEP 4: Validate ───────────────────────────────
            logger.info("Step 3: Validating text quality...")
            if not self._is_valid_text(full_text):
                result['errors'].append('Text validation failed')
                return result

            # ── STEP 5: Chunk ──────────────────────────────────
            logger.info("Step 4: Chunking (semantic + table-aware)...")
            chunks_with_metadata = self._adaptive_chunk(
                documents=documents,
                full_text=full_text,
                file_path=file_path,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
            )

            if not chunks_with_metadata:
                result['errors'].append('Chunking produced no results')
                return result

            # Mark as processed
            self.processed_hashes.add(text_hash)

            plain_chunks      = [c['text'] for c in chunks_with_metadata]
            table_chunk_count = sum(
                1 for c in chunks_with_metadata if c.get('is_table', False)
            )

            result.update({
                'success':              True,
                'processed_text':       full_text,
                'processed_length':     len(full_text),
                'original_length':      len(full_text),
                'chunks':               plain_chunks,
                'chunks_with_metadata': chunks_with_metadata,
                'chunk_count':          len(chunks_with_metadata),
                'content_hash':         text_hash,
                'compression_ratio':    1.0,
                'metadata': {
                    'file_size':           file_path.stat().st_size,
                    'file_type':           extension,
                    'processed_at':        datetime.now().isoformat(),
                    'is_scanned':          self._is_scanned(extension),
                    'is_image':            extension in IMAGE_EXTENSIONS,
                    'chunking_method':     self.chunking_method,
                    'similarity_threshold': similarity_threshold,
                    'dynamic_threshold':   dynamic_threshold,
                    'total_pages':         len(documents),
                    'table_chunks':        table_chunk_count,
                    'preserve_tables':     self.preserve_tables,
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
        file_paths:           List[str],
        similarity_threshold: float = 0.5,
        min_chunk_size:       int   = 100,
        max_chunk_size:       int   = 700,    # ✅ aligned with 700 token limit
        dynamic_threshold:    bool  = False
    ) -> Dict[str, Any]:
        """Process multiple files"""
        logger.info("=" * 70)
        logger.info(f"📁 Batch processing: {len(file_paths)} files")
        logger.info("=" * 70)

        results = {
            'total_files':            len(file_paths),
            'successful':             0,
            'failed':                 0,
            'duplicate':              0,
            'total_chunks':           0,
            'total_unique_chunks':    0,
            'scanned_count':          0,
            'ocr_count':              0,
            'total_size':             0,
            'total_original_length':  0,
            'total_processed_length': 0,
            'files':                  [],
            'all_chunks':             [],
            'all_chunks_with_metadata': [],
            'errors':                 [],
            'processing_time':        None,
            'statistics':             {}
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
                dynamic_threshold=dynamic_threshold,
            )

            results['files'].append(result)

            if result['success']:
                results['successful']             += 1
                results['total_chunks']           += result['chunk_count']
                results['all_chunks'].extend(result['chunks'])
                results['all_chunks_with_metadata'].extend(
                    result['chunks_with_metadata']
                )
                results['total_original_length']  += result['original_length']
                results['total_processed_length'] += result['processed_length']
                results['total_size']             += (
                    result.get('metadata', {}).get('file_size', 0)
                )
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
        results['all_chunks']          = self._remove_duplicates(results['all_chunks'])
        results['total_unique_chunks'] = len(results['all_chunks'])
        duplicates_removed             = before - results['total_unique_chunks']

        duration = (datetime.now() - start_time).total_seconds()
        results['processing_time'] = duration

        table_chunks = sum(
            1 for c in results['all_chunks_with_metadata']
            if c.get('is_table', False)
        )

        results['statistics'] = {
            'average_chunks_per_file': (
                results['total_unique_chunks'] / results['successful']
                if results['successful'] > 0 else 0
            ),
            'success_rate': (
                results['successful'] / results['total_files'] * 100
                if results['total_files'] > 0 else 0
            ),
            'duplicate_rate': (
                results['duplicate'] / results['total_files'] * 100
                if results['total_files'] > 0 else 0
            ),
            'scanned_rate': (
                results['scanned_count'] / results['total_files'] * 100
                if results['total_files'] > 0 else 0
            ),
            'average_processing_time': (
                duration / results['total_files']
                if results['total_files'] > 0 else 0
            ),
            'chunks_removed_as_duplicates':  duplicates_removed,
            'chunking_method':               self.chunking_method,
            'table_chunks':                  table_chunks,
            'table_preservation_enabled':    self.preserve_tables,
        }

        logger.info("=" * 70)
        logger.info("✅ Batch Processing Complete!")
        logger.info(f"   Total       : {results['total_files']}")
        logger.info(f"   Success     : {results['successful']}")
        logger.info(f"   Failed      : {results['failed']}")
        logger.info(f"   Duplicate   : {results['duplicate']}")
        logger.info(f"   Chunks      : {results['total_unique_chunks']}")
        logger.info(f"   Table chunks: {table_chunks}")
        logger.info(f"   Time        : {duration:.2f}s")
        logger.info("=" * 70)

        return results

    def get_processing_stats(self) -> Dict[str, Any]:
        return {
            'unique_documents':       len(self.processed_hashes),
            'scanned_documents':      self.scanned_count,
            'ocr_operations':         self.ocr_count,
            'chunking_method':        self.chunking_method,
            'semantic_model_available': self.splitter is not None,
            'table_preservation':     self.preserve_tables,
        }

    def is_model_available(self) -> bool:
        return self.splitter is not None

    def reset_duplicate_tracking(self):
        self.processed_hashes.clear()
        self.scanned_count = 0
        self.ocr_count     = 0
        logger.info("✅ Duplicate tracking reset")

    # ============================================================
    # FILE LOADERS
    # ============================================================

    def _load_pdf(self, path: Path) -> List[Document]:
        """
        Load PDF.

        Strategy:
            1. PyPDFLoader for text extraction
            2. If scanned → OCR
            3. Camelot for table extraction (if tables exist)
            4. Merge text + table documents
        """
        try:
            from langchain_community.document_loaders import PyPDFLoader

            # ── Pass 1: Text extraction ────────────────────────
            loader    = PyPDFLoader(str(path))
            documents = loader.load()
            full_text = " ".join(d.page_content for d in documents)
            meaningful = sum(1 for c in full_text if c.isalnum())

            # ── Scanned PDF → OCR ──────────────────────────────
            if meaningful < 50:
                logger.info(f"🔍 Scanned PDF: {path.name}")
                self.scanned_count += 1
                return self._ocr_pdf(path)

            # ── Pass 2: Try Camelot for tables ─────────────────
            if self.preserve_tables:
                table_docs = self._extract_tables_camelot(path)
                if table_docs:
                    logger.info(
                        f"📊 Camelot extracted {len(table_docs)} table(s)"
                    )
                    return documents + table_docs

            return documents

        except Exception as e:
            logger.error(f"❌ PDF load error: {e}")
            return []

    def _extract_tables_camelot(self, path: Path) -> List[Document]:
        """
        Extract tables using Camelot.
        Returns list of table Documents (empty if no tables or Camelot unavailable).
        """
        try:
            import camelot

            logger.info(f"📊 Running Camelot on: {path.name}")

            # Try lattice first (for bordered tables)
            try:
                tables = camelot.read_pdf(
                    str(path),
                    pages="all",
                    flavor="lattice",
                    strip_text="\n"
                )
            except Exception:
                # Fallback to stream (for borderless tables)
                tables = camelot.read_pdf(
                    str(path),
                    pages="all",
                    flavor="stream",
                    strip_text="\n"
                )

            if not tables or len(tables) == 0:
                logger.info("   Camelot: no tables found")
                return []

            logger.info(f"   Camelot found: {len(tables)} table(s)")

            table_docs = []
            for idx, table in enumerate(tables):
                df = table.df

                if df.empty:
                    continue

                # ✅ Only include tables with enough data
                if df.shape[0] < 2 or df.shape[1] < 2:
                    continue

                try:
                    markdown = df.to_markdown(index=False)
                except Exception:
                    # Fallback if tabulate not installed
                    markdown = df.to_csv(index=False)

                table_docs.append(Document(
                    page_content=markdown,
                    metadata={
                        "source":      str(path),
                        "type":        "table",
                        "is_table":    True,
                        "table_index": idx,
                        "page_no":     table.page,
                        "row_count":   df.shape[0],
                        "col_count":   df.shape[1],
                        "caption":     f"Table {idx + 1} (page {table.page})",
                    }
                ))

            return table_docs

        except ImportError:
            logger.warning("⚠️  Camelot not installed — no table extraction")
            return []
        except Exception as e:
            logger.warning(f"⚠️  Camelot error: {e} — skipping table extraction")
            return []

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """Convert list-of-lists table to markdown"""
        if not table or len(table) < 2:
            return ""

        def clean_cell(cell):
            if cell is None:
                return ""
            return str(cell).strip().replace('\n', ' ')

        cleaned = [[clean_cell(c) for c in row] for row in table]
        max_cols = max(len(row) for row in cleaned)
        for row in cleaned:
            while len(row) < max_cols:
                row.append("")

        lines = []
        lines.append("| " + " | ".join(cleaned[0]) + " |")
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        for row in cleaned[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _load_docx(self, path: Path) -> List[Document]:
        """Load DOCX"""
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(str(path))
            return loader.load()
        except Exception as e:
            logger.error(f"❌ DOCX error: {e}")
            return []

    def _load_doc(self, path: Path) -> List[Document]:
        """Load DOC"""
        try:
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(str(path))
            return loader.load()
        except Exception as e:
            logger.error(f"❌ DOC error: {e}")
            return []

    def _load_txt(self, path: Path) -> List[Document]:
        """Load TXT with multi-encoding"""
        for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                from langchain_community.document_loaders import TextLoader
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
            from langchain_community.document_loaders import CSVLoader
            loader = CSVLoader(str(path))
            return loader.load()
        except Exception as e:
            logger.error(f"❌ CSV error: {e}")
            return []

    def _load_email(self, path: Path) -> List[Document]:
        """Load EML"""
        try:
            from langchain_community.document_loaders import UnstructuredEmailLoader
            loader = UnstructuredEmailLoader(str(path))
            return loader.load()
        except Exception as e:
            logger.error(f"❌ EML error: {e}")
            return []

    def _load_msg(self, path: Path) -> List[Document]:
        """Load MSG"""
        try:
            from langchain_community.document_loaders import OutlookMessageLoader
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
                            'source':         str(path),
                            'is_image':       True,
                            'ocr_confidence': result.get('confidence', 0),
                            'word_count':     result.get('word_count', 0),
                        }
                    )]

            from langchain_community.document_loaders import UnstructuredImageLoader
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
                                'source':         str(path),
                                'page':           i,
                                'is_scanned':     True,
                                'ocr_confidence': result.get('confidence', 0),
                            }
                        ))
                        self.ocr_count += 1
                finally:
                    if Path(temp_path).exists():
                        Path(temp_path).unlink()

            logger.info(
                f"✅ OCR complete: {len(documents)} pages from {path.name}"
            )

        except Exception as e:
            logger.error(f"❌ OCR PDF error: {e}")

        return documents

    # ============================================================
    # UTILS
    # ============================================================

    def _generate_hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _is_valid_text(self, text: str, min_length: int = 5) -> bool:
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
        seen   = set()
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
        return extension in IMAGE_EXTENSIONS