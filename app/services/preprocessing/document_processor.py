import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import hashlib
from datetime import datetime

from .text_cleaner import TextCleaner
from .file_handler import FileHandler
from .semantic_chunker import SemanticChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Main document processing pipeline with semantic chunking"""
    
    def __init__(
        self, 
        lang: str = 'en', 
        use_gpu: bool = False,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunking_method: str = "semantic"
    ):
        """
        Initialize Document Processor
        
        Args:
            lang: OCR language code ('en', 'ch', 'french', 'german', etc.)
            use_gpu: Whether to use GPU for OCR and embeddings
            embedding_model: HuggingFace model for semantic embeddings
            chunking_method: "semantic" or "simple"
        """
        self.text_cleaner = TextCleaner()
        self.file_handler = FileHandler(lang=lang, use_gpu=use_gpu)
        self.chunking_method = chunking_method
        
        # Initialize semantic chunker
        self.semantic_chunker = SemanticChunker(
            model_name=embedding_model,
            use_gpu=use_gpu
        )
        
        self.processed_hashes = set()
        self.scanned_count = 0
        self.ocr_count = 0
    
    def process_file(
        self, 
        file_path: str,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        dynamic_threshold: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single file through the complete pipeline with semantic chunking
        
        Args:
            file_path: Path to the file to process
            similarity_threshold: For semantic chunking (0-1, lower = more chunks)
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            dynamic_threshold: Whether to adjust threshold based on text length
            
        Returns:
            Dictionary with processing results and metadata
        """
        logger.info(f"Processing file: {file_path}")
        
        result = {
            'filename': Path(file_path).name,
            'filepath': file_path,
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
        
        try:
            # Step 1: Extract text from file
            logger.info(f"Step 1: Extracting text from {Path(file_path).name}")
            extraction_result = self.file_handler.extract_text(file_path)
            
            if not extraction_result['success']:
                error_msg = extraction_result.get('error', 'Unknown extraction error')
                result['errors'].append(error_msg)
                logger.error(f"Extraction failed: {error_msg}")
                return result
            
            raw_text = extraction_result['text']
            result['original_length'] = len(raw_text)
            
            # Populate metadata
            result['metadata'] = {
                'file_size': extraction_result.get('size', 0),
                'file_type': extraction_result.get('extension', ''),
                'processed_at': datetime.now().isoformat(),
                'is_scanned': extraction_result.get('is_scanned', False),
                'is_image': extraction_result.get('is_image', False),
                'ocr_confidence': extraction_result.get('ocr_confidence', None),
                'total_pages': extraction_result.get('total_pages', None),
                'ocr_pages': extraction_result.get('ocr_pages', []),
                'chunking_method': self.chunking_method,
                'similarity_threshold': similarity_threshold,
                'dynamic_threshold': dynamic_threshold
            }
            
            # Track scanned documents
            if extraction_result.get('is_scanned') or extraction_result.get('is_image'):
                self.scanned_count += 1
                if extraction_result.get('ocr_pages'):
                    self.ocr_count += len(extraction_result['ocr_pages'])
                elif extraction_result.get('is_image'):
                    self.ocr_count += 1
            
            # Check if extraction yielded meaningful content
            if not raw_text or len(raw_text.strip()) < 10:
                result['errors'].append('Extracted text is too short or empty')
                logger.warning(f"Insufficient text extracted from {file_path}")
                return result
            
            # Step 2: Clean and normalize text
            logger.info(f"Step 2: Cleaning and normalizing text")
            cleaned_text = self.text_cleaner.clean_text(raw_text)
            
            if not cleaned_text:
                result['errors'].append('Text cleaning resulted in empty content')
                logger.warning(f"Text cleaning failed for {file_path}")
                return result
            
            # Step 3: Check for duplicates
            logger.info(f"Step 3: Checking for duplicates")
            text_hash = self._generate_hash(cleaned_text)
            
            if text_hash in self.processed_hashes:
                result['errors'].append('Duplicate content detected')
                result['warnings'].append('This file contains content that was already processed')
                logger.warning(f"Duplicate content in {file_path}")
                return result
            
            # Step 4: Validate text quality
            logger.info(f"Step 4: Validating text quality")
            if not self.text_cleaner.is_valid_text(cleaned_text):
                result['errors'].append('Text validation failed - content too short or invalid')
                logger.warning(f"Invalid text in {file_path}")
                return result
            
            # Step 5: Semantic chunking
            logger.info(f"Step 5: Creating semantic chunks (threshold={similarity_threshold})")
            chunks_with_metadata = self._semantic_chunk_text(
                cleaned_text,
                similarity_threshold=similarity_threshold,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                dynamic_threshold=dynamic_threshold
            )
            
            if not chunks_with_metadata:
                result['errors'].append('Text chunking produced no results')
                logger.warning(f"Chunking failed for {file_path}")
                return result
            
            # Mark as processed
            self.processed_hashes.add(text_hash)
            
            # Extract plain chunks for backward compatibility
            chunks = [chunk['text'] for chunk in chunks_with_metadata]
            
            # Update result with success
            result.update({
                'success': True,
                'processed_text': cleaned_text,
                'processed_length': len(cleaned_text),
                'chunks': chunks,
                'chunks_with_metadata': chunks_with_metadata,
                'chunk_count': len(chunks),
                'content_hash': text_hash,
                'compression_ratio': len(cleaned_text) / len(raw_text) if len(raw_text) > 0 else 1.0
            })
            
            logger.info(f"✓ Successfully processed {Path(file_path).name}: "
                       f"{len(chunks)} semantic chunks created, "
                       f"{len(cleaned_text)} characters")
            
            return result
        
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            result['errors'].append(error_msg)
            logger.error(f"Exception in process_file: {error_msg}", exc_info=True)
            return result
    
    def _semantic_chunk_text(
        self,
        text: str,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        dynamic_threshold: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Apply semantic chunking with adaptive behavior for short texts
        
        Args:
            text: Text to chunk
            similarity_threshold: Similarity threshold for chunking
            min_chunk_size: Minimum chunk size (will be adapted for short texts)
            max_chunk_size: Maximum chunk size
            dynamic_threshold: Whether to dynamically adjust threshold
            
        Returns:
            List of chunks with metadata (never empty if text has content)
        """
        text_length = len(text.strip())
        
        logger.info(f"🔍 Text length: {text_length} characters")
        logger.info(f"🔍 Original min/max chunk size: {min_chunk_size}/{max_chunk_size}")
        
        # ================================================================
        # ADAPTIVE STRATEGY 1: Very short text (< 50 chars)
        # Just return the whole text as one chunk
        # ================================================================
        if text_length < 50:
            logger.info(f"📝 Very short text ({text_length} chars), returning as single chunk")
            return [{
                'text': text.strip(),
                'similarity_score': None,
                'min_similarity': None,
                'max_similarity': None,
                'sentence_count': 1,
                'length': text_length,
                'model_type': 'simple',
                'model_name': 'adaptive-fallback',
                'device': 'cpu'
            }]
        
        # ================================================================
        # ADAPTIVE STRATEGY 2: Short text (50-200 chars)
        # Adjust chunk size to be more permissive
        # ================================================================
        if text_length < 200:
            logger.info(f"📝 Short text ({text_length} chars), using adapted chunking")
            adapted_min_size = max(20, text_length // 3)  # At least 20 chars, or 1/3 of text
            adapted_max_size = text_length
            logger.info(f"🔧 Adapted chunk size: {adapted_min_size}/{adapted_max_size}")
            
            chunks = self._simple_chunk_text(
                text, 
                chunk_size=adapted_max_size,
                overlap=20,
                min_chunk_size=adapted_min_size
            )
            
            return [
                {
                    'text': chunk,
                    'similarity_score': None,
                    'min_similarity': None,
                    'max_similarity': None,
                    'sentence_count': chunk.count('.') + chunk.count('!') + chunk.count('?'),
                    'length': len(chunk),
                    'model_type': 'simple-adaptive',
                    'model_name': 'adaptive-fallback',
                    'device': 'cpu'
                }
                for chunk in chunks if chunk.strip()
            ]
        
        # ================================================================
        # ADAPTIVE STRATEGY 3: Medium text (200-500 chars)
        # Use semantic if available, but with relaxed constraints
        # ================================================================
        if text_length < 500:
            logger.info(f"📝 Medium text ({text_length} chars), using relaxed semantic chunking")
            adapted_min_size = max(50, text_length // 5)
            adapted_max_size = max(200, text_length // 2)
        else:
            # Use original settings for longer texts
            adapted_min_size = min_chunk_size
            adapted_max_size = max_chunk_size
        
        # ================================================================
        # STRATEGY 4: Try semantic chunking (for medium to long texts)
        # ================================================================
        if self.chunking_method == "semantic" and self.semantic_chunker.is_model_available():
            try:
                logger.info(f"🤖 Using semantic chunking with adapted sizes: {adapted_min_size}/{adapted_max_size}")
                
                chunks = self.semantic_chunker.chunk_by_similarity_with_metadata(
                    text,
                    threshold=similarity_threshold,
                    min_chunk_size=adapted_min_size,
                    max_chunk_size=adapted_max_size,
                    dynamic_threshold=dynamic_threshold
                )
                
                if chunks:
                    logger.info(f"✅ Semantic chunker returned {len(chunks)} chunks")
                    return chunks
                else:
                    logger.warning(f"⚠️  Semantic chunker returned 0 chunks, falling back to simple chunking")
                    
            except Exception as e:
                logger.error(f"❌ Semantic chunking exception: {str(e)}")
                import traceback
                traceback.print_exc()
                logger.info("Falling back to simple chunking...")
        
        # ================================================================
        # FALLBACK: Simple chunking (always works)
        # ================================================================
        logger.info(f"📋 Using simple chunking as fallback")
        chunks = self._simple_chunk_text(
            text,
            chunk_size=adapted_max_size,
            overlap=min(100, adapted_max_size // 5),
            min_chunk_size=max(20, adapted_min_size // 2)  # Very permissive minimum
        )
        
        result = [
            {
                'text': chunk,
                'similarity_score': None,
                'min_similarity': None,
                'max_similarity': None,
                'sentence_count': chunk.count('.') + chunk.count('!') + chunk.count('?'),
                'length': len(chunk),
                'model_type': 'simple-fallback',
                'model_name': 'regex-chunker',
                'device': 'cpu'
            }
            for chunk in chunks if chunk.strip()
        ]
        
        # ================================================================
        # SAFETY NET: If still no chunks, return the whole text
        # ================================================================
        if not result:
            logger.warning("⚠️  All chunking failed, returning entire text as single chunk")
            result = [{
                'text': text.strip(),
                'similarity_score': None,
                'min_similarity': None,
                'max_similarity': None,
                'sentence_count': text.count('.') + text.count('!') + text.count('?'),
                'length': text_length,
                'model_type': 'emergency-fallback',
                'model_name': 'whole-text',
                'device': 'cpu'
            }]
        
        logger.info(f"✅ Final result: {len(result)} chunk(s)")
        return result

    def _simple_chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        min_chunk_size: int = 20  # ✅ Changed from 100 to 20
    ) -> List[str]:
        """
        Simple chunking method (fallback) - now more permissive
        
        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk
            overlap: Number of characters to overlap
            min_chunk_size: Minimum chunk size (lowered to 20)
            
        Returns:
            List of text chunks
        """
        text = text.strip()
        
        # If text is shorter than chunk_size, return as-is
        if len(text) <= chunk_size:
            if len(text) >= min_chunk_size:
                return [text]
            else:
                # Even if below min_chunk_size, return it anyway
                logger.info(f"⚠️  Text shorter than min_chunk_size ({len(text)} < {min_chunk_size}), but returning anyway")
                return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to find a sentence boundary
            if end < len(text):
                search_start = max(start, end - 100)
                search_end = min(len(text), end + 100)
                search_text = text[search_start:search_end]
                
                sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
                last_ending = -1
                last_ending_length = 0
                
                for ending in sentence_endings:
                    pos = search_text.rfind(ending)
                    if pos > last_ending:
                        last_ending = pos
                        last_ending_length = len(ending)
                
                if last_ending != -1:
                    end = search_start + last_ending + last_ending_length
            
            chunk = text[start:end].strip()
            
            # ✅ CHANGED: Accept chunk even if below min_chunk_size
            if chunk:
                chunks.append(chunk)
            
            # Move start position
            if end - overlap <= start:
                start = end
            else:
                start = end - overlap
        
        # ✅ SAFETY: If no chunks created, return whole text
        if not chunks and text:
            logger.warning("Simple chunking produced no chunks, returning whole text")
            return [text]
        
        return chunks
    
    def process_multiple_files(
        self, 
        file_paths: List[str],
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        dynamic_threshold: bool = False
    ) -> Dict[str, Any]:
        """
        Process multiple files with semantic chunking
        
        Args:
            file_paths: List of file paths
            similarity_threshold: Similarity threshold for semantic chunking
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            dynamic_threshold: Whether to dynamically adjust threshold
            
        Returns:
            Aggregated results
        """
        logger.info(f"Starting batch processing of {len(file_paths)} files")
        logger.info(f"Chunking method: {self.chunking_method}")
        logger.info(f"Semantic chunker available: {self.semantic_chunker.is_model_available()}")
        
        results = {
            'total_files': len(file_paths),
            'successful': 0,
            'failed': 0,
            'duplicate': 0,
            'total_chunks': 0,
            'scanned_count': 0,
            'ocr_count': 0,
            'total_size': 0,
            'total_original_length': 0,
            'total_processed_length': 0,
            'files': [],
            'all_chunks': [],
            'all_chunks_with_metadata': [],
            'errors': [],
            'processing_time': None
        }
        
        start_time = datetime.now()
        
        for idx, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing file {idx}/{len(file_paths)}: {Path(file_path).name}")
            
            result = self.process_file(
                file_path,
                similarity_threshold=similarity_threshold,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                dynamic_threshold=dynamic_threshold
            )
            
            if result['success']:
                results['successful'] += 1
                results['total_chunks'] += result['chunk_count']
                results['all_chunks'].extend(result['chunks'])
                results['all_chunks_with_metadata'].extend(result['chunks_with_metadata'])
                results['total_original_length'] += result['original_length']
                results['total_processed_length'] += result['processed_length']
                
                if result.get('metadata', {}).get('is_scanned', False):
                    results['scanned_count'] += 1
                if result.get('metadata', {}).get('is_image', False):
                    results['scanned_count'] += 1
                
                results['total_size'] += result.get('metadata', {}).get('file_size', 0)
                
            elif 'Duplicate' in str(result.get('errors', [])):
                results['duplicate'] += 1
            else:
                results['failed'] += 1
                results['errors'].extend(result.get('errors', []))
            
            results['files'].append(result)
        
        results['ocr_count'] = self.ocr_count
        
        # Remove duplicate chunks
        logger.info(f"Removing duplicate chunks across all files")
        unique_chunks_before = len(results['all_chunks'])
        results['all_chunks'] = self.text_cleaner.remove_duplicates(results['all_chunks'])
        results['total_unique_chunks'] = len(results['all_chunks'])
        duplicates_removed = unique_chunks_before - results['total_unique_chunks']
        
        end_time = datetime.now()
        processing_duration = (end_time - start_time).total_seconds()
        results['processing_time'] = processing_duration
        
        results['statistics'] = {
            'average_chunks_per_file': results['total_unique_chunks'] / results['successful'] if results['successful'] > 0 else 0,
            'success_rate': (results['successful'] / results['total_files'] * 100) if results['total_files'] > 0 else 0,
            'duplicate_rate': (results['duplicate'] / results['total_files'] * 100) if results['total_files'] > 0 else 0,
            'scanned_rate': (results['scanned_count'] / results['total_files'] * 100) if results['total_files'] > 0 else 0,
            'average_processing_time': processing_duration / results['total_files'] if results['total_files'] > 0 else 0,
            'compression_ratio': results['total_processed_length'] / results['total_original_length'] if results['total_original_length'] > 0 else 1.0,
            'chunks_removed_as_duplicates': duplicates_removed,
            'chunking_method': self.chunking_method
        }
        
        logger.info(f"="*80)
        logger.info(f"Batch Processing Complete!")
        logger.info(f"Total Files: {results['total_files']}")
        logger.info(f"Successful: {results['successful']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Duplicates: {results['duplicate']}")
        logger.info(f"Chunking Method: {self.chunking_method}")
        logger.info(f"Total Unique Chunks: {results['total_unique_chunks']}")
        logger.info(f"Processing Time: {processing_duration:.2f} seconds")
        logger.info(f"Success Rate: {results['statistics']['success_rate']:.2f}%")
        logger.info(f"="*80)
        
        return results
    
    def _generate_hash(self, text: str) -> str:
        """Generate hash for duplicate detection"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_chunk_with_context(
        self,
        chunks: List[str],
        query: str,
        top_k: int = 3,
        use_semantic_search: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get most relevant chunks for a query using semantic similarity
        
        Args:
            chunks: List of text chunks
            query: Search query
            top_k: Number of top chunks to return
            use_semantic_search: Whether to use semantic similarity
            
        Returns:
            List of relevant chunks with scores
        """
        if not chunks or not query:
            return []
        
        if use_semantic_search and self.semantic_chunker.is_model_available():
            return self.semantic_chunker.get_most_similar_chunks(query, chunks, top_k)
        else:
            return self._keyword_search(chunks, query, top_k)
    
    def _keyword_search(
        self,
        chunks: List[str],
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Fallback keyword-based search"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_chunks = []
        for idx, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            chunk_words = set(chunk_lower.split())
            
            common_words = query_words.intersection(chunk_words)
            score = len(common_words)
            
            if query_lower in chunk_lower:
                score += 10
            
            if score > 0:
                scored_chunks.append({
                    'chunk': chunk,
                    'similarity': score,
                    'index': idx,
                    'length': len(chunk),
                    'preview': chunk[:200] + '...' if len(chunk) > 200 else chunk
                })
        
        scored_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_chunks[:top_k]
    
    def reset_duplicate_tracking(self):
        """Reset duplicate tracking"""
        self.processed_hashes.clear()
        self.scanned_count = 0
        self.ocr_count = 0
        logger.info("Duplicate tracking and counters reset")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            'unique_documents': len(self.processed_hashes),
            'scanned_documents': self.scanned_count,
            'ocr_operations': self.ocr_count,
            'chunking_method': self.chunking_method,
            'semantic_model_available': self.semantic_chunker.is_model_available()
        }
    
    def export_chunks_to_file(
        self,
        chunks: List[Dict[str, Any]],
        output_path: str,
        format: str = 'jsonl',
        include_metadata: bool = True
    ) -> bool:
        """
        Export processed chunks to a file
        
        Args:
            chunks: List of chunks (with or without metadata)
            output_path: Output file path
            format: Export format ('txt', 'json', 'jsonl')
            include_metadata: Whether to include chunk metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Normalize chunks to dict format
            normalized_chunks = []
            for idx, chunk in enumerate(chunks, 1):
                if isinstance(chunk, dict):
                    chunk_dict = chunk.copy()
                    chunk_dict['id'] = idx
                else:
                    chunk_dict = {
                        'id': idx,
                        'text': chunk,
                        'length': len(chunk)
                    }
                normalized_chunks.append(chunk_dict)
            
            if format == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for chunk_dict in normalized_chunks:
                        f.write(f"=== Chunk {chunk_dict['id']} ===\n")
                        if include_metadata and 'similarity_score' in chunk_dict:
                            f.write(f"Similarity: {chunk_dict['similarity_score']}\n")
                            f.write(f"Sentences: {chunk_dict.get('sentence_count', 'N/A')}\n")
                        f.write(chunk_dict['text'])
                        f.write("\n\n" + "="*80 + "\n\n")
            
            elif format == 'json':
                data = {
                    'chunks': normalized_chunks,
                    'total_chunks': len(normalized_chunks),
                    'exported_at': datetime.now().isoformat()
                }
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            elif format == 'jsonl':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for chunk_dict in normalized_chunks:
                        f.write(json.dumps(chunk_dict, ensure_ascii=False) + '\n')
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported {len(normalized_chunks)} chunks to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting chunks: {str(e)}")
            return False