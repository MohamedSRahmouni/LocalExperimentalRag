"""
Batch Embedding Processor - Handles batch processing of multiple documents
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .chunk_embedder import ChunkEmbedder

logger = logging.getLogger(__name__)


class BatchEmbeddingProcessor:
    """
    Process multiple documents and their chunks for embedding
    """
    
    def __init__(
        self,
        embedder: Optional[ChunkEmbedder] = None,
        max_workers: int = 2,
        save_intermediate: bool = True,
        output_dir: Optional[str] = None
    ):
        """
        Initialize batch processor
        
        Args:
            embedder: ChunkEmbedder instance
            max_workers: Number of parallel workers
            save_intermediate: Save intermediate results
            output_dir: Directory to save intermediate results
        """
        self.embedder = embedder or ChunkEmbedder()
        self.max_workers = max_workers
        self.save_intermediate = save_intermediate
        self.output_dir = Path(output_dir) if output_dir else Path("embeddings_output")
        
        if self.save_intermediate:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Intermediate results will be saved to: {self.output_dir}")
    
    def process_document(
        self,
        document_data: Dict[str, Any],
        show_progress: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single document's chunks
        
        Args:
            document_data: Document data from DocumentProcessor
            show_progress: Show progress for this document
            
        Returns:
            Processed document with embeddings
        """
        if not document_data.get('success'):
            logger.warning(f"Skipping failed document: {document_data.get('filename')}")
            return document_data
        
        filename = document_data.get('filename', 'unknown')
        logger.info(f"Processing document: {filename}")
        
        try:
            # Get chunks with metadata
            chunks_with_metadata = document_data.get('chunks_with_metadata', [])
            
            if not chunks_with_metadata:
                # Fallback to plain chunks
                plain_chunks = document_data.get('chunks', [])
                if plain_chunks:
                    chunks_with_metadata = [{'text': chunk} for chunk in plain_chunks]
                else:
                    logger.warning(f"No chunks found in document: {filename}")
                    return document_data
            
            # Embed chunks
            embedded_chunks = self.embedder.embed_chunks_with_metadata(
                chunks_with_metadata,
                show_progress=show_progress
            )
            
            # Calculate stats
            embedding_stats = self.embedder.get_embedding_stats(embedded_chunks)
            
            # Update document data
            result = {
                **document_data,
                'embedded_chunks': embedded_chunks,
                'embedding_stats': embedding_stats,
                'embedding_complete': True,
                'embedding_timestamp': datetime.now().isoformat()
            }
            
            # Save intermediate result if enabled
            if self.save_intermediate:
                self._save_document_embeddings(result)
            
            logger.info(f"✓ Document '{filename}': {len(embedded_chunks)} chunks embedded")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document '{filename}': {str(e)}")
            document_data['embedding_error'] = str(e)
            document_data['embedding_complete'] = False
            return document_data
    
    def process_batch(
        self,
        documents: List[Dict[str, Any]],
        parallel: bool = False,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple documents
        
        Args:
            documents: List of document data from DocumentProcessor
            parallel: Use parallel processing
            show_progress: Show progress bars
            
        Returns:
            Batch processing results
        """
        logger.info(f"Processing batch of {len(documents)} documents")
        logger.info(f"Parallel processing: {parallel}")
        
        start_time = datetime.now()
        results = {
            'total_documents': len(documents),
            'successful': 0,
            'failed': 0,
            'total_chunks_embedded': 0,
            'documents': [],
            'errors': [],
            'processing_time': None
        }
        
        if parallel and self.max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_doc = {
                    executor.submit(self.process_document, doc, False): doc
                    for doc in documents
                }
                
                for future in as_completed(future_to_doc):
                    try:
                        result = future.result()
                        results['documents'].append(result)
                        
                        if result.get('embedding_complete'):
                            results['successful'] += 1
                            results['total_chunks_embedded'] += len(result.get('embedded_chunks', []))
                        else:
                            results['failed'] += 1
                            if 'embedding_error' in result:
                                results['errors'].append({
                                    'filename': result.get('filename'),
                                    'error': result['embedding_error']
                                })
                    except Exception as e:
                        logger.error(f"Error in parallel processing: {str(e)}")
                        results['failed'] += 1
        else:
            # Sequential processing
            for i, doc in enumerate(documents, 1):
                logger.info(f"Processing document {i}/{len(documents)}")
                result = self.process_document(doc, show_progress=show_progress)
                results['documents'].append(result)
                
                if result.get('embedding_complete'):
                    results['successful'] += 1
                    results['total_chunks_embedded'] += len(result.get('embedded_chunks', []))
                else:
                    results['failed'] += 1
                    if 'embedding_error' in result:
                        results['errors'].append({
                            'filename': result.get('filename'),
                            'error': result['embedding_error']
                        })
        
        # Calculate processing time
        end_time = datetime.now()
        processing_duration = (end_time - start_time).total_seconds()
        results['processing_time'] = processing_duration
        
        # Calculate statistics
        results['statistics'] = {
            'success_rate': (results['successful'] / results['total_documents'] * 100) if results['total_documents'] > 0 else 0,
            'avg_chunks_per_document': results['total_chunks_embedded'] / results['successful'] if results['successful'] > 0 else 0,
            'avg_processing_time': processing_duration / results['total_documents'] if results['total_documents'] > 0 else 0,
            'chunks_per_second': results['total_chunks_embedded'] / processing_duration if processing_duration > 0 else 0
        }
        
        # Save batch summary
        if self.save_intermediate:
            self._save_batch_summary(results)
        
        # Log summary
        logger.info("="*70)
        logger.info("BATCH EMBEDDING COMPLETE")
        logger.info(f"Total Documents: {results['total_documents']}")
        logger.info(f"Successful: {results['successful']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Total Chunks Embedded: {results['total_chunks_embedded']}")
        logger.info(f"Processing Time: {processing_duration:.2f}s")
        logger.info(f"Success Rate: {results['statistics']['success_rate']:.2f}%")
        logger.info("="*70)
        
        return results
    
    def _save_document_embeddings(self, document_data: Dict[str, Any]):
        """Save document embeddings to file"""
        try:
            filename = document_data.get('filename', 'unknown')
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_file = self.output_dir / f"{safe_filename}_embeddings.json"
            
            # Prepare data for saving (exclude large arrays if needed)
            save_data = {
                'filename': filename,
                'chunk_count': len(document_data.get('embedded_chunks', [])),
                'embedding_stats': document_data.get('embedding_stats'),
                'embedded_at': document_data.get('embedding_timestamp'),
                'embedded_chunks': document_data.get('embedded_chunks', [])
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved embeddings to: {output_file}")
            
        except Exception as e:
            logger.warning(f"Could not save intermediate results: {str(e)}")
    
    def _save_batch_summary(self, results: Dict[str, Any]):
        """Save batch processing summary"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = self.output_dir / f"batch_summary_{timestamp}.json"
            
            # Exclude embedded_chunks from summary to keep file small
            summary = {
                'total_documents': results['total_documents'],
                'successful': results['successful'],
                'failed': results['failed'],
                'total_chunks_embedded': results['total_chunks_embedded'],
                'processing_time': results['processing_time'],
                'statistics': results['statistics'],
                'errors': results['errors'],
                'processed_at': datetime.now().isoformat()
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Batch summary saved to: {summary_file}")
            
        except Exception as e:
            logger.warning(f"Could not save batch summary: {str(e)}")