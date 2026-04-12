import logging
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class ChunkEmbedder:
    """
    Handles embedding generation for text chunks
    Uses sentence-transformers for embeddings
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        use_gpu: bool = False,
        cache_dir: Optional[str] = None,
        batch_size: int = 8
    ):
        """
        Initialize chunk embedder
        
        Args:
            model_name: Embedding model name
            use_gpu: Use GPU for embedding
            cache_dir: Cache directory for model
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.embedder = None
        self.embedding_dim = None
        
        self._load_embedder()
    
    def _load_embedder(self):
        """Load the embedding model"""
        try:
            # Import TextEmbedder from preprocessing
            from app.services.preprocessing.embedder import TextEmbedder
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.embedder = TextEmbedder(
                model_name=self.model_name,
                use_gpu=self.use_gpu,
                cache_dir=self.cache_dir
            )
            
            if self.embedder.is_model_available():
                self.embedding_dim = self.embedder.get_embedding_dimension()
                logger.info(f"✓ Embedder loaded successfully")
                logger.info(f"✓ Model: {self.model_name}")
                logger.info(f"✓ Embedding dimension: {self.embedding_dim}")
                logger.info(f"✓ Device: {'GPU' if self.use_gpu else 'CPU'}")
            else:
                logger.error("❌ Embedder failed to load")
                self.embedder = None
                
        except Exception as e:
            logger.error(f"❌ Error loading embedder: {str(e)}")
            import traceback
            traceback.print_exc()
            self.embedder = None
    
    def is_available(self) -> bool:
        """Check if embedder is available"""
        return self.embedder is not None and self.embedder.is_model_available()
    
    def embed_chunk(
        self,
        chunk_text: str,
        chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Embed a single chunk
        
        Args:
            chunk_text: Text to embed
            chunk_metadata: Optional metadata about the chunk
            
        Returns:
            Dictionary with embedding and metadata
        """
        if not self.is_available():
            raise RuntimeError("Embedder not available")
        
        try:
            # Generate embedding
            embedding = self.embedder.encode(
                [chunk_text],
                batch_size=1,
                show_progress=False
            )[0]
            
            # Generate chunk ID
            chunk_id = self._generate_chunk_id(chunk_text)
            
            # Prepare result
            result = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'embedding': embedding.tolist(),
                'embedding_dim': len(embedding),
                'text_length': len(chunk_text),
                'embedded_at': datetime.now().isoformat(),
                'model_name': self.model_name,
                'model_type': 'sentence-transformer',
                'metadata': chunk_metadata or {}
            }
            
            logger.debug(f"Embedded chunk {chunk_id[:8]}... (dim: {len(embedding)})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error embedding chunk: {str(e)}")
            raise
    
    def embed_chunks_batch(
        self,
        chunks: List[str],
        chunks_metadata: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Embed multiple chunks in batch
        
        Args:
            chunks: List of chunk texts
            chunks_metadata: Optional list of metadata dicts
            show_progress: Show progress bar
            
        Returns:
            List of embedded chunk dictionaries
        """
        if not self.is_available():
            raise RuntimeError("Embedder not available")
        
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []
        
        logger.info(f"Embedding {len(chunks)} chunks in batches of {self.batch_size}")
        
        try:
            # Generate embeddings for all chunks
            embeddings = self.embedder.encode(
                chunks,
                batch_size=self.batch_size,
                show_progress=show_progress
            )
            
            # Prepare results
            results = []
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = self._generate_chunk_id(chunk_text)
                metadata = chunks_metadata[i] if chunks_metadata and i < len(chunks_metadata) else {}
                
                result = {
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'embedding': embedding.tolist(),
                    'embedding_dim': len(embedding),
                    'text_length': len(chunk_text),
                    'embedded_at': datetime.now().isoformat(),
                    'model_name': self.model_name,
                    'model_type': 'sentence-transformer',
                    'metadata': metadata
                }
                
                results.append(result)
            
            logger.info(f"✓ Successfully embedded {len(results)} chunks")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            raise
    
    def embed_chunks_with_metadata(
        self,
        chunks_with_metadata: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Embed chunks that already have metadata attached
        
        Args:
            chunks_with_metadata: List of dicts with 'text' and other metadata
            show_progress: Show progress bar
            
        Returns:
            List of embedded chunks with original metadata preserved
        """
        if not chunks_with_metadata:
            return []
        
        # Extract texts and validate
        texts = []
        valid_chunks = []
        
        for chunk in chunks_with_metadata:
            text = chunk.get('text', '')
            if text and text.strip():
                texts.append(text)
                valid_chunks.append(chunk)
        
        if not texts:
            logger.warning("All chunks had empty text")
            return []
        
        logger.info(f"Embedding {len(texts)} chunks with metadata...")
        
        # Generate embeddings
        embeddings = self.embedder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress=show_progress
        )
        
        # Merge embeddings with metadata
        results = []
        for chunk_dict, embedding in zip(valid_chunks, embeddings):
            chunk_text = chunk_dict.get('text', '')
            chunk_id = self._generate_chunk_id(chunk_text)
            
            result = {
                **chunk_dict,  # Preserve original metadata
                'chunk_id': chunk_id,
                'embedding': embedding.tolist(),
                'embedding_dim': len(embedding),
                'text_length': len(chunk_text),
                'embedded_at': datetime.now().isoformat(),
                'embedding_model': self.model_name,
                'model_type': 'sentence-transformer'
            }
            
            results.append(result)
        
        logger.info(f"✓ Embedded {len(results)} chunks with metadata preserved")
        
        return results
    
    def _generate_chunk_id(self, text: str) -> str:
        """Generate unique ID for chunk based on content"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_embedding_stats(self, embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics about embeddings
        
        Args:
            embeddings: List of embedding dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not embeddings:
            return {}
        
        embedding_arrays = [np.array(e['embedding']) for e in embeddings]
        text_lengths = [e.get('text_length', len(e.get('text', ''))) for e in embeddings]
        
        stats = {
            'total_chunks': len(embeddings),
            'embedding_dim': embeddings[0].get('embedding_dim', 0),
            'model_name': embeddings[0].get('model_name') or embeddings[0].get('embedding_model', 'unknown'),
            'model_type': embeddings[0].get('model_type', 'sentence-transformer'),
            'text_stats': {
                'min_length': min(text_lengths) if text_lengths else 0,
                'max_length': max(text_lengths) if text_lengths else 0,
                'avg_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
                'total_characters': sum(text_lengths)
            },
            'embedding_stats': {
                'mean_norm': float(np.mean([np.linalg.norm(e) for e in embedding_arrays])) if embedding_arrays else 0,
                'std_norm': float(np.std([np.linalg.norm(e) for e in embedding_arrays])) if embedding_arrays else 0
            }
        }
        
        return stats
    
    def clear_cache(self):
        """Clear embedder cache"""
        if self.embedder:
            self.embedder.clear_cache()
            logger.info("✓ Embedder cache cleared")