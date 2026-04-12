"""
Text Embedder - Handles text embedding using sentence-transformers
Optimized for CPU inference with high-quality semantic embeddings
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch

logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Text Embedding Model Handler using sentence-transformers
    Provides semantic embeddings for RAG systems
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        use_gpu: bool = False,
        cache_dir: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize text embedder
        
        Args:
            model_name: Sentence transformer model name
            use_gpu: Whether to use GPU (auto-detects availability)
            cache_dir: Cache directory for model files
            max_length: Maximum sequence length for encoding
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.cache_dir = cache_dir
        self.max_length = max_length
        
        self.model = None
        self.embedding_dim = None
        self._model_loaded = False
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load embedding model using sentence-transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info("="*70)
            logger.info("Loading Embedding Model")
            logger.info("="*70)
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Device: {self.device.upper()}")
            
            # Load model
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir
            )
            
            # Set max sequence length
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.max_length
            
            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            self._model_loaded = True
            
            logger.info("="*70)
            logger.info("✓ Embedding model loaded successfully")
            logger.info(f"  Model: {self.model_name}")
            logger.info(f"  Embedding dimension: {self.embedding_dim}")
            logger.info(f"  Device: {self.device.upper()}")
            logger.info(f"  Max sequence length: {self.max_length}")
            logger.info("="*70)
            
        except ImportError as e:
            logger.error("="*70)
            logger.error("❌ Failed to import sentence-transformers")
            logger.error("="*70)
            logger.error(f"Error: {str(e)}")
            logger.error("\nPlease install:")
            logger.error("  pip install sentence-transformers torch")
            logger.error("="*70)
            self._model_loaded = False
            
        except Exception as e:
            logger.error("="*70)
            logger.error("❌ Failed to load embedding model")
            logger.error("="*70)
            logger.error(f"Error: {str(e)}")
            logger.error(f"Model: {self.model_name}")
            logger.error("="*70)
            self._model_loaded = False
            import traceback
            traceback.print_exc()
    
    def is_model_available(self) -> bool:
        """Check if model is successfully loaded"""
        return self._model_loaded and self.model is not None
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 8,
        max_length: Optional[int] = None,
        show_progress: bool = False,
        normalize: bool = True,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            max_length: Max sequence length
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length
            convert_to_numpy: Convert to numpy array
            
        Returns:
            NumPy array of embeddings [num_texts, embedding_dim]
        """
        if not self.is_model_available():
            raise RuntimeError("Embedding model not available")
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        # Filter empty strings
        valid_texts = [t for t in texts if t and t.strip()]
        
        if not valid_texts:
            logger.warning("All texts were empty")
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=convert_to_numpy,
                device=self.device
            )
            
            logger.debug(f"Encoded {len(valid_texts)} texts → shape: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error during encoding: {str(e)}")
            raise
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text"""
        return self.encode([text], normalize=normalize)[0]
    
    def find_most_similar(
        self,
        query: str,
        texts: List[str],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find most similar texts to a query
        
        Args:
            query: Query text
            texts: List of texts to search
            top_k: Number of top results
            threshold: Minimum similarity (0-1)
            
        Returns:
            List of dicts with similarity scores
        """
        if not texts:
            return []
        
        try:
            # Encode
            query_embedding = self.encode([query], normalize=True)[0]
            text_embeddings = self.encode(texts, normalize=True, show_progress=False)
            
            # Cosine similarity
            similarities = np.dot(text_embeddings, query_embedding)
            
            # Top-k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Build results
            results = []
            for idx in top_indices:
                idx = int(idx)
                similarity = float(similarities[idx])
                
                if similarity >= threshold:
                    text = texts[idx]
                    results.append({
                        'text': text,
                        'similarity': similarity,
                        'index': idx,
                        'chunk': text,
                        'preview': text[:200] + '...' if len(text) > 200 else text,
                        'length': len(text)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Get embedding dimension"""
        return self.embedding_dim
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✓ GPU cache cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'is_available': self.is_model_available(),
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'model_type': 'sentence-transformer',
            'gpu_available': torch.cuda.is_available(),
            'using_gpu': self.use_gpu
        }