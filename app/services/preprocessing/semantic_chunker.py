import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Semantic chunking using sentence-transformers embeddings on CPU"""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        use_gpu: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize semantic chunker with embedder
        
        Args:
            model_name: Sentence-transformer model name
                - sentence-transformers/all-mpnet-base-v2 (recommended, 768 dim)
                - sentence-transformers/all-MiniLM-L6-v2 (smaller, faster, 384 dim)
                - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (multilingual)
            use_gpu: Whether to use GPU (default: False for CPU)
            cache_dir: Cache directory for model files
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.cache_dir = cache_dir
        self.embedder = None
        self.embedding_dim = None
        self._sentence_tokenizer = None
        
        self._load_embedder()
        self._load_sentence_tokenizer()
    
    def _load_embedder(self):
        """Load embedder from separate module"""
        try:
            from .embedder import TextEmbedder
            
            logger.info("Loading embedder...")
            logger.info(f"Model: {self.model_name}")
            
            self.embedder = TextEmbedder(
                model_name=self.model_name,
                use_gpu=self.use_gpu,
                cache_dir=self.cache_dir
            )
            
            if self.embedder.is_model_available():
                self.embedding_dim = self.embedder.get_embedding_dimension()
                
                device = "GPU" if self.use_gpu else "CPU"
                
                logger.info(f"✓ Embedder loaded successfully on {device}")
                logger.info(f"✓ Model: {self.model_name}")
                logger.info(f"✓ Embedding dimension: {self.embedding_dim}")
            else:
                logger.error("❌ Embedder failed to load")
                self.embedder = None
            
        except ImportError as e:
            logger.error(f"❌ Cannot import TextEmbedder: {str(e)}")
            logger.error("Make sure embedder.py exists in the same directory")
            self.embedder = None
        except Exception as e:
            logger.error(f"❌ Error loading embedder: {str(e)}")
            import traceback
            traceback.print_exc()
            self.embedder = None
    
    def _load_sentence_tokenizer(self):
        """Load NLTK sentence tokenizer"""
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer...")
                try:
                    nltk.download('punkt', quiet=True)
                except:
                    nltk.download('punkt_tab', quiet=True)
            
            self._sentence_tokenizer = sent_tokenize
            logger.debug("✓ NLTK sentence tokenizer loaded")
            
        except ImportError:
            logger.warning(
                "⚠️  NLTK not installed. Using regex fallback for sentence splitting. "
                "Install with: pip install nltk"
            )
            self._sentence_tokenizer = None
        except Exception as e:
            logger.warning(f"⚠️  Could not load NLTK: {str(e)}. Using regex fallback.")
            self._sentence_tokenizer = None
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts into embeddings
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            show_progress: Show progress bar
            
        Returns:
            NumPy array of embeddings
        """
        if not self.is_model_available():
            raise RuntimeError("Embedder not available. Cannot generate embeddings.")
        
        if not isinstance(texts, list):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        return self.embedder.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            show_progress=show_progress
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if self._sentence_tokenizer:
            try:
                sentences = self._sentence_tokenizer(text)
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {str(e)}. Using regex fallback.")
                sentences = self._regex_sentence_split(text)
        else:
            sentences = self._regex_sentence_split(text)
        
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _regex_sentence_split(self, text: str) -> List[str]:
        """Fallback regex-based sentence splitting"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def is_model_available(self) -> bool:
        """Check if embedder is available"""
        return self.embedder is not None and self.embedder.is_model_available()
    
    def chunk_by_similarity(
        self,
        text: str,
        threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        dynamic_threshold: bool = False
    ) -> List[str]:
        """
        Chunk text based on semantic similarity between sentences
        
        Args:
            text: Input text to chunk
            threshold: Similarity threshold for chunk boundaries (0-1)
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
            dynamic_threshold: If True, adjust threshold based on sentence count
            
        Returns:
            List of semantic chunks (text only)
        """
        if not self.is_model_available():
            logger.error("❌ Embedder not available")
            raise RuntimeError("Embedder not available. Cannot perform semantic chunking.")
        
        sentences = self._split_into_sentences(text)
        if not sentences:
            logger.warning("No sentences found in text")
            return []
        
        if len(sentences) == 1:
            return sentences if len(sentences[0]) >= min_chunk_size else []
        
        if dynamic_threshold:
            threshold = self._calculate_dynamic_threshold(len(sentences), threshold)
            logger.debug(f"Dynamic threshold adjusted to: {threshold:.3f}")
        
        device = "GPU" if self.use_gpu else "CPU"
        logger.info(f"Computing embeddings for {len(sentences)} sentences on {device}")
        embeddings = self.encode(sentences, show_progress=len(sentences) > 100)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_chunk_length = len(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_length = len(sentence)
            
            similarity = self._calculate_similarity(embeddings[i], embeddings[i - 1])
            
            would_exceed_max = current_chunk_length + sentence_length > max_chunk_size
            is_similar = similarity >= threshold
            is_below_min = current_chunk_length < min_chunk_size
            
            if (is_similar and not would_exceed_max) or is_below_min:
                current_chunk.append(sentence)
                current_chunk_length += sentence_length
            else:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk).strip()
                    if len(chunk_text) >= min_chunk_size:
                        chunks.append(chunk_text)
                
                current_chunk = [sentence]
                current_chunk_length = sentence_length
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)
        
        logger.info(f"✓ Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
        return chunks
    
    def chunk_by_similarity_with_metadata(
        self,
        text: str,
        threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        dynamic_threshold: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Chunk text and include similarity scores as metadata
        
        Args:
            text: Input text to chunk
            threshold: Similarity threshold for chunk boundaries
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
            dynamic_threshold: If True, adjust threshold based on sentence count
            
        Returns:
            List of chunks with metadata including similarity scores
        """
        if not self.is_model_available():
            logger.error("❌ Embedder not available")
            raise RuntimeError("Embedder not available. Cannot perform semantic chunking.")
        logger.info(f"🔍 Chunking text: {len(text)} chars, min={min_chunk_size}, max={max_chunk_size}")
    
        sentences = self._split_into_sentences(text)
        
        # ✅ ADD THIS
        logger.info(f"🔍 Split into {len(sentences)} sentences")

        sentences = self._split_into_sentences(text)
        if not sentences:
            logger.warning("No sentences found in text")
            return []
        
        if len(sentences) == 1:
            logger.info(f"🔍 Only 1 sentence found, length={len(sentences[0])}")
            if len(sentences[0]) < min_chunk_size:
                logger.warning(f"❌ Single sentence too short: {len(sentences[0])} < {min_chunk_size}")
                return []
            
        model_info = self.embedder.get_model_info()
        device = "gpu" if self.use_gpu else "cpu"
        
        if len(sentences) == 1:
            return [
                {
                    'text': sentences[0],
                    'similarity_score': None,
                    'min_similarity': None,
                    'max_similarity': None,
                    'sentence_count': 1,
                    'length': len(sentences[0]),
                    'model_type': 'sentence-transformer',
                    'model_name': self.model_name,
                    'device': device
                }
            ] if len(sentences[0]) >= min_chunk_size else []
        
        if dynamic_threshold:
            threshold = self._calculate_dynamic_threshold(len(sentences), threshold)
        
        logger.info(f"Computing embeddings for {len(sentences)} sentences on {device.upper()}")
        embeddings = self.encode(sentences, show_progress=len(sentences) > 100)
        
        chunks_with_metadata = []
        current_chunk = [sentences[0]]
        current_chunk_length = len(sentences[0])
        chunk_similarities = []
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            similarity = self._calculate_similarity(embeddings[i], embeddings[i - 1])
            
            would_exceed_max = current_chunk_length + len(sentence) > max_chunk_size
            is_similar = similarity >= threshold
            is_below_min = current_chunk_length < min_chunk_size
            
            if (is_similar and not would_exceed_max) or is_below_min:
                current_chunk.append(sentence)
                current_chunk_length += len(sentence)
                chunk_similarities.append(similarity)
            else:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk).strip()
                    if len(chunk_text) >= min_chunk_size:
                        chunks_with_metadata.append(
                            self._create_chunk_metadata(
                                chunk_text,
                                current_chunk,
                                chunk_similarities,
                                device
                            )
                        )
                
                current_chunk = [sentence]
                current_chunk_length = len(sentence)
                chunk_similarities = []
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if len(chunk_text) >= min_chunk_size:
                chunks_with_metadata.append(
                    self._create_chunk_metadata(
                        chunk_text,
                        current_chunk,
                        chunk_similarities,
                        device
                    )
                )
        
        logger.info(f"✓ Created {len(chunks_with_metadata)} semantic chunks with metadata")
        return chunks_with_metadata
    
    def _create_chunk_metadata(
        self,
        chunk_text: str,
        sentences: List[str],
        similarities: List[float],
        device: str
    ) -> Dict[str, Any]:
        """Create metadata dictionary for a chunk"""
        avg_similarity = float(np.mean(similarities)) if similarities else None
        min_similarity = float(np.min(similarities)) if similarities else None
        max_similarity = float(np.max(similarities)) if similarities else None
        
        return {
            'text': chunk_text,
            'similarity_score': avg_similarity,
            'min_similarity': min_similarity,
            'max_similarity': max_similarity,
            'sentence_count': len(sentences),
            'length': len(chunk_text),
            'model_type': 'sentence-transformer',
            'model_name': self.model_name,
            'device': device
        }
    
    def _calculate_dynamic_threshold(self, sentence_count: int, base_threshold: float) -> float:
        """Calculate dynamic threshold based on sentence count"""
        if sentence_count < 10:
            return min(base_threshold + 0.15, 0.95)
        elif sentence_count < 50:
            return min(base_threshold + 0.05, 0.95)
        elif sentence_count > 200:
            return max(base_threshold - 0.05, 0.1)
        else:
            return base_threshold
    
    def calculate_chunk_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Calculate embeddings for a list of chunks"""
        if not self.is_model_available():
            raise RuntimeError("Embedder not available")
        
        device = "GPU" if self.use_gpu else "CPU"
        logger.info(f"Computing embeddings for {len(chunks)} chunks on {device}")
        embeddings = self.encode(chunks, show_progress=len(chunks) > 50)
        return embeddings
    
    def get_most_similar_chunks(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find chunks most similar to a query using semantic similarity
        
        Args:
            query: Query text
            chunks: List of chunks to search
            top_k: Number of top results to return
            
        Returns:
            List of most similar chunks with similarity scores
        """
        if not self.is_model_available():
            logger.error("❌ Embedder not available, cannot perform semantic search")
            raise RuntimeError("Embedder not available")
        
        if not chunks:
            return []
        
        device = "GPU" if self.use_gpu else "CPU"
        logger.info(f"🔍 Using embeddings for semantic search on {device}")
        
        # Use embedder's built-in similarity search
        return self.embedder.find_most_similar(
            query=query,
            texts=chunks,
            top_k=top_k,
            threshold=0.0
        )
    
    def get_related_chunks(
        self,
        chunk_index: int,
        chunks: List[str],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Find chunks related to a specific chunk"""
        if not self.is_model_available():
            logger.error("❌ Embedder not available")
            raise RuntimeError("Embedder not available")
        
        if chunk_index >= len(chunks) or chunk_index < 0:
            logger.error(f"Invalid chunk index: {chunk_index}")
            return []
        
        query_chunk = chunks[chunk_index]
        logger.debug(f"Finding chunks related to chunk {chunk_index}")
        
        similar_chunks = self.get_most_similar_chunks(query_chunk, chunks, top_k=top_k + 1)
        
        # Remove the reference chunk itself
        similar_chunks = [c for c in similar_chunks if c.get('index') != chunk_index]
        
        return similar_chunks[:top_k]
    
    def clear_cache(self):
        """Clear embedder cache"""
        if self.embedder:
            self.embedder.clear_cache()
            logger.info("✓ Embedder cache cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.embedder:
            return self.embedder.get_model_info()
        return {
            'model_name': self.model_name,
            'is_available': False,
            'error': 'Embedder not loaded'
        }