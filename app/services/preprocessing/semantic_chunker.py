import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Semantic chunking using sentence embeddings"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", use_gpu: bool = False):
        """
        Initialize semantic chunker
        
        Args:
            model_name: HuggingFace model name for embeddings
            use_gpu: Whether to use GPU for embeddings
            
        Raises:
            ImportError: If sentence-transformers is not installed
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model = None
        self.embedding_dim = None
        self._sentence_tokenizer = None
        
        self._load_model()
        self._load_sentence_tokenizer()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            device = "cuda" if self.use_gpu else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None
            self.embedding_dim = None
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            self.model = None
            self.embedding_dim = None
    
    def _load_sentence_tokenizer(self):
        """Load NLTK sentence tokenizer"""
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            
            # Try to use punkt tokenizer
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
            
            self._sentence_tokenizer = sent_tokenize
            logger.debug("NLTK sentence tokenizer loaded")
            
        except ImportError:
            logger.warning(
                "NLTK not installed. Using regex fallback for sentence splitting. "
                "Install with: pip install nltk"
            )
            self._sentence_tokenizer = None
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if self._sentence_tokenizer:
            try:
                sentences = self._sentence_tokenizer(text)
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {str(e)}. Using regex fallback.")
                sentences = self._regex_sentence_split(text)
        else:
            sentences = self._regex_sentence_split(text)
        
        # Filter empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _regex_sentence_split(self, text: str) -> List[str]:
        """
        Fallback regex-based sentence splitting
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        import re
        # Split on sentence endings followed by space
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1 (typically 0 to 1 for normalized embeddings)
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def is_model_available(self) -> bool:
        """
        Check if embedding model is available
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        return self.model is not None
    
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
        
        Uses a moving window approach where sentences are grouped together
        if their semantic similarity is above the threshold.
        
        Args:
            text: Input text to chunk
            threshold: Similarity threshold for chunk boundaries (0-1)
                     Lower threshold = more aggressive chunking (smaller chunks)
                     Higher threshold = less aggressive chunking (larger chunks)
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
            dynamic_threshold: If True, adjust threshold based on sentence count
            
        Returns:
            List of semantic chunks (text only)
            
        Example:
            >>> chunker = SemanticChunker()
            >>> text = "Sentence 1. Sentence 2. Sentence 3."
            >>> chunks = chunker.chunk_by_similarity(text, threshold=0.5)
        """
        if not self.model:
            logger.warning("Embedding model not available, using simple chunking")
            return self._fallback_chunking(text, max_chunk_size)
        
        sentences = self._split_into_sentences(text)
        if not sentences:
            logger.warning("No sentences found in text")
            return []
        
        if len(sentences) == 1:
            return sentences if len(sentences[0]) >= min_chunk_size else []
        
        # Adjust threshold dynamically if requested
        if dynamic_threshold:
            threshold = self._calculate_dynamic_threshold(len(sentences), threshold)
            logger.debug(f"Dynamic threshold adjusted to: {threshold:.3f}")
        
        logger.debug(f"Computing embeddings for {len(sentences)} sentences")
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_chunk_length = len(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_length = len(sentence)
            
            # Calculate similarity with the last sentence in current chunk
            similarity = self._calculate_similarity(embeddings[i], embeddings[i - 1])
            
            # Determine if we should add to current chunk or start new one
            would_exceed_max = current_chunk_length + sentence_length > max_chunk_size
            is_similar = similarity >= threshold
            is_below_min = current_chunk_length < min_chunk_size
            
            if (is_similar and not would_exceed_max) or is_below_min:
                # Add to current chunk
                current_chunk.append(sentence)
                current_chunk_length += sentence_length
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunk_text = ' '.join(current_chunk).strip()
                    if len(chunk_text) >= min_chunk_size:
                        chunks.append(chunk_text)
                
                current_chunk = [sentence]
                current_chunk_length = sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)
        
        logger.debug(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
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
            
        Example:
            >>> chunker = SemanticChunker()
            >>> text = "Sentence 1. Sentence 2. Sentence 3."
            >>> chunks = chunker.chunk_by_similarity_with_metadata(text)
            >>> for chunk in chunks:
            ...     print(f"Similarity: {chunk['similarity_score']}")
        """
        if not self.model:
            chunks = self._fallback_chunking(text, max_chunk_size)
            return [
                {
                    'text': chunk,
                    'similarity_score': None,
                    'min_similarity': None,
                    'max_similarity': None,
                    'sentence_count': len(self._split_into_sentences(chunk)),
                    'length': len(chunk)
                }
                for chunk in chunks
            ]
        
        sentences = self._split_into_sentences(text)
        if not sentences:
            logger.warning("No sentences found in text")
            return []
        
        if len(sentences) == 1:
            return [
                {
                    'text': sentences[0],
                    'similarity_score': None,
                    'min_similarity': None,
                    'max_similarity': None,
                    'sentence_count': 1,
                    'length': len(sentences[0])
                }
            ] if len(sentences[0]) >= min_chunk_size else []
        
        # Adjust threshold dynamically if requested
        if dynamic_threshold:
            threshold = self._calculate_dynamic_threshold(len(sentences), threshold)
        
        logger.debug(f"Computing embeddings for {len(sentences)} sentences")
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        
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
                                chunk_similarities
                            )
                        )
                
                current_chunk = [sentence]
                current_chunk_length = len(sentence)
                chunk_similarities = []
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if len(chunk_text) >= min_chunk_size:
                chunks_with_metadata.append(
                    self._create_chunk_metadata(
                        chunk_text,
                        current_chunk,
                        chunk_similarities
                    )
                )
        
        logger.debug(f"Created {len(chunks_with_metadata)} semantic chunks with metadata")
        return chunks_with_metadata
    
    def _create_chunk_metadata(
        self,
        chunk_text: str,
        sentences: List[str],
        similarities: List[float]
    ) -> Dict[str, Any]:
        """
        Create metadata dictionary for a chunk
        
        Args:
            chunk_text: The chunk text
            sentences: List of sentences in chunk
            similarities: List of similarity scores between consecutive sentences
            
        Returns:
            Dictionary with chunk metadata
        """
        avg_similarity = float(np.mean(similarities)) if similarities else None
        min_similarity = float(np.min(similarities)) if similarities else None
        max_similarity = float(np.max(similarities)) if similarities else None
        
        return {
            'text': chunk_text,
            'similarity_score': avg_similarity,
            'min_similarity': min_similarity,
            'max_similarity': max_similarity,
            'sentence_count': len(sentences),
            'length': len(chunk_text)
        }
    
    def _calculate_dynamic_threshold(self, sentence_count: int, base_threshold: float) -> float:
        """
        Calculate dynamic threshold based on sentence count
        
        For documents with fewer sentences, increase threshold to create fewer chunks.
        For documents with many sentences, decrease threshold for more granular chunking.
        
        Args:
            sentence_count: Number of sentences in text
            base_threshold: Base threshold value
            
        Returns:
            Adjusted threshold value
        """
        if sentence_count < 10:
            # Very small texts: increase threshold to avoid over-chunking
            return min(base_threshold + 0.15, 0.95)
        elif sentence_count < 50:
            # Small texts: slightly increase threshold
            return min(base_threshold + 0.05, 0.95)
        elif sentence_count > 200:
            # Large texts: slightly decrease threshold for more chunks
            return max(base_threshold - 0.05, 0.1)
        else:
            # Medium texts: use base threshold as-is
            return base_threshold
    
    def _fallback_chunking(self, text: str, max_chunk_size: int) -> List[str]:
        """
        Fallback chunking when embeddings not available
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum chunk size
            
        Returns:
            List of chunks
        """
        logger.debug("Using fallback chunking (no embeddings)")
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= max_chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk).strip())
                current_chunk = [sentence]
                current_length = sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk).strip())
        
        return chunks
    
    def calculate_chunk_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Calculate embeddings for a list of chunks
        
        Useful for semantic search and similarity calculations
        
        Args:
            chunks: List of text chunks
            
        Returns:
            NumPy array of shape (len(chunks), embedding_dim)
            
        Raises:
            RuntimeError: If model is not available
        """
        if not self.model:
            raise RuntimeError("Embedding model not available")
        
        logger.debug(f"Computing embeddings for {len(chunks)} chunks")
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
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
            
        Example:
            >>> chunker = SemanticChunker()
            >>> chunks = ["Chunk 1 text", "Chunk 2 text"]
            >>> results = chunker.get_most_similar_chunks("search query", chunks, top_k=2)
        """
        if not self.model:
            logger.warning("Model not available, cannot perform semantic search")
            return []
        
        if not chunks:
            return []
        
        logger.debug(f"Computing query embedding")
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        logger.debug(f"Computing chunk embeddings for {len(chunks)} chunks")
        chunk_embeddings = self.model.encode(chunks, convert_to_numpy=True)
        
        scored_chunks = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            similarity = self._calculate_similarity(query_embedding, embedding)
            
            scored_chunks.append({
                'chunk': chunk,
                'similarity': float(similarity),
                'index': idx,
                'length': len(chunk),
                'preview': chunk[:200] + '...' if len(chunk) > 200 else chunk
            })
        
        scored_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        logger.debug(f"Returning top {min(top_k, len(scored_chunks))} results")
        
        return scored_chunks[:top_k]
    
    def get_related_chunks(
        self,
        chunk_index: int,
        chunks: List[str],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find chunks related to a specific chunk
        
        Args:
            chunk_index: Index of the reference chunk
            chunks: List of all chunks
            top_k: Number of related chunks to return (excluding the reference chunk)
            
        Returns:
            List of related chunks with similarity scores
        """
        if not self.model:
            logger.warning("Model not available")
            return []
        
        if chunk_index >= len(chunks) or chunk_index < 0:
            logger.error(f"Invalid chunk index: {chunk_index}")
            return []
        
        query_chunk = chunks[chunk_index]
        logger.debug(f"Finding chunks related to chunk {chunk_index}")
        
        similar_chunks = self.get_most_similar_chunks(query_chunk, chunks, top_k=top_k + 1)
        
        # Remove the reference chunk itself
        similar_chunks = [c for c in similar_chunks if c['index'] != chunk_index]
        
        return similar_chunks[:top_k]