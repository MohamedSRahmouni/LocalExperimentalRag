"""
Retrieval Service
Advanced retrieval with re-ranking and context construction
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval"""
    top_k: int = 3
    min_score: float = 0.5
    max_context_length: int = 1200
    enable_reranking: bool = True
    diversity_weight: float = 0.3
    recency_weight: float = 0.1


@dataclass
class RetrievedChunk:
    """Retrieved chunk with metadata"""
    chunk_id: str
    text: str
    score: float
    source: str
    metadata: Dict[str, Any]
    rank: int


@dataclass
class RetrievalResult:
    """Complete retrieval result"""
    query: str
    chunks: List[RetrievedChunk]
    context: str
    total_found: int
    retrieval_time: float
    metadata: Dict[str, Any]


class RetrievalService:
    """
    Advanced retrieval service with re-ranking and context construction
    Prepares context for LLM consumption
    """
    
    def __init__(
        self,
        search_engine,
        embedder,
        config: Optional[RetrievalConfig] = None
    ):
        """
        Initialize retrieval service
        
        Args:
            search_engine: Weaviate search engine
            embedder: Embedding model
            config: Retrieval configuration
        """
        self.search_engine = search_engine
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        
        logger.info("✅ Retrieval service initialized")
        logger.info(f"   Top-K: {self.config.top_k}")
        logger.info(f"   Min Score: {self.config.min_score}")
        logger.info(f"   Re-ranking: {self.config.enable_reranking}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve (override config)
            min_score: Minimum similarity score (override config)
            filters: Optional metadata filters
            
        Returns:
            RetrievalResult with chunks and constructed context
        """
        import time
        start_time = time.time()
        
        top_k = top_k or self.config.top_k
        min_score = min_score or self.config.min_score
        
        logger.info(f"🔍 Retrieving for query: '{query[:100]}...'")
        logger.info(f"   Top-K: {top_k}, Min Score: {min_score}")
        
        # ================================================================
        # STEP 1: Semantic Search in Weaviate
        # ================================================================
        try:
            raw_chunks = self.search_engine.semantic_search(
                query_text=query,
                embedder=self.embedder,
                top_k=top_k * 2,  # Get more for re-ranking
                min_score=min_score,
                filters=filters
            )
            
            logger.info(f"✅ Found {len(raw_chunks)} chunks from Weaviate")
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return self._empty_result(query, str(e))
        
        if not raw_chunks:
            logger.warning("No chunks found")
            return self._empty_result(query, "No relevant documents found")
        
        # ================================================================
        # STEP 2: Re-ranking (optional)
        # ================================================================
        if self.config.enable_reranking and len(raw_chunks) > top_k:
            logger.info("🔄 Re-ranking results...")
            ranked_chunks = self._rerank_chunks(query, raw_chunks, top_k)
        else:
            ranked_chunks = raw_chunks[:top_k]
        
        # ================================================================
        # STEP 3: Convert to RetrievedChunk objects
        # ================================================================
        retrieved_chunks = []
        for rank, chunk_data in enumerate(ranked_chunks, 1):
            retrieved_chunks.append(RetrievedChunk(
                chunk_id=chunk_data.get('chunk_id', f'chunk_{rank}'),
                text=chunk_data.get('text', ''),
                score=chunk_data.get('similarity', 0.0),
                source=chunk_data.get('metadata', {}).get('filename', 'unknown'),
                metadata=chunk_data.get('metadata', {}),
                rank=rank
            ))
        
        # ================================================================
        # STEP 4: Construct context for LLM
        # ================================================================
        context = self._construct_context(retrieved_chunks)
        
        retrieval_time = time.time() - start_time
        
        logger.info(f"✅ Retrieval complete in {retrieval_time:.3f}s")
        logger.info(f"   Final chunks: {len(retrieved_chunks)}")
        logger.info(f"   Context length: {len(context)} chars")
        
        return RetrievalResult(
            query=query,
            chunks=retrieved_chunks,
            context=context,
            total_found=len(raw_chunks),
            retrieval_time=retrieval_time,
            metadata={
                'top_k': top_k,
                'min_score': min_score,
                'reranked': self.config.enable_reranking,
                'filters': filters
            }
        )
    
    def _rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks using multiple signals
        
        Signals:
        - Semantic similarity (primary)
        - Source diversity
        - Chunk position (recency)
        """
        scored_chunks = []
        
        for chunk in chunks:
            base_score = chunk.get('similarity', 0.0)
            
            # Diversity bonus: prefer chunks from different sources
            source = chunk.get('metadata', {}).get('filename', '')
            diversity_bonus = 0.0
            
            existing_sources = [
                c['metadata'].get('filename', '')
                for c in scored_chunks
            ]
            if source not in existing_sources:
                diversity_bonus = self.config.diversity_weight
            
            # Recency bonus: prefer chunks from beginning of document
            chunk_index = chunk.get('metadata', {}).get('chunk_index', 0)
            total_chunks = chunk.get('metadata', {}).get('total_chunks', 1)
            
            if total_chunks > 0:
                recency_bonus = (1 - (chunk_index / total_chunks)) * self.config.recency_weight
            else:
                recency_bonus = 0.0
            
            # Final score
            final_score = base_score + diversity_bonus + recency_bonus
            
            chunk['rerank_score'] = final_score
            scored_chunks.append(chunk)
        
        # Sort by rerank score
        scored_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"🔄 Re-ranked {len(scored_chunks)} chunks")
        
        return scored_chunks[:top_k]
    
    def _construct_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Construct context string for LLM
        
        Format:
        ---
        Source: filename.pdf
        Relevance: 95%
        
        [Chunk text]
        ---
        """
        if not chunks:
            return ""
        
        context_parts = []
        total_length = 0
        
        for chunk in chunks:
            # Format chunk with metadata
            chunk_text = f"""---
Source: {chunk.source}
Relevance: {chunk.score * 100:.1f}%
Rank: {chunk.rank}

{chunk.text}
---
"""
            
            # Check if adding this chunk exceeds max length
            if total_length + len(chunk_text) > self.config.max_context_length:
                logger.warning(f"⚠️  Context length limit reached, truncating at {chunk.rank} chunks")
                break
            
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        context = "\n".join(context_parts)
        
        # Add header
        header = f"""# Retrieved Context ({len(context_parts)} chunks)

"""
        
        return header + context
    
    def _empty_result(self, query: str, reason: str) -> RetrievalResult:
        """Create empty result for failed retrieval"""
        return RetrievalResult(
            query=query,
            chunks=[],
            context="",
            total_found=0,
            retrieval_time=0.0,
            metadata={'error': reason}
        )
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval configuration stats"""
        return {
            'config': {
                'top_k': self.config.top_k,
                'min_score': self.config.min_score,
                'max_context_length': self.config.max_context_length,
                'enable_reranking': self.config.enable_reranking,
                'diversity_weight': self.config.diversity_weight,
                'recency_weight': self.config.recency_weight
            },
            'status': 'ready'
        }