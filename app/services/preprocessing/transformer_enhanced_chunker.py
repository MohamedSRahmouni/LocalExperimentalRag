# transformer_enhanced_chunker.py
"""
Enhanced semantic chunker using transformer-based context-aware synonyms
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
 
from .semantic_chunker import SemanticChunker
from .transformer_synonym_handler import ContextAwareSynonymHandler

logger = logging.getLogger(__name__)


class TransformerEnhancedChunker(SemanticChunker):
    """
    Semantic chunker with transformer-based context-aware synonym handling
    Best performance for synonym extraction
    """
    
    def __init__(
        self,
        semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        transformer_model: str = "bert-base-uncased",
        use_gpu: bool = False
    ):
        """
        Initialize transformer-enhanced chunker
        
        Args:
            semantic_model: Model for semantic similarity
            transformer_model: Transformer model for synonym extraction
            use_gpu: Use GPU
        """
        super().__init__(model_name=semantic_model, use_gpu=use_gpu)
        
        self.synonym_handler = ContextAwareSynonymHandler(
            model_name=transformer_model,
            embedding_model=self.model,
            device="cuda" if use_gpu else "cpu"
        )
        
        logger.info("Transformer-enhanced chunker initialized")
    
    def get_most_similar_chunks_transformer(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 5,
        use_context_expansion: bool = True,
        context_corpus: Optional[List[str]] = None,
        merge_strategy: str = "reciprocal_rank_fusion"
    ) -> List[Dict[str, Any]]:
        """
        Find similar chunks using transformer-based contextual synonyms
        
        Args:
            query: Search query
            chunks: List of chunks
            top_k: Top K results
            use_context_expansion: Use contextual synonym expansion
            context_corpus: Corpus for context
            merge_strategy: How to merge results
            
        Returns:
            List of similar chunks
        """
        if not self.model:
            logger.warning("Model not available")
            return []
        
        logger.info(f"Searching with transformer context: '{query}'")
        
        # Get original results
        original_results = self.get_most_similar_chunks(query, chunks, top_k=top_k*2)
        
        if not use_context_expansion:
            return original_results[:top_k]
        
        # Generate contextual variants
        logger.info("Generating contextual query variants...")
        context_corpus = context_corpus or chunks
        query_variants = self.synonym_handler.generate_variant_queries(
            query,
            context_corpus=context_corpus,
            max_variants=3
        )
        
        logger.info(f"Generated {len(query_variants)} query variants")
        
        # Get results for variants
        expanded_results = []
        for i, variant in enumerate(query_variants[1:], 1):
            logger.debug(f"Variant {i}: '{variant}'")
            results = self.get_most_similar_chunks(variant, chunks, top_k=top_k*2)
            expanded_results.append(results)
        
        # Merge results
        logger.info(f"Merging results using {merge_strategy}...")
        merged = self._merge_results(
            original_results,
            expanded_results,
            merge_strategy
        )
        
        return merged[:top_k]
    
    def analyze_query_synonyms(
        self,
        query: str,
        context_corpus: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze synonyms available for query terms
        
        Args:
            query: Query to analyze
            context_corpus: Optional context corpus
            
        Returns:
            Analysis dictionary
        """
        logger.info(f"Analyzing synonyms for: '{query}'")
        
        expansions = self.synonym_handler.expand_query_contextual(
            query,
            context_corpus=context_corpus,
            top_k_per_word=5
        )
        
        analysis = {
            'query': query,
            'words_analyzed': len(expansions),
            'total_synonyms': sum(len(v) for v in expansions.values()),
            'word_synonyms': expansions,
            'coverage_rate': len(expansions) / len(query.split()) if query.split() else 0
        }
        
        return analysis
    
    def compare_synonym_methods(
        self,
        word: str,
        context: str
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compare synonym extraction using different methods
        
        Args:
            word: Target word
            context: Context
            
        Returns:
            Dictionary of method -> [(synonym, score)]
        """
        methods = ["mlm", "embedding", "hybrid"]
        results = {}
        
        for method in methods:
            logger.info(f"Extracting with method: {method}")
            synonyms = self.synonym_handler.extract_synonyms_from_context(
                word,
                context,
                method=method
            )
            results[method] = synonyms
        
        return results
    
    def _merge_results(
        self,
        original: List[Dict],
        expanded_sets: List[List[Dict]],
        strategy: str
    ) -> List[Dict]:
        """Merge results from multiple queries"""
        if strategy == "reciprocal_rank_fusion":
            return self._merge_rrf(original, expanded_sets)
        else:
            return original
    
    def _merge_rrf(self, original: List[Dict], expanded_sets: List[List[Dict]]) -> List[Dict]:
        """RRF merging"""
        k = 60
        scores = {}
        
        for rank, result in enumerate(original, 1):
            doc_id = result['index']
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        
        for expanded_set in expanded_sets:
            for rank, result in enumerate(expanded_set, 1):
                doc_id = result['index']
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        
        # Reconstruct results
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        result_map = {}
        for result_set in [original] + expanded_sets:
            for result in result_set:
                doc_id = result['index']
                if doc_id not in result_map:
                    result_map[doc_id] = result
        
        merged = [result_map[doc_id] for doc_id, _ in sorted_docs]
        return merged