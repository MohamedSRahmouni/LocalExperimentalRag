# services/preprocessing/dynamic_synonym_handler.py
"""
Dynamic multi-strategy synonym handler with adaptive quality scoring
FIXED: Proper score calibration for realistic synonym quality metrics
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import numpy as np
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SynonymCandidate:
    """Enhanced synonym candidate with provenance"""
    word: str
    score: float
    source: str  # 'wordnet', 'conceptnet', 'mlm', 'embedding'
    context_score: float = 0.0
    semantic_score: float = 0.0
    final_score: float = 0.0


class DynamicSynonymEngine:
    """
    Multi-source dynamic synonym engine with quality fusion
    FIXED: Calibrated scoring for realistic quality ranges
    """
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        device: Optional[str] = None,
        enable_wordnet: bool = True,
        enable_conceptnet: bool = True,
        enable_mlm: bool = True,
        enable_embedding: bool = True,
        min_quality_threshold: float = 0.40  # FIXED: Realistic threshold
    ):
        """
        Initialize dynamic synonym engine
        
        Args:
            model_name: Transformer model
            device: cuda/cpu
            enable_*: Enable specific strategies
            min_quality_threshold: Minimum quality score (0-1) - LOWERED to 0.40
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_quality_threshold = min_quality_threshold
        
        # Strategy flags
        self.enable_wordnet = enable_wordnet
        self.enable_conceptnet = enable_conceptnet
        self.enable_mlm = enable_mlm
        self.enable_embedding = enable_embedding
        
        # Caches
        self.synonym_cache: Dict[str, List[SynonymCandidate]] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Initialize components
        self._init_wordnet()
        self._init_conceptnet()
        self._init_mlm()
        self._init_embeddings()
        
        logger.info(f"✓ DynamicSynonymEngine initialized")
        logger.info(f"  Quality threshold: {self.min_quality_threshold}")
        logger.info(f"  WordNet: {self.wordnet_available}")
        logger.info(f"  ConceptNet: {self.conceptnet_available}")
        logger.info(f"  MLM: {self.mlm_available}")
        logger.info(f"  Embeddings: {self.embedding_available}")
    
    def _init_wordnet(self):
        """Initialize WordNet"""
        self.wordnet_available = False
        if not self.enable_wordnet:
            return
        
        try:
            from nltk.corpus import wordnet as wn
            import nltk
            
            try:
                wn.synsets('test')
                self.wn = wn
                self.wordnet_available = True
                logger.info("✓ WordNet loaded")
            except LookupError:
                logger.info("Downloading WordNet...")
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                self.wn = wn
                self.wordnet_available = True
                logger.info("✓ WordNet downloaded and loaded")
                
        except Exception as e:
            logger.warning(f"WordNet not available: {e}")
            self.wn = None
    
    def _init_conceptnet(self):
        """Initialize ConceptNet API client"""
        self.conceptnet_available = False
        if not self.enable_conceptnet:
            return
        
        try:
            import requests
            self.requests = requests
            # Test connection
            response = requests.get('http://api.conceptnet.io/c/en/test', timeout=2)
            if response.status_code == 200:
                self.conceptnet_available = True
                logger.info("✓ ConceptNet API accessible")
        except Exception as e:
            logger.warning(f"ConceptNet not available: {e}")
            self.requests = None
    
    def _init_mlm(self):
        """Initialize MLM model"""
        self.mlm_available = False
        if not self.enable_mlm:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            
            logger.info(f"Loading {self.model_name} for MLM...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self.mlm_model.to(self.device)
            self.mlm_model.eval()
            
            self.mlm_available = True
            logger.info("✓ MLM model loaded")
            
        except Exception as e:
            logger.warning(f"MLM not available: {e}")
            self.tokenizer = None
            self.mlm_model = None
    
    def _init_embeddings(self):
        """Initialize embedding model"""
        self.embedding_available = False
        if not self.enable_embedding:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info("Loading sentence transformer...")
            self.embedding_model = SentenceTransformer(
                'all-mpnet-base-v2',
                device=self.device
            )
            self.embedding_available = True
            logger.info("✓ Embedding model loaded (all-mpnet-base-v2)")
            
        except Exception as e:
            logger.warning(f"Embeddings not available: {e}")
            self.embedding_model = None
    
    def get_synonyms_wordnet(self, word: str, pos: Optional[str] = None) -> List[SynonymCandidate]:
        """
        Get synonyms from WordNet
        FIXED: Better scoring calibration
        """
        if not self.wordnet_available:
            return []
        
        candidates = []
        seen = {word.lower()}
        
        try:
            synsets = self.wn.synsets(word, pos=pos) if pos else self.wn.synsets(word)
            
            for synset_idx, synset in enumerate(synsets):
                # Base score diminishes with synset position
                position_factor = 1.0 / (synset_idx + 1)
                
                for lemma_idx, lemma in enumerate(synset.lemmas()):
                    syn = lemma.name().replace('_', ' ').lower()
                    
                    if syn not in seen and self._is_valid_token(syn):
                        # FIXED: More generous scoring
                        # Frequency score (if available)
                        if lemma.count() > 0:
                            freq_score = min(lemma.count() / 50.0, 1.0)  # Changed from 100
                        else:
                            freq_score = 0.5  # Changed from 0.3
                        
                        # Lemma position within synset
                        lemma_factor = 1.0 / (lemma_idx + 1)
                        
                        # Combined score - MORE GENEROUS
                        score = (
                            freq_score * 0.4 +           # Frequency
                            position_factor * 0.35 +     # Synset position
                            lemma_factor * 0.25          # Lemma position
                        )
                        
                        # Boost first synset significantly
                        if synset_idx == 0:
                            score *= 1.3  # 30% boost
                        
                        # Ensure minimum viable score
                        score = max(score, 0.35)
                        score = min(score, 1.0)
                        
                        candidates.append(SynonymCandidate(
                            word=syn,
                            score=score,
                            source='wordnet',
                            semantic_score=score
                        ))
                        seen.add(syn)
            
            logger.debug(f"WordNet: {len(candidates)} synonyms for '{word}' (avg: {np.mean([c.score for c in candidates]):.3f})")
            
        except Exception as e:
            logger.debug(f"WordNet lookup failed for '{word}': {e}")
        
        return candidates
    
    def get_synonyms_conceptnet(self, word: str, limit: int = 20) -> List[SynonymCandidate]:
        """
        Get synonyms from ConceptNet API
        FIXED: Better weight normalization
        """
        if not self.conceptnet_available:
            return []
        
        candidates = []
        seen = {word.lower()}
        
        try:
            url = f'http://api.conceptnet.io/query?node=/c/en/{word.replace(" ", "_")}&rel=/r/Synonym&limit={limit}'
            response = self.requests.get(url, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                
                for edge in data.get('edges', []):
                    start = edge.get('start', {}).get('label', '').lower()
                    end = edge.get('end', {}).get('label', '').lower()
                    
                    syn = end if start == word.lower() else start
                    
                    if syn and syn not in seen and self._is_valid_token(syn):
                        # FIXED: Better normalization
                        weight = edge.get('weight', 1.0)
                        
                        # ConceptNet weights typically range 1-4
                        # Normalize to 0.4-0.9 range
                        score = 0.4 + (min(weight, 4.0) / 4.0) * 0.5
                        
                        candidates.append(SynonymCandidate(
                            word=syn,
                            score=score,
                            source='conceptnet',
                            semantic_score=score
                        ))
                        seen.add(syn)
                
                logger.debug(f"ConceptNet: {len(candidates)} synonyms for '{word}' (avg: {np.mean([c.score for c in candidates]):.3f})")
                
        except Exception as e:
            logger.debug(f"ConceptNet lookup failed for '{word}': {e}")
        
        return candidates
    
    def get_synonyms_mlm(
        self,
        word: str,
        context: str,
        top_k: int = 30
    ) -> List[SynonymCandidate]:
        """
        Get synonyms using MLM predictions
        MLM scores are already probabilities (0-1)
        """
        if not self.mlm_available:
            return []
        
        candidates = []
        
        try:
            word_lower = word.lower()
            context_lower = context.lower()
            
            if word_lower in context_lower:
                masked = context_lower.replace(word_lower, self.tokenizer.mask_token, 1)
            else:
                masked = f"The {self.tokenizer.mask_token} is important. {context_lower}"
            
            inputs = self.tokenizer(
                masked,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            mask_positions = (inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            
            if len(mask_positions[0]) == 0:
                return []
            
            mask_pos = mask_positions[1][0].item()
            
            with torch.no_grad():
                outputs = self.mlm_model(**inputs)
                logits = outputs.logits[0, mask_pos]
                probs = torch.softmax(logits, dim=0)
            
            top_probs, top_indices = torch.topk(probs, k=top_k)
            
            seen = {word.lower()}
            for prob, idx in zip(top_probs, top_indices):
                token = self.tokenizer.decode([idx]).strip().lower()
                score = prob.item()
                
                if token not in seen and self._is_valid_token(token) and score >= 0.01:
                    candidates.append(SynonymCandidate(
                        word=token,
                        score=score,
                        source='mlm',
                        context_score=score
                    ))
                    seen.add(token)
            
            logger.debug(f"MLM: {len(candidates)} candidates for '{word}' (avg: {np.mean([c.score for c in candidates]):.3f})")
            
        except Exception as e:
            logger.debug(f"MLM prediction failed for '{word}': {e}")
        
        return candidates
    
    def get_synonyms_embedding(
        self,
        word: str,
        candidate_pool: List[str],
        top_k: int = 20,
        min_similarity: float = 0.4  # FIXED: Lowered from 0.5
    ) -> List[SynonymCandidate]:
        """
        Get synonyms using embedding similarity
        FIXED: Lower minimum threshold
        """
        if not self.embedding_available or not candidate_pool:
            return []
        
        candidates = []
        
        try:
            word_emb = self._get_embedding(word)
            
            similarities = []
            for candidate in candidate_pool:
                if candidate.lower() != word.lower():
                    cand_emb = self._get_embedding(candidate)
                    sim = self._cosine_similarity(word_emb, cand_emb)
                    similarities.append((candidate, sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            for cand, sim in similarities[:top_k]:
                if sim >= min_similarity and self._is_valid_token(cand):
                    candidates.append(SynonymCandidate(
                        word=cand,
                        score=sim,
                        source='embedding',
                        semantic_score=sim
                    ))
            
            logger.debug(f"Embedding: {len(candidates)} synonyms for '{word}' (avg: {np.mean([c.score for c in candidates]):.3f})")
            
        except Exception as e:
            logger.debug(f"Embedding similarity failed for '{word}': {e}")
        
        return candidates
    
    def get_multi_source_synonyms(
        self,
        word: str,
        context: str = "",
        top_k: int = 10,
        fusion_strategy: str = "adaptive"
    ) -> List[Tuple[str, float]]:
        """
        Get synonyms from all sources and fuse them intelligently
        FIXED: Better fusion scores
        """
        cache_key = f"{word}:{context[:50]}"
        if cache_key in self.synonym_cache:
            cached = self.synonym_cache[cache_key]
            return [(c.word, c.final_score) for c in cached[:top_k]]
        
        all_candidates = []
        
        # Collect from all sources
        if self.wordnet_available:
            all_candidates.extend(self.get_synonyms_wordnet(word))
        
        if self.conceptnet_available:
            all_candidates.extend(self.get_synonyms_conceptnet(word))
        
        if self.mlm_available and context:
            all_candidates.extend(self.get_synonyms_mlm(word, context))
        
        if self.embedding_available and all_candidates:
            candidate_words = list({c.word for c in all_candidates})
            all_candidates.extend(self.get_synonyms_embedding(word, candidate_words))
        
        if not all_candidates:
            logger.warning(f"No synonyms found for '{word}'")
            return []
        
        # Fuse candidates
        fused = self._fuse_candidates(all_candidates, word, context, fusion_strategy)
        
        # Filter by quality
        quality_filtered = [c for c in fused if c.final_score >= self.min_quality_threshold]
        
        # Sort by final score
        quality_filtered.sort(key=lambda x: x.final_score, reverse=True)
        
        # Cache
        if len(self.synonym_cache) < 1000:
            self.synonym_cache[cache_key] = quality_filtered
        
        result = [(c.word, c.final_score) for c in quality_filtered[:top_k]]
        
        if result:
            logger.info(f"'{word}' → {len(result)} quality synonyms (threshold: {self.min_quality_threshold:.2f})")
            for syn, score in result[:5]:
                logger.info(f"  • {syn}: {score:.3f}")
        
        return result
    
    def _fuse_candidates(
        self,
        candidates: List[SynonymCandidate],
        original_word: str,
        context: str,
        strategy: str
    ) -> List[SynonymCandidate]:
        """
        Fuse candidates from multiple sources
        FIXED: More generous scoring
        """
        grouped = defaultdict(list)
        for c in candidates:
            grouped[c.word].append(c)
        
        fused = []
        
        for word, group in grouped.items():
            scores = {c.source: c.score for c in group}
            
            if strategy == "adaptive":
                # FIXED: More balanced weights
                weights = {
                    'wordnet': 1.0,       # High quality
                    'conceptnet': 0.95,   # Good quality
                    'mlm': 0.90,          # Context-aware
                    'embedding': 0.95     # Semantic match
                }
                
                # Weighted average
                weighted_sum = sum(
                    scores.get(src, 0) * weights.get(src, 0.8)
                    for src in scores
                )
                
                # Source count boost - MORE GENEROUS
                source_count = len(scores)
                if source_count >= 3:
                    boost = 0.25  # 25% boost for 3+ sources
                elif source_count == 2:
                    boost = 0.15  # 15% boost for 2 sources
                else:
                    boost = 0.0
                
                final_score = min(weighted_sum + boost, 1.0)
                
            elif strategy == "vote":
                # Average with boost
                final_score = sum(scores.values()) / len(scores)
                
                if len(scores) >= 2:
                    final_score *= 1.3  # 30% boost
                
                final_score = min(final_score, 1.0)
                
            else:  # "max"
                final_score = max(scores.values())
                
                # Small boost for multiple sources
                if len(scores) > 1:
                    final_score = min(final_score * 1.1, 1.0)
            
            best = max(group, key=lambda c: c.score)
            best.final_score = final_score
            
            fused.append(best)
        
        return fused
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        
        if len(self.embedding_cache) < 5000:
            self.embedding_cache[text] = embedding
        
        return embedding
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Cosine similarity"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _is_valid_token(self, token: str) -> bool:
        """Validate token"""
        token = token.strip().lower()
        
        if len(token) < 2:
            return False
        
        if token.startswith(('[', '<', '##', '▁')):
            return False
        
        if token.isdigit():
            return False
        
        if not any(c.isalpha() for c in token):
            return False
        
        if len(token) > 20:
            return False
        
        return True
    
    def get_score_statistics(self, word: str, context: str = "") -> Dict[str, Any]:
        """
        Get detailed statistics about synonym scores
        Useful for debugging and calibration
        """
        stats = {
            'word': word,
            'sources': {}
        }
        
        if self.wordnet_available:
            wn_syns = self.get_synonyms_wordnet(word)
            if wn_syns:
                scores = [s.score for s in wn_syns]
                stats['sources']['wordnet'] = {
                    'count': len(wn_syns),
                    'min': min(scores),
                    'max': max(scores),
                    'mean': np.mean(scores),
                    'median': np.median(scores)
                }
        
        if self.conceptnet_available:
            cn_syns = self.get_synonyms_conceptnet(word)
            if cn_syns:
                scores = [s.score for s in cn_syns]
                stats['sources']['conceptnet'] = {
                    'count': len(cn_syns),
                    'min': min(scores),
                    'max': max(scores),
                    'mean': np.mean(scores),
                    'median': np.median(scores)
                }
        
        if self.mlm_available and context:
            mlm_syns = self.get_synonyms_mlm(word, context)
            if mlm_syns:
                scores = [s.score for s in mlm_syns]
                stats['sources']['mlm'] = {
                    'count': len(mlm_syns),
                    'min': min(scores),
                    'max': max(scores),
                    'mean': np.mean(scores),
                    'median': np.median(scores)
                }
        
        # Get final scores
        final_syns = self.get_multi_source_synonyms(word, context, top_k=20)
        if final_syns:
            final_scores = [score for _, score in final_syns]
            stats['final'] = {
                'count': len(final_syns),
                'min': min(final_scores),
                'max': max(final_scores),
                'mean': np.mean(final_scores),
                'median': np.median(final_scores),
                'threshold': self.min_quality_threshold
            }
        
        return stats
    
    def explain_synonyms(self, word: str, context: str = "") -> Dict[str, Any]:
        """Get detailed explanation of synonym generation"""
        explanation = {
            'word': word,
            'context': context,
            'sources': {}
        }
        
        if self.wordnet_available:
            wn_syns = self.get_synonyms_wordnet(word)
            explanation['sources']['wordnet'] = {
                'count': len(wn_syns),
                'synonyms': [(s.word, s.score) for s in wn_syns[:5]]
            }
        
        if self.conceptnet_available:
            cn_syns = self.get_synonyms_conceptnet(word)
            explanation['sources']['conceptnet'] = {
                'count': len(cn_syns),
                'synonyms': [(s.word, s.score) for s in cn_syns[:5]]
            }
        
        if self.mlm_available and context:
            mlm_syns = self.get_synonyms_mlm(word, context)
            explanation['sources']['mlm'] = {
                'count': len(mlm_syns),
                'synonyms': [(s.word, s.score) for s in mlm_syns[:5]]
            }
        
        final = self.get_multi_source_synonyms(word, context, top_k=10)
        explanation['final'] = {
            'count': len(final),
            'synonyms': final,
            'threshold': self.min_quality_threshold
        }
        
        return explanation


class AdaptiveQueryExpander:
    """
    Adaptive query expander using dynamic synonym engine
    FIXED: Adjusted thresholds
    """
    
    def __init__(
        self,
        synonym_engine: Optional[DynamicSynonymEngine] = None,
        min_expansion_quality: float = 0.45,  # FIXED: Lowered from 0.70
        max_expansions_per_word: int = 3
    ):
        """Initialize adaptive query expander"""
        self.engine = synonym_engine or DynamicSynonymEngine()
        self.min_expansion_quality = min_expansion_quality
        self.max_expansions_per_word = max_expansions_per_word
        
        logger.info(f"AdaptiveQueryExpander initialized")
        logger.info(f"  Min quality: {min_expansion_quality}")
        logger.info(f"  Max expansions/word: {max_expansions_per_word}")
    
    def expand_query(
        self,
        query: str,
        context: str = "",
        strategy: str = "selective"
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Expand query adaptively with FIXED thresholds"""
        try:
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(query.lower())
        except:
            tokens = query.lower().split()
        
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been'
        }
        
        content_words = [t for t in tokens if t not in stopwords and len(t) > 2]
        
        # FIXED: Adjusted thresholds
        if strategy == "aggressive":
            top_k = self.max_expansions_per_word + 2
            quality_threshold = max(self.min_expansion_quality - 0.1, 0.35)
        elif strategy == "conservative":
            top_k = max(self.max_expansions_per_word - 1, 1)
            quality_threshold = min(self.min_expansion_quality + 0.1, 0.60)
        else:  # selective
            top_k = self.max_expansions_per_word
            quality_threshold = self.min_expansion_quality
        
        expansions = {}
        
        for word in content_words:
            synonyms = self.engine.get_multi_source_synonyms(
                word,
                context=context or query,
                top_k=top_k,
                fusion_strategy="adaptive"
            )
            
            quality_synonyms = [
                (syn, score) for syn, score in synonyms
                if score >= quality_threshold
            ]
            
            if quality_synonyms:
                expansions[word] = quality_synonyms
                logger.info(f"'{word}' → {[s for s, _ in quality_synonyms]}")
        
        return expansions
    
    def generate_expanded_queries(
        self,
        query: str,
        context: str = "",
        max_variants: int = 5,
        strategy: str = "selective"
    ) -> List[Tuple[str, float]]:
        """Generate expanded query variants with quality scores"""
        expansions = self.expand_query(query, context, strategy)
        
        if not expansions:
            return [(query, 1.0)]
        
        variants = [(query, 1.0)]
        
        # Single-word replacements
        for word, synonyms in expansions.items():
            for syn, score in synonyms[:2]:
                variant = query.lower().replace(word, syn)
                if variant not in [v[0] for v in variants]:
                    variants.append((variant, score * 0.9))
        
        # Multi-word replacements
        if len(expansions) >= 2:
            words = list(expansions.keys())[:2]
            if len(expansions[words[0]]) > 0 and len(expansions[words[1]]) > 0:
                syn1, score1 = expansions[words[0]][0]
                syn2, score2 = expansions[words[1]][0]
                
                variant = query.lower()
                variant = variant.replace(words[0], syn1)
                variant = variant.replace(words[1], syn2)
                
                combined_score = (score1 * score2) ** 0.5
                variants.append((variant, combined_score * 0.85))
        
        variants.sort(key=lambda x: x[1], reverse=True)
        
        return variants[:max_variants]


# Usage example with diagnostics
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("DYNAMIC SYNONYM ENGINE - SCORE CALIBRATION TEST")
    print("="*70)
    
    # Initialize engine
    engine = DynamicSynonymEngine(
        enable_wordnet=True,
        enable_conceptnet=True,
        enable_mlm=True,
        enable_embedding=True,
        min_quality_threshold=0.40  # Realistic threshold
    )
    
    # Test words
    test_cases = [
        ("important", "This is a very important document about machine learning"),
        ("algorithm", "The algorithm processes data efficiently"),
        ("performance", "We need to improve system performance")
    ]
    
    for word, context in test_cases:
        print(f"\n{'─'*70}")
        print(f"TESTING: '{word}'")
        print(f"Context: '{context}'")
        print(f"{'─'*70}")
        
        # Get score statistics
        stats = engine.get_score_statistics(word, context)
        
        print("\n📊 SCORE STATISTICS BY SOURCE:")
        for source, data in stats.get('sources', {}).items():
            print(f"\n  {source.upper()}:")
            print(f"    Count: {data['count']}")
            print(f"    Range: {data['min']:.3f} - {data['max']:.3f}")
            print(f"    Mean:  {data['mean']:.3f}")
            print(f"    Median: {data['median']:.3f}")
        
        if 'final' in stats:
            print(f"\n  FINAL (FUSED):")
            print(f"    Count: {stats['final']['count']}")
            print(f"    Range: {stats['final']['min']:.3f} - {stats['final']['max']:.3f}")
            print(f"    Mean:  {stats['final']['mean']:.3f}")
            print(f"    Median: {stats['final']['median']:.3f}")
            print(f"    Threshold: {stats['final']['threshold']:.3f}")
        
        # Get top synonyms
        print(f"\n🎯 TOP SYNONYMS:")
        synonyms = engine.get_multi_source_synonyms(word, context, top_k=10)
        for syn, score in synonyms:
            print(f"    • {syn:20s} {score:.3f}")
        
        if not synonyms:
            print("    ⚠️  No synonyms passed the quality threshold!")
    
    # Test query expansion
    print(f"\n{'='*70}")
    print("QUERY EXPANSION TEST")
    print(f"{'='*70}\n")
    
    expander = AdaptiveQueryExpander(engine, min_expansion_quality=0.45)
    
    query = "find important machine learning papers"
    print(f"Original query: '{query}'\n")
    
    variants = expander.generate_expanded_queries(query, max_variants=5, strategy="selective")
    
    print("Generated variants:")
    for i, (variant, score) in enumerate(variants, 1):
        print(f"  {i}. [{score:.3f}] {variant}")