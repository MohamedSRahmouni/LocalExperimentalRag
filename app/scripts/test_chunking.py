#!/usr/bin/env python3
"""
Test complet du Dynamic Synonym Handler avec TOUTES les métriques
Teste UNIQUEMENT le système actuel (pas de comparaison baseline)
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.preprocessing.semantic_chunker import SemanticChunker
from services.preprocessing.transformer_synonym_handler import (
    DynamicSynonymEngine,
    AdaptiveQueryExpander
)


@dataclass
class EvaluationMetrics:
    """Container for ALL evaluation metrics"""
    # Retrieval & Ranking
    precision_at_k: Dict[int, float]
    recall: float
    ndcg: float
    mrr: float
    
    # Latency
    avg_search_time: float
    p99_latency: float
    
    # Semantic Robustness
    synonym_handling_score: float
    polysemy_handling_score: float
    concept_drift_score: float
    adversarial_robustness_score: float
    
    # Synonym Quality
    avg_synonym_quality: float
    max_synonym_quality: float
    min_synonym_quality: float
    median_synonym_quality: float
    synonym_diversity: float
    multi_source_coverage: float
    synonym_score_distribution: Dict[str, float]
    
    # Source Statistics
    wordnet_coverage: float
    conceptnet_coverage: float
    mlm_coverage: float
    embedding_coverage: float
    
    # Behavior
    zero_result_rate: float
    
    # Overall
    overall_score: float


class DynamicSynonymEvaluator:
    """Évaluateur complet pour Dynamic Synonym Engine"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        # Initialize chunker
        self.chunker = SemanticChunker(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            use_gpu=False
        )
        
        # Initialize Dynamic Synonym Engine with FIXED calibration
        self.logger.info("\n" + "="*80)
        self.logger.info("Initializing Dynamic Synonym Engine (FIXED)")
        self.logger.info("="*80)
        
        self.engine = DynamicSynonymEngine(
            enable_wordnet=True,
            enable_conceptnet=True,
            enable_mlm=True,
            enable_embedding=True,
            min_quality_threshold=0.40  # FIXED threshold
        )
        
        self.expander = AdaptiveQueryExpander(
            synonym_engine=self.engine,
            min_expansion_quality=0.45,  # FIXED threshold
            max_expansions_per_word=3
        )
        
        # Test data
        self.test_corpus = self._load_test_corpus()
        self.test_queries = self._load_test_queries()
        self.relevance_judgments = self._load_relevance_judgments()
    
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_test_corpus(self) -> List[str]:
        """Extended test corpus"""
        return [
            # Technology (0-3)
            "A laptop is a portable personal computer designed for mobility and convenience. "
            "It typically features a built-in keyboard, trackpad, and screen. "
            "Laptops come in various sizes and configurations for different use cases.",
            
            "A notebook computer, often called a laptop, is a small computing device. "
            "It provides portability while maintaining computing power comparable to desktop systems. "
            "Modern notebooks feature advanced processors and sufficient RAM for most tasks.",
            
            "The Apple iPhone is a smartphone developed by Apple Inc. "
            "It revolutionized the mobile industry with its touch interface and app ecosystem. "
            "iPhones are known for their premium build quality and performance.",
            
            "Apple Inc. is a technology company that designs and manufactures consumer electronics. "
            "The company is one of the largest in the world by market capitalization. "
            "Apple products include computers, phones, tablets, and wearables.",
            
            # Home Repair (4-6)
            "A leaky faucet can waste significant water over time. "
            "Common causes include worn washers, damaged seals, or corrosion. "
            "Many people can fix a leaky faucet without calling a plumber.",
            
            "To fix a dripping tap, you may need to replace internal components. "
            "First, turn off the water supply to prevent water damage. "
            "Then disassemble the faucet carefully and inspect for damaged parts.",
            
            "Plumbing maintenance is essential for home upkeep. "
            "Regular inspections can prevent costly repairs in the future. "
            "DIY plumbing projects range from simple fixes to complex installations.",
            
            # Cloud (7-9)
            "Cloud storage services allow users to store data on remote servers. "
            "Benefits include accessibility, backup, and scalability. "
            "Popular cloud storage providers include Google Drive, Dropbox, and OneDrive.",
            
            "AWS (Amazon Web Services) is a leading cloud computing platform. "
            "It provides services including computing, storage, databases, and networking. "
            "AWS is widely used by enterprises and startups worldwide.",
            
            "Cloud computing offers cost efficiency and flexibility. "
            "Organizations can scale resources up or down based on demand. "
            "Security and compliance are important considerations in cloud adoption.",
            
            # Fitness (10-11)
            "Running is an excellent form of aerobic exercise. "
            "It improves cardiovascular health and burns calories effectively. "
            "Proper running technique and footwear are important for injury prevention.",
            
            "Jogging regularly can improve overall fitness levels. "
            "It's a low-cost activity that requires minimal equipment. "
            "Consistent jogging builds endurance and strengthens leg muscles.",
            
            # Winter (12-13)
            "Winter athletic gear includes insulated jackets, thermal wear, and appropriate footwear. "
            "Cold weather running requires special considerations for safety and comfort. "
            "Layering is key to maintaining body temperature during winter exercise.",
            
            "Winter boots provide insulation and traction in cold, snowy conditions. "
            "Good winter footwear prevents slips and keeps feet warm. "
            "Different boot styles suit various winter activities.",
            
            # Fruit (14-15)
            "An apple is a sweet, edible fruit produced by an apple tree. "
            "Apples are grown worldwide and are the most widely cultivated species in the genus Malus. "
            "They are rich in fiber and vitamin C, making them a healthy snack choice.",
            
            "Fresh apples can be eaten raw or used in various recipes. "
            "Apple pie is a classic dessert enjoyed in many countries. "
            "Organic apples are grown without synthetic pesticides or fertilizers.",
        ]
    
    def _load_test_queries(self) -> List[Dict[str, Any]]:
        """All test queries"""
        return [
            # Standard queries
            {"query": "portable computer", "query_id": "q1", "category": "standard"},
            {"query": "mobile phone from Apple", "query_id": "q2", "category": "standard"},
            {"query": "fix dripping water tap", "query_id": "q3", "category": "standard"},
            {"query": "online data storage services", "query_id": "q4", "category": "standard"},
            {"query": "aerobic running exercise", "query_id": "q5", "category": "standard"},
            
            # Synonym queries (CRITICAL TEST)
            {"query": "notebook computer", "query_id": "q6", "category": "synonym"},
            {"query": "leaky faucet repair", "query_id": "q7", "category": "synonym"},
            {"query": "winter shoes for athletes", "query_id": "q8", "category": "synonym"},
            
            # Polysemy queries
            {"query": "Apple technology company products", "query_id": "q9", "category": "polysemy", "sense": "company"},
            {"query": "Apple iPhone smartphone", "query_id": "q10", "category": "polysemy", "sense": "product"},
            {"query": "fresh apple fruit nutrition", "query_id": "q11", "category": "polysemy", "sense": "fruit"},
            
            # Concept drift
            {"query": "How to fix a water leak from kitchen faucet during winter?", "query_id": "q12", "category": "concept_drift"},
            {"query": "What cloud computing services does Amazon provide for enterprise data storage?", "query_id": "q13", "category": "concept_drift"},
            
            # Adversarial queries
            {"query": "portabel computr machne", "query_id": "q14", "category": "adversarial"},
            {"query": "the bestest way for to runn fast", "query_id": "q15", "category": "adversarial"},
            {"query": "clou storag servce onlin", "query_id": "q16", "category": "adversarial"},
        ]
    
    def _load_relevance_judgments(self) -> Dict[str, List[int]]:
        """Ground truth"""
        return {
            "q1": [0, 1], "q2": [2, 3], "q3": [4, 5], "q4": [7, 8, 9], "q5": [10, 11],
            "q6": [0, 1], "q7": [4, 5], "q8": [12, 13],
            "q9": [2, 3], "q10": [2, 3], "q11": [14, 15],
            "q12": [4, 5, 12], "q13": [8, 9],
            "q14": [0, 1], "q15": [10, 11], "q16": [7, 8, 9],
        }
    
    def analyze_synonym_quality_detailed(self) -> Dict[str, Any]:
        """Analyse détaillée de la qualité des synonymes"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("DETAILED SYNONYM QUALITY ANALYSIS")
        self.logger.info(f"{'='*80}\n")
        
        test_words = [
            ("portable", "portable computer devices"),
            ("important", "this is very important information"),
            ("fix", "how to fix broken things"),
            ("running", "running is good exercise"),
            ("computer", "computer technology advances"),
            ("storage", "data storage solutions"),
            ("mobile", "mobile phone technology"),
        ]
        
        all_scores = []
        score_ranges = {'0.0-0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7-1.0': 0}
        source_stats = defaultdict(lambda: {'count': 0, 'scores': [], 'words': []})
        
        for word, context in test_words:
            self.logger.info(f"\n📝 Word: '{word}'")
            self.logger.info(f"   Context: '{context}'")
            
            # Get score statistics
            stats = self.engine.get_score_statistics(word, context)
            
            # Display per-source stats
            for source, data in stats.get('sources', {}).items():
                self.logger.info(f"\n   {source.upper()}:")
                self.logger.info(f"     Count:  {data['count']}")
                self.logger.info(f"     Range:  {data['min']:.3f} - {data['max']:.3f}")
                self.logger.info(f"     Mean:   {data['mean']:.3f}")
                self.logger.info(f"     Median: {data['median']:.3f}")
                
                source_stats[source]['count'] += data['count']
                source_stats[source]['scores'].append(data['mean'])
                source_stats[source]['words'].append(word)
            
            # Get final synonyms
            synonyms = self.engine.get_multi_source_synonyms(word, context, top_k=10)
            
            if synonyms:
                self.logger.info(f"\n   🎯 TOP SYNONYMS:")
                for syn, score in synonyms[:5]:
                    self.logger.info(f"      • {syn:20s} {score:.3f}")
                    all_scores.append(score)
                    
                    # Categorize score
                    if score >= 0.7:
                        score_ranges['0.7-1.0'] += 1
                    elif score >= 0.5:
                        score_ranges['0.5-0.7'] += 1
                    elif score >= 0.3:
                        score_ranges['0.3-0.5'] += 1
                    else:
                        score_ranges['0.0-0.3'] += 1
            else:
                self.logger.warning(f"   ⚠️  No synonyms found!")
        
        # Overall statistics
        self.logger.info(f"\n{'='*80}")
        self.logger.info("OVERALL STATISTICS")
        self.logger.info(f"{'='*80}\n")
        
        if all_scores:
            self.logger.info(f"📊 Score Distribution:")
            self.logger.info(f"   Total synonyms: {len(all_scores)}")
            self.logger.info(f"   Min:    {min(all_scores):.3f}")
            self.logger.info(f"   Max:    {max(all_scores):.3f}")
            self.logger.info(f"   Mean:   {np.mean(all_scores):.3f}")
            self.logger.info(f"   Median: {np.median(all_scores):.3f}")
            self.logger.info(f"   Std:    {np.std(all_scores):.3f}")
            
            self.logger.info(f"\n📈 Score Ranges:")
            for range_name, count in sorted(score_ranges.items()):
                pct = count / len(all_scores) * 100
                bar = '█' * int(pct / 5)
                self.logger.info(f"   {range_name}: {count:3d} ({pct:5.1f}%) {bar}")
            
            # Check if target is met
            high_quality = score_ranges['0.7-1.0'] + score_ranges['0.5-0.7']
            high_quality_pct = high_quality / len(all_scores) * 100
            
            self.logger.info(f"\n🎯 Quality Check:")
            self.logger.info(f"   High quality (≥0.5): {high_quality} ({high_quality_pct:.1f}%)")
            
            if np.mean(all_scores) >= 0.6:
                self.logger.info(f"   ✅ EXCELLENT! Average score {np.mean(all_scores):.3f} ≥ 0.6")
            elif np.mean(all_scores) >= 0.5:
                self.logger.info(f"   ✓ Good! Average score {np.mean(all_scores):.3f} ≥ 0.5")
            elif np.mean(all_scores) >= 0.4:
                self.logger.info(f"   → Fair. Average score {np.mean(all_scores):.3f} ≥ 0.4")
            else:
                self.logger.info(f"   ⚠️ LOW! Average score {np.mean(all_scores):.3f} < 0.4")
        
        # Source comparison
        self.logger.info(f"\n🔍 Source Comparison:")
        source_coverage = {}
        for source, data in source_stats.items():
            if data['scores']:
                avg_score = np.mean(data['scores'])
                coverage = data['count'] / len(test_words)
                source_coverage[source] = coverage
                self.logger.info(f"   {source:12s}: {data['count']:3d} synonyms, avg {avg_score:.3f}, coverage {coverage:.1%}")
        
        return {
            'avg_quality': np.mean(all_scores) if all_scores else 0.0,
            'max_quality': max(all_scores) if all_scores else 0.0,
            'min_quality': min(all_scores) if all_scores else 0.0,
            'median_quality': float(np.median(all_scores)) if all_scores else 0.0,
            'std_quality': float(np.std(all_scores)) if all_scores else 0.0,
            'high_quality_pct': high_quality_pct if all_scores else 0.0,
            'score_distribution': {k: v / len(all_scores) * 100 if all_scores else 0.0 
                                   for k, v in score_ranges.items()},
            'source_coverage': source_coverage,
            'diversity': len(set(all_scores)) / len(all_scores) if all_scores else 0.0
        }
    
    def _search_with_expansion(self, query: str, chunks: List[str], top_k: int):
        """Search with synonym expansion"""
        # Generate expanded variants
        variants = self.expander.generate_expanded_queries(
            query,
            context=" ".join(chunks[:3]),
            max_variants=5,
            strategy="selective"
        )
        
        # Log variants
        self.logger.debug(f"\nQuery variants for '{query}':")
        for i, (variant, score) in enumerate(variants):
            self.logger.debug(f"  {i+1}. [{score:.3f}] {variant}")
        
        # Search with RRF fusion
        k = 60  # RRF parameter
        doc_scores = defaultdict(float)
        
        for variant, quality_score in variants:
            results = self.chunker.get_most_similar_chunks(
                variant, chunks, top_k=top_k * 2
            )
            
            for rank, result in enumerate(results, 1):
                doc_id = result['index']
                rrf_score = 1.0 / (k + rank)
                doc_scores[doc_id] += rrf_score * quality_score
        
        # Sort and format
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            results.append({
                'index': doc_id,
                'similarity': score,
                'text': chunks[doc_id][:100] + "..."
            })
        
        return results
    
    def calculate_precision_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """Precision@K"""
        top_k = retrieved[:k]
        return len(set(top_k) & set(relevant)) / k if k > 0 else 0.0
    
    def calculate_recall(self, retrieved: List[int], relevant: List[int]) -> float:
        """Recall"""
        return len(set(retrieved) & set(relevant)) / len(relevant) if relevant else 0.0
    
    def calculate_ndcg(self, retrieved: List[int], relevant: List[int], k: int = 10) -> float:
        """NDCG@K"""
        dcg = sum(1.0 / np.log2(i + 1) for i, doc in enumerate(retrieved[:k], 1) if doc in relevant)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_mrr(self, retrieved: List[int], relevant: List[int]) -> float:
        """MRR"""
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant:
                return 1.0 / rank
        return 0.0
    
    def test_synonym_handling(self) -> float:
        """Test synonym handling (CRITICAL)"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("TESTING SYNONYM HANDLING (CRITICAL TEST)")
        self.logger.info(f"{'='*80}\n")
        
        synonym_queries = [q for q in self.test_queries if q.get("category") == "synonym"]
        scores = []
        
        for query_data in synonym_queries:
            query = query_data["query"]
            query_id = query_data["query_id"]
            relevant_docs = self.relevance_judgments[query_id]
            
            self.logger.info(f"\n📝 Query: '{query}'")
            self.logger.info(f"   Expected docs: {relevant_docs}")
            
            results = self._search_with_expansion(query, self.test_corpus, top_k=5)
            retrieved_indices = [r["index"] for r in results]
            
            precision = self.calculate_precision_at_k(retrieved_indices, relevant_docs, k=5)
            recall = self.calculate_recall(retrieved_indices, relevant_docs)
            ndcg = self.calculate_ndcg(retrieved_indices, relevant_docs, k=5)
            
            scores.append(ndcg)
            
            self.logger.info(f"   Retrieved: {retrieved_indices}")
            self.logger.info(f"   Precision@5: {precision:.3f}")
            self.logger.info(f"   Recall:      {recall:.3f}")
            self.logger.info(f"   NDCG@5:      {ndcg:.3f}")
            
            if ndcg >= 0.8:
                self.logger.info(f"   ✅ Excellent!")
            elif ndcg >= 0.6:
                self.logger.info(f"   ✓ Good")
            else:
                self.logger.info(f"   ⚠️ Needs improvement")
        
        avg_score = np.mean(scores) if scores else 0.0
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"⭐ SYNONYM HANDLING SCORE: {avg_score:.3f}")
        self.logger.info(f"{'='*80}")
        
        return avg_score
    
    def test_polysemy_handling(self) -> float:
        """Test polysemy/word sense disambiguation"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("TESTING POLYSEMY HANDLING")
        self.logger.info(f"{'='*80}\n")
        
        polysemy_queries = [q for q in self.test_queries if q.get("category") == "polysemy"]
        scores = []
        
        for query_data in polysemy_queries:
            query = query_data["query"]
            query_id = query_data["query_id"]
            sense = query_data.get("sense", "unknown")
            relevant_docs = self.relevance_judgments[query_id]
            
            self.logger.info(f"\n📝 Query: '{query}' (sense: {sense})")
            self.logger.info(f"   Expected docs: {relevant_docs}")
            
            results = self._search_with_expansion(query, self.test_corpus, top_k=5)
            retrieved_indices = [r["index"] for r in results]
            
            ndcg = self.calculate_ndcg(retrieved_indices, relevant_docs, k=5)
            scores.append(ndcg)
            
            self.logger.info(f"   Retrieved: {retrieved_indices}")
            self.logger.info(f"   NDCG@5: {ndcg:.3f}")
        
        avg_score = np.mean(scores) if scores else 0.0
        self.logger.info(f"\n⭐ POLYSEMY HANDLING SCORE: {avg_score:.3f}")
        
        return avg_score
    
    def test_concept_drift(self) -> float:
        """Test handling of complex multi-concept queries"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("TESTING CONCEPT DRIFT")
        self.logger.info(f"{'='*80}\n")
        
        drift_queries = [q for q in self.test_queries if q.get("category") == "concept_drift"]
        scores = []
        
        for query_data in drift_queries:
            query = query_data["query"]
            query_id = query_data["query_id"]
            relevant_docs = self.relevance_judgments[query_id]
            
            self.logger.info(f"\n📝 Query: '{query[:60]}...'")
            self.logger.info(f"   Expected docs: {relevant_docs}")
            
            results = self._search_with_expansion(query, self.test_corpus, top_k=5)
            retrieved_indices = [r["index"] for r in results]
            
            ndcg = self.calculate_ndcg(retrieved_indices, relevant_docs, k=5)
            scores.append(ndcg)
            
            self.logger.info(f"   Retrieved: {retrieved_indices}")
            self.logger.info(f"   NDCG@5: {ndcg:.3f}")
        
        avg_score = np.mean(scores) if scores else 0.0
        self.logger.info(f"\n⭐ CONCEPT DRIFT SCORE: {avg_score:.3f}")
        
        return avg_score
    
    def test_adversarial_robustness(self) -> float:
        """Test robustness to typos and noise"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("TESTING ADVERSARIAL ROBUSTNESS")
        self.logger.info(f"{'='*80}\n")
        
        adversarial_queries = [q for q in self.test_queries if q.get("category") == "adversarial"]
        scores = []
        
        for query_data in adversarial_queries:
            query = query_data["query"]
            query_id = query_data["query_id"]
            relevant_docs = self.relevance_judgments[query_id]
            
            self.logger.info(f"\n📝 Query: '{query}'")
            self.logger.info(f"   Expected docs: {relevant_docs}")
            
            results = self._search_with_expansion(query, self.test_corpus, top_k=5)
            retrieved_indices = [r["index"] for r in results]
            
            ndcg = self.calculate_ndcg(retrieved_indices, relevant_docs, k=5)
            scores.append(ndcg)
            
            self.logger.info(f"   Retrieved: {retrieved_indices}")
            self.logger.info(f"   NDCG@5: {ndcg:.3f}")
        
        avg_score = np.mean(scores) if scores else 0.0
        self.logger.info(f"\n⭐ ADVERSARIAL ROBUSTNESS SCORE: {avg_score:.3f}")
        
        return avg_score
    
    def measure_latency(self, num_iterations: int = 30) -> Tuple[float, float]:
        """Measure search latency"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("MEASURING LATENCY")
        self.logger.info(f"{'='*80}\n")
        
        test_query = "portable computer for work"
        latencies = []
        
        # Warmup
        for _ in range(5):
            self._search_with_expansion(test_query, self.test_corpus, top_k=5)
        
        # Measure
        for i in range(num_iterations):
            start = time.perf_counter()
            self._search_with_expansion(test_query, self.test_corpus, top_k=5)
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        
        self.logger.info(f"  Iterations: {num_iterations}")
        self.logger.info(f"  Average latency: {avg_latency:.2f} ms")
        self.logger.info(f"  P99 latency:     {p99_latency:.2f} ms")
        self.logger.info(f"  Min latency:     {min(latencies):.2f} ms")
        self.logger.info(f"  Max latency:     {max(latencies):.2f} ms")
        
        return avg_latency, p99_latency
    
    def calculate_zero_result_rate(self) -> float:
        """Calculate zero-result rate"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("CALCULATING ZERO-RESULT RATE")
        self.logger.info(f"{'='*80}\n")
        
        zero_count = 0
        total = len(self.test_queries)
        
        for query_data in self.test_queries:
            query = query_data["query"]
            results = self._search_with_expansion(query, self.test_corpus, top_k=5)
            
            if len(results) == 0:
                zero_count += 1
                self.logger.warning(f"  ⚠️  Zero results for: '{query}'")
        
        zero_rate = zero_count / total if total > 0 else 0.0
        
        self.logger.info(f"\n  Total queries: {total}")
        self.logger.info(f"  Zero results:  {zero_count}")
        self.logger.info(f"  Zero rate:     {zero_rate:.3f} ({zero_rate*100:.1f}%)")
        
        if zero_rate == 0:
            self.logger.info(f"  ✅ Perfect! No zero results")
        elif zero_rate < 0.05:
            self.logger.info(f"  ✓ Good! Very low zero rate")
        else:
            self.logger.info(f"  ⚠️ High zero rate")
        
        return zero_rate
    
    def run_full_evaluation(self) -> EvaluationMetrics:
        """Run comprehensive evaluation with ALL metrics"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("COMPREHENSIVE EVALUATION - ALL METRICS")
        self.logger.info(f"{'='*80}\n")
        
        # 1. Synonym Quality Analysis
        syn_quality = self.analyze_synonym_quality_detailed()
        
        # 2. Semantic Robustness Tests
        synonym_score = self.test_synonym_handling()
        polysemy_score = self.test_polysemy_handling()
        concept_drift_score = self.test_concept_drift()
        adversarial_score = self.test_adversarial_robustness()
        
        # 3. Standard retrieval metrics
        precision_scores = {5: [], 10: []}
        recall_scores = []
        ndcg_scores = []
        mrr_scores = []
        
        standard_queries = [q for q in self.test_queries if q.get("category") == "standard"]
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("TESTING STANDARD RETRIEVAL METRICS")
        self.logger.info(f"{'='*80}\n")
        
        for query_data in standard_queries:
            query = query_data["query"]
            query_id = query_data["query_id"]
            relevant_docs = self.relevance_judgments[query_id]
            
            results = self._search_with_expansion(query, self.test_corpus, top_k=10)
            retrieved_indices = [r["index"] for r in results]
            
            p5 = self.calculate_precision_at_k(retrieved_indices, relevant_docs, k=5)
            p10 = self.calculate_precision_at_k(retrieved_indices, relevant_docs, k=10)
            recall = self.calculate_recall(retrieved_indices, relevant_docs)
            ndcg = self.calculate_ndcg(retrieved_indices, relevant_docs, k=10)
            mrr = self.calculate_mrr(retrieved_indices, relevant_docs)
            
            precision_scores[5].append(p5)
            precision_scores[10].append(p10)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)
            
            self.logger.info(f"Query: '{query}'")
            self.logger.info(f"  P@5: {p5:.3f}, P@10: {p10:.3f}, Recall: {recall:.3f}, NDCG: {ndcg:.3f}, MRR: {mrr:.3f}")
        
        # 4. Latency
        avg_latency, p99_latency = self.measure_latency()
        
        # 5. Zero-result rate
        zero_result_rate = self.calculate_zero_result_rate()
        
        # 6. Calculate overall score
        overall = (
            0.25 * synonym_score +              # 25% - Most important
            0.15 * np.mean(ndcg_scores) +       # 15%
            0.10 * np.mean(recall_scores) +     # 10%
            0.10 * np.mean(mrr_scores) +        # 10%
            0.10 * polysemy_score +             # 10%
            0.10 * concept_drift_score +        # 10%
            0.10 * adversarial_score +          # 10%
            0.10 * syn_quality['avg_quality']   # 10%
        ) * 100
        
        return EvaluationMetrics(
            precision_at_k={k: np.mean(v) for k, v in precision_scores.items()},
            recall=np.mean(recall_scores),
            ndcg=np.mean(ndcg_scores),
            mrr=np.mean(mrr_scores),
            avg_search_time=avg_latency,
            p99_latency=p99_latency,
            synonym_handling_score=synonym_score,
            polysemy_handling_score=polysemy_score,
            concept_drift_score=concept_drift_score,
            adversarial_robustness_score=adversarial_score,
            avg_synonym_quality=syn_quality['avg_quality'],
            max_synonym_quality=syn_quality['max_quality'],
            min_synonym_quality=syn_quality['min_quality'],
            median_synonym_quality=syn_quality['median_quality'],
            synonym_diversity=syn_quality['diversity'],
            multi_source_coverage=1.0,
            synonym_score_distribution=syn_quality['score_distribution'],
            wordnet_coverage=syn_quality['source_coverage'].get('wordnet', 0.0),
            conceptnet_coverage=syn_quality['source_coverage'].get('conceptnet', 0.0),
            mlm_coverage=syn_quality['source_coverage'].get('mlm', 0.0),
            embedding_coverage=syn_quality['source_coverage'].get('embedding', 0.0),
            zero_result_rate=zero_result_rate,
            overall_score=overall
        )
    
    def print_summary(self, metrics: EvaluationMetrics):
        """Print comprehensive summary with ALL metrics"""
        print(f"""
{'='*80}
COMPREHENSIVE EVALUATION SUMMARY - DYNAMIC SYNONYM ENGINE (FIXED)
{'='*80}

SYNONYM QUALITY METRICS (KEY)
{'-'*80}
  Average Quality:       {metrics.avg_synonym_quality:.3f} {'✅ EXCELLENT' if metrics.avg_synonym_quality >= 0.6 else '✓ Good' if metrics.avg_synonym_quality >= 0.5 else '→ Fair' if metrics.avg_synonym_quality >= 0.4 else '⚠️ LOW'}
  Max Quality:           {metrics.max_synonym_quality:.3f}
  Min Quality:           {metrics.min_synonym_quality:.3f}
  Median Quality:        {metrics.median_synonym_quality:.3f}
  Diversity:             {metrics.synonym_diversity:.3f}
  Multi-Source Coverage: {metrics.multi_source_coverage:.3f}

SCORE DISTRIBUTION
{'-'*80}""")
        for range_name, pct in sorted(metrics.synonym_score_distribution.items()):
            bar = '█' * int(pct / 5)
            print(f"  {range_name}: {pct:5.1f}% {bar}")
        
        print(f"""
SOURCE COVERAGE
{'-'*80}
  WordNet:               {metrics.wordnet_coverage:.1%}
  ConceptNet:            {metrics.conceptnet_coverage:.1%}
  MLM:                   {metrics.mlm_coverage:.1%}
  Embedding:             {metrics.embedding_coverage:.1%}

SEMANTIC ROBUSTNESS METRICS
{'-'*80}
  Synonym Handling:      {metrics.synonym_handling_score:.3f} {'⭐⭐⭐ EXCELLENT' if metrics.synonym_handling_score >= 0.8 else '⭐⭐ Good' if metrics.synonym_handling_score >= 0.6 else '⭐ Fair' if metrics.synonym_handling_score >= 0.4 else '⚠️ LOW'}
  Polysemy Handling:     {metrics.polysemy_handling_score:.3f} {'✅' if metrics.polysemy_handling_score >= 0.7 else '✓' if metrics.polysemy_handling_score >= 0.5 else '→'}
  Concept Drift:         {metrics.concept_drift_score:.3f} {'✅' if metrics.concept_drift_score >= 0.7 else '✓' if metrics.concept_drift_score >= 0.5 else '→'}
  Adversarial Robustness:{metrics.adversarial_robustness_score:.3f} {'✅' if metrics.adversarial_robustness_score >= 0.7 else '✓' if metrics.adversarial_robustness_score >= 0.5 else '→'}

RETRIEVAL & RANKING METRICS
{'-'*80}
  Precision@5:           {metrics.precision_at_k.get(5, 0.0):.3f}
  Precision@10:          {metrics.precision_at_k.get(10, 0.0):.3f}
  Recall:                {metrics.recall:.3f}
  NDCG@10:               {metrics.ndcg:.3f}
  MRR:                   {metrics.mrr:.3f}

LATENCY METRICS
{'-'*80}
  Average Latency:       {metrics.avg_search_time:.2f} ms {'✅' if metrics.avg_search_time < 100 else '✓' if metrics.avg_search_time < 200 else '⚠️'}
  P99 Latency:           {metrics.p99_latency:.2f} ms

BEHAVIORAL METRICS
{'-'*80}
  Zero-Result Rate:      {metrics.zero_result_rate:.3f} ({metrics.zero_result_rate*100:.1f}%) {'✅ Perfect' if metrics.zero_result_rate == 0 else '✓ Good' if metrics.zero_result_rate < 0.05 else '⚠️'}

OVERALL SCORE
{'-'*80}
  Overall Score:         {metrics.overall_score:.1f}/100 {'🎉 EXCELLENT' if metrics.overall_score >= 80 else '✅ Good' if metrics.overall_score >= 70 else '✓ Fair' if metrics.overall_score >= 60 else '⚠️ Needs Work'}

{'='*80}
""")


def main():
    print("\n" + "="*80)
    print("DYNAMIC SYNONYM ENGINE - COMPLETE EVALUATION")
    print("Testing: WordNet + ConceptNet + MLM + Embeddings (FIXED)")
    print("="*80)
    
    evaluator = DynamicSynonymEvaluator()
    
    if not evaluator.chunker.is_model_available():
        print("✗ Chunker model not available")
        return False
    
    # Run full evaluation
    metrics = evaluator.run_full_evaluation()
    
    # Print summary
    evaluator.print_summary(metrics)
    
    # Save results
    results = {
        "metrics": asdict(metrics),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "min_quality_threshold": 0.40,
            "min_expansion_quality": 0.45,
            "sources": ["wordnet", "conceptnet", "mlm", "embedding"]
        }
    }
    
    output_path = Path(__file__).parent.parent / "dynamic_synonym_complete_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Final verdict
    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print(f"{'='*80}\n")
    
    verdicts = []
    
    # Synonym Quality
    if metrics.avg_synonym_quality >= 0.6:
        verdicts.append(("✅ SYNONYM QUALITY", "EXCELLENT", metrics.avg_synonym_quality))
    elif metrics.avg_synonym_quality >= 0.5:
        verdicts.append(("✓ SYNONYM QUALITY", "GOOD", metrics.avg_synonym_quality))
    elif metrics.avg_synonym_quality >= 0.4:
        verdicts.append(("→ SYNONYM QUALITY", "FAIR", metrics.avg_synonym_quality))
    else:
        verdicts.append(("⚠️ SYNONYM QUALITY", "LOW", metrics.avg_synonym_quality))
    
    # Synonym Handling
    if metrics.synonym_handling_score >= 0.8:
        verdicts.append(("✅ SYNONYM HANDLING", "EXCELLENT", metrics.synonym_handling_score))
    elif metrics.synonym_handling_score >= 0.6:
        verdicts.append(("✓ SYNONYM HANDLING", "GOOD", metrics.synonym_handling_score))
    else:
        verdicts.append(("⚠️ SYNONYM HANDLING", "NEEDS WORK", metrics.synonym_handling_score))
    
    # Overall Performance
    if metrics.overall_score >= 80:
        verdicts.append(("🎉 OVERALL", "EXCELLENT", metrics.overall_score / 100))
    elif metrics.overall_score >= 70:
        verdicts.append(("✅ OVERALL", "GOOD", metrics.overall_score / 100))
    elif metrics.overall_score >= 60:
        verdicts.append(("✓ OVERALL", "FAIR", metrics.overall_score / 100))
    else:
        verdicts.append(("⚠️ OVERALL", "NEEDS WORK", metrics.overall_score / 100))
    
    for symbol_metric, rating, score in verdicts:
        print(f"{symbol_metric:25s} {rating:15s} ({score:.3f})")
    
    print(f"\n{'='*80}\n")
    
    return metrics.avg_synonym_quality >= 0.5 and metrics.synonym_handling_score >= 0.6


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)