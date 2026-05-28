"""
RAG Chain Pipeline - LangGraph Powered
Enhanced with better retrieval strategies
"""

import logging
import re
import unicodedata
from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass
import time

from langchain_openai import ChatOpenAI

from .memory        import ConversationMemoryManager
from .graph_builder import build_rag_graph

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class RAGConfig:
    top_k:              int   = 5
    min_score:          float = 0.3
    enable_reranking:   bool  = True
    use_hybrid:         bool  = True
    bm25_weight:        float = 0.5
    vector_weight:      float = 0.5
    temperature:        float = 0.1
    max_tokens:         int   = 1000
    model:              str   = "local-model"
    base_url:           str   = "http://localhost:1234/v1"
    include_sources:    bool  = True
    language:           str   = "français"
    max_context_length: int   = 4000


# ============================================================
# RETRIEVAL CONFIG
# ============================================================

class RetrievalConfig:
    def __init__(
        self,
        top_k:              int   = 5,
        min_score:          float = 0.3,
        max_context_length: int   = 4000,
        enable_reranking:   bool  = True,
        diversity_weight:   float = 0.0,
        recency_weight:     float = 0.0,
        use_hybrid:         bool  = True,
        bm25_weight:        float = 0.5,
        vector_weight:      float = 0.5,
        rrf_k:              int   = 20,
    ):
        self.top_k              = top_k
        self.min_score          = min_score
        self.max_context_length = max_context_length
        self.enable_reranking   = enable_reranking
        self.diversity_weight   = diversity_weight
        self.recency_weight     = recency_weight
        self.use_hybrid         = use_hybrid
        self.bm25_weight        = bm25_weight
        self.vector_weight      = vector_weight
        self.rrf_k              = rrf_k


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class RetrievedChunk:
    chunk_id:     str
    text:         str
    score:        float
    source:       str
    metadata:     Dict[str, Any]
    rank:         int
    vector_score: float = 0.0
    bm25_score:   float = 0.0
    hybrid_score: float = 0.0


@dataclass
class RetrievalResult:
    query:          str
    chunks:         List[RetrievedChunk]
    context:        str
    total_found:    int
    retrieval_time: float
    metadata:       Dict[str, Any]


# ============================================================
# BM25 INDEX - CORPUS-WIDE
# ============================================================

class BM25Index:
    """
    Robust corpus-wide BM25 index.

    Handles:
    - Accent normalization  (ingénieur == ingenieur)
    - Multilingual stop words (FR + EN + AR basics)
    - Aggressive tokenization (camelCase, snake_case, dots, slashes)
    - Graceful corpus loading (get_all_chunks → get_random_samples → fallback)
    - Debug diagnostics built in
    """

    # ── Stop words (post-normalization, no accented chars) ─────────────────
    STOP_WORDS = {
        # French
        "le","la","les","un","une","des","du","de","d",
        "et","ou","en","a","au","aux","ce","se","cet","cette",
        "est","son","sa","ses","mon","ma","mes","ton","ta","tes",
        "que","qui","quoi","dont","je","tu","il","elle",
        "nous","vous","ils","elles","on","y",
        "sur","dans","par","pour","avec","sans","sous","vers",
        "plus","mais","donc","car","ni","or","si","bien",
        "tout","tous","toute","toutes","aussi","tres","meme",
        "encore","alors","comme","avait","avoir","avais",
        "avons","avez","etait","etaient","etre",
        "fait","faire","va","vais","sont","ont","ete","eu",
        "apres","avant","entre","lors","depuis","pendant",
        # English
        "the","an","is","are","was","were","be","been","being",
        "of","in","on","at","to","for","with","by","from",
        "this","that","these","those","it","its","as","into",
        "have","has","had","do","does","did","will","would",
        "could","should","may","might","can","shall","not",
        "and","or","but","so","yet","about","through","over",
        "he","she","they","we","you","i","me","him","her","us",
        "his","their","our","your","my",
        # Arabic basics (transliterated)
        "wa","fi","min","ila","an","ma","la","li","ala","bi",
        "had","maa","kan","haa","aw","thm","fy",
    }

    def __init__(self, vectorstore=None):
        self.index:             Any                  = None
        self.chunks:            List[Dict]           = []
        self.tokenized_corpus:  List[List[str]]      = []
        self.vectorstore                             = vectorstore
        self._initialized:      bool                 = False
        self._vocab_size:       int                  = 0

    # ── Normalization ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Unicode NFD → strip combining marks → lowercase.
        ingénieur → ingenieur, Full-Stack → full stack, etc.
        """
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        return text.lower()

    # ── Tokenizer ───────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """
        General-purpose tokenizer:
        1. Normalize accents + lowercase
        2. Split camelCase / PascalCase
        3. Replace all non-alphanumeric with spaces
        4. Remove stop words + short tokens
        """
        if not text or not text.strip():
            return []

        text = self._normalize(text)

        # Split camelCase / PascalCase (fullStack → full stack)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        # Replace non-alphanumeric with space
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        tokens = text.split()
        return [
            t for t in tokens
            if t not in self.STOP_WORDS and len(t) >= 2
        ]

    # ── Corpus loading ──────────────────────────────────────────────────────

    def initialize_from_corpus(self, max_chunks: int = 2000) -> bool:
        """
        Build a corpus-wide BM25 index.
        Tries multiple vectorstore APIs in order of preference.
        """
        if self._initialized:
            return True

        if not self.vectorstore:
            logger.warning("⚠️  BM25: no vectorstore provided")
            return False

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error("❌ BM25: rank-bm25 not installed (pip install rank-bm25)")
            return False

        chunks = self._load_chunks(max_chunks)
        if not chunks:
            logger.warning("⚠️  BM25: corpus is empty — index not built")
            return False

        self.chunks           = chunks
        self.tokenized_corpus = [self._tokenize(c.get("text", "")) for c in chunks]
        non_empty             = sum(1 for t in self.tokenized_corpus if t)

        if non_empty == 0:
            logger.warning(
                "⚠️  BM25: all documents tokenized to empty — "
                "check language / stop word list"
            )
            return False

        self.index        = BM25Okapi(self.tokenized_corpus)
        self._initialized = True

        vocab            = {t for tokens in self.tokenized_corpus for t in tokens}
        self._vocab_size = len(vocab)

        logger.info(
            f"✅ BM25 index built: {len(chunks)} docs | "
            f"{non_empty} non-empty | vocab={self._vocab_size}"
        )
        return True

    def _load_chunks(self, max_chunks: int) -> List[Dict]:
        """
        Try every reasonable vectorstore API to get the full corpus.
        Returns a list of dicts with at least {'text': ..., 'chunk_id': ...}.
        """
        methods = [
            ("get_all_chunks",     lambda: self.vectorstore.get_all_chunks(limit=max_chunks)),
            ("get_chunks",         lambda: self.vectorstore.get_chunks(limit=max_chunks)),
            ("list_chunks",        lambda: self.vectorstore.list_chunks(limit=max_chunks)),
            ("get_random_samples", lambda: self.vectorstore.get_random_samples(count=max_chunks)),
            ("similarity_search",  lambda: self._load_via_search()),
        ]

        for name, fn in methods:
            attr = name.split("(")[0]
            if not hasattr(self.vectorstore, attr):
                continue
            try:
                chunks = fn()
                if chunks:
                    logger.info(f"   BM25 corpus loaded via {name}(): {len(chunks)} chunks")
                    return chunks
            except Exception as e:
                logger.debug(f"   {name}() failed: {e}")

        logger.error("❌ BM25: no corpus-loading method succeeded")
        return []

    def _load_via_search(self, sample_queries: Optional[List[str]] = None) -> List[Dict]:
        """
        Last-resort: run broad semantic searches to collect representative chunks.
        Not ideal (may miss chunks), but better than nothing.
        """
        queries = sample_queries or [
            "experience", "formation", "competence",
            "projet", "skill", "education", "work",
        ]
        seen: Dict[str, Dict] = {}
        for q in queries:
            try:
                results = self.vectorstore.semantic_search(
                    query_text=q, embedder=None, top_k=50, min_score=0.0
                )
                for r in results:
                    cid = r.get("chunk_id", "")
                    if cid and cid not in seen:
                        seen[cid] = r
            except Exception:
                pass
        return list(seen.values())

    # ── Search ──────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search the BM25 index.
        Returns chunks enriched with 'bm25_score' (0–1 normalized).
        """
        if not self._initialized or not self.index:
            logger.warning("⚠️  BM25: index not initialized")
            return []

        tokens = self._tokenize(query)
        if not tokens:
            logger.warning(f"⚠️  BM25: query '{query}' tokenized to nothing")
            return []

        scores  = self.index.get_scores(tokens)
        max_s   = max(scores) if max(scores) > 0 else 1.0
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx].copy()
            chunk["bm25_score"]     = float(scores[idx]) / max_s
            chunk["bm25_raw_score"] = float(scores[idx])
            results.append(chunk)

        if results:
            logger.info(
                f"   BM25: '{query}' → tokens={tokens} | "
                f"hits={len(results)} | top_score={results[0]['bm25_score']:.3f}"
            )
        else:
            logger.info(f"   BM25: '{query}' → tokens={tokens} | hits=0")

        return results

    # ── Diagnostics ─────────────────────────────────────────────────────────

    def debug_query(self, query: str) -> Dict[str, Any]:
        """
        Call this when BM25 returns 0 results to understand why.
        Shows tokens produced and whether they exist in the vocab.
        """
        tokens   = self._tokenize(query)
        vocab    = {t for toks in self.tokenized_corpus for t in toks}
        in_vocab = [t for t in tokens if t in vocab]
        missing  = [t for t in tokens if t not in vocab]

        info = {
            "query":            query,
            "normalized_query": self._normalize(query),
            "tokens":           tokens,
            "tokens_in_vocab":  in_vocab,
            "tokens_missing":   missing,
            "corpus_size":      len(self.chunks),
            "vocab_size":       self._vocab_size,
            "index_ready":      self._initialized,
        }

        logger.info("🔎 BM25 debug:")
        for k, v in info.items():
            logger.info(f"   {k}: {v}")

        return info


# ============================================================
# RETRIEVAL SERVICE
# ============================================================

class RetrievalService:
    def __init__(
        self,
        vectorstore,
        config: Optional[RetrievalConfig] = None
    ):
        self.vectorstore = vectorstore
        self.config      = config or RetrievalConfig()

        self.bm25_index = BM25Index(vectorstore=vectorstore)
        if self.config.use_hybrid:
            self.bm25_index.initialize_from_corpus(max_chunks=2000)

        logger.info("✅ RetrievalService initialized")
        logger.info(f"   top_k     : {self.config.top_k}")
        logger.info(f"   min_score : {self.config.min_score}")
        logger.info(f"   hybrid    : {self.config.use_hybrid}")
        logger.info(f"   RRF k     : {self.config.rrf_k}")

    def retrieve(
        self,
        query:     str,
        top_k:     Optional[int]   = None,
        min_score: Optional[float] = None,
        filters:   Optional[Dict]  = None,
    ) -> RetrievalResult:

        start_time = time.time()

        top_k     = top_k     if top_k     is not None else self.config.top_k
        min_score = min_score if min_score is not None else self.config.min_score

        logger.info(f"🔍 Retrieving: '{query[:80]}'")
        logger.info(f"   top_k={top_k} | min_score={min_score}")

        query_variants = self._generate_query_variants(query)
        logger.info(f"   variants: {query_variants}")

        # ── Primary search ─────────────────────────────────────
        if self.config.use_hybrid:
            results = self._hybrid_retrieve(
                query=query,
                query_variants=query_variants,
                top_k=top_k,
                min_score=min_score,
                filters=filters,
            )
        else:
            results = self._vector_retrieve(
                query_variants=query_variants,
                top_k=top_k,
                min_score=min_score,
                filters=filters,
            )

        logger.info(f"   Primary search: {len(results)} results")

        # ── Fallback 1: lower threshold ────────────────────────
        if not results:
            fallback_score = max(0.2, min_score * 0.7)
            logger.warning(
                f"⚠️  No results at {min_score:.2f} → "
                f"trying {fallback_score:.2f}"
            )
            results = self._vector_retrieve(
                query_variants=query_variants,
                top_k=top_k,
                min_score=fallback_score,
                filters=filters,
            )
            logger.info(f"   Fallback 1: {len(results)} results")

        # ── Fallback 2: keyword-only search ───────────────────
        if not results:
            logger.warning("⚠️  Still no results → trying keyword-only")
            results = self._keyword_only_retrieve(query, top_k)
            logger.info(f"   Fallback 2 (keywords): {len(results)} results")

        if not results:
            logger.error("❌ No results found even with fallbacks")
            return self._empty_result(query, "No results found")

        # ── Rerank ─────────────────────────────────────────────
        if self.config.enable_reranking and len(results) > top_k:
            results = self._rerank_chunks(query, results, top_k)
        else:
            results = results[:top_k]

        # ── Log score distribution ─────────────────────────────
        scores = [
            c.get("final_score", c.get("similarity", 0.0))
            for c in results
        ]
        if scores:
            logger.info(
                f"   Score range: "
                f"min={min(scores):.4f} "
                f"max={max(scores):.4f} "
                f"avg={sum(scores)/len(scores):.4f}"
            )

        # ── Convert to RetrievedChunk ──────────────────────────
        retrieved_chunks = []
        for rank, c in enumerate(results, 1):
            chunk = RetrievedChunk(
                chunk_id=c.get("chunk_id", f"c_{rank}"),
                text=c.get("text", ""),
                score=c.get("final_score", c.get("similarity", 0.0)),
                source=c.get("metadata", {}).get("filename", "unknown"),
                metadata=c.get("metadata", {}),
                rank=rank,
                vector_score=c.get("similarity", 0.0),
                bm25_score=c.get("bm25_score", 0.0),
                hybrid_score=c.get("final_score", 0.0),
            )
            retrieved_chunks.append(chunk)

            logger.info(
                f"   #{rank} score={chunk.score:.4f} "
                f"(vec={chunk.vector_score:.3f}, bm25={chunk.bm25_score:.3f}) | "
                f"src={chunk.source} | "
                f"preview={chunk.text[:60].strip()}..."
            )

        context        = self._construct_context(retrieved_chunks)
        retrieval_time = time.time() - start_time

        logger.info(
            f"✅ Retrieved {len(retrieved_chunks)} chunks "
            f"in {retrieval_time:.3f}s | "
            f"context={len(context)} chars"
        )

        return RetrievalResult(
            query=query,
            chunks=retrieved_chunks,
            context=context,
            total_found=len(results),
            retrieval_time=retrieval_time,
            metadata={
                "top_k":          top_k,
                "min_score":      min_score,
                "query_variants": query_variants,
                "hybrid":         self.config.use_hybrid,
            },
        )

    # ── Hybrid retrieve ────────────────────────────────────────

    def _hybrid_retrieve(
        self,
        query,
        query_variants,
        top_k,
        min_score,
        filters
    ) -> List[Dict]:
        """
        True hybrid search:
        - Vector search with reasonable threshold
        - BM25 search on ENTIRE corpus
        - RRF fusion
        """
        vector_results = self._vector_retrieve(
            query_variants=query_variants,
            top_k=top_k * 2,
            min_score=max(0.2, min_score * 0.8),
            filters=filters,
        )
        logger.info(f"   Vector results: {len(vector_results)}")

        bm25_results = []
        for variant in query_variants:
            results = self.bm25_index.search(variant, top_k=top_k * 2)
            for r in results:
                cid = r.get("chunk_id", "")
                if cid and not any(b.get("chunk_id") == cid for b in bm25_results):
                    bm25_results.append(r)

        # Debug if BM25 returns nothing
        if not bm25_results:
            self.bm25_index.debug_query(query)

        logger.info(f"   BM25 results: {len(bm25_results)}")

        fused = self._rrf(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=top_k * 2,
            k=self.config.rrf_k,
        )
        logger.info(f"   Fused results: {len(fused)}")
        return fused

    # ── Vector retrieve ────────────────────────────────────────

    def _vector_retrieve(
        self,
        query_variants,
        top_k,
        min_score,
        filters
    ) -> List[Dict]:
        all_chunks: Dict[str, Dict] = {}

        for variant in query_variants:
            chunks = self.vectorstore.semantic_search(
                query_text=variant,
                embedder=None,
                top_k=top_k,
                min_score=min_score,
                filters=filters,
            )
            logger.debug(f"   variant='{variant}' → {len(chunks)} chunks")

            for c in chunks:
                cid = c.get("chunk_id", "")
                if (cid not in all_chunks or
                        c.get("similarity", 0) >
                        all_chunks[cid].get("similarity", 0)):
                    all_chunks[cid] = c

        results = list(all_chunks.values())
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return results

    # ── Keyword-only fallback ──────────────────────────────────

    def _keyword_only_retrieve(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict]:
        """Pure keyword search fallback when vector search fails completely."""
        logger.info(f"   🔑 Keyword-only search: '{query}'")

        results = self.bm25_index.search(query, top_k=top_k * 2)

        for r in results:
            if "similarity" not in r:
                r["similarity"] = r.get("bm25_score", 0.0)
            r["final_score"] = r.get("bm25_score", 0.0)

        return results

    # ── RRF ────────────────────────────────────────────────────

    def _rrf(
        self,
        vector_results,
        bm25_results,
        top_k,
        k: int = 20
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion with lower k value for better score distribution.
        k=20 → scores range from 0.025 to 0.05 (more discriminative than k=60).
        """
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, Dict]  = {}

        for rank, chunk in enumerate(vector_results, 1):
            cid   = chunk.get("chunk_id", f"v{rank}")
            score = self.config.vector_weight * (1.0 / (k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score
            if cid not in chunk_data:
                chunk_data[cid] = chunk.copy()
                chunk_data[cid]["bm25_score"] = 0.0

        for rank, chunk in enumerate(bm25_results, 1):
            cid   = chunk.get("chunk_id", f"b{rank}")
            score = self.config.bm25_weight * (1.0 / (k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score
            if cid not in chunk_data:
                chunk_data[cid] = chunk.copy()
                chunk_data[cid]["similarity"] = 0.0
            chunk_data[cid]["bm25_score"] = chunk.get("bm25_score", 0.0)

        # Normalize RRF scores to 0-1
        if rrf_scores:
            max_rrf = max(rrf_scores.values())
            if max_rrf > 0:
                rrf_scores = {cid: s / max_rrf for cid, s in rrf_scores.items()}

        results = []
        for cid, rrf_score in rrf_scores.items():
            c = chunk_data[cid].copy()
            c["final_score"] = rrf_score
            results.append(c)

        results.sort(key=lambda x: x["final_score"], reverse=True)

        if results:
            scores = [r["final_score"] for r in results[:10]]
            logger.info(
                f"   RRF scores (top 10): "
                f"max={max(scores):.4f}, min={min(scores):.4f}"
            )

        return results[:top_k]

    # ── Rerank ─────────────────────────────────────────────────

    def _rerank_chunks(
        self,
        query,
        chunks,
        top_k
    ) -> List[Dict]:
        return sorted(
            chunks,
            key=lambda x: x.get("final_score", x.get("similarity", 0.0)),
            reverse=True,
        )[:top_k]

    # ── Query variants ─────────────────────────────────────────

    def _generate_query_variants(self, query: str) -> List[str]:
        """Generate comprehensive query variants for better recall."""
        variants  = [query]
        query_low = query.lower().strip()

        stop_words = {
            "le","la","les","un","une","des","du","de",
            "et","ou","en","a","au","aux","ce","se",
            "est","son","sa","ses","c'est","qu'est",
            "que","qui","quoi","dont","quel","quelle",
            "comment","pourquoi","quand","combien",
            "dans","sur","par","pour","avec","sans",
            "comme","avait","avoir","depuis","pendant",
            "the","an","is","are","of","in","on",
            "what","how","why","who","when","where",
            "and","or","but","not","with","from",
        }

        # Variant 1: keywords only (remove stop words)
        words = [
            w for w in query_low.split()
            if w not in stop_words and len(w) >= 2
        ]

        if words and len(words) < len(query_low.split()):
            kw = " ".join(words)
            if kw not in variants:
                variants.append(kw)

        # Variant 2: core nouns/verbs (remove question words too)
        question_words = {
            "qui","quoi","quel","quelle","comment","pourquoi","quand","ou",
            "what","how","why","who","when","where","which",
        }
        core_words = [w for w in words if w not in question_words]

        if core_words and len(core_words) >= 2:
            core = " ".join(core_words)
            if core not in variants:
                variants.append(core)

        return variants[:3]

    # ── Context builder ────────────────────────────────────────

    def _construct_context(
        self,
        chunks: List[RetrievedChunk]
    ) -> str:
        if not chunks:
            return ""

        parts = []
        total = 0

        for chunk in chunks:
            is_table = chunk.metadata.get("is_table", False)
            caption  = chunk.metadata.get("caption", "")

            if is_table:
                header = f"[TABLE — Source: {chunk.source}"
                if caption:
                    header += f" | {caption}"
                header += "]"
                block = f"{header}\n\n{chunk.text}"
            else:
                block = f"[Source: {chunk.source}]\n{chunk.text}\n"

            if total + len(block) > self.config.max_context_length:
                logger.warning(f"⚠️  Context limit reached at chunk #{chunk.rank}")
                break

            parts.append(block)
            total += len(block)

        return "\n\n---\n\n".join(parts)

    def _empty_result(self, query: str, reason: str) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            chunks=[],
            context="",
            total_found=0,
            retrieval_time=0.0,
            metadata={"error": reason},
        )


# ============================================================
# RAG SERVICE — LangGraph Powered
# ============================================================

class LangChainRAGService:
    """
    RAG Service powered by LangGraph.

    ask()        → runs full LangGraph
    ask_stream() → runs procedural stream
    """

    def __init__(
        self,
        vectorstore,
        config:          Optional[RAGConfig] = None,
        lm_studio_url:   str = "http://localhost:1234/v1",
        lm_studio_model: str = "local-model",
        memory_manager:  Optional[ConversationMemoryManager] = None,
    ):
        self.vectorstore     = vectorstore
        self.config          = config or RAGConfig()
        self.lm_studio_url   = lm_studio_url
        self.lm_studio_model = lm_studio_model
        self.config.model    = lm_studio_model
        self.config.base_url = lm_studio_url

        self.memory_manager = memory_manager or ConversationMemoryManager(
            memory_type="buffer_window",
            window_size=6,
        )

        self.llm = self._init_llm()

        self.retrieval_service = RetrievalService(
            vectorstore=vectorstore,
            config=RetrievalConfig(
                top_k=self.config.top_k,
                min_score=self.config.min_score,
                max_context_length=self.config.max_context_length,
                enable_reranking=self.config.enable_reranking,
                use_hybrid=self.config.use_hybrid,
                bm25_weight=self.config.bm25_weight,
                vector_weight=self.config.vector_weight,
                rrf_k=20,
            ),
        )

        # ── Build LangGraph ────────────────────────────────────
        self.graph = build_rag_graph(
            retrieval_service=self.retrieval_service,
            memory_manager=self.memory_manager,
            llm=self.llm,
        )

        logger.info("=" * 70)
        logger.info("✅ LangChainRAGService initialized (LangGraph)")
        logger.info(f"   LM Studio  : {lm_studio_url}")
        logger.info(f"   Model      : {lm_studio_model}")
        logger.info(f"   Top-K      : {self.config.top_k}")
        logger.info(f"   Min Score  : {self.config.min_score}")
        logger.info(f"   Hybrid     : {self.config.use_hybrid}")
        logger.info(f"   RRF k      : 20")
        logger.info("=" * 70)

    def _init_llm(self):
        try:
            llm = ChatOpenAI(
                base_url=self.lm_studio_url,
                api_key="not-needed",
                model=self.lm_studio_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=120,
                max_retries=2,
                streaming=False,
            )
            logger.info("✅ LLM initialized")
            return llm
        except Exception as e:
            logger.error(f"❌ LLM init: {e}")
            return None

    def is_available(self) -> bool:
        if not self.llm:
            return False
        try:
            self.llm.invoke("ping")
            return True
        except Exception:
            return False

    # ============================================================
    # ASK — LangGraph Powered
    # ============================================================

    def ask(
        self,
        question:             str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id:           str = "default",
    ) -> Dict[str, Any]:
        """Ask using full LangGraph pipeline."""

        logger.info("=" * 60)
        logger.info(f"💬 [LangGraph] Question: '{question}'")
        logger.info(f"   Session: {session_id}")
        logger.info("=" * 60)

        initial_state = {
            "question":          question,
            "session_id":        session_id,
            "history":           conversation_history or [],
            "retrieval_result":  None,
            "context":           "",
            "chunks_found":      0,
            "has_table_chunks":  False,
            "retrieval_time":    0.0,
            "history_text":      "",
            "full_prompt":       "",
            "raw_answer":        "",
            "answer":            "",
            "generation_time":   0.0,
            "sources":           [],
            "table_chunks_used": 0,
            "route":             "standard",
            "error":             None,
            "fallback_used":     False,
            "success":           False,
            "metadata":          {},
        }

        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"❌ LangGraph execution error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "answer":  "Erreur lors de l'exécution du pipeline.",
                "sources": [],
                "metadata": {"error": str(e)},
            }

        return {
            "success": final_state.get("success", False),
            "answer":  final_state.get("answer", ""),
            "sources": final_state.get("sources", []),
            "metadata": {
                **final_state.get("metadata", {}),
                "model":         self.lm_studio_model,
                "temperature":   self.config.temperature,
                "hybrid_search": self.config.use_hybrid,
                "query_variants": (
                    final_state
                    .get("retrieval_result")
                    .metadata
                    .get("query_variants", [])
                    if final_state.get("retrieval_result")
                    else []
                ),
            },
        }

    # ============================================================
    # STREAM — Procedural
    # ============================================================

    def ask_stream(
        self,
        question:             str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id:           str = "default",
    ) -> Iterator[Dict]:
        """Streaming ask."""

        retrieval_result = self.retrieval_service.retrieve(
            query=question,
            top_k=self.config.top_k,
            min_score=self.config.min_score,
        )

        if not retrieval_result.chunks:
            yield {
                "type":    "error",
                "content": "Aucune information pertinente trouvée.",
            }
            return

        yield {
            "type": "sources",
            "content": [
                {
                    "source":     c.source,
                    "relevance":  round(c.score, 3),
                    "bm25_score": round(c.bm25_score, 3),
                    "type":       "table" if c.metadata.get("is_table") else "text",
                    "caption":    c.metadata.get("caption"),
                }
                for c in retrieval_result.chunks
            ],
        }

        if conversation_history:
            self.memory_manager.load_history(session_id, conversation_history)

        history      = self.memory_manager.get_history(session_id)
        history_text = ""
        if history:
            for msg in history[-4:]:
                role          = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

        has_tables  = any(c.metadata.get("is_table") for c in retrieval_result.chunks)
        full_prompt = self._build_prompt(
            question=question,
            context=retrieval_result.context,
            history_text=history_text,
            has_tables=has_tables,
        )

        try:
            streaming_llm = ChatOpenAI(
                base_url=self.lm_studio_url,
                api_key="not-needed",
                model=self.lm_studio_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=True,
                timeout=120,
            )

            full_answer = ""
            for chunk in streaming_llm.stream(full_prompt):
                if chunk.content:
                    full_answer += chunk.content
                    yield {"type": "text", "content": chunk.content}

            self.memory_manager.add_message(
                session_id=session_id,
                user_message=question,
                assistant_message=full_answer,
            )

            yield {
                "type": "metadata",
                "content": {
                    "chunks_found":      len(retrieval_result.chunks),
                    "table_chunks_used": sum(
                        1 for c in retrieval_result.chunks
                        if c.metadata.get("is_table")
                    ),
                    "model":         self.lm_studio_model,
                    "hybrid_search": self.config.use_hybrid,
                },
            }

        except Exception as e:
            logger.error(f"❌ Stream error: {e}")
            yield {"type": "error", "content": str(e)}

    def _build_prompt(
        self,
        question:     str,
        context:      str,
        history_text: str,
        has_tables:   bool = False,
    ) -> str:
        """Build prompt — used by ask_stream only."""
        if has_tables:
            rules = (
                "You are a precise and helpful assistant.\n"
                "Answer ONLY using the information in CONTEXT below.\n\n"
                "CRITICAL RULES:\n"
                "• Reproduce markdown tables EXACTLY with | pipes |\n"
                "• Do NOT convert tables to prose\n"
                "• If answer not in context, say so clearly\n"
            )
        else:
            rules = (
                "You are a precise and helpful assistant.\n"
                "Answer ONLY using the information in CONTEXT below.\n\n"
                "CRITICAL RULES:\n"
                "• Be concise and direct\n"
                "• If answer not in context, say so clearly\n"
            )

        prompt = f"{rules}\nCONTEXT:\n{context}\n\n"
        if history_text:
            prompt += f"CONVERSATION HISTORY:\n{history_text}\n\n"
        prompt += f"QUESTION: {question}\n\nANSWER:"
        return prompt

    # ── Session helpers ────────────────────────────────────────

    def clear_session(self, session_id: str):
        self.memory_manager.clear_session(session_id)

    def get_session_history(self, session_id: str) -> List[Dict]:
        return self.memory_manager.get_history(session_id)

    def get_memory_stats(self) -> Dict:
        return self.memory_manager.get_stats()