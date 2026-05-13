"""
RAG Chain Pipeline
LangChain-powered RAG with BM25 + Vector Hybrid Search
"""

import logging
import re
from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass
import time

from langchain_openai import ChatOpenAI

from .memory import ConversationMemoryManager

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class RAGConfig:
    """RAG Configuration"""

    # Retrieval
    top_k: int = 5
    min_score: float = 0.3
    enable_reranking: bool = True

    # Hybrid Search
    use_hybrid: bool = True
    bm25_weight: float = 0.4
    vector_weight: float = 0.6

    # Generation
    temperature: float = 0.1
    max_tokens: int = 1000

    # LM Studio
    model: str = "local-model"
    base_url: str = "http://localhost:1234/v1"

    # Prompt
    include_sources: bool = True
    language: str = "français"

    # Context
    max_context_length: int = 4000


# ============================================================
# RETRIEVAL CONFIG
# ============================================================

class RetrievalConfig:
    """Retrieval Configuration"""

    def __init__(
        self,
        top_k: int = 5,
        min_score: float = 0.3,
        max_context_length: int = 4000,
        enable_reranking: bool = True,
        diversity_weight: float = 0.0,   # ← FIXED: was 0.1, kills RRF order
        recency_weight: float = 0.0,     # ← FIXED: was 0.05, irrelevant noise
        use_hybrid: bool = True,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6
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


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class RetrievedChunk:
    """Single retrieved chunk"""
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
    """Complete retrieval result"""
    query:          str
    chunks:         List[RetrievedChunk]
    context:        str
    total_found:    int
    retrieval_time: float
    metadata:       Dict[str, Any]


# ============================================================
# BM25 INDEX
# ============================================================

class BM25Index:
    """BM25 index built from retrieved chunks"""

    def __init__(self):
        self.index            = None
        self.chunks           = []
        self.tokenized_corpus = []

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        text   = text.lower()
        text   = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()

        stop_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de',
            'et', 'ou', 'en', 'à', 'au', 'aux', 'ce', 'se',
            'est', 'son', 'sa', 'ses', 'mon', 'ma', 'mes',
            'que', 'qui', 'quoi', 'dont', 'où', 'je', 'tu',
            'il', 'elle', 'nous', 'vous', 'ils', 'elles',
            'sur', 'dans', 'par', 'pour', 'avec', 'sans',
            'plus', 'mais', 'donc', 'car', 'ni', 'or',
            # English - removed 'a' to keep it as standalone term
            'the', 'an', 'is', 'are', 'was', 'were',
            'of', 'in', 'on', 'at', 'to', 'for', 'with',
            'this', 'that', 'these', 'those', 'it', 'its',
            'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall',
            'and', 'or', 'but', 'not', 'so', 'yet',
            'from', 'by', 'about', 'as', 'into', 'through'
        }

        # ← FIXED: was len(t) > 2 (excluded 'rag', 'ai', etc.)
        #          now len(t) >= 2 keeps short technical terms
        return [
            t for t in tokens
            if t not in stop_words and len(t) >= 2
        ]

    def build(self, chunks: List[Dict[str, Any]]) -> bool:
        """Build BM25 index"""
        try:
            from rank_bm25 import BM25Okapi

            if not chunks:
                logger.warning("⚠️  No chunks for BM25")
                return False

            self.chunks           = chunks
            self.tokenized_corpus = [
                self._tokenize(c.get('text', ''))
                for c in chunks
            ]
            self.index = BM25Okapi(self.tokenized_corpus)

            logger.info(f"✅ BM25 index: {len(chunks)} docs")
            return True

        except ImportError:
            logger.error("❌ rank-bm25 not installed!")
            return False
        except Exception as e:
            logger.error(f"❌ BM25 build: {e}")
            return False

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search BM25"""
        if not self.index or not self.chunks:
            return []

        try:
            tokenized = self._tokenize(query)
            if not tokenized:
                return []

            logger.info(f"🔑 BM25 tokens: {tokenized}")
            scores = self.index.get_scores(tokenized)

            top_idx = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

            max_score = max(scores) if max(scores) > 0 else 1.0
            results   = []

            for idx in top_idx:
                if scores[idx] <= 0:
                    continue
                chunk = self.chunks[idx].copy()
                chunk['bm25_score']     = scores[idx] / max_score
                chunk['bm25_raw_score'] = float(scores[idx])
                results.append(chunk)

            logger.info(f"✅ BM25: {len(results)} results")
            if results:
                logger.info(
                    f"   Top: bm25={results[0]['bm25_score']:.3f} | "
                    f"{results[0].get('text', '')[:80]}..."
                )
            return results

        except Exception as e:
            logger.error(f"❌ BM25 search: {e}")
            return []


# ============================================================
# RETRIEVAL SERVICE
# ============================================================

class RetrievalService:
    """Retrieval with BM25 + Vector + RRF"""

    def __init__(
        self,
        vectorstore,
        config: Optional[RetrievalConfig] = None
    ):
        self.vectorstore = vectorstore
        self.config      = config or RetrievalConfig()
        self.bm25_index  = BM25Index()
        self._bm25_built = False

        logger.info("✅ RetrievalService initialized")
        logger.info(f"   Top-K      : {self.config.top_k}")
        logger.info(f"   Min Score  : {self.config.min_score}")
        logger.info(f"   Hybrid     : {self.config.use_hybrid}")
        logger.info(f"   BM25 weight: {self.config.bm25_weight}")
        logger.info(f"   Vec weight : {self.config.vector_weight}")

    # ============================================================
    # MAIN RETRIEVE
    # ============================================================

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filters: Optional[Dict] = None
    ) -> RetrievalResult:
        """Main retrieval"""

        start_time = time.time()
        top_k      = top_k     or self.config.top_k
        min_score  = min_score or self.config.min_score

        logger.info(f"🔍 Retrieving: '{query[:80]}'")
        logger.info(
            f"   Mode: "
            f"{'Hybrid BM25+Vector' if self.config.use_hybrid else 'Vector'}"
        )

        query_variants = self._generate_query_variants(query)
        logger.info(f"📝 Variants: {query_variants}")

        # ── Search ─────────────────────────────────────────────
        if self.config.use_hybrid:
            results = self._hybrid_retrieve(
                query=query,
                query_variants=query_variants,
                top_k=top_k,
                min_score=min_score,
                filters=filters
            )
        else:
            results = self._vector_retrieve(
                query_variants=query_variants,
                top_k=top_k,
                min_score=min_score,
                filters=filters
            )

        # ── Fallback ───────────────────────────────────────────
        if not results:
            logger.warning("⚠️  No results, lowering threshold...")
            results = self._vector_retrieve(
                query_variants=query_variants,
                top_k=top_k,
                min_score=0.1,
                filters=filters
            )

        if not results:
            return self._empty_result(query, "No results found")

        # ── Re-rank ────────────────────────────────────────────
        if self.config.enable_reranking and len(results) > top_k:
            results = self._rerank_chunks(query, results, top_k)
        else:
            results = results[:top_k]

        # ── Convert ────────────────────────────────────────────
        retrieved_chunks = []
        for rank, c in enumerate(results, 1):
            retrieved_chunks.append(RetrievedChunk(
                chunk_id=c.get('chunk_id', f'c_{rank}'),
                text=c.get('text', ''),
                score=c.get('final_score', c.get('similarity', 0.0)),
                source=c.get('metadata', {}).get('filename', 'unknown'),
                metadata=c.get('metadata', {}),
                rank=rank,
                vector_score=c.get('similarity', 0.0),
                bm25_score=c.get('bm25_score', 0.0),
                hybrid_score=c.get('final_score', 0.0)
            ))

        # ── Log ────────────────────────────────────────────────
        logger.info("📋 Final chunks:")
        for i, chunk in enumerate(retrieved_chunks, 1):
            logger.info(
                f"   #{i} final={chunk.score:.4f} | "
                f"vec={chunk.vector_score:.3f} | "
                f"bm25={chunk.bm25_score:.3f} | "
                f"src={chunk.source}"
            )
            logger.info(f"       {chunk.text[:120].strip()}...")

        context        = self._construct_context(retrieved_chunks)
        retrieval_time = time.time() - start_time

        logger.info(
            f"✅ Done in {retrieval_time:.3f}s | "
            f"{len(retrieved_chunks)} chunks | "
            f"{len(context)} chars"
        )

        return RetrievalResult(
            query=query,
            chunks=retrieved_chunks,
            context=context,
            total_found=len(results),
            retrieval_time=retrieval_time,
            metadata={
                'top_k': top_k,
                'min_score': min_score,
                'query_variants': query_variants,
                'hybrid': self.config.use_hybrid
            }
        )

    # ============================================================
    # HYBRID
    # ============================================================

    def _hybrid_retrieve(
        self,
        query: str,
        query_variants: List[str],
        top_k: int,
        min_score: float,
        filters: Optional[Dict]
    ) -> List[Dict]:
        """BM25 + Vector + RRF"""

        # Vector
        logger.info("🔢 Vector search...")
        vector_results = self._vector_retrieve(
            query_variants=query_variants,
            top_k=top_k * 4,
            min_score=min_score * 0.6,
            filters=filters
        )
        logger.info(f"   Vector: {len(vector_results)}")

        # BM25
        bm25_results = []
        if vector_results:
            logger.info("📚 Building BM25...")
            built = self.bm25_index.build(vector_results)
            self._bm25_built = built

        if self._bm25_built:
            logger.info("🔑 BM25 search...")
            all_bm25 = {}
            for variant in query_variants:
                for r in self.bm25_index.search(variant, top_k * 2):
                    cid = r.get('chunk_id', '')
                    if (cid not in all_bm25 or
                            r.get('bm25_score', 0) >
                            all_bm25[cid].get('bm25_score', 0)):
                        all_bm25[cid] = r
            bm25_results = list(all_bm25.values())
            logger.info(f"   BM25: {len(bm25_results)}")

        # RRF
        logger.info("🔀 RRF fusion...")
        fused = self._rrf(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=top_k * 2
        )
        logger.info(f"   Fused: {len(fused)}")
        return fused

    def _vector_retrieve(
        self,
        query_variants: List[str],
        top_k: int,
        min_score: float,
        filters: Optional[Dict]
    ) -> List[Dict]:
        """Vector search"""
        all_chunks = {}
        for variant in query_variants:
            chunks = self.vectorstore.semantic_search(
                query_text=variant,
                embedder=None,
                top_k=top_k,
                min_score=min_score,
                filters=filters
            )
            for c in chunks:
                cid = c.get('chunk_id', '')
                if (cid not in all_chunks or
                        c.get('similarity', 0) >
                        all_chunks[cid].get('similarity', 0)):
                    all_chunks[cid] = c

        results = list(all_chunks.values())
        results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return results

    # ============================================================
    # RRF
    # ============================================================

    def _rrf(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        top_k: int,
        k: int = 60
    ) -> List[Dict]:
        """Reciprocal Rank Fusion"""

        rrf_scores = {}
        chunk_data  = {}

        # Vector contribution
        for rank, chunk in enumerate(vector_results, 1):
            cid   = chunk.get('chunk_id', f'v{rank}')
            score = self.config.vector_weight * (1.0 / (k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0) + score
            if cid not in chunk_data:
                chunk_data[cid] = chunk.copy()
                chunk_data[cid]['bm25_score'] = 0.0

        # BM25 contribution
        for rank, chunk in enumerate(bm25_results, 1):
            cid   = chunk.get('chunk_id', f'b{rank}')
            score = self.config.bm25_weight * (1.0 / (k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0) + score
            if cid not in chunk_data:
                chunk_data[cid] = chunk.copy()
                chunk_data[cid]['similarity'] = 0.0
            chunk_data[cid]['bm25_score'] = chunk.get('bm25_score', 0.0)

        # Build results
        results = []
        for cid, rrf_score in rrf_scores.items():
            c = chunk_data[cid].copy()
            c['final_score'] = rrf_score
            results.append(c)

        results.sort(key=lambda x: x['final_score'], reverse=True)

        # Log top 5
        logger.info("🔀 RRF top results:")
        for i, r in enumerate(results[:5], 1):
            logger.info(
                f"   #{i} rrf={r['final_score']:.4f} | "
                f"vec={r.get('similarity', 0):.3f} | "
                f"bm25={r.get('bm25_score', 0):.3f} | "
                f"src={r.get('metadata', {}).get('filename', '?')} | "
                f"text={r.get('text', '')[:60]}..."
            )

        return results[:top_k]

    # ============================================================
    # RE-RANK - FIXED
    # ============================================================

    def _rerank_chunks(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Re-rank: ONLY by RRF/final score.
        Diversity and recency removed — they were demoting relevant chunks.
        """
        # Sort purely by the score RRF already computed
        chunks_sorted = sorted(
            chunks,
            key=lambda x: x.get('final_score', x.get('similarity', 0.0)),
            reverse=True
        )

        logger.info(f"🔄 Re-ranked {len(chunks_sorted)} → top {top_k}")

        # Log to verify correct order is preserved
        for i, c in enumerate(chunks_sorted[:top_k], 1):
            logger.info(
                f"   #{i} final={c.get('final_score', 0):.4f} | "
                f"vec={c.get('similarity', 0):.3f} | "
                f"bm25={c.get('bm25_score', 0):.3f} | "
                f"src={c.get('metadata', {}).get('filename', '?')} | "
                f"text={c.get('text', '')[:60]}..."
            )

        return chunks_sorted[:top_k]

    # ============================================================
    # QUERY VARIANTS - FIXED
    # ============================================================

    def _generate_query_variants(self, query: str) -> List[str]:
        """
        Generate query variants.
        Keeps technical short terms (rag, ai, llm, etc.)
        Adds individual term variants for better BM25 recall.
        """
        variants  = [query]
        query_low = query.lower().strip()

        # Technical terms that must NEVER be filtered out
        technical_terms = {
            'rag', 'llm', 'ai', 'ml', 'nlp', 'api',
            'bm25', 'rrf', 'gpu', 'cpu', 'sql', 'cv',
            'ocr', 'pdf', 'ui', 'ux', 'db', 'qa'
        }

        stop_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de',
            'et', 'ou', 'en', 'à', 'au', 'aux', 'ce', 'se',
            'est', 'son', 'sa', 'ses', "c'est", "qu'est",
            'que', 'qui', 'quoi', 'dont', 'quel', 'quelle',
            'comment', 'pourquoi', 'quand', 'combien',
            'dans', 'sur', 'par', 'pour', 'avec', 'sans',
            # English — removed 'a' to avoid filtering 'a' in acronyms
            'the', 'an', 'is', 'are', 'of', 'in', 'on',
            'what', 'how', 'why', 'who', 'when', 'where',
            'and', 'or', 'but', 'not', 'with', 'from',
        }

        words    = query_low.split()
        keywords = []

        for w in words:
            # Always keep technical terms regardless of length
            if w in technical_terms:
                keywords.append(w)
            # Keep if not a stop word and length >= 2 (was > 2)
            elif w not in stop_words and len(w) >= 2:
                keywords.append(w)

        # Add combined keyword variant
        if keywords:
            kw = " ".join(keywords)
            if kw.lower() not in [v.lower() for v in variants]:
                variants.append(kw)
                logger.info(f"🔑 Keyword variant: '{kw}'")

        # Add individual term variants for better BM25 single-term recall
        if len(keywords) > 1:
            for kw in keywords:
                if kw not in [v.lower() for v in variants]:
                    variants.append(kw)
                    logger.info(f"🔑 Single term variant: '{kw}'")

        return variants

    # ============================================================
    # CONTEXT
    # ============================================================

    def _construct_context(
        self,
        chunks: List[RetrievedChunk]
    ) -> str:
        """Build context string"""
        if not chunks:
            return ""

        parts = []
        total = 0

        for chunk in chunks:
            text = f"[Source: {chunk.source}]\n{chunk.text}\n"
            if total + len(text) > self.config.max_context_length:
                logger.warning(f"⚠️  Context limit at chunk #{chunk.rank}")
                break
            parts.append(text)
            total += len(text)

        return "\n---\n".join(parts)

    def _empty_result(self, query: str, reason: str) -> RetrievalResult:
        return RetrievalResult(
            query=query, chunks=[], context="",
            total_found=0, retrieval_time=0.0,
            metadata={'error': reason}
        )


# ============================================================
# RAG SERVICE
# ============================================================

class LangChainRAGService:
    """Complete RAG with Hybrid Search"""

    def __init__(
        self,
        vectorstore,
        config: Optional[RAGConfig] = None,
        lm_studio_url: str = "http://localhost:1234/v1",
        lm_studio_model: str = "local-model",
        memory_manager: Optional[ConversationMemoryManager] = None
    ):
        self.vectorstore     = vectorstore
        self.config          = config or RAGConfig()
        self.lm_studio_url   = lm_studio_url
        self.lm_studio_model = lm_studio_model
        self.config.model    = lm_studio_model
        self.config.base_url = lm_studio_url

        self.memory_manager = memory_manager or ConversationMemoryManager(
            memory_type="buffer_window",
            window_size=6
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
                diversity_weight=0.0,   # ← FIXED: was 0.1
                recency_weight=0.0      # ← FIXED: was 0.05
            )
        )

        logger.info("=" * 70)
        logger.info("✅ LangChainRAGService initialized")
        logger.info(f"   LM Studio  : {lm_studio_url}")
        logger.info(f"   Model      : {lm_studio_model}")
        logger.info(f"   Temperature: {self.config.temperature}")
        logger.info(f"   Max tokens : {self.config.max_tokens}")
        logger.info(f"   Top-K      : {self.config.top_k}")
        logger.info(f"   Min Score  : {self.config.min_score}")
        logger.info(f"   Hybrid     : {self.config.use_hybrid}")
        logger.info(f"   BM25       : {self.config.bm25_weight}")
        logger.info(f"   Vector     : {self.config.vector_weight}")
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
                streaming=False
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
    # ASK
    # ============================================================

    def ask(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """Ask with hybrid search"""

        logger.info("=" * 60)
        logger.info(f"💬 Question : '{question}'")
        logger.info(f"   Session  : {session_id}")
        logger.info("=" * 60)

        # ── Step 1: Retrieve ───────────────────────────────────
        logger.info("📚 Step 1: Retrieving...")

        retrieval_result = self.retrieval_service.retrieve(
            query=question,
            top_k=self.config.top_k,
            min_score=self.config.min_score
        )

        if not retrieval_result.chunks:
            return {
                "success": False,
                "answer": (
                    "Désolé, je n'ai trouvé aucune information "
                    "pertinente dans les documents."
                ),
                "sources": [],
                "metadata": {
                    "retrieval_time": retrieval_result.retrieval_time,
                    "chunks_found": 0,
                    "session_id": session_id
                }
            }

        # ── Step 2: History ────────────────────────────────────
        if conversation_history:
            self.memory_manager.load_history(
                session_id, conversation_history
            )

        history      = self.memory_manager.get_history(session_id)
        history_text = ""
        if history:
            for msg in history[-4:]:
                role = "User" if msg['role'] == 'user' else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

        # ── Step 3: Prompt ─────────────────────────────────────
        full_prompt = (
            "You are a helpful assistant. "
            "Answer the question using ONLY the context below. "
            "Be direct and specific. "
            "If the answer is not in the context, say so.\n\n"
            f"CONTEXT:\n{retrieval_result.context}\n\n"
        )
        if history_text:
            full_prompt += f"CONVERSATION HISTORY:\n{history_text}\n"
        full_prompt += f"QUESTION: {question}\n\nANSWER:"

        logger.info("=" * 70)
        logger.info("📤 PROMPT:")
        logger.info("=" * 70)
        logger.info(full_prompt)
        logger.info("=" * 70)
        logger.info(f"📏 {len(full_prompt)} chars")

        # ── Step 4: Generate ───────────────────────────────────
        try:
            t0         = time.time()
            raw_answer = self.llm.invoke(full_prompt).content
            gen_time   = time.time() - t0

            logger.info(f"⏱️  {gen_time:.2f}s")
            logger.info(f"🤖 Raw: '{raw_answer}'")

            if not raw_answer or len(raw_answer.strip()) < 3:
                return {
                    "success": False,
                    "answer": "Le modèle n'a pas généré de réponse.",
                    "sources": [],
                    "metadata": {"error": "Empty response"}
                }

            answer = self._post_process(raw_answer)
            logger.info(f"✅ Answer: '{answer}'")

            self.memory_manager.add_message(
                session_id=session_id,
                user_message=question,
                assistant_message=answer
            )

        except Exception as e:
            logger.error(f"❌ Generation: {e}")
            return {
                "success": False,
                "answer": "Erreur lors de la génération.",
                "sources": [],
                "metadata": {"error": str(e)}
            }

        # ── Step 5: Response ───────────────────────────────────
        return {
            "success": True,
            "answer":  answer,
            "sources": [
                {
                    "source":       c.source,
                    "relevance":    round(c.score, 3),
                    "vector_score": round(c.vector_score, 3),
                    "bm25_score":   round(c.bm25_score, 3),
                    "text_preview": (
                        c.text[:200] + "..."
                        if len(c.text) > 200 else c.text
                    )
                }
                for c in retrieval_result.chunks
            ],
            "metadata": {
                "retrieval_time": round(retrieval_result.retrieval_time, 3),
                "chunks_found":   len(retrieval_result.chunks),
                "model":          self.lm_studio_model,
                "temperature":    self.config.temperature,
                "session_id":     session_id,
                "hybrid_search":  self.config.use_hybrid,
                "conversation_length": (
                    self.memory_manager.get_session_message_count(session_id)
                )
            }
        }

    # ============================================================
    # STREAM
    # ============================================================

    def ask_stream(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: str = "default"
    ) -> Iterator[Dict]:
        """Streaming ask"""

        retrieval_result = self.retrieval_service.retrieve(
            query=question,
            top_k=self.config.top_k,
            min_score=self.config.min_score
        )

        if not retrieval_result.chunks:
            yield {"type": "error", "content": "Aucune information trouvée."}
            return

        yield {
            "type": "sources",
            "content": [
                {
                    "source":     c.source,
                    "relevance":  round(c.score, 3),
                    "bm25_score": round(c.bm25_score, 3)
                }
                for c in retrieval_result.chunks
            ]
        }

        if conversation_history:
            self.memory_manager.load_history(session_id, conversation_history)

        history      = self.memory_manager.get_history(session_id)
        history_text = ""
        if history:
            for msg in history[-4:]:
                role = "User" if msg['role'] == 'user' else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

        full_prompt = (
            "You are a helpful assistant. "
            "Answer ONLY from the context below.\n\n"
            f"CONTEXT:\n{retrieval_result.context}\n\n"
        )
        if history_text:
            full_prompt += f"HISTORY:\n{history_text}\n"
        full_prompt += f"QUESTION: {question}\n\nANSWER:"

        try:
            streaming_llm = ChatOpenAI(
                base_url=self.lm_studio_url,
                api_key="not-needed",
                model=self.lm_studio_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=True,
                timeout=120
            )

            full_answer = ""
            for chunk in streaming_llm.stream(full_prompt):
                if chunk.content:
                    full_answer += chunk.content
                    yield {"type": "text", "content": chunk.content}

            self.memory_manager.add_message(
                session_id=session_id,
                user_message=question,
                assistant_message=full_answer
            )

            yield {
                "type": "metadata",
                "content": {
                    "chunks_found":  len(retrieval_result.chunks),
                    "model":         self.lm_studio_model,
                    "hybrid_search": self.config.use_hybrid
                }
            }

        except Exception as e:
            logger.error(f"❌ Stream: {e}")
            yield {"type": "error", "content": str(e)}

    # ============================================================
    # POST PROCESS
    # ============================================================

    def _post_process(self, answer: str) -> str:
        """Clean output"""
        if not answer:
            return "Information non trouvée"

        answer = answer.strip()

        for prefix in [
            "Based on the context,",
            "Based on the provided context,",
            "According to the context,",
            "D'après le contexte,",
            "Selon le contexte,",
            "ANSWER:", "RÉPONSE:",
        ]:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
                if answer:
                    answer = answer[0].upper() + answer[1:]

        return answer.strip()

    # ── backward compat alias ──────────────────────────────────
    def _post_process_answer(self, answer: str, question: str) -> str:
        return self._post_process(answer)

    # ============================================================
    # SESSION
    # ============================================================

    def clear_session(self, session_id: str):
        self.memory_manager.clear_session(session_id)

    def get_session_history(self, session_id: str) -> List[Dict]:
        return self.memory_manager.get_history(session_id)

    def get_memory_stats(self) -> Dict:
        return self.memory_manager.get_stats()
