"""
RAG Chain Pipeline - LangGraph Powered
"""

import logging
import re
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
    min_score:          float = 0.1    # ✅ lowered from 0.3
    enable_reranking:   bool  = True
    use_hybrid:         bool  = True
    bm25_weight:        float = 0.4
    vector_weight:      float = 0.6
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
        min_score:          float = 0.1,    # ✅ lowered from 0.3
        max_context_length: int   = 4000,
        enable_reranking:   bool  = True,
        diversity_weight:   float = 0.0,
        recency_weight:     float = 0.0,
        use_hybrid:         bool  = True,
        bm25_weight:        float = 0.4,
        vector_weight:      float = 0.6,
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
# BM25 INDEX
# ============================================================

class BM25Index:
    def __init__(self):
        self.index            = None
        self.chunks           = []
        self.tokenized_corpus = []

    def _tokenize(self, text: str) -> List[str]:
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
            'the', 'an', 'is', 'are', 'was', 'were',
            'of', 'in', 'on', 'at', 'to', 'for', 'with',
            'this', 'that', 'these', 'those', 'it', 'its',
            'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall',
            'and', 'or', 'but', 'not', 'so', 'yet',
            'from', 'by', 'about', 'as', 'into', 'through'
        }

        return [
            t for t in tokens
            if t not in stop_words and len(t) >= 2
        ]

    def build(self, chunks: List[Dict[str, Any]]) -> bool:
        try:
            from rank_bm25 import BM25Okapi

            if not chunks:
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

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.index or not self.chunks:
            return []

        try:
            tokenized = self._tokenize(query)
            if not tokenized:
                return []

            scores  = self.index.get_scores(tokenized)
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

            return results

        except Exception as e:
            logger.error(f"❌ BM25 search: {e}")
            return []


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
        self.bm25_index  = BM25Index()
        self._bm25_built = False

        logger.info("✅ RetrievalService initialized")
        logger.info(f"   top_k     : {self.config.top_k}")
        logger.info(f"   min_score : {self.config.min_score}")
        logger.info(f"   hybrid    : {self.config.use_hybrid}")

    def retrieve(
        self,
        query:     str,
        top_k:     Optional[int]   = None,
        min_score: Optional[float] = None,
        filters:   Optional[Dict]  = None,
    ) -> RetrievalResult:

        start_time = time.time()

        # ✅ FIXED: use 'is None' instead of 'or'
        # prevents 0.0 being treated as falsy
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
            fallback_score = min_score * 0.5
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

        # ── Fallback 2: no threshold at all ───────────────────
        if not results:
            logger.warning("⚠️  Still no results → trying min_score=0.0")
            results = self._vector_retrieve(
                query_variants=query_variants,
                top_k=top_k,
                min_score=0.0,
                filters=filters,
            )
            logger.info(f"   Fallback 2: {len(results)} results")

        if not results:
            logger.error("❌ No results found even at score=0.0")
            return self._empty_result(query, "No results found")

        # ── Rerank ─────────────────────────────────────────────
        if self.config.enable_reranking and len(results) > top_k:
            results = self._rerank_chunks(query, results, top_k)
        else:
            results = results[:top_k]

        # ── Log score distribution ─────────────────────────────
        scores = [
            c.get('final_score', c.get('similarity', 0.0))
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
                chunk_id=c.get('chunk_id', f'c_{rank}'),
                text=c.get('text', ''),
                score=c.get('final_score', c.get('similarity', 0.0)),
                source=c.get('metadata', {}).get('filename', 'unknown'),
                metadata=c.get('metadata', {}),
                rank=rank,
                vector_score=c.get('similarity', 0.0),
                bm25_score=c.get('bm25_score', 0.0),
                hybrid_score=c.get('final_score', 0.0),
            )
            retrieved_chunks.append(chunk)

            logger.info(
                f"   #{rank} score={chunk.score:.4f} | "
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
                'top_k':          top_k,
                'min_score':      min_score,
                'query_variants': query_variants,
                'hybrid':         self.config.use_hybrid,
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

        # Vector search with relaxed threshold
        vector_results = self._vector_retrieve(
            query_variants=query_variants,
            top_k=top_k * 3,
            min_score=min_score * 0.5,   # ✅ more relaxed for hybrid
            filters=filters,
        )
        logger.info(f"   Vector results: {len(vector_results)}")

        # BM25 on top of vector results
        bm25_results = []
        if vector_results:
            built = self.bm25_index.build(vector_results)
            self._bm25_built = built

        if self._bm25_built:
            all_bm25 = {}
            for variant in query_variants:
                for r in self.bm25_index.search(variant, top_k * 2):
                    cid = r.get('chunk_id', '')
                    if (cid not in all_bm25 or
                            r.get('bm25_score', 0) >
                            all_bm25[cid].get('bm25_score', 0)):
                        all_bm25[cid] = r
            bm25_results = list(all_bm25.values())
            logger.info(f"   BM25 results: {len(bm25_results)}")

        # RRF fusion
        fused = self._rrf(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=top_k * 2,
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
            logger.debug(
                f"   variant='{variant}' → {len(chunks)} chunks"
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

    # ── RRF ────────────────────────────────────────────────────

    def _rrf(
        self,
        vector_results,
        bm25_results,
        top_k,
        k: int = 60
    ) -> List[Dict]:
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, Dict]  = {}

        for rank, chunk in enumerate(vector_results, 1):
            cid   = chunk.get('chunk_id', f'v{rank}')
            score = self.config.vector_weight * (1.0 / (k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score
            if cid not in chunk_data:
                chunk_data[cid] = chunk.copy()
                chunk_data[cid]['bm25_score'] = 0.0

        for rank, chunk in enumerate(bm25_results, 1):
            cid   = chunk.get('chunk_id', f'b{rank}')
            score = self.config.bm25_weight * (1.0 / (k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score
            if cid not in chunk_data:
                chunk_data[cid] = chunk.copy()
                chunk_data[cid]['similarity'] = 0.0
            chunk_data[cid]['bm25_score'] = chunk.get('bm25_score', 0.0)

        results = []
        for cid, rrf_score in rrf_scores.items():
            c = chunk_data[cid].copy()
            c['final_score'] = rrf_score
            results.append(c)

        results.sort(key=lambda x: x['final_score'], reverse=True)
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
            key=lambda x: x.get('final_score', x.get('similarity', 0.0)),
            reverse=True,
        )[:top_k]

    # ── Query variants ─────────────────────────────────────────

    def _generate_query_variants(self, query: str) -> List[str]:
        variants  = [query]
        query_low = query.lower().strip()

        stop_words = {
            'le','la','les','un','une','des','du','de',
            'et','ou','en','à','au','aux','ce','se',
            'est','son','sa','ses',"c'est","qu'est",
            'que','qui','quoi','dont','quel','quelle',
            'comment','pourquoi','quand','combien',
            'dans','sur','par','pour','avec','sans',
            'the','an','is','are','of','in','on',
            'what','how','why','who','when','where',
            'and','or','but','not','with','from',
            'give','show','tell','explain',
        }

        words = [
            w for w in query_low.split()
            if w not in stop_words and len(w) >= 2
        ]

        if words and len(words) < len(query_low.split()):
            kw = " ".join(words)
            if kw not in variants:
                variants.append(kw)

        return variants[:2]

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
                logger.warning(
                    f"⚠️  Context limit reached at chunk #{chunk.rank}"
                )
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
            metadata={'error': reason},
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
                role          = "User" if msg['role'] == 'user' else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

        has_tables  = any(
            c.metadata.get("is_table") for c in retrieval_result.chunks
        )
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