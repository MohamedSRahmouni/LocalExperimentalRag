"""
RAG Graph Nodes — with Prometheus metrics
"""

import logging
import time
from typing import Dict, Any, List

from .graph_state import RAGState
from app.core.metrics import (
    rag_node_duration_seconds,
    retrieval_duration_seconds,
    retrieval_chunks_found,
    retrieval_score_max,
    retrieval_fallback_total,
    retrieval_empty_total,
    retrieval_table_chunks_total,
    llm_requests_total,
    llm_duration_seconds,
    llm_answer_length,
    llm_prompt_length,
    llm_errors_total,
    memory_active_sessions,
    memory_total_messages,
)

logger = logging.getLogger(__name__)


# ============================================================
# NODE 1: MEMORY NODE
# ============================================================

def memory_node(state: RAGState, memory_manager) -> Dict[str, Any]:
    """Load conversation history into state."""
    start = time.time()
    logger.info(f"🧠 [Memory Node] session='{state['session_id']}'")

    session_id = state["session_id"]
    history    = state.get("history", [])

    if history:
        memory_manager.load_history(session_id, history)

    stored_history = memory_manager.get_history(session_id)
    history_text   = ""

    if stored_history:
        for msg in stored_history[-4:]:
            role          = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    # ── Metrics ───────────────────────────────────────────────
    memory_active_sessions.set(memory_manager.get_session_count())
    rag_node_duration_seconds.labels(node_name="memory").observe(
        time.time() - start
    )

    logger.info(
        f"   History: {len(stored_history)} messages → "
        f"{len(history_text)} chars"
    )

    return {"history_text": history_text}


# ============================================================
# NODE 2: RETRIEVAL NODE
# ============================================================

def retrieval_node(
    state: RAGState,
    retrieval_service
) -> Dict[str, Any]:
    """Retrieve relevant chunks from vector store."""
    start = time.time()
    logger.info(f"📚 [Retrieval Node] query='{state['question'][:80]}'")

    retrieval_result = retrieval_service.retrieve(
        query=state["question"]
    )

    chunks_found     = len(retrieval_result.chunks)
    has_table_chunks = any(
        c.metadata.get("is_table", False)
        for c in retrieval_result.chunks
    )

    # Routing decision
    if chunks_found == 0:
        route = "fallback"
        logger.warning("⚠️  No chunks found → routing to fallback")
    elif has_table_chunks:
        route = "table_aware"
        logger.info(f"📊 Table chunks detected → route=table_aware")
    else:
        route = "standard"
        logger.info(f"✅ {chunks_found} chunks → route=standard")

    # ── Metrics ───────────────────────────────────────────────
    mode = "hybrid" if retrieval_service.config.use_hybrid else "vector"

    retrieval_duration_seconds.labels(mode=mode).observe(
        retrieval_result.retrieval_time
    )
    retrieval_chunks_found.labels(mode=mode).observe(chunks_found)

    if retrieval_result.chunks:
        max_score = max(c.score for c in retrieval_result.chunks)
        retrieval_score_max.labels(mode=mode).observe(max_score)

    if chunks_found == 0:
        retrieval_empty_total.inc()

    if has_table_chunks:
        table_count = sum(
            1 for c in retrieval_result.chunks
            if c.metadata.get("is_table", False)
        )
        retrieval_table_chunks_total.inc(table_count)

    rag_node_duration_seconds.labels(node_name="retrieval").observe(
        time.time() - start
    )

    return {
        "retrieval_result": retrieval_result,
        "context":          retrieval_result.context,
        "chunks_found":     chunks_found,
        "has_table_chunks": has_table_chunks,
        "retrieval_time":   retrieval_result.retrieval_time,
        "route":            route,
    }


# ============================================================
# NODE 3: FALLBACK NODE
# ============================================================

def fallback_node(state: RAGState) -> Dict[str, Any]:
    """Handle case where retrieval found nothing."""
    start = time.time()
    logger.warning("⚠️  [Fallback Node] No relevant information found")

    rag_node_duration_seconds.labels(node_name="fallback").observe(
        time.time() - start
    )

    return {
        "answer":        (
            "Désolé, je n'ai trouvé aucune information pertinente "
            "dans les documents pour répondre à votre question."
        ),
        "sources":       [],
        "success":       False,
        "fallback_used": True,
        "route":         "end",
    }


# ============================================================
# NODE 4: PROMPT NODE
# ============================================================

def prompt_node(state: RAGState) -> Dict[str, Any]:
    """Build the full LLM prompt."""
    start = time.time()
    logger.info(
        f"📝 [Prompt Node] "
        f"context={len(state['context'])} chars | "
        f"tables={state.get('has_table_chunks', False)}"
    )

    question     = state["question"]
    context      = state["context"]
    history_text = state.get("history_text", "")
    has_tables   = state.get("has_table_chunks", False)

    if has_tables:
        system_rules = (
            "You are a precise and helpful assistant.\n"
            "Answer ONLY using the information in CONTEXT below.\n\n"
            "CRITICAL RULES:\n"
            "• The context contains markdown tables — reproduce them EXACTLY\n"
            "• Do NOT convert tables to prose or bullet points\n"
            "• Keep all | pipes |, headers, and separator rows intact\n"
            "• Place the table in your answer as-is\n"
            "• Only add brief explanation before/after the table if needed\n"
            "• If answer not in context, say so clearly\n"
        )
    else:
        system_rules = (
            "You are a precise and helpful assistant.\n"
            "Answer ONLY using the information in CONTEXT below.\n\n"
            "CRITICAL RULES:\n"
            "• Be concise and direct\n"
            "• Do not hallucinate information\n"
            "• If answer not in context, say so clearly\n"
        )

    prompt  = f"{system_rules}\n"
    prompt += f"CONTEXT:\n{context}\n\n"

    if history_text:
        prompt += f"CONVERSATION HISTORY:\n{history_text}\n\n"

    prompt += f"QUESTION: {question}\n\nANSWER:"

    # ── Metrics ───────────────────────────────────────────────
    llm_prompt_length.observe(len(prompt))
    rag_node_duration_seconds.labels(node_name="prompt").observe(
        time.time() - start
    )

    logger.info(f"   Prompt length: {len(prompt)} chars")

    return {"full_prompt": prompt}


# ============================================================
# NODE 5: GENERATION NODE
# ============================================================

def generation_node(state: RAGState, llm) -> Dict[str, Any]:
    """Generate answer using LLM."""
    start = time.time()
    logger.info("🤖 [Generation Node] Calling LLM...")

    try:
        raw_answer = llm.invoke(state["full_prompt"]).content
        gen_time   = time.time() - start

        logger.info(
            f"   Generated: {len(raw_answer)} chars "
            f"in {gen_time:.2f}s"
        )

        if not raw_answer or len(raw_answer.strip()) < 3:
            # ── Metrics: empty ────────────────────────────────
            llm_requests_total.labels(status="empty").inc()
            llm_duration_seconds.observe(gen_time)
            rag_node_duration_seconds.labels(node_name="generation").observe(
                time.time() - start
            )
            return {
                "raw_answer":      "",
                "generation_time": gen_time,
                "error":           "LLM returned empty response",
                "route":           "error",
            }

        # ── Metrics: success ──────────────────────────────────
        llm_requests_total.labels(status="success").inc()
        llm_duration_seconds.observe(gen_time)
        llm_answer_length.observe(len(raw_answer))
        rag_node_duration_seconds.labels(node_name="generation").observe(
            time.time() - start
        )

        return {
            "raw_answer":      raw_answer,
            "generation_time": gen_time,
            "error":           None,
        }

    except Exception as e:
        gen_time = time.time() - start
        error_type = type(e).__name__

        logger.error(f"❌ [Generation Node] Error: {e}")

        # ── Metrics: error ────────────────────────────────────
        llm_requests_total.labels(status="error").inc()
        llm_errors_total.labels(error_type=error_type).inc()
        llm_duration_seconds.observe(gen_time)
        rag_node_duration_seconds.labels(node_name="generation").observe(
            time.time() - start
        )

        return {
            "raw_answer":      "",
            "generation_time": gen_time,
            "error":           str(e),
            "route":           "error",
        }


# ============================================================
# NODE 6: POST-PROCESS NODE
# ============================================================

def postprocess_node(state: RAGState) -> Dict[str, Any]:
    """Clean and post-process LLM output."""
    start = time.time()
    logger.info("✨ [PostProcess Node]")

    error = state.get("error")

    if error or not state.get("raw_answer", "").strip():
        rag_node_duration_seconds.labels(node_name="postprocess").observe(
            time.time() - start
        )
        return {
            "answer":  "Erreur lors de la génération de la réponse.",
            "success": False,
        }

    answer = state["raw_answer"].strip()

    prefixes = [
        "Based on the context,",
        "Based on the provided context,",
        "According to the context,",
        "D'après le contexte,",
        "Selon le contexte,",
        "Voici le tableau",
        "Le tableau suivant",
        "ANSWER:", "RÉPONSE:",
    ]

    for prefix in prefixes:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
            if answer and not answer[0].isupper():
                answer = answer[0].upper() + answer[1:]

    rag_node_duration_seconds.labels(node_name="postprocess").observe(
        time.time() - start
    )
    logger.info(f"   Final answer: {len(answer)} chars")

    return {
        "answer":  answer,
        "success": True,
    }


# ============================================================
# NODE 7: BUILD SOURCES NODE
# ============================================================

def sources_node(state: RAGState) -> Dict[str, Any]:
    """Build structured sources list."""
    start = time.time()
    logger.info("📋 [Sources Node]")

    retrieval_result = state.get("retrieval_result")

    if not retrieval_result or not retrieval_result.chunks:
        rag_node_duration_seconds.labels(node_name="build_sources").observe(
            time.time() - start
        )
        return {
            "sources":           [],
            "table_chunks_used": 0,
        }

    table_chunks = 0
    sources      = []

    for chunk in retrieval_result.chunks:
        is_table = chunk.metadata.get("is_table", False)
        if is_table:
            table_chunks += 1

        sources.append({
            "source":       chunk.source,
            "relevance":    round(chunk.score, 3),
            "vector_score": round(chunk.vector_score, 3),
            "bm25_score":   round(chunk.bm25_score, 3),
            "type":         "table" if is_table else "text",
            "is_table":     is_table,
            "caption":      chunk.metadata.get("caption"),
            "page_no":      chunk.metadata.get("page_no"),
            "text_preview": (
                chunk.text[:150] + "..."
                if len(chunk.text) > 150
                else chunk.text
            ),
        })

    rag_node_duration_seconds.labels(node_name="build_sources").observe(
        time.time() - start
    )
    logger.info(f"   Sources: {len(sources)} ({table_chunks} tables)")

    return {
        "sources":           sources,
        "table_chunks_used": table_chunks,
    }


# ============================================================
# NODE 8: MEMORY SAVE NODE
# ============================================================

def memory_save_node(
    state: RAGState,
    memory_manager,
) -> Dict[str, Any]:
    """Save conversation exchange to memory."""
    start = time.time()
    logger.info("💾 [Memory Save Node]")

    if state.get("success") and state.get("answer"):
        memory_manager.add_message(
            session_id=state["session_id"],
            user_message=state["question"],
            assistant_message=state["answer"],
        )

        msg_count = memory_manager.get_session_message_count(
            state["session_id"]
        )

        # ── Metrics ───────────────────────────────────────────
        memory_total_messages.inc(2)
        memory_active_sessions.set(memory_manager.get_session_count())
        logger.info(f"   Saved → {msg_count} messages in session")

    metadata = {
        "retrieval_time":      round(state.get("retrieval_time", 0.0), 3),
        "generation_time":     round(state.get("generation_time", 0.0), 3),
        "chunks_found":        state.get("chunks_found", 0),
        "table_chunks_used":   state.get("table_chunks_used", 0),
        "fallback_used":       state.get("fallback_used", False),
        "session_id":          state["session_id"],
        "has_table_chunks":    state.get("has_table_chunks", False),
        "route_taken":         state.get("route", "unknown"),
        "conversation_length": memory_manager.get_session_message_count(
            state["session_id"]
        ),
    }

    rag_node_duration_seconds.labels(node_name="memory_save").observe(
        time.time() - start
    )

    return {"metadata": metadata}