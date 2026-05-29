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
    start = time.time()

    question     = state["question"]
    context      = state["context"]
    history_text = state.get("history_text", "")
    has_tables   = state.get("has_table_chunks", False)

    # ── Use override rules if provided (e.g. web search) ───────
    if state.get("_prompt_override"):
        system_rules = state["_prompt_override"]
    elif has_tables:
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




# ============================================================
# NODE 9: MCP NODE
# ============================================================


def web_search_node(
    state: RAGState,
    mcp_client=None,
) -> Dict[str, Any]:
    """Execute web search via external MCP server."""
    start = time.time()
    logger.info(
        f"🌐 [Web Search Node] query='{state['question'][:80]}'"
    )

    if not mcp_client:
        logger.warning("⚠️  MCP client not configured")
        return {
            "web_search_result":  None,
            "web_search_context": "",
            "web_search_used":    False,
            "route":              "fallback",
        }

    if not mcp_client.is_server_online("websearch"):
        logger.warning("⚠️  WebSearch MCP server offline")
        return {
            "web_search_result":  None,
            "web_search_context": "",
            "web_search_used":    False,
            "route":              "fallback",
        }

    # ── Call external MCP server ───────────────────────────────
    result_text = mcp_client.web_search(
        query=state["question"],
        max_results=5,
    )

    elapsed = time.time() - start

    rag_node_duration_seconds.labels(
        node_name="web_search"
    ).observe(elapsed)

    if result_text and "No results found" not in result_text:
        logger.info(
            f"✅ Web search: {len(result_text)} chars "
            f"in {elapsed:.2f}s"
        )
        return {
            "web_search_result":  {"raw": result_text},
            "web_search_context": result_text,
            "web_search_used":    True,
            "route":              "prompt",
        }

    logger.warning("⚠️  Web search returned no results")
    return {
        "web_search_result":  None,
        "web_search_context": "",
        "web_search_used":    False,
        "route":              "fallback",
    }

def intent_detection_node(
    state:      RAGState,
    mcp_client=None,
) -> Dict[str, Any]:
    """Detect file/web intent using MCP client availability."""
    start = time.time()
    logger.info(
        f"🎯 [Intent Detection] query='{state['question'][:80]}'"
    )

    needs_web_search  = False
    needs_file_read   = False
    detected_filename = None

    query = state["question"]

    # ── File intent ────────────────────────────────────────────
    if mcp_client and mcp_client.is_server_online("filesystem"):
        needs_file_read, detected_filename = _detect_file_intent(query)

    # ── Web intent (explicit) ──────────────────────────────────
    if mcp_client and mcp_client.is_server_online("websearch"):
        needs_web_search = _detect_web_intent(query)

    if not needs_file_read and not needs_web_search:
        logger.info(
            "   📚 Standard RAG → "
            "web search auto-triggers if 0 results"
        )

    rag_node_duration_seconds.labels(
        node_name="intent_detection"
    ).observe(time.time() - start)

    logger.info(
        f"   file_read={needs_file_read} "
        f"(file='{detected_filename}') | "
        f"web_search={needs_web_search}"
    )

    return {
        "needs_web_search":  needs_web_search,
        "needs_file_read":   needs_file_read,
        "detected_filename": detected_filename,
    }

def filesystem_node(
    state: RAGState,
    mcp_client=None,
) -> Dict[str, Any]:
    """Execute file read via external MCP server."""
    start = time.time()
    filename = state.get("detected_filename", "")
    logger.info(f"📁 [FileSystem Node] file='{filename}'")

    if not mcp_client:
        logger.warning("⚠️  MCP client not configured")
        return {
            "file_result":    None,
            "file_context":   "",
            "file_read_used": False,
            "route":          "retrieve",
        }

    if not mcp_client.is_server_online("filesystem"):
        logger.warning("⚠️  FileSystem MCP server offline")
        return {
            "file_result":    None,
            "file_context":   "",
            "file_read_used": False,
            "route":          "retrieve",
        }

    if not filename:
        logger.warning("⚠️  No filename detected")
        return {
            "file_result":    None,
            "file_context":   "",
            "file_read_used": False,
            "route":          "retrieve",
        }

    # ── Call external MCP server ───────────────────────────────
    content = mcp_client.read_file(filename)
    elapsed = time.time() - start

    rag_node_duration_seconds.labels(
        node_name="filesystem"
    ).observe(elapsed)

    success = not content.startswith("File not found") and \
              not content.startswith("Access denied") and \
              not content.startswith("Error")

    if success:
        logger.info(
            f"✅ File read: '{filename}' "
            f"({len(content)} chars) in {elapsed:.2f}s"
        )
        file_context = (
            f"[FILE CONTENT: {filename}]\n\n{content}\n"
        )
        return {
            "file_result": {
                "filename":   filename,
                "content":    content,
                "char_count": len(content),
                "source":     "filesystem",
            },
            "file_context":   file_context,
            "file_read_used": True,
            "route":          "prompt",
        }

    logger.warning(f"⚠️  File read failed: {content[:100]}")
    return {
        "file_result":    None,
        "file_context":   content,
        "file_read_used": False,
        "route":          "retrieve",
    }



import re

_FILE_PATTERNS = [
    r"\bread\s+(?:the\s+)?(?:file\s+)?['\"]?([^\s'\"]+\.\w+)['\"]?",
    r"\bopen\s+(?:the\s+)?(?:file\s+)?['\"]?([^\s'\"]+\.\w+)['\"]?",
    r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:content\s+of\s+)?['\"]?([^\s'\"]+\.\w+)['\"]?",
    r"\blire\s+(?:le\s+)?(?:fichier\s+)?['\"]?([^\s'\"]+\.\w+)['\"]?",
    r"\bouvrir\s+(?:le\s+)?(?:fichier\s+)?['\"]?([^\s'\"]+\.\w+)['\"]?",
    r"\bafficher\s+(?:le\s+)?(?:fichier\s+)?['\"]?([^\s'\"]+\.\w+)['\"]?",
    r"['\"]([^\s'\"]+\.\w{2,5})['\"]",
    r"\b([\w\-\/\\]+\.(?:txt|pdf|md|csv|json|docx|log|yml|yaml))\b",
]

_FILE_KEYWORDS = [
    "read file","open file","show file","load file",
    "file content","content of","from file",
    "lire fichier","ouvrir fichier","afficher fichier",
    "contenu du fichier","depuis le fichier",
    "lire le","montre moi","montrez moi",
]

_WEB_KEYWORDS = [
    "search the web","search online","find online","look up",
    "latest","recent","current","news","today",
    "from the web","on the web","internet",
    "cherche sur le web","recherche sur internet",
    "actualité","récent","dernière","aujourd'hui",
    "nouvelles","sur internet","en ligne","recherche web",
]


def _detect_file_intent(query: str):
    """Returns (has_intent, filename)."""
    q = query.lower()
    has_intent = any(kw in q for kw in _FILE_KEYWORDS)
    filename   = None
    for pattern in _FILE_PATTERNS:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            filename   = m.group(1)
            has_intent = True
            break
    return has_intent, filename


def _detect_web_intent(query: str) -> bool:
    """Returns True if explicit web search requested."""
    q = query.lower()
    return any(kw in q for kw in _WEB_KEYWORDS)