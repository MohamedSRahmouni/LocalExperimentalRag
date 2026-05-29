"""
RAG Graph Builder
Assembles the LangGraph execution graph
"""

import logging
from functools import partial
from typing import Literal

from langgraph.graph import StateGraph, END

from .graph_state  import RAGState
from .graph_nodes  import (
    filesystem_node,
    intent_detection_node,
    memory_node,
    retrieval_node,
    fallback_node,
    prompt_node,
    generation_node,
    postprocess_node,
    sources_node,
    memory_save_node,
    web_search_node,
)

logger = logging.getLogger(__name__)


# ============================================================
# ROUTING FUNCTIONS
# ============================================================

def route_after_intent(
    state: RAGState,
) -> Literal["filesystem", "retrieve"]:
    """Route after intent detection."""
    if state.get("needs_file_read") and state.get("detected_filename"):
        logger.info("🔀 Route → filesystem")
        return "filesystem"
    logger.info("🔀 Route → retrieve")
    return "retrieve"


def route_after_filesystem(
    state: RAGState,
) -> Literal["prompt", "retrieve"]:
    """Route after filesystem node."""
    if state.get("file_read_used"):
        logger.info("🔀 Route → prompt (file context ready)")
        return "prompt"
    logger.info("🔀 Route → retrieve (file read failed, fallback to RAG)")
    return "retrieve"


def route_after_retrieval(
    state: RAGState,
) -> Literal["web_search", "prompt"]:
    """Route after retrieval — web search if 0 results OR explicit web intent."""
    route             = state.get("route", "standard")
    needs_web_search  = state.get("needs_web_search", False)

    if route == "fallback" or needs_web_search:
        logger.info(
            f"🔀 Route → web_search "
            f"(reason: {'no_results' if route == 'fallback' else 'explicit_intent'})"
        )
        return "web_search"

    logger.info(f"🔀 Route → prompt (mode={route})")
    return "prompt"


def route_after_web_search(
    state: RAGState,
) -> Literal["prompt", "fallback"]:
    """Route after web search."""
    if state.get("web_search_used"):
        logger.info("🔀 Route → prompt (web search context ready)")
        return "prompt"
    logger.info("🔀 Route → fallback (web search also failed)")
    return "fallback"


def route_after_generation(
    state: RAGState,
) -> Literal["postprocess", "error_handler"]:
    """Route after generation."""
    if state.get("route") == "error" or state.get("error"):
        logger.warning("🔀 Route → error_handler")
        return "error_handler"
    logger.info("🔀 Route → postprocess")
    return "postprocess"


# ============================================================
# PROMPT NODE WRAPPER
# ============================================================

def prompt_node_with_context(state: RAGState) -> dict:
    merged_context = state.get("context", "")
    web_used  = state.get("web_search_used", False)
    file_used = state.get("file_read_used", False)

    if web_used and state.get("web_search_context"):
        web_ctx = state["web_search_context"]
        merged_context = (
            f"{web_ctx}\n\n---\n\n{merged_context}"
            if merged_context
            else web_ctx
        )

    if file_used and state.get("file_context"):
        file_ctx = state["file_context"]
        merged_context = (
            f"{file_ctx}\n\n---\n\n{merged_context}"
            if merged_context
            else file_ctx
        )

    state = {**state, "context": merged_context}

    # ── Patch prompt rules based on context source ─────────────
    if web_used:
        state = {
            **state,
            "_prompt_override": (
                "Tu es un assistant précis et utile.\n"
                "Utilise les résultats de recherche web ci-dessous "
                "pour répondre directement à la question.\n\n"
                "RÈGLES:\n"
                "• Réponds directement avec l'information trouvée\n"
                "• Cite la source si pertinent\n"
                "• Si les résultats ne contiennent pas la réponse, "
                "dis-le clairement\n"
                "• Réponds en français\n"
            ),
        }

    return prompt_node(state)

# ============================================================
# ERROR HANDLER NODE
# ============================================================

def error_handler_node(state: RAGState):
    """Handle generation errors gracefully."""
    logger.error(
        f"❌ [Error Handler] error='{state.get('error', 'unknown')}'"
    )
    return {
        "answer":  "Une erreur s'est produite lors de la génération.",
        "success": False,
    }


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_rag_graph(
    retrieval_service,
    memory_manager,
    llm,
    mcp_client=None,
):
    """
    Build and compile the full RAG LangGraph.

    Flow:
        memory
          → intent_detection
          → [filesystem OR retrieve]
          filesystem → [prompt OR retrieve]
          retrieve   → [web_search OR prompt]
          web_search → [prompt OR fallback]
          fallback   → build_sources
          prompt     → generate → [postprocess OR error_handler]
          [postprocess / error_handler] → build_sources
          build_sources → memory_save → END
    """
    from app.pipeline.langsmith_callbacks import LangSmithRAGCallbackHandler
    callback_handler = LangSmithRAGCallbackHandler(project_name="rag-traces")

    logger.info("🔧 Building RAG LangGraph...")

    # ── Bind services to nodes ─────────────────────────────────
    _memory_node      = partial(memory_node,      memory_manager=memory_manager)
    _retrieval_node   = partial(retrieval_node,   retrieval_service=retrieval_service)
    _generation_node  = partial(generation_node,  llm=llm)
    _memory_save_node = partial(memory_save_node, memory_manager=memory_manager)
    _web_search_node  = partial(web_search_node,  mcp_client=mcp_client)
    _filesystem_node  = partial(filesystem_node,  mcp_client=mcp_client)
    _intent_node      = partial(intent_detection_node, mcp_client=mcp_client)

    # ── Build graph ────────────────────────────────────────────
    graph = StateGraph(RAGState)

    # ── Add nodes ──────────────────────────────────────────────
    graph.add_node("memory",           _memory_node)
    graph.add_node("intent_detection", _intent_node)
    graph.add_node("filesystem",       _filesystem_node)
    graph.add_node("retrieve",         _retrieval_node)
    graph.add_node("web_search",       _web_search_node)
    graph.add_node("fallback",         fallback_node)
    graph.add_node("prompt",           prompt_node_with_context)
    graph.add_node("generate",         _generation_node)
    graph.add_node("error_handler",    error_handler_node)
    graph.add_node("postprocess",      postprocess_node)
    graph.add_node("build_sources",    sources_node)
    graph.add_node("memory_save",      _memory_save_node)

    # ── Entry point ────────────────────────────────────────────
    graph.set_entry_point("memory")

    # ── Edges ──────────────────────────────────────────────────
    graph.add_edge("memory", "intent_detection")

    graph.add_conditional_edges(
        "intent_detection",
        route_after_intent,
        {
            "filesystem": "filesystem",
            "retrieve":   "retrieve",
        }
    )

    graph.add_conditional_edges(
        "filesystem",
        route_after_filesystem,
        {
            "prompt":   "prompt",
            "retrieve": "retrieve",
        }
    )

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "web_search": "web_search",
            "prompt":     "prompt",
        }
    )

    graph.add_conditional_edges(
        "web_search",
        route_after_web_search,
        {
            "prompt":   "prompt",
            "fallback": "fallback",
        }
    )

    graph.add_edge("fallback",      "build_sources")
    graph.add_edge("prompt",        "generate")

    graph.add_conditional_edges(
        "generate",
        route_after_generation,
        {
            "postprocess":   "postprocess",
            "error_handler": "error_handler",
        }
    )

    graph.add_edge("postprocess",   "build_sources")
    graph.add_edge("error_handler", "build_sources")
    graph.add_edge("build_sources", "memory_save")
    graph.add_edge("memory_save",   END)

    # ── Compile ────────────────────────────────────────────────
    compiled = graph.compile()

    logger.info("✅ RAG LangGraph compiled")
    logger.info("   Flow:")
    logger.info("     memory → intent_detection")
    logger.info("     intent_detection → [filesystem OR retrieve]")
    logger.info("     filesystem → [prompt OR retrieve]")
    logger.info("     retrieve → [web_search OR prompt]")
    logger.info("     web_search → [prompt OR fallback]")
    logger.info("     fallback → build_sources")
    logger.info("     prompt → generate → [postprocess OR error_handler]")
    logger.info("     [postprocess/error_handler] → build_sources")
    logger.info("     build_sources → memory_save → END")

    return compiled