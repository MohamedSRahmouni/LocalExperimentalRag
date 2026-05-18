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
    memory_node,
    retrieval_node,
    fallback_node,
    prompt_node,
    generation_node,
    postprocess_node,
    sources_node,
    memory_save_node,
)

logger = logging.getLogger(__name__)


# ============================================================
# ROUTING FUNCTIONS
# ============================================================

def route_after_retrieval(
    state: RAGState,
) -> Literal["fallback", "prompt"]:
    """Route after retrieval."""
    route = state.get("route", "standard")
    
    if route == "fallback":
        logger.info("🔀 Route → fallback")
        return "fallback"
    
    logger.info(f"🔀 Route → prompt (mode={route})")
    return "prompt"


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
):
    """
    Build and compile the full RAG LangGraph.
    
    Args:
        retrieval_service : RetrievalService instance
        memory_manager    : ConversationMemoryManager instance
        llm               : ChatOpenAI instance
        
    Returns:
        Compiled LangGraph
    """
    from app.pipeline.langsmith_callbacks import LangSmithRAGCallbackHandler
    
    callback_handler = LangSmithRAGCallbackHandler(project_name="rag-traces")
    
    logger.info("🔧 Building RAG LangGraph...")
    
    # ── Bind services to nodes ─────────────────────────────────
    _memory_node      = partial(memory_node,      memory_manager=memory_manager)
    _retrieval_node   = partial(retrieval_node,   retrieval_service=retrieval_service)
    _generation_node  = partial(generation_node,  llm=llm)
    _memory_save_node = partial(memory_save_node, memory_manager=memory_manager)
    
    # ── Build graph ────────────────────────────────────────────
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("memory",        _memory_node)
    graph.add_node("retrieve",      _retrieval_node)
    graph.add_node("fallback",      fallback_node)
    graph.add_node("prompt",        prompt_node)
    graph.add_node("generate",      _generation_node)
    graph.add_node("error_handler", error_handler_node)
    graph.add_node("postprocess",   postprocess_node)
    graph.add_node("build_sources", sources_node)
    graph.add_node("memory_save",   _memory_save_node)
    
    # ── Entry point ────────────────────────────────────────────
    graph.set_entry_point("memory")
    
    # ── Linear flow ────────────────────────────────────────────
    graph.add_edge("memory", "retrieve")
    
    # ── After retrieval: fallback OR prompt ────────────────────
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "fallback": "fallback",
            "prompt":   "prompt",
        }
    )
    
    # ── Fallback path: fallback → build_sources ────────────────
    graph.add_edge("fallback", "build_sources")
    
    # ── Standard path: prompt → generate ───────────────────────
    graph.add_edge("prompt", "generate")
    
    # ── After generation: postprocess OR error_handler ─────────
    graph.add_conditional_edges(
        "generate",
        route_after_generation,
        {
            "postprocess":   "postprocess",
            "error_handler": "error_handler",
        }
    )
    
    # ── Both success paths merge at build_sources ──────────────
    graph.add_edge("postprocess",   "build_sources")
    graph.add_edge("error_handler", "build_sources")
    
    # ── Final path: build_sources → memory_save → END ─────────
    graph.add_edge("build_sources", "memory_save")
    graph.add_edge("memory_save",   END)
    
    # ── Compile ────────────────────────────────────────────────
    compiled = graph.compile()
    
    logger.info("✅ RAG LangGraph compiled")
    logger.info("   Flow:")
    logger.info("     memory → retrieve → [fallback OR prompt]")
    logger.info("     fallback → build_sources")
    logger.info("     prompt → generate → [postprocess OR error_handler]")
    logger.info("     [postprocess/error_handler] → build_sources")
    logger.info("     build_sources → memory_save → END")
    
    return compiled