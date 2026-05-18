"""
RAG Graph State
Shared state object for LangGraph RAG pipeline
"""

from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass, field


class RAGState(TypedDict):
    """
    Shared state flowing through LangGraph nodes.
    
    Every node reads from and writes to this state.
    LangGraph merges updates automatically.
    """
    
    # ── Input ─────────────────────────────────────────────────────
    question:     str
    session_id:   str
    history:      List[Dict[str, str]]
    
    # ── Retrieval ─────────────────────────────────────────────────
    retrieval_result:  Optional[Any]     # RetrievalResult object
    context:           str               # Formatted context string
    chunks_found:      int               # Number of chunks retrieved
    has_table_chunks:  bool              # Whether tables were found
    retrieval_time:    float
    
    # ── Prompt ────────────────────────────────────────────────────
    history_text:  str                   # Formatted history for prompt
    full_prompt:   str                   # Final prompt sent to LLM
    
    # ── Generation ────────────────────────────────────────────────
    raw_answer:    str                   # Raw LLM output
    answer:        str                   # Post-processed answer
    generation_time: float
    
    # ── Sources ───────────────────────────────────────────────────
    sources:       List[Dict[str, Any]]  # Formatted sources
    table_chunks_used: int
    
    # ── Control Flow ──────────────────────────────────────────────
    route:         str                   # Current routing decision
    error:         Optional[str]         # Error message if any
    fallback_used: bool                  # Whether fallback was triggered
    
    # ── Final ─────────────────────────────────────────────────────
    success:       bool
    metadata:      Dict[str, Any]