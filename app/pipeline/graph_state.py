"""
RAG Graph State
Shared state object for LangGraph RAG pipeline
"""

from typing import TypedDict, List, Dict, Any, Optional


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

    # ── Intent Detection ──────────────────────────────────────────
    needs_web_search:   bool             # Explicit web search requested
    needs_file_read:    bool             # File read intent detected
    detected_filename:  Optional[str]    # Filename extracted from query

    # ── Retrieval ─────────────────────────────────────────────────
    retrieval_result:  Optional[Any]     # RetrievalResult object
    context:           str               # Formatted context string
    chunks_found:      int               # Number of chunks retrieved
    has_table_chunks:  bool              # Whether tables were found
    retrieval_time:    float

    # ── MCP: Web Search ───────────────────────────────────────────
    web_search_result:  Optional[Any]    # Raw web search response
    web_search_context: str              # Formatted web search context
    web_search_used:    bool             # Whether web search was used

    # ── MCP: Filesystem ───────────────────────────────────────────
    file_result:        Optional[Any]    # Raw file read response
    file_context:       str              # Formatted file content
    file_read_used:     bool             # Whether file read was used

    # ── Prompt ────────────────────────────────────────────────────
    history_text:  str                   # Formatted history for prompt
    full_prompt:   str                   # Final prompt sent to LLM
    _prompt_override: Optional[str]

    # ── Generation ────────────────────────────────────────────────
    raw_answer:       str                # Raw LLM output
    answer:           str                # Post-processed answer
    generation_time:  float

    # ── Sources ───────────────────────────────────────────────────
    sources:           List[Dict[str, Any]]
    table_chunks_used: int

    # ── Control Flow ──────────────────────────────────────────────
    route:         str                   # Current routing decision
    error:         Optional[str]         # Error message if any
    fallback_used: bool                  # Whether fallback was triggered

    # ── Final ─────────────────────────────────────────────────────
    success:  bool
    metadata: Dict[str, Any]