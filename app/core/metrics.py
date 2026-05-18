"""
Prometheus Metrics
Central metrics registry for the RAG system
"""

import logging
import time
from typing import Optional
from functools import wraps
from contextlib import contextmanager

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

logger = logging.getLogger(__name__)


# ============================================================
# REGISTRY
# ============================================================

# Use default registry
registry = REGISTRY


# ============================================================
# HTTP METRICS
# ============================================================

http_requests_total = Counter(
    name='rag_http_requests_total',
    documentation='Total HTTP requests',
    labelnames=['method', 'endpoint', 'status_code'],
)

http_request_duration_seconds = Histogram(
    name='rag_http_request_duration_seconds',
    documentation='HTTP request duration in seconds',
    labelnames=['method', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)


# ============================================================
# RAG PIPELINE METRICS
# ============================================================

rag_requests_total = Counter(
    name='rag_requests_total',
    documentation='Total RAG ask() requests',
    labelnames=['status', 'session_id'],
)

rag_pipeline_duration_seconds = Histogram(
    name='rag_pipeline_duration_seconds',
    documentation='Full RAG pipeline duration (retrieval + generation)',
    labelnames=['route_taken'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# LangGraph node durations
rag_node_duration_seconds = Histogram(
    name='rag_node_duration_seconds',
    documentation='Duration of each LangGraph node',
    labelnames=['node_name'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
)


# ============================================================
# RETRIEVAL METRICS
# ============================================================

retrieval_requests_total = Counter(
    name='rag_retrieval_requests_total',
    documentation='Total retrieval requests',
    labelnames=['mode'],   # hybrid, vector, fallback
)

retrieval_duration_seconds = Histogram(
    name='rag_retrieval_duration_seconds',
    documentation='Retrieval duration in seconds',
    labelnames=['mode'],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

retrieval_chunks_found = Histogram(
    name='rag_retrieval_chunks_found',
    documentation='Number of chunks retrieved per query',
    labelnames=['mode'],
    buckets=[0, 1, 2, 3, 5, 10, 15, 20],
)

retrieval_score_max = Histogram(
    name='rag_retrieval_score_max',
    documentation='Maximum similarity score per retrieval',
    labelnames=['mode'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

retrieval_fallback_total = Counter(
    name='rag_retrieval_fallback_total',
    documentation='Times retrieval fell back to lower threshold',
    labelnames=['fallback_level'],   # level_1, level_2
)

retrieval_empty_total = Counter(
    name='rag_retrieval_empty_total',
    documentation='Times retrieval found zero results',
)

retrieval_table_chunks_total = Counter(
    name='rag_retrieval_table_chunks_total',
    documentation='Total table chunks retrieved',
)


# ============================================================
# LLM METRICS
# ============================================================

llm_requests_total = Counter(
    name='rag_llm_requests_total',
    documentation='Total LLM generation requests',
    labelnames=['status'],   # success, error, empty
)

llm_duration_seconds = Histogram(
    name='rag_llm_duration_seconds',
    documentation='LLM generation duration in seconds',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

llm_answer_length = Histogram(
    name='rag_llm_answer_length_chars',
    documentation='Length of LLM answers in characters',
    buckets=[50, 100, 200, 500, 1000, 2000, 5000],
)

llm_prompt_length = Histogram(
    name='rag_llm_prompt_length_chars',
    documentation='Length of prompts sent to LLM',
    buckets=[500, 1000, 2000, 4000, 8000],
)

llm_errors_total = Counter(
    name='rag_llm_errors_total',
    documentation='Total LLM errors',
    labelnames=['error_type'],
)


# ============================================================
# DOCUMENT PROCESSING METRICS
# ============================================================

documents_processed_total = Counter(
    name='rag_documents_processed_total',
    documentation='Total documents processed',
    labelnames=['status', 'file_type'],  # success/failed/duplicate
)

document_chunks_created_total = Counter(
    name='rag_document_chunks_created_total',
    documentation='Total chunks created during processing',
    labelnames=['chunk_type'],   # text, table
)

document_processing_duration_seconds = Histogram(
    name='rag_document_processing_duration_seconds',
    documentation='Document processing duration',
    labelnames=['file_type'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

ocr_operations_total = Counter(
    name='rag_ocr_operations_total',
    documentation='Total OCR operations performed',
    labelnames=['status'],
)


# ============================================================
# WEAVIATE METRICS
# ============================================================

weaviate_queries_total = Counter(
    name='rag_weaviate_queries_total',
    documentation='Total Weaviate queries',
    labelnames=['query_type'],   # semantic, hybrid, by_file
)

weaviate_query_duration_seconds = Histogram(
    name='rag_weaviate_query_duration_seconds',
    documentation='Weaviate query duration',
    labelnames=['query_type'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

weaviate_connection_status = Gauge(
    name='rag_weaviate_connection_status',
    documentation='Weaviate connection status (1=connected, 0=disconnected)',
)

weaviate_chunks_stored_total = Counter(
    name='rag_weaviate_chunks_stored_total',
    documentation='Total chunks stored in Weaviate',
    labelnames=['status'],   # success, failed
)

weaviate_total_chunks = Gauge(
    name='rag_weaviate_total_chunks',
    documentation='Current total chunks in Weaviate collection',
)


# ============================================================
# MEMORY METRICS
# ============================================================

memory_active_sessions = Gauge(
    name='rag_memory_active_sessions',
    documentation='Number of active conversation sessions',
)

memory_total_messages = Counter(
    name='rag_memory_total_messages',
    documentation='Total messages processed across all sessions',
)


# ============================================================
# SYSTEM METRICS
# ============================================================

system_startup_time = Gauge(
    name='rag_system_startup_time_seconds',
    documentation='System startup time in seconds',
)

system_ready = Gauge(
    name='rag_system_ready',
    documentation='System readiness (1=ready, 0=not ready)',
)


# ============================================================
# HELPER DECORATORS & CONTEXT MANAGERS
# ============================================================

@contextmanager
def track_duration(histogram, labels: dict = None):
    """
    Context manager to track duration of any operation.
    
    Usage:
        with track_duration(retrieval_duration_seconds, {'mode': 'hybrid'}):
            result = retrieve(query)
    """
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        if labels:
            histogram.labels(**labels).observe(duration)
        else:
            histogram.observe(duration)


def track_node_duration(node_name: str):
    """
    Decorator for LangGraph nodes to track duration.
    
    Usage:
        @track_node_duration("retrieval_node")
        def retrieval_node(state, retrieval_service):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                rag_node_duration_seconds.labels(
                    node_name=node_name
                ).observe(duration)
        return wrapper
    return decorator


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    return generate_latest(registry)


def get_content_type() -> str:
    """Get Prometheus content type header."""
    return CONTENT_TYPE_LATEST


# ============================================================
# INITIALIZATION
# ============================================================

def init_metrics():
    """Initialize default metric values."""
    system_ready.set(0)
    weaviate_connection_status.set(0)
    memory_active_sessions.set(0)
    weaviate_total_chunks.set(0)
    logger.info("✅ Prometheus metrics initialized")