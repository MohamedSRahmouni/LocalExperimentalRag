"""
LangSmith Callback Handlers
Integrates tracing into LangGraph nodes and LLM calls
"""

import logging
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from contextlib import contextmanager

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

from .graph_state import RAGState
from ..core.langsmith_config import get_langsmith_config

logger = logging.getLogger(__name__)


class LangSmithRAGCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for LangSmith RAG tracing.
    
    Tracks:
        - LLM calls (prompts, responses, tokens)
        - Retrieval operations
        - Graph node execution
        - Errors
    """
    
    def __init__(self, project_name: str = "rag-traces"):
        self.config = get_langsmith_config()
        self.project_name = project_name
        self.runs: Dict[str, Any] = {}
        self.current_run_id: Optional[str] = None
        self.start_time: Optional[float] = None
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Called when LLM starts generating."""
        if not self.config.is_enabled():
            return
        
        try:
            run_id = self.config.client.create_run(
                name="llm_generation",
                inputs={"prompts": prompts},
                run_type="llm",
                project_name=self.project_name,
                serialized=serialized,
                tags=["llm", "generation"],
            )
            self.runs["llm"] = run_id
            self.start_time = time.time()
            
            logger.debug(f"📊 LangSmith: LLM started (run_id={run_id.id})")
        except Exception as e:
            logger.error(f"❌ LangSmith LLM start error: {e}")
    
    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any
    ) -> None:
        """Called when LLM finishes generating."""
        if not self.config.is_enabled():
            return
        
        try:
            llm_run = self.runs.get("llm")
            if llm_run:
                generation_time = time.time() - self.start_time
                
                self.config.client.update_run(
                    id=llm_run.id,
                    outputs={"generations": response.generations},
                    end_time=datetime.utcnow(),
                    execution_time=generation_time,
                )
                
                logger.debug(f"📊 LangSmith: LLM completed (time={generation_time:.2f}s)")
        except Exception as e:
            logger.error(f"❌ LangSmith LLM end error: {e}")
    
    def on_llm_error(
        self,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """Called when LLM errors."""
        if not self.config.is_enabled():
            return
        
        try:
            llm_run = self.runs.get("llm")
            if llm_run:
                self.config.client.update_run(
                    id=llm_run.id,
                    error=str(error),
                    end_time=datetime.utcnow(),
                )
                
                logger.error(f"❌ LangSmith: LLM error recorded")
        except Exception as e:
            logger.error(f"❌ LangSmith LLM error recording: {e}")
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Called when chain/graph starts."""
        if not self.config.is_enabled():
            return
        
        try:
            run_id = self.config.client.create_run(
                name=f"rag_pipeline_{inputs.get('session_id', 'default')}",
                inputs=inputs,
                run_type="chain",
                project_name=self.project_name,
                serialized=serialized,
                tags=["rag", "pipeline"],
            )
            self.current_run_id = run_id.id
            self.runs["pipeline"] = run_id
            self.start_time = time.time()
            
            logger.debug(f"📊 LangSmith: Pipeline started (run_id={run_id.id})")
        except Exception as e:
            logger.error(f"❌ LangSmith chain start error: {e}")
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Called when chain/graph ends."""
        if not self.config.is_enabled():
            return
        
        try:
            pipeline_run = self.runs.get("pipeline")
            if pipeline_run:
                total_time = time.time() - self.start_time
                
                self.config.client.update_run(
                    id=pipeline_run.id,
                    outputs=outputs,
                    end_time=datetime.utcnow(),
                    execution_time=total_time,
                )
                
                logger.debug(f"📊 LangSmith: Pipeline completed (time={total_time:.2f}s)")
        except Exception as e:
            logger.error(f"❌ LangSmith chain end error: {e}")
    
    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        **kwargs: Any
    ) -> None:
        """Called when retrieval starts."""
        if not self.config.is_enabled():
            return
        
        try:
            run_id = self.config.client.create_run(
                name=f"retrieval_{kwargs.get('node_name', 'unknown')}",
                inputs={"query": query},
                run_type="retriever",
                project_name=self.project_name,
                serialized=serialized,
                tags=["retrieval"],
            )
            self.runs[f"retrieval_{kwargs.get('node_name', 'unknown')}"] = run_id
        except Exception as e:
            logger.error(f"❌ LangSmith retriever start error: {e}")
    
    def on_retriever_end(
        self,
        documents: List[Any],
        **kwargs: Any
    ) -> None:
        """Called when retrieval ends."""
        if not self.config.is_enabled():
            return
        
        try:
            key = f"retrieval_{kwargs.get('node_name', 'unknown')}"
            run = self.runs.get(key)
            if run:
                self.config.client.update_run(
                    id=run.id,
                    outputs={"documents": [doc.page_content[:500] for doc in documents]},
                    end_time=datetime.utcnow(),
                )
        except Exception as e:
            logger.error(f"❌ LangSmith retriever end error: {e}")


# Helper: Wrap LLM for automatic tracing
def wrap_llm_for_langsmith(llm, model_name: str = "rag-llm"):
    """
    Wrap an LLM instance with LangSmith tracing.
    
    Usage:
        from langchain_openai import ChatOpenAI
        from app.pipeline.langsmith_callbacks import wrap_llm_for_langsmith
        
        llm = ChatOpenAI(...)
        llm_wrapped = wrap_llm_for_langsmith(llm)
    """
    config = get_langsmith_config()
    
    if not config.is_enabled():
        return llm
    
    try:
        wrapped = wrap_openai(llm, client=config.client)
        logger.info(f"✅ LangSmith: Wrapped LLM '{model_name}'")
        return wrapped
    except Exception as e:
        logger.error(f"❌ Failed to wrap LLM: {e}")
        return llm


@contextmanager
def trace_rag_node(node_name: str, state: RAGState = None):
    """
    Context manager to trace individual LangGraph nodes.
    
    Usage:
        with trace_rag_node("retrieve"):
            result = retrieval_service.retrieve(query)
    """
    config = get_langsmith_config()
    
    if not config.is_enabled():
        yield
        return
    
    try:
        run_id = config.client.create_run(
            name=node_name,
            inputs={} if state is None else dict(state),
            run_type="tool",
            project_name=config.project,
            tags=["rag", "node", node_name],
        )
        start_time = time.time()
        
        yield run_id
        
        duration = time.time() - start_time
        config.client.update_run(
            id=run_id.id,
            end_time=datetime.utcnow(),
            execution_time=duration,
        )
        
    except Exception as e:
        logger.error(f"❌ Trace error for {node_name}: {e}")
        raise