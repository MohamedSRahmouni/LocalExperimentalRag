"""
LangSmith Configuration
Enables distributed tracing, logging, and observability
"""

import os
import logging
from typing import Optional

from langsmith import Client

logger = logging.getLogger(__name__)


class LangSmithConfig:
    """Centralized LangSmith configuration."""
    
    def __init__(self):
        self.enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        self.api_key = os.getenv("LANGSMITH_API_KEY")  
        self.endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")  
        self.project = os.getenv("LANGSMITH_PROJECT", "RAG")  
        self.session = os.getenv("LANGSMIFY_SESSION", "production")
        
        self._client: Optional[Client] = None
        
        if not self.enabled:
            logger.warning("⚠️  LangSmith tracing disabled")
            return
        
        if not self.api_key:
            logger.error("❌ LangSmith enabled but no API_KEY configured")
            logger.info("   Set LANGSMIFY_API_KEY in .env")
            self.enabled = False
            return
        
        try:
            self._client = Client(
                api_key=self.api_key,
                api_url=self.endpoint,
            )
            
            # Verify connection
            list(self._client.list_projects())
            logger.info("✅ LangSmith connected")
            logger.info(f"   Project: {self.project}")
            logger.info(f"   Session: {self.session}")
            
        except Exception as e:
            logger.error(f"❌ LangSmith init failed: {e}")
            self._client = None
            self.enabled = False
    
    @property
    def client(self) -> Optional[Client]:
        """Get LangSmith client."""
        return self._client
    
    def get_client(self) -> Optional[Client]:
        """Get LangSmith client (alias)."""
        return self._client
    
    def is_enabled(self) -> bool:
        """Check if LangSmith is enabled."""
        return self.enabled and self._client is not None
    
    def create_experiment(self, name: str) -> Optional[str]:
        """
        Create a new experiment for evaluation.
        
        Args:
            name: Experiment name
            
        Returns:
            Experiment ID or None
        """
        if not self.is_enabled():
            return None
        
        try:
            from langsmith.evaluation import evaluate
            logger.info(f"🧪 Creating experiment: {name}")
            return name  # Simplified - returns experiment name
        except Exception as e:
            logger.error(f"❌ Failed to create experiment: {e}")
            return None


# Single instance
_langsmith_config = None

def get_langsmith_config() -> LangSmithConfig:
    """Get singleton LangSmithConfig instance."""
    global _langsmith_config
    if _langsmith_config is None:
        _langsmith_config = LangSmithConfig()
    return _langsmith_config