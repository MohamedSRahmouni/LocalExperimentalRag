"""
Prometheus Middleware
Automatically tracks all HTTP requests
"""

import time
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .metrics import (
    http_requests_total,
    http_request_duration_seconds,
)

logger = logging.getLogger(__name__)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that tracks all HTTP requests.
    
    Tracks:
        - Request count by method, endpoint, status
        - Request duration by method, endpoint
    """

    def __init__(self, app: ASGIApp, app_name: str = "rag"):
        super().__init__(app)
        self.app_name = app_name

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:

        # Normalize endpoint path
        endpoint = self._normalize_path(request.url.path)
        method   = request.method

        start_time = time.time()

        try:
            response = await call_next(request)
            status_code = response.status_code

        except Exception as e:
            status_code = 500
            logger.error(f"❌ Request error: {e}")
            raise

        finally:
            duration = time.time() - start_time

            # Record metrics
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
            ).inc()

            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

        return response

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path for grouping metrics.
        
        Replaces dynamic segments to avoid cardinality explosion.
        e.g. /api/users/123 → /api/users/{id}
        """
        # Skip metrics endpoint itself
        if path == "/metrics":
            return "/metrics"

        # Group known endpoints
        path_map = {
            "/api/ask":              "/api/ask",
            "/api/ask/stream":       "/api/ask/stream",
            "/api/chat":             "/api/chat",
            "/api/retrieve":         "/api/retrieve",
            "/api/stats":            "/api/stats",
            "/api/rag/status":       "/api/rag/status",
            "/upload":               "/upload",
            "/api/weaviate/stats":   "/api/weaviate/stats",
        }

        for known_path, label in path_map.items():
            if path.startswith(known_path):
                return label

        return path