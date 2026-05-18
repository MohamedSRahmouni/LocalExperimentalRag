"""
Metrics Route
Exposes Prometheus metrics endpoint
"""

import logging
from fastapi import APIRouter
from fastapi.responses import Response

from app.core.metrics import get_metrics, get_content_type

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Scrape this endpoint with your Prometheus server:
    
    prometheus.yml:
        scrape_configs:
          - job_name: 'rag'
            static_configs:
              - targets: ['localhost:8000']
            metrics_path: '/metrics'
    """
    logger.debug("📊 Metrics scraped")
    return Response(
        content=get_metrics(),
        media_type=get_content_type(),
    )


@router.get("/metrics/health")
async def metrics_health():
    """Quick health check via metrics."""
    from app.core.metrics import (
        weaviate_connection_status,
        system_ready,
        memory_active_sessions,
        weaviate_total_chunks,
    )
    from app.api.dependencies import get_service_health

    health = get_service_health()

    return {
        "status":   "ok" if health.get("all_ready") else "degraded",
        "services": health,
    }