from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter

from app.config.runtime import get_settings

router = APIRouter(tags=["Health"])
settings = get_settings()


def _utc_timestamp() -> str:
    """
    Timestamp UTC en formato ISO-8601.

    Ejemplo:
      2026-05-02T03:15:27+00:00
    """
    return datetime.now(timezone.utc).isoformat()


def _health_payload() -> dict[str, Any]:
    """
    Payload común de health check.

    Compatible con el endpoint local original, pero enriquecido con metadata
    del servicio Cloud Run.
    """
    return {
        "status": "healthy",
        "success": True,
        "timestamp": _utc_timestamp(),
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.ENV,
    }


def _readiness_payload() -> dict[str, Any]:
    """
    Payload de readiness.

    No fuerza carga de FAISS, BigQuery ni Gemini para evitar cold start costoso.
    Solo confirma que la app arrancó y que la configuración base está disponible.
    """
    return {
        "status": "ready",
        "success": True,
        "timestamp": _utc_timestamp(),
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.ENV,
        "config": {
            "project_id": settings.PROJECT_ID,
            "gcp_location": settings.GCP_LOCATION,
            "index_bucket": settings.INDEX_BUCKET,
            "index_prefix": settings.INDEX_PREFIX,
            "default_collection": settings.DEFAULT_COLLECTION,
            "bq_project_id": settings.BQ_PROJECT_ID,
            "bq_dataset_id": settings.BQ_DATASET_ID,
            "bq_listings_table": settings.BQ_LISTINGS_TABLE,
        },
    }


# -------------------------------------------------------------------------
# Health endpoints
# -------------------------------------------------------------------------
# /api/v1/health mantiene compatibilidad con _realstate_ragas local.
# /health es práctico para Cloud Run, balanceadores o pruebas rápidas.
# -------------------------------------------------------------------------

@router.get("/api/v1/health")
def health_check_v1() -> dict[str, Any]:
    return _health_payload()


@router.get("/health")
def health_check() -> dict[str, Any]:
    return _health_payload()


# -------------------------------------------------------------------------
# Readiness endpoints
# -------------------------------------------------------------------------
# No ejecutan dependencias externas. Solo indican que el proceso está listo.
# -------------------------------------------------------------------------

@router.get("/api/v1/ready")
def readiness_check_v1() -> dict[str, Any]:
    return _readiness_payload()


@router.get("/ready")
def readiness_check() -> dict[str, Any]:
    return _readiness_payload()
