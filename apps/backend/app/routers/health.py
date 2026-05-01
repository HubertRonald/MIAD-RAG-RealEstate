from __future__ import annotations

from fastapi import APIRouter

from app.config.runtime import get_settings

router = APIRouter(tags=["Health"])
settings = get_settings()


@router.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.ENV,
    }


@router.get("/ready")
async def readiness_check() -> dict:
    return {
        "status": "ready",
        "service": settings.APP_NAME,
    }
