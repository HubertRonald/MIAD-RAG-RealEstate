from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.runtime import get_settings
from app.routers import ask, health, recommend
from miad_rag_common.logging.structured_logging import configure_logging

settings = get_settings()
logger = configure_logging(level=settings.LOG_LEVEL, json_logs=settings.JSON_LOGS)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Backend RAG para recomendación inmobiliaria en Montevideo.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ask.router)
app.include_router(recommend.router)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info(
        "backend_startup",
        extra={
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "env": settings.ENV,
            "project_id": settings.PROJECT_ID,
            "index_bucket": settings.INDEX_BUCKET,
        },
    )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("backend_shutdown", extra={"app": settings.APP_NAME})

@app.get("/")
def root() -> dict:
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
    }
