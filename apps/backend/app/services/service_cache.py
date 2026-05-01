from __future__ import annotations

from threading import Lock
from typing import Any

from app.config.runtime import get_settings
from app.services.bigquery_listing_service import BigQueryListingService
from app.services.embedding_service import EmbeddingService
from app.services.gcs_index_service import GCSIndexService
from app.services.generation_service import GenerationService
from app.services.preference_extraction_service import PreferenceExtractionService
from app.services.query_rewriting_service import QueryRewritingService
from app.services.rag_graph_service import RAGGraphService
from app.services.reranking_service import RerankingService
from app.services.retrieval_service import RetrievalService
from miad_rag_common.logging.structured_logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

_services_cache: dict[str, dict[str, Any]] = {}
_cache_lock = Lock()


def get_collection_services(collection: str | None = None) -> dict[str, Any]:
    """
    Retorna servicios cacheados por colección.

    Lo costoso es cargar FAISS, por eso se cachea por colección.
    RAGGraphService NO se cachea porque sus flags varían por request.
    """
    collection_name = collection or settings.DEFAULT_COLLECTION

    with _cache_lock:
        if collection_name in _services_cache:
            return _services_cache[collection_name]

        logger.info(
            "service_cache_cold_start",
            extra={"collection": collection_name},
        )

        index_service = GCSIndexService()
        local_index_path = index_service.ensure_local_index(collection_name)

        embedding_service = EmbeddingService(
            model_name=settings.GEMINI_EMBEDDING_MODEL,
        )
        embedding_service.load_vectorstore(str(local_index_path))

        services = {
            "index": index_service,
            "embedding": embedding_service,
            "retrieval": RetrievalService(
                embedding_service=embedding_service,
                k=settings.RETRIEVAL_K,
                fetch_k=settings.RETRIEVAL_FETCH_K,
            ),
            "generation": GenerationService(
                model=settings.GEMINI_GENERATION_MODEL,
                temperature=settings.GEMINI_TEMPERATURE,
            ),
            "preference": PreferenceExtractionService(
                model=settings.GEMINI_GENERATION_MODEL,
            ),
            "qrewrite": QueryRewritingService(
                model=settings.GEMINI_GENERATION_MODEL,
            ),
            "reranking": RerankingService(
                top_k=settings.RERANKING_TOP_K,
                model_name=settings.RERANKING_MODEL,
                enabled=settings.ENABLE_RERANKING_MODEL,
            ),
            "bq_listing": BigQueryListingService(),
        }

        _services_cache[collection_name] = services

        logger.info(
            "service_cache_collection_ready",
            extra={"collection": collection_name, "index_path": str(local_index_path)},
        )

        return services


def clear_services_cache() -> None:
    """
    Útil para tests o para endpoints administrativos futuros.
    """
    with _cache_lock:
        _services_cache.clear()
