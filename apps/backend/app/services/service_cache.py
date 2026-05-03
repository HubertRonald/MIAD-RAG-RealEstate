from __future__ import annotations

from threading import Lock
from typing import Any, Optional

from app.config.runtime import get_settings
from app.services.bigquery_listing_service import BigQueryListingService
from app.services.embedding_service import EmbeddingService
from app.services.gcs_index_service import GCSIndexService
from app.services.generation_service import GenerationService
from app.services.preference_extraction_service import PreferenceExtractionService
from app.services.query_rewriting_service import QueryRewritingService
from app.services.retrieval_service import RetrievalService
from miad_rag_common.logging.structured_logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

_services_cache: dict[str, dict[str, Any]] = {}
_cache_lock = Lock()


def _resolve_collection(collection: Optional[str] = None) -> str:
    """
    Resuelve la colección FAISS a usar.

    Si no se recibe collection, usa settings.DEFAULT_COLLECTION.
    """
    collection_name = (collection or settings.DEFAULT_COLLECTION or "").strip()

    if not collection_name:
        raise ValueError(
            "No se recibió collection y DEFAULT_COLLECTION está vacío."
        )

    return collection_name


def get_collection_services(
    collection: Optional[str] = None,
    force_reload: bool = False,
    refresh_index: bool = False,
) -> dict[str, Any]:
    """
    Retorna servicios cacheados por colección.

    Responsabilidad:
      - Descargar/cargar índice FAISS desde GCS.
      - Crear EmbeddingService con vectorstore cargado.
      - Crear RetrievalService.
      - Crear GenerationService.
      - Crear PreferenceExtractionService.
      - Crear QueryRewritingService.
      - Crear BigQueryListingService.

    No crea RAGGraphService porque sus flags cambian por request:
      - use_query_rewriting
      - use_reranking
      - rewriting_strategy

    No crea RerankingService por defecto. El reranker se carga lazy cuando
    /ask lo pide, usando ensure_reranking_service().

    Args:
        collection:
            Nombre de la colección FAISS. Si None, usa DEFAULT_COLLECTION.

        force_reload:
            Si True, reconstruye el cache de servicios para la colección y
            fuerza nueva descarga del índice desde GCS.

        refresh_index:
            Si True, compara manifest local/remoto y refresca si latest cambió.
            Útil para escenarios donde el job-indexer publicó un índice nuevo y
            la instancia Cloud Run sigue viva.

    Returns:
        Diccionario de servicios compartidos por colección dentro de esta
        instancia Cloud Run.
    """
    collection_name = _resolve_collection(collection)

    with _cache_lock:
        if force_reload and collection_name in _services_cache:
            logger.info(
                "service_cache_force_reload_requested",
                extra={"collection": collection_name},
            )
            _services_cache.pop(collection_name, None)

        if collection_name in _services_cache:
            logger.info(
                "service_cache_hit",
                extra={"collection": collection_name},
            )
            return _services_cache[collection_name]

        logger.info(
            "service_cache_cold_start",
            extra={
                "collection": collection_name,
                "force_reload": force_reload,
                "refresh_index": refresh_index,
            },
        )

        index_service = GCSIndexService()

        if refresh_index:
            local_index_path = index_service.refresh_if_remote_changed(
                collection=collection_name,
                version="latest",
            )
        else:
            local_index_path = index_service.ensure_local_index(
                collection=collection_name,
                version="latest",
                force_download=force_reload,
            )

        embedding_service = EmbeddingService(
            model_name=settings.GEMINI_EMBEDDING_MODEL,
        )

        embedding_service.load_vectorstore(
            persist_path=str(local_index_path),
        )

        retrieval_service = RetrievalService(
            embedding_service=embedding_service,
            k=settings.RETRIEVAL_K,
            fetch_k=settings.RETRIEVAL_FETCH_K,
        )

        generation_service = GenerationService(
            model=settings.GEMINI_GENERATION_MODEL,
            temperature=settings.GEMINI_TEMPERATURE,
            max_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS,
        )

        preference_service = PreferenceExtractionService(
            model=settings.GEMINI_GENERATION_MODEL,
        )

        query_rewriting_service = QueryRewritingService(
            model=settings.GEMINI_GENERATION_MODEL,
        )

        bq_listing_service = BigQueryListingService()

        services: dict[str, Any] = {
            "collection": collection_name,
            "index_path": str(local_index_path),

            # Storage / index
            "index": index_service,

            # Embeddings + vectorstore
            "embedding": embedding_service,
            "retrieval": retrieval_service,

            # LLM services
            "generation": generation_service,
            "preference": preference_service,
            "qrewrite": query_rewriting_service,

            # Lazy: se crea solo si /ask pide reranking.
            "reranking": None,

            # BigQuery enrichment
            "bq_listing": bq_listing_service,
        }

        _services_cache[collection_name] = services

        logger.info(
            "service_cache_collection_ready",
            extra={
                "collection": collection_name,
                "index_path": str(local_index_path),
                "services": sorted(services.keys()),
            },
        )

        return services


def ensure_reranking_service(
    services: dict[str, Any],
) -> Any:
    """
    Crea o retorna el RerankingService de forma lazy.

    Se usa desde /ask cuando payload.use_reranking=True.

    Si ENABLE_RERANKING_MODEL=False, el servicio sigue existiendo, pero usará
    fallback léxico. Si ENABLE_RERANKING_MODEL=True, intentará cargar el modelo
    CrossEncoder en lazy load.
    """
    with _cache_lock:
        if services.get("reranking") is not None:
            return services["reranking"]

        from app.services.reranking_service import RerankingService

        logger.info(
            "service_cache_lazy_reranking_initialization",
            extra={
                "model_name": settings.RERANKING_MODEL,
                "enabled": settings.ENABLE_RERANKING_MODEL,
                "top_k": settings.RERANKING_TOP_K,
            },
        )

        services["reranking"] = RerankingService(
            top_k=settings.RERANKING_TOP_K,
            model_name=settings.RERANKING_MODEL,
            enabled=settings.ENABLE_RERANKING_MODEL,
            lazy_load=True,
        )

        return services["reranking"]


def clear_services_cache(
    collection: Optional[str] = None,
) -> None:
    """
    Limpia el cache de servicios.

    Uso:
      - tests;
      - endpoint administrativo futuro;
      - recarga manual tras publicar un nuevo índice.

    Args:
        collection:
            Si se pasa, limpia solo esa colección.
            Si None, limpia todo el cache.
    """
    with _cache_lock:
        if collection:
            collection_name = _resolve_collection(collection)
            _services_cache.pop(collection_name, None)

            logger.info(
                "service_cache_collection_cleared",
                extra={"collection": collection_name},
            )
            return

        _services_cache.clear()

        logger.info("service_cache_cleared")


def reload_collection_services(
    collection: Optional[str] = None,
) -> dict[str, Any]:
    """
    Fuerza recarga completa de servicios para una colección.

    Descarga nuevamente latest desde GCS y recarga FAISS.
    """
    collection_name = _resolve_collection(collection)

    clear_services_cache(collection_name)

    return get_collection_services(
        collection=collection_name,
        force_reload=True,
    )


def get_services_cache_status() -> dict[str, Any]:
    """
    Retorna estado resumido del cache para debugging.
    """
    with _cache_lock:
        return {
            "collections": list(_services_cache.keys()),
            "collections_count": len(_services_cache),
            "details": {
                collection: {
                    "index_path": services.get("index_path"),
                    "has_embedding": services.get("embedding") is not None,
                    "has_retrieval": services.get("retrieval") is not None,
                    "has_generation": services.get("generation") is not None,
                    "has_preference": services.get("preference") is not None,
                    "has_qrewrite": services.get("qrewrite") is not None,
                    "has_reranking": services.get("reranking") is not None,
                    "has_bq_listing": services.get("bq_listing") is not None,
                }
                for collection, services in _services_cache.items()
            },
        }
