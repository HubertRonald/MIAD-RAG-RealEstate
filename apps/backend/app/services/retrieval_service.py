from __future__ import annotations

from typing import Any, Optional

from langchain.schema import Document

from app.services.embedding_service import EmbeddingService
from miad_rag_common.logging.structured_logging import get_logger
from miad_rag_common.schemas.filters import PropertyFilters

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator


logger = get_logger(__name__)


def l2_relevance_to_cosine(relevance_score: float) -> float:
    """
    Convierte un relevance_score de FAISS/LangChain a similitud coseno aproximada.

    LangChain puede normalizar distancia L2 así:

        relevance_score = 1 / (1 + L2_distance)

    Para vectores normalizados:

        cosine_similarity = 1 - (L2_distance² / 2)

    Args:
        relevance_score:
            Score normalizado de LangChain en (0, 1].

    Returns:
        Similitud coseno aproximada en [0, 1].
    """
    if relevance_score is None:
        return 0.0

    if relevance_score <= 0:
        return 0.0

    l2_distance = (1.0 / relevance_score) - 1.0
    cosine = 1.0 - (l2_distance**2) / 2.0

    return max(0.0, min(1.0, cosine))


class RetrievalService:
    """
    Servicio de recuperación semántica sobre FAISS.

    Responsabilidades:
      - Ejecutar búsqueda semántica pura.
      - Ejecutar búsqueda semántica con filtros estructurados.
      - Ejecutar búsqueda con scores de relevancia.
      - Exponer logs explícitos para depuración.

    Este servicio vive solo en backend.

    No construye índices.
    No lee BigQuery.
    No descarga desde GCS.
    No genera respuestas con LLM.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        k: int = 10,
        fetch_k: Optional[int] = None,
    ) -> None:
        """
        Inicializa el servicio con un vectorstore FAISS ya cargado.

        Args:
            embedding_service:
                Servicio de embeddings del backend. Debe tener FAISS cargado.
            k:
                Número final de documentos a recuperar.
            fetch_k:
                Número de candidatos previos cuando se usan filtros.
                Si no se especifica, usa k * 20.
        """
        self.embedding_service = embedding_service
        self.vectorstore = embedding_service.get_vectorstore()

        if self.vectorstore is None:
            raise ValueError(
                "El embedding_service debe tener un vectorstore cargado. "
                "Ejecuta embedding_service.load_vectorstore() primero."
            )

        self.k = int(k)
        self.fetch_k = int(fetch_k) if fetch_k is not None else self.k * 20
        self.retriever = self._build_retriever()

        logger.info(
            "retrieval_service_initialized",
            extra={
                "k": self.k,
                "fetch_k": self.fetch_k,
                "vectorstore_type": type(self.vectorstore).__name__,
            },
        )

    def _build_retriever(self):
        """
        Construye retriever base sin filtros.
        """
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.k,
            },
        )

    @staticmethod
    def _validate_query(query: str) -> str:
        """
        Valida y normaliza la consulta.
        """
        normalized_query = (query or "").strip()

        if not normalized_query:
            raise ValueError("La consulta de retrieval no puede estar vacía.")

        return normalized_query

    @staticmethod
    def _filters_are_empty(filters: Optional[PropertyFilters]) -> bool:
        """
        Evalúa si no hay filtros activos.
        """
        return filters is None or filters.is_empty()

    @staticmethod
    def _filters_summary(filters: Optional[PropertyFilters]) -> dict[str, Any]:
        """
        Serializa filtros activos para logging.

        Compatible con PropertyFilters.active_dict() si existe.
        """
        if filters is None:
            return {}

        if hasattr(filters, "active_dict"):
            try:
                return filters.active_dict()
            except Exception:
                pass

        result: dict[str, Any] = {}

        candidate_fields = [
            "operation_type",
            "property_type",
            "barrio",
            "min_price",
            "max_price",
            "max_price_m2",
            "min_bedrooms",
            "max_bedrooms",
            "min_surface",
            "max_surface",
            "max_dist_plaza",
            "max_dist_playa",
            "has_pool",
            "has_gym",
            "has_elevator",
            "has_parrillero",
            "has_terrace",
            "has_rooftop",
            "has_security",
            "has_storage",
            "has_parking",
            "has_party_room",
            "has_green_area",
            "has_playground",
            "has_visitor_parking",
        ]

        for field_name in candidate_fields:
            value = getattr(filters, field_name, None)

            if value is None:
                continue

            if isinstance(value, bool) and value is False:
                continue

            result[field_name] = value

        return result

    def _build_search_kwargs(
        self,
        filters: Optional[PropertyFilters] = None,
        include_filter: bool = True,
    ) -> dict[str, Any]:
        """
        Construye search_kwargs para FAISS.

        Si hay filtros activos:
          - usa k;
          - usa fetch_k para ampliar candidatos;
          - usa filter callable generado por PropertyFilters.

        Si no hay filtros:
          - usa únicamente k.
        """
        search_kwargs: dict[str, Any] = {
            "k": self.k,
        }

        if include_filter and not self._filters_are_empty(filters):
            filter_fn = filters.to_filter_fn() if filters else None

            if filter_fn is not None:
                search_kwargs["fetch_k"] = self.fetch_k
                search_kwargs["filter"] = filter_fn

        return search_kwargs

    def _estimate_query_cost(self, query: str) -> dict[str, Any]:
        """
        Estima costo de embedding de la query si el EmbeddingService lo soporta.

        Esto es solo para logging operativo.
        """
        if hasattr(self.embedding_service, "estimate_query_cost"):
            try:
                return self.embedding_service.estimate_query_cost(query)
            except Exception as exc:
                logger.warning(
                    "retrieval_query_cost_estimation_failed",
                    extra={"error": str(exc)},
                )

        return {}

    @staticmethod
    def _extract_doc_ids(documents: list[Document]) -> list[str]:
        """
        Extrae IDs o sources para logs de depuración.
        """
        ids: list[str] = []

        for doc in documents:
            metadata = doc.metadata or {}
            value = (
                metadata.get("id")
                or metadata.get("listing_id")
                or metadata.get("property_id")
                or metadata.get("source")
                or metadata.get("source_file")
            )

            if value is not None:
                ids.append(str(value))

        return ids

    @staticmethod
    def average_relevance_score(
        scored_docs: list[tuple[Document, float]],
    ) -> Optional[float]:
        """
        Promedio de scores crudos devueltos por LangChain/FAISS.
        """
        if not scored_docs:
            return None

        scores = [score for _, score in scored_docs if score is not None]

        if not scores:
            return None

        return sum(scores) / len(scores)

    @staticmethod
    def average_cosine_similarity(
        scored_docs: list[tuple[Document, float]],
    ) -> Optional[float]:
        """
        Promedio de similitud coseno aproximada a partir de relevance_score.

        Útil para evaluación offline o logging tipo MLflow/RAGAS.
        """
        if not scored_docs:
            return None

        cosine_scores = [
            l2_relevance_to_cosine(score)
            for _, score in scored_docs
            if score is not None
        ]

        if not cosine_scores:
            return None

        return sum(cosine_scores) / len(cosine_scores)

    # =========================================================================
    # Retrieval público
    # =========================================================================

    @traceable(name="faiss_retrieval_plain")
    def retrieve_documents(self, query: str) -> list[Document]:
        """
        Recupera documentos por similitud semántica pura.

        No aplica filtros estructurados.

        Args:
            query:
                Consulta del usuario.

        Returns:
            Lista de hasta k documentos.
        """
        normalized_query = self._validate_query(query)
        query_cost = self._estimate_query_cost(normalized_query)

        logger.info(
            "retrieval_plain_started",
            extra={
                "query_length": len(normalized_query),
                "k": self.k,
                "query_cost_estimate": query_cost,
            },
        )

        documents = self.retriever.invoke(normalized_query)

        logger.info(
            "retrieval_plain_completed",
            extra={
                "documents_count": len(documents),
                "document_ids": self._extract_doc_ids(documents),
            },
        )

        return documents

    @traceable(name="faiss_retrieval_filtered")
    def retrieve_with_filters(
        self,
        query: str,
        filters: Optional[PropertyFilters] = None,
    ) -> list[Document]:
        """
        Recupera documentos combinando filtros estructurados + similitud semántica.

        Flujo:
          1. Si no hay filtros, usa retrieve_documents().
          2. Si hay filtros, convierte PropertyFilters en callable.
          3. FAISS aplica el filtro sobre metadata.
          4. Retorna top-k documentos más similares entre los que pasan el filtro.

        Args:
            query:
                Consulta del usuario.
            filters:
                Filtros estructurados opcionales.

        Returns:
            Lista de hasta k documentos.
        """
        normalized_query = self._validate_query(query)

        if self._filters_are_empty(filters):
            logger.info(
                "retrieval_filtered_without_filters_delegating",
                extra={
                    "query_length": len(normalized_query),
                    "k": self.k,
                },
            )
            return self.retrieve_documents(normalized_query)

        search_kwargs = self._build_search_kwargs(
            filters=filters,
            include_filter=True,
        )

        logger.info(
            "retrieval_filtered_started",
            extra={
                "query_length": len(normalized_query),
                "k": self.k,
                "fetch_k": self.fetch_k,
                "filters": self._filters_summary(filters),
                "search_kwargs_keys": list(search_kwargs.keys()),
            },
        )

        filtered_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

        documents = filtered_retriever.invoke(normalized_query)

        logger.info(
            "retrieval_filtered_completed",
            extra={
                "documents_count": len(documents),
                "document_ids": self._extract_doc_ids(documents),
                "filters": self._filters_summary(filters),
            },
        )

        return documents

    @traceable(name="faiss_retrieval_with_scores")
    def retrieve_with_scores(
        self,
        query: str,
        filters: Optional[PropertyFilters] = None,
    ) -> list[tuple[Document, float]]:
        """
        Recupera documentos junto con scores de relevancia.

        Usado por /recommend para:
          - construir listings_used con semantic_score;
          - calcular match_score en el router;
          - alimentar análisis de calidad de retrieval.

        Args:
            query:
                Consulta del usuario.
            filters:
                Filtros estructurados opcionales.

        Returns:
            Lista de tuplas (Document, relevance_score).
        """
        normalized_query = self._validate_query(query)

        search_kwargs = self._build_search_kwargs(
            filters=filters,
            include_filter=True,
        )

        logger.info(
            "retrieval_with_scores_started",
            extra={
                "query_length": len(normalized_query),
                "k": self.k,
                "fetch_k": self.fetch_k
                if not self._filters_are_empty(filters)
                else None,
                "filters": self._filters_summary(filters),
                "search_kwargs_keys": list(search_kwargs.keys()),
            },
        )

        scored_docs = self.vectorstore.similarity_search_with_relevance_scores(
            normalized_query,
            **search_kwargs,
        )

        documents = [doc for doc, _score in scored_docs]
        relevance_scores = [score for _doc, score in scored_docs]

        avg_relevance = self.average_relevance_score(scored_docs)
        avg_cosine = self.average_cosine_similarity(scored_docs)

        logger.info(
            "retrieval_with_scores_completed",
            extra={
                "documents_count": len(scored_docs),
                "document_ids": self._extract_doc_ids(documents),
                "relevance_scores": relevance_scores,
                "avg_relevance_score": avg_relevance,
                "avg_cosine_similarity": avg_cosine,
                "filters": self._filters_summary(filters),
            },
        )

        return scored_docs
