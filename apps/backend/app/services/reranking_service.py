from __future__ import annotations

import re
from typing import Any, Optional

from langchain.schema import Document

from app.config.runtime import get_settings
from miad_rag_common.logging.structured_logging import get_logger

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator


settings = get_settings()
logger = get_logger(__name__)


class RerankingService:
    """
    Reranker opcional para el backend RAG inmobiliario.

    Responsabilidades:
      - Recibir documentos recuperados por FAISS.
      - Reordenarlos según relevancia frente a la query original.
      - Escribir rerank_score en metadata.
      - Retornar top-k documentos reordenados.
      - Ofrecer fallback léxico si el modelo CrossEncoder no está habilitado
        o no está disponible.

    Este servicio vive solo en backend.

    No construye índices.
    No descarga FAISS.
    No consulta BigQuery.
    No se usa en job-indexer.
    """

    def __init__(
        self,
        top_k: Optional[int] = None,
        model_name: Optional[str] = None,
        enabled: Optional[bool] = None,
        lazy_load: bool = True,
    ) -> None:
        """
        Inicializa el servicio de reranking.

        Args:
            top_k:
                Número de documentos a retornar después del reranking.
                Si None, usa settings.RERANKING_TOP_K.

            model_name:
                Modelo CrossEncoder de sentence-transformers.
                Si None, usa settings.RERANKING_MODEL.

            enabled:
                Si True, intenta usar CrossEncoder.
                Si False, usa fallback léxico.
                Si None, usa settings.ENABLE_RERANKING_MODEL.

            lazy_load:
                Si True, carga el CrossEncoder solo en el primer uso.
                Recomendado para Cloud Run.
        """
        self.top_k = int(top_k if top_k is not None else settings.RERANKING_TOP_K)
        self.model_name = model_name or settings.RERANKING_MODEL
        self.enabled = (
            bool(settings.ENABLE_RERANKING_MODEL)
            if enabled is None
            else bool(enabled)
        )
        self.lazy_load = lazy_load

        self.model = None
        self.model_available = False

        logger.info(
            "reranking_service_initialized",
            extra={
                "enabled": self.enabled,
                "lazy_load": self.lazy_load,
                "model_name": self.model_name,
                "top_k": self.top_k,
            },
        )

        if self.enabled and not self.lazy_load:
            self._load_model()

    # =========================================================================
    # Carga de modelo
    # =========================================================================

    def _load_model(self) -> None:
        """
        Carga CrossEncoder si el reranking está habilitado.

        En Cloud Run esto puede aumentar:
          - tiempo de cold start;
          - memoria;
          - tamaño efectivo de dependencias;
          - descarga inicial desde HuggingFace si no hay caché.

        Por eso se recomienda lazy_load=True.
        """
        if not self.enabled:
            return

        if self.model is not None:
            self.model_available = True
            return

        try:
            from sentence_transformers import CrossEncoder

            logger.info(
                "reranking_cross_encoder_loading",
                extra={"model_name": self.model_name},
            )

            self.model = CrossEncoder(self.model_name)
            self.model_available = True

            logger.info(
                "reranking_cross_encoder_loaded",
                extra={"model_name": self.model_name},
            )

        except Exception as exc:
            logger.warning(
                "reranking_cross_encoder_unavailable_using_lexical_fallback",
                extra={
                    "model_name": self.model_name,
                    "error": str(exc),
                },
            )

            self.model = None
            self.model_available = False
            self.enabled = False

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _validate_query(query: str) -> str:
        normalized_query = (query or "").strip()

        if not normalized_query:
            raise ValueError("La pregunta no puede estar vacía.")

        return normalized_query

    @staticmethod
    def _validate_documents(documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        return [
            document
            for document in documents
            if document is not None and document.page_content
        ]

    @staticmethod
    def _copy_document_with_metadata(
        document: Document,
        extra_metadata: dict[str, Any],
    ) -> Document:
        metadata = dict(document.metadata or {})
        metadata.update(extra_metadata)

        return Document(
            page_content=document.page_content,
            metadata=metadata,
        )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        """
        Tokenizador simple para fallback léxico.

        Mantiene letras españolas, números y descarta tokens muy cortos.
        """
        return {
            token
            for token in re.findall(
                r"[a-záéíóúñü0-9]+",
                (text or "").lower(),
            )
            if len(token) > 2
        }

    @staticmethod
    def _source_from_doc(document: Document) -> str:
        metadata = document.metadata or {}

        source = (
            metadata.get("source")
            or metadata.get("source_file")
            or metadata.get("id")
            or metadata.get("listing_id")
            or metadata.get("property_id")
            or "unknown"
        )

        return str(source)

    @staticmethod
    def _preview(text: str, max_chars: int = 160) -> str:
        value = (text or "").replace("\n", " ").strip()

        if len(value) <= max_chars:
            return value

        return value[:max_chars] + "..."

    # =========================================================================
    # API principal compatible con backend actual
    # =========================================================================

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """
        Alias práctico para RAGGraphService.

        Mantiene compatibilidad con el backend migrado, donde se invoca
        reranking_service.rerank(...).
        """
        return self.rerank_documents(
            query=query,
            documents=documents,
            top_k=top_k,
        )

    @traceable(
        name="rerank_documents",
        run_type="llm",
        metadata={"strategy": "cross-encoder-or-lexical-fallback"},
    )
    def rerank_documents(
        self,
        query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """
        Reordena documentos por relevancia.

        Si CrossEncoder está habilitado y disponible:
          - usa modelo de sentence-transformers.

        Si no:
          - usa fallback léxico por overlap de tokens.

        Args:
            query:
                Pregunta original del usuario.

            documents:
                Documentos recuperados por FAISS.

            top_k:
                Número de documentos finales.

        Returns:
            Lista de Document con metadata["rerank_score"].
        """
        normalized_query = self._validate_query(query)
        valid_documents = self._validate_documents(documents)

        if not valid_documents:
            logger.info(
                "reranking_skipped_empty_documents",
                extra={"query_length": len(normalized_query)},
            )
            return []

        effective_top_k = int(top_k if top_k is not None else self.top_k)
        effective_top_k = max(1, min(effective_top_k, len(valid_documents)))

        logger.info(
            "reranking_started",
            extra={
                "documents_count": len(valid_documents),
                "top_k": effective_top_k,
                "enabled": self.enabled,
                "model_available": self.model_available,
                "model_name": self.model_name,
            },
        )

        if self.enabled:
            self._load_model()

        if self.enabled and self.model is not None:
            reranked = self._rerank_with_model(
                query=normalized_query,
                documents=valid_documents,
                top_k=effective_top_k,
            )
            strategy = "cross_encoder"
        else:
            reranked = self._rerank_lexical(
                query=normalized_query,
                documents=valid_documents,
                top_k=effective_top_k,
            )
            strategy = "lexical_fallback"

        logger.info(
            "reranking_completed",
            extra={
                "strategy": strategy,
                "returned_count": len(reranked),
                "top_scores": [
                    doc.metadata.get("rerank_score")
                    for doc in reranked
                ],
                "sources": [
                    self._source_from_doc(doc)
                    for doc in reranked
                ],
            },
        )

        return reranked

    # =========================================================================
    # CrossEncoder
    # =========================================================================

    def _rerank_with_model(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        """
        Reranking con CrossEncoder.

        Evalúa pares:
          (query, document.page_content)
        """
        if self.model is None:
            return self._rerank_lexical(
                query=query,
                documents=documents,
                top_k=top_k,
            )

        pairs = [
            (query, document.page_content)
            for document in documents
        ]

        scores = self.model.predict(pairs)

        scored_documents: list[tuple[float, int, Document]] = []

        for original_rank, (score, document) in enumerate(zip(scores, documents)):
            rerank_score = round(float(score), 4)

            enriched_doc = self._copy_document_with_metadata(
                document=document,
                extra_metadata={
                    "rerank_score": rerank_score,
                    "rerank_strategy": "cross_encoder",
                    "rerank_model": self.model_name,
                    "original_rank": original_rank + 1,
                },
            )

            scored_documents.append(
                (
                    rerank_score,
                    original_rank,
                    enriched_doc,
                )
            )

        scored_documents.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        reranked: list[Document] = []

        for reranked_rank, (_score, _original_rank, document) in enumerate(
            scored_documents[:top_k],
            start=1,
        ):
            metadata = dict(document.metadata or {})
            metadata["reranked_rank"] = reranked_rank

            reranked.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        return reranked

    # =========================================================================
    # Fallback léxico
    # =========================================================================

    def _rerank_lexical(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        """
        Fallback liviano por overlap de tokens.

        No reemplaza un CrossEncoder, pero evita romper el backend si:
          - sentence-transformers no está instalado;
          - el modelo no se puede descargar;
          - ENABLE_RERANKING_MODEL=False.
        """
        query_terms = self._tokens(query)

        scored_documents: list[tuple[float, int, Document]] = []

        for original_rank, document in enumerate(documents):
            document_terms = self._tokens(document.page_content)

            overlap = len(query_terms & document_terms)
            denominator = max(len(query_terms), 1)
            score = round(overlap / denominator, 4)

            enriched_doc = self._copy_document_with_metadata(
                document=document,
                extra_metadata={
                    "rerank_score": score,
                    "rerank_strategy": "lexical_fallback",
                    "rerank_model": None,
                    "original_rank": original_rank + 1,
                },
            )

            scored_documents.append(
                (
                    score,
                    original_rank,
                    enriched_doc,
                )
            )

        scored_documents.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        reranked: list[Document] = []

        for reranked_rank, (_score, _original_rank, document) in enumerate(
            scored_documents[:top_k],
            start=1,
        ):
            metadata = dict(document.metadata or {})
            metadata["reranked_rank"] = reranked_rank

            reranked.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        return reranked

    # =========================================================================
    # Debug / análisis
    # =========================================================================

    @traceable(
        name="rerank_with_metadata",
        run_type="llm",
        metadata={"strategy": "debug"},
    )
    def rerank_with_metadata(
        self,
        query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Reordena documentos e incluye información detallada para debugging.

        Returns:
            Lista de dicts con:
              - document
              - score
              - original_rank
              - reranked_rank
              - source_file
              - content_preview
              - strategy
        """
        normalized_query = self._validate_query(query)
        reranked_docs = self.rerank_documents(
            query=normalized_query,
            documents=documents,
            top_k=top_k,
        )

        records: list[dict[str, Any]] = []

        for reranked_rank, document in enumerate(reranked_docs, start=1):
            metadata = document.metadata or {}

            records.append(
                {
                    "document": document,
                    "score": float(metadata.get("rerank_score", 0.0)),
                    "original_rank": metadata.get("original_rank"),
                    "reranked_rank": reranked_rank,
                    "source_file": metadata.get("source_file", "unknown"),
                    "source": self._source_from_doc(document),
                    "strategy": metadata.get("rerank_strategy"),
                    "model": metadata.get("rerank_model"),
                    "content_preview": self._preview(document.page_content),
                }
            )

        logger.info(
            "rerank_with_metadata_completed",
            extra={
                "records_count": len(records),
                "scores": [record["score"] for record in records],
            },
        )

        return records
