from __future__ import annotations

from typing import Any

from langchain.schema import Document

from app.services.embedding_service import EmbeddingService
from miad_rag_common.schemas.filters import PropertyFilters


def l2_relevance_to_cosine(relevance_score: float) -> float:
    """
    Convierte el relevance_score de FAISS/LangChain a similitud coseno aproximada.

    LangChain puede normalizar distancia L2 así:
      relevance_score = 1 / (1 + L2_distance)

    Para vectores normalizados:
      cosine_similarity = 1 - (L2_distance² / 2)
    """
    if relevance_score <= 0:
        return 0.0

    l2_distance = (1.0 / relevance_score) - 1.0
    cosine = 1.0 - (l2_distance**2) / 2.0

    return max(0.0, min(1.0, cosine))


class RetrievalService:
    """
    Recupera documentos desde FAISS.

    Usa PropertyFilters desde shared para no duplicar lógica entre backend
    y job-indexer.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        k: int = 10,
        fetch_k: int | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vectorstore = embedding_service.get_vectorstore()

        if self.vectorstore is None:
            raise ValueError(
                "El embedding_service debe tener un vectorstore cargado. "
                "Ejecuta embedding_service.load_vectorstore() primero."
            )

        self.k = k
        self.fetch_k = fetch_k if fetch_k is not None else k * 20
        self.retriever = self._build_retriever()

    def _build_retriever(self):
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k},
        )

    def retrieve_documents(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def retrieve_with_filters(
        self,
        query: str,
        filters: PropertyFilters | None = None,
    ) -> list[Document]:
        if filters is None or filters.is_empty():
            return self.retrieve_documents(query)

        filter_fn = filters.to_filter_fn()

        filtered_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.k,
                "fetch_k": self.fetch_k,
                "filter": filter_fn,
            },
        )

        return filtered_retriever.invoke(query)

    def retrieve_with_scores(
        self,
        query: str,
        filters: PropertyFilters | None = None,
    ) -> list[tuple[Document, float]]:
        search_kwargs: dict[str, Any] = {"k": self.k}

        if filters is not None and not filters.is_empty():
            search_kwargs["fetch_k"] = self.fetch_k
            search_kwargs["filter"] = filters.to_filter_fn()

        return self.vectorstore.similarity_search_with_relevance_scores(
            query,
            **search_kwargs,
        )
