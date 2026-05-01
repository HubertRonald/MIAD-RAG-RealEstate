from __future__ import annotations

import logging
import re
from typing import Any

from langchain.schema import Document

logger = logging.getLogger(__name__)


class RerankingService:
    """
    Reranker opcional.

    Si enabled=False o sentence-transformers no está disponible,
    usa un fallback léxico simple para no romper el backend.
    """

    def __init__(
        self,
        top_k: int = 3,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enabled: bool = False,
    ) -> None:
        self.top_k = top_k
        self.model_name = model_name
        self.enabled = enabled
        self.model = None

        if enabled:
            try:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(model_name)
            except Exception as exc:
                logger.warning("[RerankingService] modelo no disponible: %s", exc)
                self.model = None
                self.enabled = False

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        if not documents:
            return []

        limit = top_k or self.top_k

        if self.enabled and self.model is not None:
            return self._rerank_with_model(query, documents, limit)

        return self._rerank_lexical(query, documents, limit)

    def _rerank_with_model(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        scored = []

        for doc, score in zip(documents, scores):
            metadata = dict(doc.metadata or {})
            metadata["rerank_score"] = float(score)
            scored.append(
                Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )

        scored.sort(
            key=lambda doc: float(doc.metadata.get("rerank_score", 0.0)),
            reverse=True,
        )

        return scored[:top_k]

    def _rerank_lexical(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        query_terms = self._tokens(query)

        scored: list[tuple[Document, float]] = []

        for doc in documents:
            doc_terms = self._tokens(doc.page_content)
            overlap = len(query_terms & doc_terms)
            score = overlap / max(len(query_terms), 1)

            metadata = dict(doc.metadata or {})
            metadata["rerank_score"] = score

            scored.append(
                (
                    Document(
                        page_content=doc.page_content,
                        metadata=metadata,
                    ),
                    score,
                )
            )

        scored.sort(key=lambda pair: pair[1], reverse=True)

        return [doc for doc, _score in scored[:top_k]]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-záéíóúñü0-9]+", (text or "").lower())
            if len(token) > 2
        }
