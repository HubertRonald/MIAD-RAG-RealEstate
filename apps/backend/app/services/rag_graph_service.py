from __future__ import annotations

from collections import OrderedDict
from typing import Any

from langchain.schema import Document


class RAGGraphService:
    """
    Orquestador RAG para /ask.

    Conserva la interfaz del backend original:
    - retrieval_service
    - generation_service
    - query_rewriting_service
    - reranking_service
    - process_question()
    """

    def __init__(
        self,
        retrieval_service,
        generation_service,
        query_rewriting_service=None,
        reranking_service=None,
        rewriting_strategy: str = "few_shot_rewrite",
        use_query_rewriting: bool = False,
        use_reranking: bool = False,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
        self.query_rewriting_service = query_rewriting_service
        self.reranking_service = reranking_service
        self.rewriting_strategy = rewriting_strategy
        self.use_query_rewriting = use_query_rewriting
        self.use_reranking = use_reranking

    def process_question(self, question: str) -> dict[str, Any]:
        rewritten_queries = self._build_queries(question)

        docs: list[Document] = []

        for query in rewritten_queries:
            docs.extend(self.retrieval_service.retrieve_documents(query))

        docs = self._dedupe_documents(docs)

        if self.use_reranking and self.reranking_service is not None:
            docs = self.reranking_service.rerank(question, docs)

        result = self.generation_service.generate_response(
            question=question,
            retrieved_docs=docs,
        )

        context = [doc.page_content for doc in docs]
        sources = [
            str((doc.metadata or {}).get("source_file") or (doc.metadata or {}).get("source") or "unknown")
            for doc in docs
        ]

        return {
            "answer": result["answer"],
            "context": context,
            "sources": sources,
            "rewritten_queries": rewritten_queries,
            "final_query": rewritten_queries[0] if rewritten_queries else question,
        }

    def _build_queries(self, question: str) -> list[str]:
        queries = [question]

        if not self.use_query_rewriting or self.query_rewriting_service is None:
            return queries

        strategy = getattr(
            self.query_rewriting_service,
            self.rewriting_strategy,
            None,
        )

        if strategy is None:
            return queries

        rewritten = strategy(question)

        if isinstance(rewritten, str):
            rewritten_queries = [rewritten]
        else:
            rewritten_queries = list(rewritten or [])

        for query in rewritten_queries:
            if query and query not in queries:
                queries.append(query)

        return queries

    @staticmethod
    def _dedupe_documents(documents: list[Document]) -> list[Document]:
        """
        Deduplica por id/listing_id si existe; si no, por contenido.
        """
        deduped: OrderedDict[str, Document] = OrderedDict()

        for doc in documents:
            metadata = doc.metadata or {}
            key = (
                metadata.get("id")
                or metadata.get("listing_id")
                or metadata.get("source_file")
                or doc.page_content[:300]
            )

            deduped[str(key)] = doc

        return list(deduped.values())
