from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional, TypedDict

from langchain.schema import Document
from langgraph.graph import END, StateGraph

from miad_rag_common.logging.structured_logging import get_logger

if TYPE_CHECKING:
    from app.services.generation_service import GenerationService
    from app.services.query_rewriting_service import QueryRewritingService
    from app.services.reranking_service import RerankingService
    from app.services.retrieval_service import RetrievalService


logger = get_logger(__name__)


class RAGState(TypedDict):
    """
    Estado compartido del flujo RAG para /ask.

    question:
        Pregunta original del usuario.

    rewritten_queries:
        Lista de queries usadas para retrieval.
        Incluye la pregunta original y, si aplica, reformulaciones.

    documents:
        Documents recuperados desde FAISS, antes o después de reranking.

    sources:
        Fuentes asociadas a los documentos usados.

    context:
        Texto final enviado/devuelto como contexto.

    answer:
        Respuesta generada por el LLM.
    """

    question: str
    rewritten_queries: list[str]
    documents: list[Document]
    sources: list[str]
    context: list[str]
    answer: str


class RAGGraphService:
    """
    Orquestador RAG para /ask usando LangGraph.

    Flujo posible:

      Sin rewriting ni reranking:
        retrieve_documents -> generate_answer

      Con rewriting:
        rewrite_query -> retrieve_documents -> generate_answer

      Con reranking:
        retrieve_documents -> rerank_documents -> generate_answer

      Con ambos:
        rewrite_query -> retrieve_documents -> rerank_documents -> generate_answer

    Este servicio solo orquesta.

    No carga FAISS.
    No consulta BigQuery.
    No crea servicios.
    No construye embeddings.
    """

    def __init__(
        self,
        retrieval_service: "RetrievalService",
        generation_service: "GenerationService",
        query_rewriting_service: Optional["QueryRewritingService"] = None,
        reranking_service: Optional["RerankingService"] = None,
        rewriting_strategy: str = "few_shot_rewrite",
        use_query_rewriting: bool = False,
        use_reranking: bool = False,
        rewrite_num_queries: int = 3,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
        self.query_rewriting_service = query_rewriting_service
        self.reranking_service = reranking_service

        self.rewriting_strategy = rewriting_strategy
        self.use_query_rewriting = use_query_rewriting
        self.use_reranking = use_reranking
        self.rewrite_num_queries = rewrite_num_queries

        self.rewriting_enabled = (
            self.use_query_rewriting
            and self.query_rewriting_service is not None
            and bool(self.rewriting_strategy)
        )

        self.reranking_enabled = (
            self.use_reranking
            and self.reranking_service is not None
        )

        self.rag_app = self._build_graph()

        logger.info(
            "rag_graph_service_initialized",
            extra={
                "rewriting_enabled": self.rewriting_enabled,
                "reranking_enabled": self.reranking_enabled,
                "rewriting_strategy": self.rewriting_strategy,
                "rewrite_num_queries": self.rewrite_num_queries,
            },
        )

    # =========================================================================
    # Nodo 1: rewrite_query
    # =========================================================================

    def _rewrite_query_node(self, state: RAGState) -> RAGState:
        question = state["question"]

        logger.info(
            "rag_rewrite_query_started",
            extra={
                "question_length": len(question or ""),
                "strategy": self.rewriting_strategy,
                "num_queries": self.rewrite_num_queries,
            },
        )

        try:
            rewritten_queries = self._build_queries(question)

        except Exception as exc:
            logger.warning(
                "rag_rewrite_query_failed_using_original",
                extra={
                    "error": str(exc),
                    "strategy": self.rewriting_strategy,
                },
            )
            rewritten_queries = [question]

        state["rewritten_queries"] = rewritten_queries

        logger.info(
            "rag_rewrite_query_completed",
            extra={
                "queries_count": len(rewritten_queries),
                "queries": rewritten_queries,
            },
        )

        return state

    # =========================================================================
    # Nodo 2: retrieve_documents
    # =========================================================================

    def _retrieve_documents_node(self, state: RAGState) -> RAGState:
        queries = state.get("rewritten_queries") or [state["question"]]

        logger.info(
            "rag_retrieve_documents_started",
            extra={
                "queries_count": len(queries),
                "queries": queries,
            },
        )

        all_documents: list[Document] = []

        for query in queries:
            try:
                docs = self.retrieval_service.retrieve_documents(query)
                all_documents.extend(docs)

                logger.info(
                    "rag_retrieve_documents_query_completed",
                    extra={
                        "query": query,
                        "documents_count": len(docs),
                    },
                )

            except Exception as exc:
                logger.warning(
                    "rag_retrieve_documents_query_failed",
                    extra={
                        "query": query,
                        "error": str(exc),
                    },
                )

        deduped_documents = self._dedupe_documents(all_documents)

        state["documents"] = deduped_documents

        logger.info(
            "rag_retrieve_documents_completed",
            extra={
                "raw_documents_count": len(all_documents),
                "deduped_documents_count": len(deduped_documents),
                "document_ids": self._extract_doc_ids(deduped_documents),
            },
        )

        return state

    # =========================================================================
    # Nodo 3: rerank_documents
    # =========================================================================

    def _rerank_documents_node(self, state: RAGState) -> RAGState:
        question = state["question"]
        documents = state.get("documents") or []

        if not documents:
            logger.info("rag_rerank_documents_skipped_empty_documents")
            return state

        if self.reranking_service is None:
            logger.info("rag_rerank_documents_skipped_no_service")
            return state

        logger.info(
            "rag_rerank_documents_started",
            extra={
                "documents_count": len(documents),
            },
        )

        try:
            if hasattr(self.reranking_service, "rerank"):
                reranked_documents = self.reranking_service.rerank(
                    query=question,
                    documents=documents,
                )
            else:
                reranked_documents = self.reranking_service.rerank_documents(
                    query=question,
                    documents=documents,
                )

            state["documents"] = reranked_documents

            logger.info(
                "rag_rerank_documents_completed",
                extra={
                    "input_documents_count": len(documents),
                    "reranked_documents_count": len(reranked_documents),
                    "document_ids": self._extract_doc_ids(reranked_documents),
                    "rerank_scores": [
                        (doc.metadata or {}).get("rerank_score")
                        for doc in reranked_documents
                    ],
                },
            )

        except Exception as exc:
            logger.warning(
                "rag_rerank_documents_failed_using_original_order",
                extra={
                    "error": str(exc),
                    "documents_count": len(documents),
                },
            )

        return state

    # =========================================================================
    # Nodo 4: generate_answer
    # =========================================================================

    def _generate_answer_node(self, state: RAGState) -> RAGState:
        question = state["question"]
        documents = state.get("documents") or []

        logger.info(
            "rag_generate_answer_started",
            extra={
                "documents_count": len(documents),
                "question_length": len(question or ""),
            },
        )

        result = self.generation_service.generate_response(
            question=question,
            retrieved_docs=documents,
        )

        context = result.get("context")
        if context is None:
            context = [doc.page_content for doc in documents]

        sources = result.get("sources")
        if sources is None:
            sources = self._extract_sources(documents)

        state["answer"] = result.get("answer", "")
        state["sources"] = sources
        state["context"] = context

        logger.info(
            "rag_generate_answer_completed",
            extra={
                "answer_length": len(state["answer"] or ""),
                "sources_count": len(state["sources"]),
                "context_count": len(state["context"]),
            },
        )

        return state

    # =========================================================================
    # Construcción del grafo
    # =========================================================================

    def _build_graph(self):
        graph = StateGraph(RAGState)

        if self.rewriting_enabled:
            graph.add_node("rewrite_query", self._rewrite_query_node)

        graph.add_node("retrieve_documents", self._retrieve_documents_node)

        if self.reranking_enabled:
            graph.add_node("rerank_documents", self._rerank_documents_node)

        graph.add_node("generate_answer", self._generate_answer_node)

        if self.rewriting_enabled:
            graph.set_entry_point("rewrite_query")
            graph.add_edge("rewrite_query", "retrieve_documents")
        else:
            graph.set_entry_point("retrieve_documents")

        if self.reranking_enabled:
            graph.add_edge("retrieve_documents", "rerank_documents")
            graph.add_edge("rerank_documents", "generate_answer")
        else:
            graph.add_edge("retrieve_documents", "generate_answer")

        graph.add_edge("generate_answer", END)

        return graph.compile()

    # =========================================================================
    # API pública
    # =========================================================================

    def process_question(self, question: str) -> dict[str, Any]:
        normalized_question = (question or "").strip()

        if not normalized_question:
            raise ValueError("La pregunta no puede estar vacía.")

        initial_state: RAGState = {
            "question": normalized_question,
            "rewritten_queries": [normalized_question]
            if not self.rewriting_enabled
            else [],
            "documents": [],
            "sources": [],
            "context": [],
            "answer": "",
        }

        logger.info(
            "rag_process_question_started",
            extra={
                "question_length": len(normalized_question),
                "rewriting_enabled": self.rewriting_enabled,
                "reranking_enabled": self.reranking_enabled,
            },
        )

        result = self.rag_app.invoke(initial_state)

        rewritten_queries = result.get("rewritten_queries") or [normalized_question]
        final_query = rewritten_queries[0] if rewritten_queries else normalized_question

        response = {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "context": result.get("context", []),
            "question": result.get("question", normalized_question),
            "rewritten_queries": rewritten_queries,
            "final_query": final_query,
            "documents_count": len(result.get("documents", [])),
            "context_count": len(result.get("context", [])),
        }

        logger.info(
            "rag_process_question_completed",
            extra={
                "answer_length": len(response["answer"] or ""),
                "sources_count": len(response["sources"]),
                "context_count": response["context_count"],
                "rewritten_queries_count": len(rewritten_queries),
            },
        )

        return response

    # =========================================================================
    # Helpers
    # =========================================================================

    def _build_queries(self, question: str) -> list[str]:
        """
        Construye la lista de queries para retrieval.

        Preferencia:
          1. Si QueryRewritingService tiene rewrite(), usa esa API.
          2. Si no, busca el método indicado por rewriting_strategy.
          3. Si falla, usa solo la pregunta original.
        """
        normalized_question = (question or "").strip()

        if not normalized_question:
            raise ValueError("La pregunta no puede estar vacía.")

        if not self.rewriting_enabled:
            return [normalized_question]

        service = self.query_rewriting_service

        if service is None:
            return [normalized_question]

        if hasattr(service, "rewrite"):
            queries = service.rewrite(
                query=normalized_question,
                strategy=self.rewriting_strategy,
                num_queries=self.rewrite_num_queries,
                include_original=True,
            )
            return self._dedupe_strings(queries)

        strategy = getattr(service, self.rewriting_strategy, None)

        if strategy is None:
            logger.warning(
                "rag_rewrite_strategy_not_found_using_original",
                extra={"strategy": self.rewriting_strategy},
            )
            return [normalized_question]

        result = strategy(normalized_question)

        if isinstance(result, str):
            generated = [result]
        else:
            generated = list(result or [])

        return self._dedupe_strings([normalized_question, *generated])

    @staticmethod
    def _dedupe_documents(documents: list[Document]) -> list[Document]:
        """
        Deduplica documentos preservando el primer resultado.

        Prioridad de clave:
          - id
          - listing_id
          - property_id
          - source
          - source_file
          - primeros 300 caracteres del contenido
        """
        deduped: OrderedDict[str, Document] = OrderedDict()

        for doc in documents:
            if doc is None:
                continue

            metadata = doc.metadata or {}

            key = (
                metadata.get("id")
                or metadata.get("listing_id")
                or metadata.get("property_id")
                or metadata.get("source")
                or metadata.get("source_file")
                or (doc.page_content or "")[:300]
            )

            if key is None:
                continue

            key_str = str(key)

            if key_str not in deduped:
                deduped[key_str] = doc

        return list(deduped.values())

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = (value or "").strip()

            if not normalized:
                continue

            key = normalized.lower()

            if key not in seen:
                seen.add(key)
                result.append(normalized)

        return result

    @staticmethod
    def _extract_sources(documents: list[Document]) -> list[str]:
        sources: list[str] = []

        for doc in documents:
            metadata = doc.metadata or {}

            source = (
                metadata.get("source_file")
                or metadata.get("source")
                or metadata.get("id")
                or metadata.get("listing_id")
                or metadata.get("property_id")
                or "unknown"
            )

            sources.append(str(source))

        return sources

    @staticmethod
    def _extract_doc_ids(documents: list[Document]) -> list[str]:
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
