from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from app.services.rag_graph_service import RAGGraphService
from app.services.service_cache import get_collection_services
from miad_rag_common.schemas.ask import AskRequest, AskResponse

router = APIRouter(prefix="/api/v1", tags=["Ask"])


def ask_processing(
    question: str,
    collection: str,
    use_reranking: bool = False,
    use_query_rewriting: bool = False,
) -> dict[str, Any]:
    """
    Procesamiento RAG para consultas generales del mercado inmobiliario.

    Mantiene la intención del endpoint original:
    - FAISS se carga desde cache.
    - RAGGraphService se instancia por request porque sus flags varían.
    - Query rewriting y reranking son opcionales.
    """
    start_time = time.time()
    svc = get_collection_services(collection)

    rag_graph_service = RAGGraphService(
        retrieval_service=svc["retrieval"],
        generation_service=svc["generation"],
        query_rewriting_service=svc["qrewrite"],
        reranking_service=svc["reranking"],
        rewriting_strategy="few_shot_rewrite",
        use_query_rewriting=use_query_rewriting,
        use_reranking=use_reranking,
    )

    rag_result = rag_graph_service.process_question(question)

    context = rag_result.get("context", [])
    sources = rag_result.get("sources", [])

    if not context:
        return {
            "question": question,
            "final_query": question,
            "answer": "No tengo información suficiente en la base de datos para responder a esta pregunta.",
            "collection": collection,
            "files_consulted": [],
            "context_docs": [],
            "reranker_used": False,
            "query_rewriting_used": False,
            "response_time_sec": round(time.time() - start_time, 3),
        }

    context_docs = []
    for i, context_text in enumerate(context):
        source_file = sources[i] if i < len(sources) else "unknown"
        context_docs.append(
            {
                "file_name": source_file,
                "chunk_type": "unknown",
                "snippet": context_text[:200],
                "content": context_text,
                "priority": "high" if i == 0 else "medium",
            }
        )

    rewritten_queries = rag_result.get("rewritten_queries", [])

    return {
        "question": question,
        "final_query": rag_result.get("final_query", question),
        "answer": rag_result.get("answer", ""),
        "collection": collection,
        "files_consulted": list(set(sources)),
        "context_docs": context_docs,
        "reranker_used": bool(use_reranking and svc["reranking"]),
        "query_rewriting_used": bool(use_query_rewriting and len(rewritten_queries) > 1),
        "response_time_sec": round(time.time() - start_time, 3),
    }


@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> dict[str, Any]:
    if not payload.question or not payload.question.strip():
        raise HTTPException(
            status_code=400,
            detail="La pregunta es requerida y no puede estar vacía.",
        )

    if not payload.collection or not payload.collection.strip():
        raise HTTPException(
            status_code=400,
            detail="La colección es requerida y no puede estar vacía.",
        )

    try:
        return await asyncio.to_thread(
            ask_processing,
            payload.question.strip(),
            payload.collection.strip(),
            payload.use_reranking,
            payload.use_query_rewriting,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la consulta: {str(exc)}",
        ) from exc
