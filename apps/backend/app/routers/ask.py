from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from app.config.runtime import get_settings
from app.services.rag_graph_service import RAGGraphService
from app.services.service_cache import (
    ensure_reranking_service,
    get_collection_services,
)
from miad_rag_common.logging.structured_logging import get_logger
from miad_rag_common.schemas.ask import AskRequest, AskResponse

router = APIRouter(prefix="/api/v1", tags=["Ask"])

settings = get_settings()
logger = get_logger(__name__)


# =============================================================================
# Guardrail de dominio
# =============================================================================

_OFFTOPIC_PATTERNS = [
    "receta",
    "cocina",
    "fútbol",
    "futbol",
    "política",
    "politica",
    "clima",
    "temperatura",
    "programación",
    "programacion",
    "código",
    "codigo",
    "música",
    "musica",
    "película",
    "pelicula",
    "restaurante",
    "médico",
    "medico",
    "doctor",
    "enfermedad",
    "viaje",
    "vuelo",
    "hotel",
]

_REALESTATE_PATTERNS = [
    "apartamento",
    "apto",
    "casa",
    "inmueble",
    "propiedad",
    "alquiler",
    "venta",
    "comprar",
    "arrendar",
    "dormitorio",
    "baño",
    "bano",
    "barrio",
    "m²",
    "m2",
    "metros",
    "precio",
    "piso",
    "ascensor",
    "terraza",
    "cochera",
    "garaje",
    "garage",
    "parrillero",
    "piscina",
    "jardín",
    "jardin",
    "rambla",
    "playa",
    "costa",
    "montevideo",
    "pocitos",
    "carrasco",
    "centro",
    "punta carretas",
    "malvín",
    "malvin",
    "cordón",
    "cordon",
    "buceo",
    "palermo",
    "busco",
    "necesito",
    "quiero",
    "buscando",
    "familia",
    "niños",
    "ninos",
]


def _is_real_estate_query(question: str) -> bool:
    """
    Clasifica si una consulta está dentro del dominio inmobiliario.

    Primero usa reglas rápidas para evitar costo innecesario.
    Si la consulta es ambigua, usa Gemini como fallback permisivo.
    """
    q = (question or "").lower()

    if any(pattern in q for pattern in _OFFTOPIC_PATTERNS):
        return False

    if any(pattern in q for pattern in _REALESTATE_PATTERNS):
        return True

    try:
        import google.generativeai as genai

        result = genai.GenerativeModel("gemini-2.0-flash").generate_content(
            f"""Clasificador binario. Responde ÚNICAMENTE con YES o NO.

¿Esta consulta podría ser de alguien buscando una propiedad para vivir
o entender el mercado inmobiliario de Montevideo?

Ante la duda, responde YES.

Consulta:
{question}
"""
        )

        raw = (result.text or "").strip()

        logger.info(
            "guardrail_llm_fallback",
            extra={
                "question": question,
                "raw_response": raw,
            },
        )

        return raw.upper().startswith("YES")

    except Exception as exc:
        # Fallback permisivo: si el clasificador falla, no bloqueamos la consulta.
        logger.warning(
            "guardrail_llm_fallback_failed_allowing_query",
            extra={
                "question": question,
                "error": str(exc),
            },
        )

        return True


# =============================================================================
# /ask — Market Q&A
# =============================================================================

def ask_processing(
    question: str,
    collection: str,
    use_reranking: bool = False,
    use_query_rewriting: bool = False,
) -> dict[str, Any]:
    """
    Procesamiento RAG para consultas generales del mercado inmobiliario.

    Flujo:
      1. Obtiene servicios cacheados por colección.
      2. Aplica guardrail de dominio.
      3. Carga reranker solo si el request lo pide.
      4. Construye RAGGraphService por request.
      5. Ejecuta recuperación + generación.
      6. Devuelve respuesta compatible con AskResponse.
    """
    start_time = time.time()

    svc = get_collection_services(collection)

    if not _is_real_estate_query(question):
        return {
            "question": question,
            "final_query": question,
            "answer": (
                "Solo puedo responder consultas sobre el mercado inmobiliario "
                "de Montevideo. Intenta de nuevo con una pregunta sobre "
                "propiedades, barrios, precios, alquileres, ventas o amenities."
            ),
            "collection": collection,
            "files_consulted": [],
            "context_docs": [],
            "reranker_used": False,
            "query_rewriting_used": False,
            "response_time_sec": round(time.time() - start_time, 3),
        }

    if use_reranking:
        ensure_reranking_service(svc)

    rag_graph_service = RAGGraphService(
        retrieval_service=svc["retrieval"],
        generation_service=svc["generation"],
        query_rewriting_service=svc["qrewrite"],
        reranking_service=svc.get("reranking"),
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
            "answer": (
                "No tengo información suficiente en la base de datos para "
                "responder a esta pregunta."
            ),
            "collection": collection,
            "files_consulted": [],
            "context_docs": [],
            "reranker_used": False,
            "query_rewriting_used": False,
            "response_time_sec": round(time.time() - start_time, 3),
        }

    rewritten_queries = rag_result.get("rewritten_queries", [])
    reranker_used = bool(use_reranking and svc.get("reranking") is not None)
    query_rewriting_used = bool(use_query_rewriting and len(rewritten_queries) > 1)

    context_docs = []

    for index, context_text in enumerate(context):
        source_file = sources[index] if index < len(sources) else "unknown"

        context_docs.append(
            {
                "file_name": source_file,
                "chunk_type": "unknown",
                "snippet": context_text[:200],
                "content": context_text,
                "priority": "high" if index == 0 else "medium",
            }
        )

    files_consulted = list(set(sources))

    return {
        "question": question,
        "final_query": rag_result.get("final_query", question),
        "answer": rag_result.get("answer", ""),
        "collection": collection,
        "files_consulted": files_consulted,
        "context_docs": context_docs,
        "reranker_used": reranker_used,
        "query_rewriting_used": query_rewriting_used,
        "response_time_sec": round(time.time() - start_time, 3),
    }


@router.post(
    "/ask",
    response_model=AskResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "precios_zona": {
                            "summary": "Consulta de mercado — precios por zona",
                            "value": {
                                "question": (
                                    "¿Qué diferencia hay entre Pocitos y Punta "
                                    "Carretas en términos de oferta?"
                                ),
                                "collection": "realstate_mvd",
                                "use_reranking": False,
                                "use_query_rewriting": False,
                            },
                        },
                        "amenities_segmento": {
                            "summary": "Consulta de mercado — amenities por segmento",
                            "value": {
                                "question": (
                                    "¿Qué amenities son más comunes en apartamentos "
                                    "en alquiler en Carrasco?"
                                ),
                                "collection": "realstate_mvd",
                                "use_reranking": False,
                                "use_query_rewriting": False,
                            },
                        },
                        "con_rewriting": {
                            "summary": "Consulta con query rewriting",
                            "value": {
                                "question": (
                                    "¿Qué zonas tienen apartamentos familiares "
                                    "cerca de espacios verdes?"
                                ),
                                "collection": "realstate_mvd",
                                "use_reranking": False,
                                "use_query_rewriting": True,
                            },
                        },
                        "con_reranking": {
                            "summary": "Consulta con reranking",
                            "value": {
                                "question": (
                                    "¿Qué opciones cerca de la rambla parecen "
                                    "más adecuadas para una familia?"
                                ),
                                "collection": "realstate_mvd",
                                "use_reranking": True,
                                "use_query_rewriting": True,
                            },
                        },
                    }
                }
            }
        }
    },
)
async def ask(payload: AskRequest) -> dict[str, Any]:
    """
    Endpoint principal para consultas sobre el mercado inmobiliario de Montevideo.

    Args:
        payload:
            AskRequest con pregunta, colección y flags opcionales.

    Raises:
        HTTPException 400:
            Pregunta o colección vacías.
        HTTPException 500:
            Error interno de recuperación/generación.
    """
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
            payload.use_reranking or False,
            payload.use_query_rewriting or False,
        )

    except Exception as exc:
        logger.exception(
            "ask_endpoint_failed",
            extra={
                "collection": payload.collection,
                "question": payload.question,
                "error": str(exc),
            },
        )

        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la consulta: {str(exc)}",
        ) from exc
