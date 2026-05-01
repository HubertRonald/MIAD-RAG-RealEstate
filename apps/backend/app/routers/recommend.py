from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from app.config.runtime import get_settings
from app.services.retrieval_service import l2_relevance_to_cosine
from app.services.service_cache import get_collection_services
from miad_rag_common.schemas.listing import ListingInfo
from miad_rag_common.schemas.recommend import RecommendRequest, RecommendResponse

router = APIRouter(prefix="/api/v1", tags=["Recommend"])
settings = get_settings()


def _build_fallback_query(payload: RecommendRequest) -> str:
    """
    Query semántica cuando el usuario usa solo filtros estructurados.
    """
    parts: list[str] = []

    if payload.property_type:
        parts.append(payload.property_type)

    if payload.operation_type:
        parts.append(f"en {payload.operation_type}")

    if payload.barrio:
        parts.append(f"en {payload.barrio}")

    if payload.min_bedrooms is not None:
        n = payload.min_bedrooms
        parts.append("monoambiente" if n == 0 else f"{n} dormitorios")

    if payload.max_price is not None:
        parts.append(f"hasta {int(payload.max_price)}")

    return " ".join(parts) if parts else "propiedades disponibles en Montevideo"


def _cosine_to_match_score(score: float) -> int:
    """
    Convierte similitud coseno a score 1-100 para frontend.
    """
    low = settings.SCORE_LOW
    high = settings.SCORE_HIGH

    clamped = max(low, min(high, score))
    return round((clamped - low) / (high - low) * 99) + 1


def recommendation_processing(payload: RecommendRequest) -> dict[str, Any]:
    """
    Flujo productivo del endpoint /recommend.

    1. Obtiene servicios cacheados.
    2. Construye filtros explícitos.
    3. Enriquece filtros con LLM si hay texto libre.
    4. Recupera documentos con FAISS + filtros.
    5. Enriquece listings con BigQuery.
    6. Genera narrativa.
    7. Devuelve cards + map_points para Streamlit.
    """
    start_time = time.time()
    svc = get_collection_services(payload.collection)

    retrieval_service = svc["retrieval"]
    generation_service = svc["generation"]
    preference_service = svc["preference"]
    bq_listing_service = svc["bq_listing"]

    explicit_filters = payload.to_explicit_filters()
    filters = preference_service.extract(payload.question, explicit_filters)

    query = payload.question.strip() if payload.question else _build_fallback_query(payload)

    doc_score_pairs = retrieval_service.retrieve_with_scores(query, filters)

    if not doc_score_pairs:
        return {
            "question": payload.question,
            "answer": (
                "No encontré propiedades que coincidan con los criterios indicados. "
                "Puedes ampliar la zona, el presupuesto o reducir amenities obligatorios."
            ),
            "collection": payload.collection,
            "listings_used": [],
            "map_points": [],
            "files_consulted": [],
            "filters_applied": filters.active_dict(),
            "response_time_sec": round(time.time() - start_time, 3),
        }

    docs = [doc for doc, _score in doc_score_pairs]
    semantic_scores = [score for _doc, score in doc_score_pairs]

    listing_ids = bq_listing_service.extract_listing_ids(docs)
    listing_overrides = bq_listing_service.fetch_by_ids(listing_ids)

    generated = generation_service.generate_recommendations(
        question=query,
        retrieved_docs=docs,
        max_recommendations=payload.max_recommendations,
        semantic_scores=semantic_scores,
        listing_overrides=listing_overrides,
    )

    enriched_listings: list[ListingInfo] = []

    for rank_pos, listing in enumerate(generated["listings_used"], start=1):
        raw_score = listing.semantic_score

        listing.match_score = (
            _cosine_to_match_score(l2_relevance_to_cosine(raw_score))
            if raw_score is not None
            else None
        )
        listing.rank = rank_pos

        enriched_listings.append(listing)

    map_points = (
        bq_listing_service.build_map_points(enriched_listings)
        if payload.include_map_points
        else []
    )

    return {
        "question": payload.question,
        "answer": generated["answer"] if payload.include_explanation else "",
        "collection": payload.collection,
        "listings_used": enriched_listings,
        "map_points": map_points,
        "files_consulted": generated["sources"],
        "filters_applied": filters.active_dict(),
        "response_time_sec": round(time.time() - start_time, 3),
    }


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(payload: RecommendRequest) -> dict[str, Any]:
    if not payload.collection or not payload.collection.strip():
        raise HTTPException(
            status_code=400,
            detail="El campo 'collection' es requerido.",
        )

    try:
        return await asyncio.to_thread(recommendation_processing, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la recomendación: {str(exc)}",
        ) from exc
