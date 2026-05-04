from __future__ import annotations

import asyncio
import time
from typing import Any, Optional, Union

from fastapi import APIRouter, HTTPException

from app.config.runtime import get_settings
from app.services.service_cache import get_collection_services
from miad_rag_common.logging.structured_logging import get_logger
from miad_rag_common.schemas.filters import PropertyFilters
from miad_rag_common.schemas.recommend import RecommendRequest, RecommendResponse
from miad_rag_common.utils.norm_barrio_utils import normalize_barrio

router = APIRouter(prefix="/api/v1", tags=["Recommend"])

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

    Solo se aplica cuando existe texto libre. Para búsquedas solo con filtros,
    no se llama este guardrail.
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
            "recommend_guardrail_llm_fallback",
            extra={
                "question": question,
                "raw_response": raw,
            },
        )

        return raw.upper().startswith("YES")

    except Exception as exc:
        # Fallback permisivo: si el clasificador falla, no bloqueamos.
        logger.warning(
            "recommend_guardrail_llm_fallback_failed_allowing_query",
            extra={
                "question": question,
                "error": str(exc),
            },
        )

        return True


# =============================================================================
# Helpers de filtros
# =============================================================================

def _zero_to_none(value: Any) -> Any:
    """
    Trata 0 como null para filtros numéricos opcionales.

    Excepción conceptual:
      - min_bedrooms=0 puede representar monoambiente.
      - Por compatibilidad con el original, se conserva la función general.
    """
    return None if value == 0 else value


def _normalize_barrio_filter(
    barrio: Optional[Union[str, list[str]]],
) -> Optional[Union[str, list[str]]]:
    """
    Normaliza barrio único o lista de barrios.
    """
    if not barrio:
        return None

    if isinstance(barrio, list):
        normalized = [normalize_barrio(item) for item in barrio]
        return [value for value in normalized if value]

    return normalize_barrio(barrio)


def _request_to_property_filters(payload: RecommendRequest) -> PropertyFilters:
    """
    Convierte el contrato público RecommendRequest en filtros internos.

    Esta lógica vive en backend, no en shared/schema, para mantener
    RecommendRequest fiel al contrato original.
    """
    return PropertyFilters(
        operation_type=payload.operation_type,
        property_type=payload.property_type,
        barrio=_normalize_barrio_filter(payload.barrio),
        min_price=_zero_to_none(payload.min_price),
        max_price=_zero_to_none(payload.max_price),
        max_price_m2=_zero_to_none(payload.max_price_m2),
        min_bedrooms=_zero_to_none(payload.min_bedrooms),
        max_bedrooms=_zero_to_none(payload.max_bedrooms),
        min_surface=_zero_to_none(payload.min_surface),
        max_surface=_zero_to_none(payload.max_surface),
        max_dist_plaza=_zero_to_none(payload.max_dist_plaza),
        max_dist_playa=_zero_to_none(payload.max_dist_playa),
        has_pool=payload.has_pool,
        has_gym=payload.has_gym,
        has_elevator=payload.has_elevator,
        has_parrillero=payload.has_parrillero,
        has_terrace=payload.has_terrace,
        has_rooftop=payload.has_rooftop,
        has_security=payload.has_security,
        has_storage=payload.has_storage,
        has_parking=payload.has_parking,
        has_party_room=payload.has_party_room,
        has_green_area=payload.has_green_area,
        has_playground=payload.has_playground,
        has_visitor_parking=payload.has_visitor_parking,
    )


def _filters_dict(filters: PropertyFilters) -> dict[str, Any]:
    """
    Serializa los filtros activos que realmente se usaron en búsqueda.

    Compatible aunque PropertyFilters tenga o no active_dict().
    """
    if hasattr(filters, "active_dict"):
        try:
            return filters.active_dict()
        except Exception:
            pass

    result: dict[str, Any] = {}

    for field in [
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
    ]:
        value = getattr(filters, field, None)
        if value is not None:
            result[field] = value

    for flag in [
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
    ]:
        if getattr(filters, flag, False):
            result[flag] = True

    return result


def _has_recommendation_input(payload: RecommendRequest) -> bool:
    """
    Valida que haya al menos una pregunta o un filtro estructurado.
    """
    has_question = bool(payload.question and payload.question.strip())

    has_filters = any(
        [
            payload.operation_type,
            payload.property_type,
            payload.barrio,
            payload.min_price is not None,
            payload.max_price is not None,
            payload.max_price_m2 is not None,
            payload.min_bedrooms is not None,
            payload.max_bedrooms is not None,
            payload.min_surface is not None,
            payload.max_surface is not None,
            payload.max_dist_plaza is not None,
            payload.max_dist_playa is not None,
            payload.has_pool,
            payload.has_gym,
            payload.has_elevator,
            payload.has_parrillero,
            payload.has_terrace,
            payload.has_rooftop,
            payload.has_security,
            payload.has_storage,
            payload.has_parking,
            payload.has_party_room,
            payload.has_green_area,
            payload.has_playground,
            payload.has_visitor_parking,
        ]
    )

    return has_question or has_filters


# =============================================================================
# Query semántica y score frontend
# =============================================================================

def _build_fallback_query(payload: RecommendRequest) -> str:
    """
    Construye una query semántica cuando question está vacía.

    Esto soporta el modo 1:
      solo filtros estructurados.
    """
    parts: list[str] = []

    if payload.property_type:
        parts.append(payload.property_type)

    if payload.operation_type:
        parts.append(f"en {payload.operation_type}")

    if payload.barrio:
        if isinstance(payload.barrio, list):
            parts.append(f"en {' o '.join(payload.barrio)}")
        else:
            parts.append(f"en {payload.barrio}")

    if payload.min_bedrooms is not None:
        n = payload.min_bedrooms
        parts.append("monoambiente" if n == 0 else f"{n} dormitorios")

    if payload.max_price is not None:
        parts.append(f"hasta {int(payload.max_price)}")

    return " ".join(parts) if parts else "propiedades disponibles en Montevideo"


def _cosine_to_match_score(score: float) -> int:
    """
    Convierte score de relevancia FAISS a score 1-100 para frontend.

    Nota:
      similarity_search_with_relevance_scores() ya retorna un relevance score
      normalizado en [0, 1]. Por compatibilidad con el comportamiento original,
      no se vuelve a convertir con l2_relevance_to_cosine().
    """
    low = settings.SCORE_LOW
    high = settings.SCORE_HIGH

    clamped = max(low, min(high, score))
    return round((clamped - low) / (high - low) * 99) + 1


def _inject_rank_and_match_score(
    listings: list[Any],
) -> list[Any]:
    """
    Inserta rank y match_score en listings.

    Soporta dos formatos:
      - dict, como en el servicio local original.
      - Pydantic model / objeto, como en la migración actual.
    """
    enriched: list[Any] = []

    for rank_pos, listing in enumerate(listings, start=1):
        if isinstance(listing, dict):
            raw_score = listing.get("semantic_score")

            listing["match_score"] = (
                _cosine_to_match_score(raw_score)
                if raw_score is not None
                else None
            )
            listing["rank"] = rank_pos
            enriched.append(listing)
            continue

        raw_score = getattr(listing, "semantic_score", None)

        setattr(
            listing,
            "match_score",
            _cosine_to_match_score(raw_score)
            if raw_score is not None
            else None,
        )
        setattr(listing, "rank", rank_pos)

        enriched.append(listing)

    return enriched


# =============================================================================
# /recommend
# =============================================================================

def recommendation_processing(payload: RecommendRequest) -> dict[str, Any]:
    """
    Procesamiento principal del endpoint /recommend.

    Flujo:
      1. Obtiene servicios cacheados.
      2. Aplica guardrail si hay texto libre.
      3. Construye filtros explícitos desde payload.
      4. Enriquece filtros con PreferenceExtractionService.
      5. Ejecuta retrieval con filtros + scores.
      6. Extrae IDs y consulta BigQuery para enriquecer datos.
      7. Genera narrativa con GenerationService.
      8. Construye payload completo para frontend desde BigQuery.
    """
    start_time = time.time()
    collection = payload.collection.strip()

    svc = get_collection_services(collection)

    if payload.question and not _is_real_estate_query(payload.question):
        return {
            "question": payload.question,
            "answer": (
                "Solo puedo ayudarte con búsquedas de propiedades en Montevideo. "
                "Intenta de nuevo con criterios como barrio, precio, dormitorios, "
                "tipo de operación o amenities."
            ),
            "collection": collection,
            "listings_used": [],
            "map_points": [],
            "files_consulted": [],
            "filters_applied": {},
            "response_time_sec": round(time.time() - start_time, 3),
        }

    retrieval_service = svc["retrieval"]
    generation_service = svc["generation"]
    preference_service = svc["preference"]
    bq_listing_service = svc["bq_listing"]

    explicit_filters = _request_to_property_filters(payload)

    filters = preference_service.extract(
        payload.question,
        explicit_filters,
    )

    query = (
        payload.question.strip()
        if payload.question and payload.question.strip()
        else _build_fallback_query(payload)
    )

    logger.info(
        "recommend_retrieval_started",
        extra={
            "collection": collection,
            "query": query,
            "filters_applied": _filters_dict(filters),
            "max_recommendations": payload.max_recommendations,
        },
    )

    doc_score_pairs = retrieval_service.retrieve_with_scores(
        query,
        filters,
    )

    if not doc_score_pairs:
        return {
            "question": payload.question,
            "answer": (
                "No encontré propiedades que coincidan con los filtros indicados. "
                "Intenta ampliar los criterios de búsqueda."
            ),
            "collection": collection,
            "listings_used": [],
            "map_points": [],
            "files_consulted": [],
            "filters_applied": _filters_dict(filters),
            "response_time_sec": round(time.time() - start_time, 3),
        }

    # -------------------------------------------------------------------------
    # Seleccion final de candidatos
    # -------------------------------------------------------------------------
    # Retrieval puede devolver más documentos que max_recommendations.
    # Para que la narrativa, las cards y el mapa conversen entre sí, usamos
    # el mismo subconjunto en generación y en payload enriquecido.
    # -------------------------------------------------------------------------
    selected_pairs = doc_score_pairs[: payload.max_recommendations]
    selected_docs = [doc for doc, _score in selected_pairs]
    selected_scores = [score for _doc, score in selected_pairs]

    listing_overrides: dict[str, dict[str, Any]] = {}

    if bq_listing_service is not None:
        listing_ids = bq_listing_service.extract_listing_ids(selected_docs)
        listing_overrides = bq_listing_service.fetch_by_ids(listing_ids)

        logger.info(
            "recommend_bigquery_enrichment_completed",
            extra={
                "collection": collection,
                "listing_ids": listing_ids,
                "listing_ids_count": len(listing_ids),
                "listing_overrides_count": len(listing_overrides),
            },
        )

    # -------------------------------------------------------------------------
    # Generacion narrativa
    # -------------------------------------------------------------------------
    # GenerationService puede seguir usando listing_overrides para producir una
    # respuesta más rica, pero la salida final para Streamlit NO depende del
    # ListingInfo compacto que retorna GenerationService.
    # -------------------------------------------------------------------------
    try:
        generated = generation_service.generate_recommendations(
            question=query,
            retrieved_docs=selected_docs,
            max_recommendations=payload.max_recommendations,
            semantic_scores=selected_scores,
            listing_overrides=listing_overrides,
        )

    except TypeError:
        # Compatibilidad con GenerationService original/local,
        # que no recibía listing_overrides.
        generated = generation_service.generate_recommendations(
            question=query,
            retrieved_docs=selected_docs,
            max_recommendations=payload.max_recommendations,
            semantic_scores=selected_scores,
        )

    # -------------------------------------------------------------------------
    # Frontend payload enriquecido desde BigQuery
    # -------------------------------------------------------------------------
    # GenerationService devuelve ListingInfo compacto para compatibilidad,
    # pero el frontend necesita el registro completo de BigQuery:
    # title, description, image_urls, lat, lon, url, thumbnail_url, etc.
    #
    # Por eso aquí construimos listings_used desde BigQueryListingService,
    # usando los mismos docs/scores seleccionados que se usaron para la narrativa.
    # -------------------------------------------------------------------------
    if bq_listing_service is not None:
        frontend_records = bq_listing_service.documents_to_frontend_records(
            docs=selected_docs,
            semantic_scores=selected_scores,
            listing_overrides=listing_overrides,
            match_score_fn=_cosine_to_match_score,
        )

        frontend_records = _inject_rank_and_match_score(frontend_records)

        map_points = bq_listing_service.build_map_points_from_records(
            frontend_records,
        )
    else:
        frontend_records = _inject_rank_and_match_score(
            generated.get("listings_used", []),
        )
        map_points = []

    response = {
        "question": payload.question,
        "answer": generated.get("answer", ""),
        "collection": collection,
        "listings_used": frontend_records,
        "map_points": map_points,
        "files_consulted": generated.get(
            "sources",
            [
                str((doc.metadata or {}).get("source") or (doc.metadata or {}).get("id"))
                for doc in selected_docs
            ],
        ),
        "filters_applied": _filters_dict(filters),
        "response_time_sec": round(time.time() - start_time, 3),
    }

    logger.info(
        "recommend_completed",
        extra={
            "collection": collection,
            "results_count": len(frontend_records),
            "map_points_count": len(map_points),
            "response_time_sec": response["response_time_sec"],
            "listing_ids": [
                record.get("id")
                for record in frontend_records
                if isinstance(record, dict)
            ],
        },
    )

    return response


@router.post(
    "/recommend",
    response_model=RecommendResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "solo_filtros": {
                            "summary": "Modo 1 — solo filtros estructurados",
                            "value": {
                                "collection": "realstate_mvd",
                                "operation_type": "venta",
                                "property_type": "apartamentos",
                                "barrio": "POCITOS",
                                "max_price": 200000,
                                "min_bedrooms": 2,
                                "max_recommendations": 5,
                            },
                        },
                        "solo_texto": {
                            "summary": "Modo 2 — solo texto libre",
                            "value": {
                                "question": (
                                    "Busco algo tranquilo cerca del mar, "
                                    "con buena luz y terraza"
                                ),
                                "collection": "realstate_mvd",
                                "max_recommendations": 5,
                            },
                        },
                        "combinado": {
                            "summary": "Modo 3 — filtros + texto libre",
                            "value": {
                                "question": (
                                    "Que tenga ascensor y sea moderno, "
                                    "pensando en una familia con niños"
                                ),
                                "collection": "realstate_mvd",
                                "operation_type": "venta",
                                "property_type": "apartamentos",
                                "barrio": ["POCITOS", "PUNTA CARRETAS"],
                                "max_price": 250000,
                                "max_recommendations": 3,
                            },
                        },
                    }
                }
            }
        }
    },
)
async def recommend(payload: RecommendRequest) -> dict[str, Any]:
    """
    Endpoint de recomendaciones inmobiliarias personalizadas.

    Soporta:
      - Modo 1: solo filtros estructurados.
      - Modo 2: solo texto libre.
      - Modo 3: texto libre + filtros.
    """
    if not payload.collection or not payload.collection.strip():
        raise HTTPException(
            status_code=400,
            detail="El campo 'collection' es requerido.",
        )

    if not _has_recommendation_input(payload):
        raise HTTPException(
            status_code=400,
            detail=(
                "Proporciona al menos una pregunta o un filtro "
                "(operation_type, property_type, barrio, max_price, "
                "min_bedrooms, amenities, etc.)."
            ),
        )

    try:
        return await asyncio.to_thread(
            recommendation_processing,
            payload,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "recommend_endpoint_failed",
            extra={
                "collection": payload.collection,
                "question": payload.question,
                "error": str(exc),
            },
        )

        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la recomendación: {str(exc)}",
        ) from exc
