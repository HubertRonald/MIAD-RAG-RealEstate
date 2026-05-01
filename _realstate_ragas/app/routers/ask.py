"""
Endpoint de Consultas RAG
=====================================================

Implementa dos endpoints:

  /ask        — Market Q&A sobre el mercado inmobiliario de Montevideo.
                Orquestado por LangGraph (RAGGraphService), con soporte
                opcional de query rewriting y reranking.

  /recommend  — Recomendaciones inmobiliarias personalizadas.
                Flujo lineal: extracción de preferencias → filtrado semántico
                → generación narrativa. Soporta tres modos de uso:
                  Modo 1: solo filtros estructurados (question vacía)
                  Modo 2: solo texto libre
                  Modo 3: combinado (filtros + texto libre)
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import asyncio
import time

from app.models.ask import (
    AskRequest, AskResponse,
    RecommendRequest, RecommendResponse, ListingInfo,
)

router = APIRouter(prefix="/api/v1", tags=["Ask"])


# =============================================================================
# SERVICE CACHE
# =============================================================================
# Keyed by collection name. Populated on first request, reused on all
# subsequent ones. Avoids reloading the FAISS index on every call.
#
# RAGGraphService is NOT cached here because its constructor receives
# per-request flags (use_query_rewriting, use_reranking). It is built
# fresh each time from the cached underlying services — cheap to construct,
# only the FAISS load is expensive.

_services_cache: dict = {}


def _get_services(collection: str) -> dict:
    """
    Returns cached services for a collection, loading from disk on first call.

    Cached services:
      retrieval   : RetrievalService (wraps FAISS index — the expensive part)
      generation  : GenerationService (Gemini chat)
      preference  : PreferenceExtractionService (Gemini, for /recommend)
      qrewrite    : QueryRewritingService (for /ask via RAGGraphService)
      reranking   : RerankingService (for /ask via RAGGraphService)
    """
    if collection not in _services_cache:
        from app.services.embedding_service import EmbeddingService
        from app.services.retrieval_service import RetrievalService
        from app.services.generation_service import GenerationService
        from app.services.preference_extraction_service import PreferenceExtractionService
        from app.services.query_rewriting_service import QueryRewritingService
        from app.services.reranking_service import RerankingService

        print(f"[ServiceCache] Cold start for '{collection}' — loading FAISS index...")

        embedding_svc = EmbeddingService()
        embedding_svc.load_vectorstore(f"./faiss_index/{collection}")

        _services_cache[collection] = {
            "retrieval":   RetrievalService(embedding_svc, k=5),
            "generation":  GenerationService(),
            "preference":  PreferenceExtractionService(),
            "qrewrite":    QueryRewritingService(),
            "reranking":   None,   # loaded on demand only when use_reranking=True
        }

        print(f"[ServiceCache] Collection '{collection}' ready.")

    return _services_cache[collection]

# =============================================================================
# Shared Classifier function 
# =============================================================================
# Used as guardrail for questions out of scope.
# Classifies whether a question is within scope before retrieval runs. Uses the 
# cached GenerationService (no extra cost). Returns True if the query is related 
# to real estate in Montevideo.

# Clearly off-topic topics — fast reject without any LLM call
_OFFTOPIC_PATTERNS = [
    "receta", "cocina", "fútbol", "política", "clima", "temperatura",
    "programación", "código", "música", "película", "restaurante",
    "médico", "doctor", "enfermedad", "viaje", "vuelo", "hotel",
]

# Strong real estate signals — fast accept without any LLM call  
_REALESTATE_PATTERNS = [
    "apartamento", "casa", "inmueble", "propiedad", "alquiler", "venta",
    "dormitorio", "baño", "barrio", "m²", "metros", "precio", "piso",
    "ascensor", "terraza", "cochera", "parrillero", "piscina", "jardín",
    "pocitos", "carrasco", "centro", "punta carretas", "malvín",
    "busco", "necesito", "quiero", "buscando", "familia", "niños",
]

def _is_real_estate_query(question: str) -> bool:
    q = question.lower()

    # Fast reject — clearly off-topic
    if any(p in q for p in _OFFTOPIC_PATTERNS):
        return False

    # Fast accept — clear real estate signal
    if any(p in q for p in _REALESTATE_PATTERNS):
        return True

    # Ambiguous — fall back to LLM only as last resort
    import google.generativeai as genai
    result = genai.GenerativeModel("gemini-2.0-flash").generate_content(
        f"""Clasificador binario. Responde ÚNICAMENTE con YES o NO.
¿Esta consulta podría ser de alguien buscando una propiedad para vivir?
Ante la duda, responde YES.
Consulta: {question}"""
    )
    raw = result.text.strip()
    print(f"[Guardrail LLM fallback] '{question}' → '{raw}'", flush=True)
    return raw.upper().startswith("YES")


# =============================================================================
# /ask — MARKET Q&A (LangGraph)
# =============================================================================

def basic_rag_processing(
    question: str,
    collection: str,
    use_reranking: bool = False,
    use_query_rewriting: bool = False,
) -> Dict[str, Any]:
    """
    Procesamiento RAG para consultas de mercado inmobiliario.

    Función síncrona — llamada con asyncio.to_thread() desde el endpoint
    para no bloquear el event loop durante las llamadas a FAISS y Gemini.

    Orquestado por RAGGraphService (LangGraph). Los servicios pesados
    se obtienen del cache; RAGGraphService se construye por request
    ya que sus flags varían (use_query_rewriting, use_reranking).

    Raises:
        Exception: propagada al endpoint, que la convierte en HTTPException 500.
    """
    from app.services.rag_graph_service import RAGGraphService

    start_time = time.time()
    svc = _get_services(collection)

    # ── Guardrail — reject out-of-scope queries before retrieval ──────────
    if not _is_real_estate_query(question):
        return {
            "question":             question,
            "final_query":          question,
            "answer":               "Solo puedo responder consultas sobre el mercado inmobiliario de Montevideo. Intenta de nuevo",
            "collection":           collection,
            "files_consulted":      [],
            "context_docs":         [],
            "reranker_used":        False,
            "query_rewriting_used": False,
            "response_time_sec":    round(time.time() - start_time, 3),
        }
    
    # Lazy-load reranker only if requested
    if use_reranking and svc["reranking"] is None:
        from app.services.reranking_service import RerankingService
        svc["reranking"] = RerankingService(top_k=3)

    # RAGGraphService es liviano — solo orquesta servicios ya cacheados.
    rag_graph_service = RAGGraphService(
        retrieval_service       = svc["retrieval"],
        generation_service      = svc["generation"],
        query_rewriting_service = svc["qrewrite"],
        reranking_service       = svc["reranking"],
        rewriting_strategy      = "few_shot_rewrite",
        use_query_rewriting     = use_query_rewriting,
        use_reranking           = use_reranking,
    )

    final_query = question
    rag_result  = rag_graph_service.process_question(final_query)

    if not rag_result["context"] or len(rag_result["context"]) == 0:
        return {
            "question":             question,
            "final_query":          final_query,
            "answer":               "No tengo información suficiente en la base de datos para responder a esta pregunta.",
            "collection":           collection,
            "files_consulted":      [],
            "context_docs":         [],
            "reranker_used":        False,
            "query_rewriting_used": False,
            "response_time_sec":    round(time.time() - start_time, 3),
        }

    rewritten_queries    = rag_result.get("rewritten_queries", [])
    reranker_used        = use_reranking and svc["reranking"] is not None
    query_rewriting_used = use_query_rewriting and len(rewritten_queries) > 1
    files_consulted      = list(set(rag_result["sources"]))

    context_docs = []
    for i, context_text in enumerate(rag_result["context"]):
        source_file = rag_result["sources"][i] if i < len(rag_result["sources"]) else "unknown"
        context_docs.append({
            "file_name":  source_file,
            "chunk_type": "unknown",
            "snippet":    context_text[:200],
            "content":    context_text,
            "priority":   "high" if i == 0 else "medium",
        })

    return {
        "question":             question,
        "final_query":          final_query,
        "answer":               rag_result["answer"],
        "collection":           collection,
        "files_consulted":      files_consulted,
        "context_docs":         context_docs,
        "reranker_used":        reranker_used,
        "query_rewriting_used": query_rewriting_used,
        "response_time_sec":    round(time.time() - start_time, 3),
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
                                "question": "¿Qué diferencia hay entre Pocitos y Punta Carretas en términos de oferta?",
                                "collection": "realstate_mvd",
                                "use_reranking": False,
                                "use_query_rewriting": False,
                            }
                        },
                        "amenities_segmento": {
                            "summary": "Consulta de mercado — amenities por segmento",
                            "value": {
                                "question": "¿Qué amenities son más comunes en apartamentos en alquiler en Carrasco?",
                                "collection": "realstate_mvd",
                                "use_reranking": False,
                                "use_query_rewriting": False,
                            }
                        },
                    }
                }
            }
        }
    }
)
async def ask(payload: AskRequest):
    """
    Endpoint principal para consultas sobre el mercado inmobiliario de Montevideo.

    Args:
        payload: AskRequest con pregunta, colección y flags opcionales.

    Raises:
        HTTPException 400: Pregunta o colección vacías.
        HTTPException 500: Error interno.
    """
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta es requerida y no puede estar vacía.")

    if not payload.collection or not payload.collection.strip():
        raise HTTPException(status_code=400, detail="La colección es requerida y no puede estar vacía.")

    try:
        data = await asyncio.to_thread(
            basic_rag_processing,
            payload.question.strip(),
            payload.collection,
            payload.use_reranking or False,
            payload.use_query_rewriting or False,
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la consulta: {str(e)}")


# =============================================================================
# /recommend — RECOMENDACIONES INMOBILIARIAS
# =============================================================================


def _zero_to_none(val):
    """Treat 0 as null for optional numeric filter fields."""
    return None if val == 0 else val


def _build_fallback_query(payload: RecommendRequest) -> str:
    """
    Construye una query semántica significativa cuando question está vacía (Modo 1).

    En lugar de usar la string genérica "propiedades disponibles", compone
    una frase con los filtros estructurados disponibles para que la búsqueda
    vectorial sea más precisa.

    Ejemplos:
      payload: operation_type=venta, property_type=apartamentos, barrio=POCITOS
      → "apartamentos en venta en POCITOS"

      payload: operation_type=alquiler, min_bedrooms=2
      → "alquiler 2 dormitorios"
    """
    parts = []
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
    return " ".join(parts) if parts else "propiedades disponibles"


# ── Score conversion ──────────────────────────────────────────────────────────

# Observed range of FAISS relevance scores for this corpus.
# Tune these bounds after the first evaluation run using the
# avg_cosine_similarity distribution logged in MLflow.
_SCORE_LOW  = 0.72   # R9 acceptance threshold — below this fails the requirement
_SCORE_HIGH = 0.92   # observed ceiling from v1 evaluation corpus


def _cosine_to_match_score(score: float) -> int:
    """
    Converts a raw FAISS relevance score to a user-facing 1–100 integer.

    Rather than a naïve score * 100 (which wastes most of the scale since
    real matches cluster between 0.50 and 0.95), this rescales the observed
    range to fill [1, 100] meaningfully.

    Formula:
        clamped = clip(score, LOW, HIGH)
        match   = round((clamped - LOW) / (HIGH - LOW) * 99) + 1

    After the first evaluation run, update _SCORE_LOW and _SCORE_HIGH
    using the per-query cosine scores logged to MLflow to reflect the
    actual distribution of this corpus.

    Args:
        score: Raw FAISS relevance score in [0, 1].

    Returns:
        Integer in [1, 100].
    """
    clamped = max(_SCORE_LOW, min(_SCORE_HIGH, score))
    return round((clamped - _SCORE_LOW) / (_SCORE_HIGH - _SCORE_LOW) * 99) + 1


def recommendation_processing(payload: RecommendRequest) -> Dict[str, Any]:
    """
    Procesamiento principal del endpoint de recomendaciones.

    Función síncrona — llamada con asyncio.to_thread() desde el endpoint
    para no bloquear el event loop durante las llamadas a FAISS y Gemini.

    Flujo:
      1. Obtener servicios del cache (FAISS se carga solo en cold start).
      2. Construir filtros explícitos desde los campos estructurados del payload.
      3. Enriquecer filtros con PreferenceExtractionService (LLM sobre texto libre).
         Modos soportados:
           Modo 1 — solo filtros estructurados (question vacía): LLM se omite.
           Modo 2 — solo texto libre: LLM extrae todos los filtros.
           Modo 3 — combinado: filtros explícitos tienen precedencia, LLM rellena gaps.
      4. retrieve_with_scores() — búsqueda semántica pre-filtrada + relevance scores.
      5. generate_recommendations() — ranking y narrativa por el LLM, scores threading.
      6. Inject match_score and rank into each listing before returning.

    Raises:
        Exception: propagada al endpoint, que la convierte en HTTPException 500.
    """
    try:
        from app.services.retrieval_service import PropertyFilters
        from app.utils.norm_barrio_utils import normalize_barrio

        start_time = time.time()
        svc = _get_services(payload.collection)
        print(f"[Timing] services: {time.time()-start_time:.2f}s", flush=True)

        t = time.time()
        # ── Guardrail — only applies when free text is present (Modos 2 & 3) ─
        if payload.question and not _is_real_estate_query(payload.question):
            print(f"[Timing] guardrail: {time.time()-t:.2f}s", flush=True)
            return {
                "question":          payload.question,
                "answer":            "Solo puedo ayudarte con búsquedas de propiedades en Montevideo. Intenta de nuevo",
                "collection":        payload.collection,
                "listings_used":     [],
                "files_consulted":   [],
                "filters_applied":   {},
                "response_time_sec": round(time.time() - start_time, 3),
            }
        print(f"[Timing] guardrail (passed): {time.time()-t:.2f}s", flush=True)

        
        retrieval_service  = svc["retrieval"]
        generation_service = svc["generation"]
        preference_service = svc["preference"]

        barrio = None
        if payload.barrio:
            if isinstance(payload.barrio, list):
                barrio = [normalize_barrio(b) for b in payload.barrio]
            else:
                barrio = normalize_barrio(payload.barrio)

        # --- Construir filtros explícitos desde el payload ---
        
        explicit_filters = PropertyFilters(
            operation_type      = payload.operation_type,
            property_type       = payload.property_type,
            barrio              = barrio,
            min_price           = _zero_to_none(payload.min_price),
            max_price           = _zero_to_none(payload.max_price),
            max_price_m2        = _zero_to_none(payload.max_price_m2),
            min_bedrooms        = _zero_to_none(payload.min_bedrooms),
            max_bedrooms        = _zero_to_none(payload.max_bedrooms),
            min_surface         = _zero_to_none(payload.min_surface),
            max_surface         = _zero_to_none(payload.max_surface),
            max_dist_plaza      = _zero_to_none(payload.max_dist_plaza),
            max_dist_playa      = _zero_to_none(payload.max_dist_playa),
            has_pool            = payload.has_pool,
            has_gym             = payload.has_gym,
            has_elevator        = payload.has_elevator,
            has_parrillero      = payload.has_parrillero,
            has_terrace         = payload.has_terrace,
            has_rooftop         = payload.has_rooftop,
            has_security        = payload.has_security,
            has_storage         = payload.has_storage,
            has_parking         = payload.has_parking,
            has_party_room      = payload.has_party_room,
            has_green_area      = payload.has_green_area,
            has_playground      = payload.has_playground,
            has_visitor_parking = payload.has_visitor_parking,
        )

        t = time.time()
        # --- Enriquecer con LLM (Modos 2 y 3; Modo 1 retorna explicit_filters sin cambios) ---
        filters = preference_service.extract(payload.question, explicit_filters)
        print(f"[Timing] preference extraction: {time.time()-t:.2f}s", flush=True)

        # --- Query semántica: texto libre si existe, fallback construido desde filtros ---
        query = payload.question.strip() if payload.question else _build_fallback_query(payload)

        # --- Recuperar listings con scores de relevancia semántica ---
        t = time.time()
        doc_score_pairs = retrieval_service.retrieve_with_scores(query, filters)
        print(f"[Timing] retrieval: {time.time()-t:.2f}s", flush=True)

        if not doc_score_pairs:
            return {
                "question":          payload.question,
                "answer":            (
                    "No encontré propiedades que coincidan con los filtros indicados. "
                    "Intenta ampliar los criterios de búsqueda."
                ),
                "collection":        payload.collection,
                "listings_used":     [],
                "files_consulted":   [],
                "filters_applied":   _filters_dict(filters),
                "response_time_sec": round(time.time() - start_time, 3),
            }

        # Unpack — keep lists aligned; generation service receives both
        docs            = [doc   for doc, _     in doc_score_pairs]
        semantic_scores = [score for _,   score in doc_score_pairs]

        # --- Generar recomendaciones (scores threaded through to listings_used) ---
        t = time.time()
        result = generation_service.generate_recommendations(
            question            = query,
            retrieved_docs      = docs,
            max_recommendations = payload.max_recommendations,
            semantic_scores     = semantic_scores,
        )
        print(f"[Timing] generation: {time.time()-t:.2f}s", flush=True)


        # --- Inject user-facing match_score and rank into each listing --------
        # This conversion lives here (router) — services deal in raw floats only.
        # Future reranking: when reranking is active, use rerank_score for the
        # match_score conversion instead and preserve semantic_score as-is.
        enriched_listings = []
        for rank_pos, listing_dict in enumerate(result["listings_used"], start=1):
            raw_score = listing_dict.get("semantic_score")
            listing_dict["match_score"] = (
                _cosine_to_match_score(raw_score) if raw_score is not None else None
            )
            listing_dict["rank"] = rank_pos
            enriched_listings.append(listing_dict)

        return {
            "question":          payload.question,
            "answer":            result["answer"],
            "collection":        payload.collection,
            "listings_used":     enriched_listings,
            "files_consulted":   result["sources"],
            "filters_applied":   _filters_dict(filters),
            "response_time_sec": round(time.time() - start_time, 3),
        }
    except Exception as e:
        raise


def _filters_dict(filters: "PropertyFilters") -> dict:
    """
    Serializa los filtros activos en un dict para incluirlos en la respuesta.

    Recibe el PropertyFilters final (ya enriquecido por preference_service),
    no el payload — así filters_applied refleja lo que realmente se usó
    en la búsqueda, incluyendo lo extraído por el LLM.
    """
    result = {}
    for field in [
        "operation_type", "property_type", "barrio",
        "min_price", "max_price", "max_price_m2",
        "min_bedrooms", "max_bedrooms", "min_surface", "max_surface",
        "max_dist_plaza", "max_dist_playa",
    ]:
        val = getattr(filters, field, None)
        if val is not None:
            result[field] = val

    for flag in [
        "has_pool", "has_gym", "has_elevator", "has_parrillero",
        "has_terrace", "has_rooftop", "has_security", "has_storage", "has_parking",
        "has_party_room", "has_green_area", "has_playground", "has_visitor_parking",
    ]:
        if getattr(filters, flag, False):
            result[flag] = True

    return result


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(payload: RecommendRequest):
    """
    Endpoint de recomendaciones inmobiliarias personalizadas.

    Soporta tres modos de uso:

    **Modo 1 — Solo filtros estructurados:**
    ```json
    {
      "collection": "realstate_mvd",
      "operation_type": "venta",
      "property_type": "apartamentos",
      "barrio": "POCITOS",
      "max_price": 200000,
      "min_bedrooms": 2
    }
    ```

    **Modo 2 — Solo texto libre:**
    ```json
    {
      "question": "Busco algo tranquilo cerca del mar, con buena luz y terraza",
      "collection": "realstate_mvd"
    }
    ```

    **Modo 3 — Combinado:**
    ```json
    {
      "question": "que tenga ascensor y sea moderno, pensando en una familia con niños",
      "collection": "realstate_mvd",
      "operation_type": "venta",
      "property_type": "apartamentos",
      "barrio": "Pocitos",
      "max_recommendations": 3
    }
    ```

    Each listing in the response includes:
      - match_score (int 1–100): semantic match quality, ready for frontend display.
      - rank (int): position in recommendation order (1 = best match).
      - semantic_score (float): raw FAISS relevance score, useful for debugging.

    Raises:
        HTTPException 400: Colección vacía, o ni question ni filtros provistos.
        HTTPException 500: Error interno.
    """
    if not payload.collection or not payload.collection.strip():
        raise HTTPException(status_code=400, detail="El campo 'collection' es requerido.")

    if not payload.question and not any([
        payload.operation_type, payload.property_type, payload.barrio,
        payload.max_price, payload.min_bedrooms,
    ]):
        raise HTTPException(
            status_code=400,
            detail="Proporciona al menos una pregunta o un filtro (operation_type, barrio, max_price, etc.)"
        )

    try:
        data = await asyncio.to_thread(recommendation_processing, payload)
        return data
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error procesando la recomendación: {str(e)}")
