from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ListingInfo(BaseModel):
    """
    Metadata estructurada de un listing usado en una recomendación.

    SCORES
    ──────
    Dos capas de scores, con propósitos distintos:

    Internos (raw floats — útiles para MLflow, debugging y reranking futuro):
      semantic_score : Relevance score de FAISS normalizado a [0, 1] por
                       LangChain. Para índices L2: 1 / (1 + distance).
                       Para índices coseno: cosine similarity directamente.
                       No exponer este valor directamente al usuario.
      rerank_score   : Score del cross-encoder cuando reranking está activo.
                       Escala no acotada (logits del modelo) — solo válido
                       como ranking ordinal, no como porcentaje absoluto.
                       None cuando reranking está desactivado.

    User-facing (conversión hecha en el router, nunca en los servicios):
      match_score    : Entero 1–100 derivado de semantic_score (o rerank_score
                       si reranking está activo).
      rank           : Posición ordinal en la lista de recomendaciones.
                       1 = mejor match.
    """

    id: Optional[str] = None
    barrio: Optional[str] = None
    barrio_confidence: Optional[str] = None
    operation_type: Optional[str] = None
    is_dual_intent: Optional[bool] = None
    property_type: Optional[str] = None

    price_fixed: Optional[float] = None
    currency_fixed: Optional[str] = None
    price_m2: Optional[float] = None

    bedrooms: Optional[float] = None
    bathrooms: Optional[float] = None
    surface_covered: Optional[float] = None
    surface_total: Optional[float] = None
    floor: Optional[float] = None
    age: Optional[float] = None
    garages: Optional[float] = None

    dist_plaza: Optional[float] = None
    dist_playa: Optional[float] = None
    n_escuelas_800m: Optional[int] = None

    source: Optional[str] = None

    # ── Internal raw scores (services + MLflow only) ──────────────────────────
    semantic_score: Optional[float] = None
    rerank_score: Optional[float] = None

    # ── User-facing scores (computed by router, exposed to frontend) ───────────
    match_score: Optional[int] = None
    rank: Optional[int] = None