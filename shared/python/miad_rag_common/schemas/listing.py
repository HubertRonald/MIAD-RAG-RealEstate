from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ListingInfo(BaseModel):
    """
    Metadata estructurada de una propiedad recomendada.

    Este modelo debe ser amigable para:
    - cards del frontend,
    - tabla de resultados,
    - mapa,
    - debugging de scores semánticos.
    """

    id: Optional[str] = None

    title: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None

    barrio: Optional[str] = None
    operation_type: Optional[str] = None
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

    latitude: Optional[float] = None
    longitude: Optional[float] = None

    dist_plaza: Optional[float] = None
    dist_playa: Optional[float] = None
    n_escuelas_800m: Optional[int] = None

    has_pool: Optional[bool] = None
    has_gym: Optional[bool] = None
    has_elevator: Optional[bool] = None
    has_parrillero: Optional[bool] = None
    has_terrace: Optional[bool] = None
    has_rooftop: Optional[bool] = None
    has_security: Optional[bool] = None
    has_storage: Optional[bool] = None
    has_parking: Optional[bool] = None
    has_party_room: Optional[bool] = None
    has_green_area: Optional[bool] = None
    has_playground: Optional[bool] = None
    has_visitor_parking: Optional[bool] = None

    # Scores internos
    semantic_score: Optional[float] = Field(
        default=None,
        description="Score crudo de similitud/relevancia semántica devuelto por FAISS/LangChain.",
    )
    rerank_score: Optional[float] = Field(
        default=None,
        description="Score crudo del reranker, si se usa.",
    )

    # Scores user-facing
    match_score: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Score normalizado 1-100 para mostrar en frontend.",
    )
    rank: Optional[int] = Field(
        default=None,
        ge=1,
        description="Posición ordinal en la recomendación.",
    )


class MapPoint(BaseModel):
    """Punto geográfico simplificado para renderizar mapa en Streamlit."""

    id: str
    lat: float
    lon: float
    label: Optional[str] = None
    barrio: Optional[str] = None
    price_fixed: Optional[float] = None
    currency_fixed: Optional[str] = None
    match_score: Optional[int] = None
    rank: Optional[int] = None