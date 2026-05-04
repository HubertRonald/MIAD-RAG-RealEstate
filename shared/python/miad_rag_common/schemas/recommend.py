from __future__ import annotations
from typing import Any, Optional, List, Union
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """
    Request para el endpoint de recomendaciones inmobiliarias.

    Soporta:
    - solo filtros estructurados,
    - solo texto libre,
    - texto libre + filtros.
    """

    question: str = ""
    collection: str

    # Segmentación
    operation_type: Optional[str] = Field(default=None, example="venta")
    property_type: Optional[str] = Field(default=None, example="apartamentos")
    barrio: Optional[Union[str, List[str]]] = Field(
        default=None,
        description=(
            "Barrio único (ej: 'POCITOS') o lista de barrios "
            "(ej: ['POCITOS', 'CORDON'])"
        ),
        example="POCITOS",
    )

    # Rangos numéricos
    min_price: Optional[float] = Field(default=None, example=None)
    max_price: Optional[float] = Field(default=None, example=200000)
    max_price_m2: Optional[float] = Field(default=None, example=None)
    min_bedrooms: Optional[int] = Field(default=None, example=2)
    max_bedrooms: Optional[int] = Field(default=None, example=None)
    min_surface: Optional[float] = Field(default=None, example=None)
    max_surface: Optional[float] = Field(default=None, example=None)
    max_dist_plaza: Optional[float] = Field(default=None, example=None)
    max_dist_playa: Optional[float] = Field(default=None, example=None)

    # Amenities
    has_pool: bool = False
    has_gym: bool = False
    has_elevator: bool = False
    has_parrillero: bool = False
    has_terrace: bool = False
    has_rooftop: bool = False
    has_security: bool = False
    has_storage: bool = False
    has_parking: bool = False
    has_party_room: bool = False
    has_green_area: bool = False
    has_playground: bool = False
    has_visitor_parking: bool = False

    # Control interno del frontend.
    # En Streamlit puede estar oculto para el usuario.
    max_recommendations: int = 5


class RecommendResponse(BaseModel):
    """
    Respuesta del endpoint de recomendaciones inmobiliarias.

    Para este primer alcance, listings_used se deja como lista de dicts para
    permitir devolver el registro completo enriquecido desde BigQuery.

    Cada item puede incluir:
    - columnas completas de real_estate_listings,
    - campos display para frontend,
    - scores de retrieval,
    - rank,
    - map_point,
    - retrieval_snippet.
    """

    question: str
    answer: str
    collection: str

    listings_used: list[dict[str, Any]] = Field(default_factory=list)
    map_points: list[dict[str, Any]] = Field(default_factory=list)

    files_consulted: list[str] = Field(default_factory=list)
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    response_time_sec: float
