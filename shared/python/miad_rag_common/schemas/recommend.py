from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from miad_rag_common.schemas.filters import PropertyFilters
from miad_rag_common.schemas.listing import ListingInfo, MapPoint
from miad_rag_common.utils.norm_barrio_utils import normalize_barrio


def zero_to_none(value):
    """Convierte 0 a None en filtros opcionales numéricos."""
    return None if value == 0 else value


class RecommendRequest(BaseModel):
    """
    Request para recomendaciones inmobiliarias.

    Soporta:
    - Modo 1: solo filtros estructurados.
    - Modo 2: solo texto libre.
    - Modo 3: texto libre + filtros.
    """

    question: str = ""
    collection: str = Field(..., min_length=1)

    # Segmentación
    operation_type: Optional[str] = Field(default=None, examples=["venta"])
    property_type: Optional[str] = Field(default=None, examples=["apartamentos"])
    barrio: Optional[str] = Field(default=None, examples=["POCITOS"])

    # Rangos numéricos
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    max_price_m2: Optional[float] = None
    min_bedrooms: Optional[int] = None
    max_bedrooms: Optional[int] = None
    min_surface: Optional[float] = None
    max_surface: Optional[float] = None
    max_dist_plaza: Optional[float] = None
    max_dist_playa: Optional[float] = None

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

    # Control
    max_recommendations: int = Field(default=5, ge=1, le=20)
    include_map_points: bool = True
    include_explanation: bool = True

    @model_validator(mode="after")
    def validate_question_or_filters(self):
        has_question = bool(self.question and self.question.strip())
        has_filters = any(
            [
                self.operation_type,
                self.property_type,
                self.barrio,
                self.min_price,
                self.max_price,
                self.max_price_m2,
                self.min_bedrooms is not None,
                self.max_bedrooms is not None,
                self.min_surface,
                self.max_surface,
                self.max_dist_plaza,
                self.max_dist_playa,
                self.has_pool,
                self.has_gym,
                self.has_elevator,
                self.has_parrillero,
                self.has_terrace,
                self.has_rooftop,
                self.has_security,
                self.has_storage,
                self.has_parking,
                self.has_party_room,
                self.has_green_area,
                self.has_playground,
                self.has_visitor_parking,
            ]
        )

        if not has_question and not has_filters:
            raise ValueError(
                "Proporciona al menos una pregunta o un filtro estructurado."
            )

        return self

    def to_explicit_filters(self) -> PropertyFilters:
        """Construye PropertyFilters a partir de los campos explícitos del request."""
        barrio = normalize_barrio(self.barrio) if self.barrio else None

        return PropertyFilters(
            operation_type=self.operation_type,
            property_type=self.property_type,
            barrio=barrio,
            min_price=zero_to_none(self.min_price),
            max_price=zero_to_none(self.max_price),
            max_price_m2=zero_to_none(self.max_price_m2),
            min_bedrooms=zero_to_none(self.min_bedrooms),
            max_bedrooms=zero_to_none(self.max_bedrooms),
            min_surface=zero_to_none(self.min_surface),
            max_surface=zero_to_none(self.max_surface),
            max_dist_plaza=zero_to_none(self.max_dist_plaza),
            max_dist_playa=zero_to_none(self.max_dist_playa),
            has_pool=self.has_pool,
            has_gym=self.has_gym,
            has_elevator=self.has_elevator,
            has_parrillero=self.has_parrillero,
            has_terrace=self.has_terrace,
            has_rooftop=self.has_rooftop,
            has_security=self.has_security,
            has_storage=self.has_storage,
            has_parking=self.has_parking,
            has_party_room=self.has_party_room,
            has_green_area=self.has_green_area,
            has_playground=self.has_playground,
            has_visitor_parking=self.has_visitor_parking,
        )


class RecommendResponse(BaseModel):
    """Response del endpoint /api/v1/recommend."""

    question: str
    answer: str
    collection: str

    listings_used: list[ListingInfo] = Field(default_factory=list)
    map_points: list[MapPoint] = Field(default_factory=list)

    files_consulted: list[str] = Field(default_factory=list)
    filters_applied: dict = Field(default_factory=dict)

    response_time_sec: float