from __future__ import annotations
from typing import Optional, List, Union
from pydantic import BaseModel, Field
from miad_rag_common.schemas.listing import ListingInfo


class RecommendRequest(BaseModel):
    """
    Request para el endpoint de recomendaciones inmobiliarias.

    Combina una pregunta en lenguaje natural con filtros estructurados
    opcionales. Los filtros se aplican ANTES de la búsqueda semántica,
    reduciendo el espacio de búsqueda al segmento relevante.
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

    # Control
    max_recommendations: int = 5


class RecommendResponse(BaseModel):
    """
    Respuesta del endpoint de recomendaciones inmobiliarias.

    Atributos:
        question            : Solicitud original del cliente.
        answer              : Texto narrativo con las recomendaciones del modelo.
        collection          : Colección FAISS consultada.
        listings_used       : Metadata estructurada de los listings recomendados.
                              Incluye match_score y rank para uso directo en frontend.
        files_consulted     : Lista de IDs de listings consultados.
        filters_applied     : Filtros que se aplicaron en la búsqueda.
        response_time_sec   : Tiempo total de procesamiento.
    """

    question: str
    answer: str
    collection: str
    listings_used: List[ListingInfo] = []
    files_consulted: List[str] = []
    filters_applied: dict = {}
    response_time_sec: float
