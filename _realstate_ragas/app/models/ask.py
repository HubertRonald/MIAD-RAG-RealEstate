"""
Modelos para el sistema RAG de consultas

Define las estructuras de datos para las consultas al sistema RAG,
incluyendo peticiones, respuestas y metadatos de contexto.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Union


class AskRequest(BaseModel):
    """
    Modelo para validar las peticiones de consulta al sistema RAG.

    Atributos:
        question (str): La pregunta que se quiere responder. Campo obligatorio.
        collection (str): Nombre de la colección de documentos a consultar. Campo obligatorio.
        use_reranking (bool, optional): No implementado.
        use_query_rewriting (bool, optional): No implementado.
    """
    question: str
    collection: str
    use_reranking: Optional[bool] = False
    use_query_rewriting: Optional[bool] = False


class FuenteContexto(BaseModel):
    """
    Modelo que representa una fuente de información utilizada para generar la respuesta.

    Atributos:
        file_name (str, optional): Nombre del archivo fuente del fragmento.
        page_number (int, optional): Número de página dentro del documento.
        chunk_type (str, optional): Tipo de fragmento (ej: 'text', 'table', 'header').
        priority (str, optional): Prioridad de relevancia ('high', 'medium', 'low').
        snippet (str, optional): Fragmento de texto extraído del documento (preview).
        content (str, optional): Contenido completo del chunk (para RAGAS).
        rerank_score (float, optional): Score de reranking CrossEncoder. (No implementado)
    """
    file_name: Optional[str] = None
    page_number: Optional[int] = None
    chunk_type: Optional[str] = None
    priority: Optional[str] = None
    snippet: Optional[str] = None
    content: Optional[str] = None
    rerank_score: Optional[float] = None


class AskResponse(BaseModel):
    """
    Modelo de respuesta que retorna el sistema RAG después de procesar una consulta.

    Atributos:
        question (str): La pregunta original que se procesó.
        final_query (str): La consulta final que le llega al RAG.
        answer (str): La respuesta generada por el sistema RAG.
        collection (str): Nombre de la colección de documentos utilizada.
        files_consulted (List[str]): Lista de nombres de archivos consultados.
        context_docs (List[FuenteContexto]): Fragmentos específicos utilizados como contexto.
        reranker_used (bool): Indica si se utilizó reordenamiento de resultados.
        query_rewriting_used (bool): Indica si se aplicó reescritura de consulta.
        response_time_sec (float): Tiempo total de procesamiento en segundos.
    """
    question: str
    final_query: str
    answer: str
    collection: str
    files_consulted: List[str] = []
    context_docs: List[FuenteContexto] = []
    reranker_used: bool = False
    query_rewriting_used: bool = False
    response_time_sec: float


# =============================================================================
# Modelos Recomendaciones Inmobiliarias
# =============================================================================

class RecommendRequest(BaseModel):
    """
    Request para el endpoint de recomendaciones inmobiliarias.

    Combina una pregunta en lenguaje natural con filtros estructurados
    opcionales. Los filtros se aplican ANTES de la búsqueda semántica,
    reduciendo el espacio de búsqueda al segmento relevante.
    """
    question:    str = ""
    collection:  str

    # Segmentación
    operation_type: Optional[str]   = Field(default=None, example="venta")
    property_type:  Optional[str]   = Field(default=None, example="apartamentos")
    barrio:         Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Barrio único (ej: 'POCITOS') o lista de barrios (ej: ['POCITOS', 'CORDON'])",
        example="POCITOS",
    )

    # Rangos numéricos
    min_price:    Optional[float]   = Field(default=None, example=None)
    max_price:    Optional[float]   = Field(default=None, example=200000)
    max_price_m2: Optional[float]   = Field(default=None, example=None)
    min_bedrooms: Optional[int]     = Field(default=None, example=2)
    max_bedrooms: Optional[int]     = Field(default=None, example=None)
    min_surface:  Optional[float]   = Field(default=None, example=None)
    max_surface:  Optional[float]   = Field(default=None, example=None)
    max_dist_plaza: Optional[float] = Field(default=None, example=None)
    max_dist_playa: Optional[float] = Field(default=None, example=None)

    # Amenities
    has_pool:            bool = False
    has_gym:             bool = False
    has_elevator:        bool = False
    has_parrillero:      bool = False
    has_terrace:         bool = False
    has_rooftop:         bool = False
    has_security:        bool = False
    has_storage:         bool = False
    has_parking:         bool = False
    has_party_room:      bool = False
    has_green_area:      bool = False
    has_playground:      bool = False
    has_visitor_parking: bool = False

    # Control
    max_recommendations: int = 5


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
                       si reranking está activo). Ver _cosine_to_match_score()
                       en routers/ask.py para la fórmula de normalización.
                       Rango real de scores coseno observado: [0.50, 0.95].
                       Actualizar LOW/HIGH en el router tras la primera evaluación.
      rank           : Posición ordinal en la lista de recomendaciones.
                       1 = mejor match. Siempre presente cuando hay resultados.
    """
    id:              Optional[str]   = None
    barrio:          Optional[str]   = None
    barrio_confidence:  Optional[str]  = None  # 'consistent' | 'genuine_ambiguity' | etc.
    operation_type:  Optional[str]   = None
    is_dual_intent:     Optional[bool] = None  # True if available for both sale and rent
    property_type:   Optional[str]   = None
    price_fixed:     Optional[float] = None
    currency_fixed:  Optional[str]   = None
    price_m2:        Optional[float] = None
    bedrooms:        Optional[float] = None
    bathrooms:       Optional[float] = None
    surface_covered: Optional[float] = None
    surface_total:   Optional[float] = None
    floor:           Optional[float] = None
    age:             Optional[float] = None
    garages:         Optional[float] = None
    dist_plaza:      Optional[float] = None
    dist_playa:      Optional[float] = None
    n_escuelas_800m: Optional[int]   = None
    source:          Optional[str]   = None

    # ── Internal raw scores (services + MLflow only) ──────────────────────────
    semantic_score:  Optional[float] = None  # FAISS relevance score [0, 1]
    rerank_score:    Optional[float] = None  # cross-encoder logit (unbounded)

    # ── User-facing scores (computed by router, exposed to frontend) ───────────
    match_score:     Optional[int]   = None  # 1–100 normalized match quality
    rank:            Optional[int]   = None  # 1 = best match


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
    question:          str
    answer:            str
    collection:        str
    listings_used:     List[ListingInfo] = []
    files_consulted:   List[str]         = []
    filters_applied:   dict              = {}
    response_time_sec: float
