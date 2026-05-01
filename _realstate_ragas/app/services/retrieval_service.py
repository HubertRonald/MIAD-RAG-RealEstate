"""
Servicio de Retrieval para el Sistema RAG
==========================================

Este módulo implementa la funcionalidad de recuperación de documentos
usando índices vectoriales FAISS.

Para listings inmobiliarios, expone retrieve_with_filters() que combina
filtrado pre-semántico por metadatos estructurados (barrio, tipo de operación,
precio, amenities) con búsqueda por similitud semántica.
"""
 
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple, Union
from langchain.schema import Document
from app.services.embedding_service import EmbeddingService
from langsmith import traceable

# FILTROS ESTRUCTURADOS PARA LISTINGS 
@dataclass
class PropertyFilters:
    """
    Filtros pre-semánticos para recuperación de listings inmobiliarios.
 
    Todos los campos son opcionales — solo los que se especifiquen
    se aplican al filtrado. Los filtros se combinan con AND lógico.
 
    Campos de segmentación (exact match):
        operation_type  : "venta" | "alquiler"
        property_type   : "apartamentos" | "casas"
        barrio          : nombre exacto del barrio (ej: "POCITOS") o lista
                          de barrios (ej: ["POCITOS", "CORDON"]) para filtrar
                          múltiples zonas simultáneamente.
 
    Campos numéricos (rangos):
        max_price       : precio máximo (en la moneda de la operación)
        min_price       : precio mínimo
        max_price_m2    : precio/m² máximo 
        min_bedrooms    : mínimo de dormitorios
        max_bedrooms    : máximo de dormitorios
        min_surface     : superficie mínima en m² (cubierta o total)
        max_surface     : superficie máxima en m²
        max_dist_plaza  : distancia máxima a una plaza en metros
        max_dist_playa  : distancia máxima a la playa en metros
 
    Amenities (flags — True = requerido):
        has_pool, has_gym, has_elevator, has_parrillero,
        has_terrace, has_rooftop, has_security, has_storage,
        has_parking (cocheras > 0)
    """
    # Segmentación
    operation_type: Optional[str] = None
    property_type:  Optional[str] = None
    barrio:         Optional[Union[str, List[str]]] = None
 
    # Precio
    min_price:    Optional[float] = None
    max_price:    Optional[float] = None
    max_price_m2: Optional[float] = None
 
    # Características físicas
    min_bedrooms: Optional[int] = None
    max_bedrooms: Optional[int] = None
    min_surface:  Optional[float] = None
    max_surface:  Optional[float] = None
 
    # Entorno urbano
    max_dist_plaza: Optional[float] = None
    max_dist_playa: Optional[float] = None
 
    # Amenities requeridas
    has_pool:            bool = False
    has_gym:             bool = False
    has_elevator:        bool = False
    has_parrillero:      bool = False
    has_terrace:         bool = False
    has_rooftop:         bool = False
    has_security:        bool = False
    has_storage:         bool = False
    has_parking:         bool = False  # cocheras > 0
    has_party_room:      bool = False
    has_green_area:      bool = False
    has_playground:      bool = False
    has_visitor_parking: bool = False
 
    def is_empty(self) -> bool:
        """True si no se especificó ningún filtro."""
        return all([
            self.operation_type is None,
            self.property_type  is None,
            self.barrio         is None,
            self.min_price      is None,
            self.max_price      is None,
            self.max_price_m2   is None,
            self.min_bedrooms   is None,
            self.max_bedrooms   is None,
            self.min_surface    is None,
            self.max_surface    is None,
            self.max_dist_plaza is None,
            self.max_dist_playa is None,
            not self.has_pool,
            not self.has_gym,
            not self.has_elevator,
            not self.has_parrillero,
            not self.has_terrace,
            not self.has_rooftop,
            not self.has_security,
            not self.has_storage,
            not self.has_parking,
            not self.has_party_room,
            not self.has_green_area,
            not self.has_playground,
            not self.has_visitor_parking,
        ])
 
    def to_filter_fn(self) -> Optional[Callable[[Dict], bool]]:
        """
        Convierte los filtros en una función callable compatible con FAISS.
 
        FAISS acepta filter=callable en search_kwargs, donde la función
        recibe el dict de metadata de cada documento y retorna True/False.
        Usar un callable (en lugar de un dict) permite rangos numéricos
        y lógica combinada, no solo exact match.
 
        Returns:
            Función de filtrado, o None si no hay filtros activos.
        """
        if self.is_empty():
            return None
 
        # Capturar self en closure
        filters = self
 
        def _filter(meta: Dict[str, Any]) -> bool:
 
            # --- Segmentación ---
            if filters.operation_type:
                if meta.get("operation_type") != filters.operation_type:
                    return False
 
            if filters.property_type:
                if meta.get("property_type") != filters.property_type:
                    return False
 
            if filters.barrio:
                barrios = (
                    [filters.barrio] if isinstance(filters.barrio, str)
                    else filters.barrio
                )
                if meta.get("barrio_fixed") not in barrios:
                    return False
 
            # --- Precio ---
            price = meta.get("price_fixed")
            if price is not None:
                if filters.min_price is not None and price < filters.min_price:
                    return False
                if filters.max_price is not None and price > filters.max_price:
                    return False
 
            if filters.max_price_m2:
                pm2 = meta.get("price_m2")
                if pm2 is not None and pm2 > filters.max_price_m2:
                    return False
 
            # --- Dormitorios ---
            bedrooms = meta.get("bedrooms")
            if bedrooms is not None:
                if filters.min_bedrooms is not None and bedrooms < filters.min_bedrooms:
                    return False
                if filters.max_bedrooms is not None and bedrooms > filters.max_bedrooms:
                    return False
 
            # --- Superficie ---
            surface = meta.get("surface_covered") or meta.get("surface_total")
            if surface is not None:
                if filters.min_surface  is not None and surface < filters.min_surface:
                    return False
                if filters.max_surface  is not None and surface > filters.max_surface:
                    return False
 
            # --- Entorno urbano ---
            if filters.max_dist_plaza is not None:
                dist = meta.get("dist_plaza")
                if dist is not None and dist > filters.max_dist_plaza:
                    return False
 
            if filters.max_dist_playa is not None:
                dist = meta.get("dist_playa")
                if dist is not None and dist > filters.max_dist_playa:
                    return False
 
            # --- Amenities requeridas ---
            amenity_checks = {
                "has_pool":             filters.has_pool,
                "has_gym":              filters.has_gym,
                "has_elevator":         filters.has_elevator,
                "has_parrillero":       filters.has_parrillero,
                "has_terrace":          filters.has_terrace,
                "has_rooftop":          filters.has_rooftop,
                "has_security":         filters.has_security,
                "has_storage":          filters.has_storage,
                "has_party_room":       filters.has_party_room,
                "has_green_area":       filters.has_green_area,
                "has_playground":       filters.has_playground,
                "has_visitor_parking":  filters.has_visitor_parking,
            }
            for flag, required in amenity_checks.items():
                if required and meta.get(flag, 0) != 1:
                    return False
 
            if filters.has_parking:
                if not (meta.get("garages") and meta.get("garages", 0) > 0):
                    return False
 
            return True
 
        return _filter
 
 
# CONVERSIÓN DE SCORES
def l2_relevance_to_cosine(relevance_score: float) -> float:
    """
    Convierte un score de relevancia L2 de LangChain a similitud coseno real.

    LangChain normaliza las distancias L2 de FAISS via:
        relevance_score = 1 / (1 + L2_distance)

    Para vectores normalizados (gemini-embedding-001 los normaliza),
    L2 y coseno se relacionan como:
        cosine_similarity = 1 - (L2_distance² / 2)

    Combinando ambas relaciones:
        L2_distance       = (1 / relevance_score) - 1
        cosine_similarity = 1 - (L2_distance² / 2)

    Args:
        relevance_score: Score normalizado de LangChain en (0, 1].

    Returns:
        Similitud coseno en [0, 1]. Valores cercanos a 1.0 indican
        vectores casi idénticos.
    """
    if relevance_score <= 0:
        return 0.0
    l2_distance = (1.0 / relevance_score) - 1.0
    cosine = 1.0 - (l2_distance ** 2) / 2.0
    return max(0.0, min(1.0, cosine))


# SERVICIO DE RETRIEVAL
class RetrievalService:
    """
    Servicio para recuperar documentos desde índices vectoriales FAISS.
    
    Expone tres métodos de recuperación:
    - retrieve_documents()      : búsqueda semántica pura
    - retrieve_with_filters()   : filtrado estructurado + búsqueda semántica
    - retrieve_with_scores()    : igual que lo anterior + scores de relevancia [0,1]
                                  (usado para la métrica avg_cosine_similarity)
    """
    
    def __init__(self, embedding_service: EmbeddingService, k: int = 3, fetch_k: int = None):
        """
        Inicializa el servicio de retrieval y construye el retriever.
        
        Args:
            embedding_service: Servicio de embeddings que contiene el vectorstore FAISS
            k: Número de documentos a recuperar (top k)
            fetch_k: vectores candidatos recuperados del indice (similitus semántica, default = 20)
            
        Raises:
            ValueError: Si el embedding_service no tiene vectorstore inicializado
        """
        self.embedding_service = embedding_service
        self.vectorstore = embedding_service.get_vectorstore()
        
        if self.vectorstore is None:
            raise ValueError(
                "El embedding_service debe tener un vectorstore inicializado. "
                "Use embedding_service.build_vectorstore() o load_vectorstore() primero."
            )
        
        # Configura el retriever una sola vez
        self.k = k
        self.fetch_k = fetch_k if fetch_k is not None else k * 20
        self.retriever = self._build_retriever()
    
    def _build_retriever(self):
        """
        Construye el retriever configurado para búsqueda semántica.

        Returns:
            Retriever configurado para búsqueda semántica
        """
        assert self.vectorstore is not None, "Vectorstore must be initialized"
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )
    
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Recupera documentos usando retriever.invoke()
        
        Método original — llamado por rag_graph_service en el nodo
        de recuperación. No aplica filtros de metadata.
        
        
        Args:
            query: Consulta del usuario en lenguaje natural
            
        Returns:
            Lista de documentos más similares a la consulta
        """
        return self.retriever.invoke(query)
    
    @traceable(name="faiss_retrieval_filtered")
    def retrieve_with_filters(
        self,
        query: str,
        filters: Optional[PropertyFilters] = None,
    ) -> List[Document]:
        """
        Recupera documentos combinando filtros de metadata + similitud semántica.
 
        Flujo:
          1. Convierte PropertyFilters en una función callable para FAISS.
          2. FAISS evalúa la función contra el metadata de cada vector
             ANTES de calcular similitud — descarta los que no pasan.
          3. Retorna los top-k más similares entre los que pasaron el filtro.
 
        Si filters es None o está vacío, se comporta igual que
        retrieve_documents() (sin filtrado).
 
        Args:
            query   : Consulta del usuario en lenguaje natural.
            filters : Filtros estructurados opcionales.
 
        Returns:
            Lista de hasta k documentos que cumplen los filtros
            y son semánticamente similares a la consulta.
        """
        if filters is None or filters.is_empty():
            return self.retrieve_documents(query)
 
        filter_fn = filters.to_filter_fn()
 
        filtered_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k":      self.k,
                "fetch_k": self.fetch_k,  # cast a wider net before filtering
                "filter": filter_fn,
            }
        )
 
        docs = filtered_retriever.invoke(query)
 
        print(
            f"[RetrievalService] retrieve_with_filters: "
            f"{len(docs)} docs retrieved "
            f"(operation={filters.operation_type}, "
            f"property={filters.property_type}, "
            f"barrio={filters.barrio})"
        )
 
        return docs

    @traceable(name="faiss_retrieval_with_scores") 
    def retrieve_with_scores(
        self,
        query: str,
        filters: Optional[PropertyFilters] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Recupera documentos junto con sus scores de relevancia semántica.

        Usado para calcular la métrica avg_cosine_similarity: mide la
        cercanía angular entre el vector de la query y los vectores
        de los documentos recuperados.

        LangChain normaliza los scores de FAISS a [0, 1] via
        similarity_search_with_relevance_scores():
          - Índice L2     : score = 1 / (1 + distance_l2)
          - Índice coseno : score = cosine_similarity directamente

        El score promedio de los k documentos por query se loguea
        a MLflow como avg_cosine_similarity.

        Args:
            query   : Consulta del usuario en lenguaje natural.
            filters : PropertyFilters opcionales. Si se especifican,
                      se aplica el mismo filtrado que en retrieve_with_filters().

        Returns:
            Lista de hasta k tuplas (Document, relevance_score).
            relevance_score ∈ [0, 1], mayor es mejor.
        """
        filter_fn = None
        if filters is not None and not filters.is_empty():
            filter_fn = filters.to_filter_fn()

        search_kwargs: Dict[str, Any] = {"k": self.k}
        if filter_fn is not None:
            search_kwargs["fetch_k"] = self.fetch_k
            search_kwargs["filter"]  = filter_fn

        return self.vectorstore.similarity_search_with_relevance_scores(
            query,
            **search_kwargs,
        )
 