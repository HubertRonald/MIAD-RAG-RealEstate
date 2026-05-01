"""
Dataset de Evaluación — Sistema RAG Inmobiliario Montevideo
===========================================================

Pares pregunta/referencia curados para evaluar:
  - /ask   : consultas de mercado respondidas por RAGGraphService
  - /recommend : recomendaciones de propiedades por RetrievalService + GenerationService

DISEÑO DE LAS REFERENCIAS
--------------------------
Las referencias NO son respuestas verbatim basadas en listings específicos,
sino descripciones de qué debe contener una buena respuesta. Esto se hace porque:
  1. El contenido del índice FAISS cambia cuando se re-indexan propiedades.
  2. RAGAS answer_correctness usa similitud semántica por LLM, no exact match.
  3. Referencias de nivel alto son más robustas a variaciones de datos.

CRITERIO DE DISEÑO DE PREGUNTAS /ask
--------------------------------------
El índice FAISS contiene únicamente listings individuales — cada documento
es una propiedad con sus características, metadata y contexto geográfico.

Preguntas APTAS para este índice:
  ✓ Características de propiedades en un barrio específico
  ✓ Disponibilidad de amenities en una zona
  ✓ Tipo de propiedades disponibles (venta/alquiler) en un barrio
  ✓ Características del entorno de un barrio (basadas en contexto geográfico)
  ✓ Preguntas sobre propiedades específicas con filtros concretos

Preguntas NO APTAS (requieren agregación sobre el índice completo):
  ✗ Rangos de precio promedio por barrio
  ✗ Comparativas de precio/m² entre zonas
  ✗ Rankings de barrios por accesibilidad
  ✗ Frecuencia de amenities en un segmento

Flujo recomendado para mejorar la calidad de referencias:
  1. Correr evaluación baseline con estas referencias.
  2. Revisar manualmente las respuestas del sistema.
  3. Para preguntas bien respondidas, reemplazar la referencia genérica
     con la respuesta real del sistema (más específica, más justa para recall).
"""

from typing import List, Dict, Any


# ====================================================================
# PREGUNTAS PARA /ask — Consultas sobre propiedades y barrios
# ====================================================================
# Todas las preguntas son respondibles directamente desde listings
# individuales sin necesidad de agregación.
#
# Tipos cubiertos:
#   - Características y amenities de propiedades en un barrio (Q1, Q4)
#   - Disponibilidad por tipo de operación y zona (Q2, Q6, Q7)
#   - Contexto geográfico y entorno del barrio (Q3, Q5)
#   - Preguntas con filtros específicos de características (Q8)
# ====================================================================

ASK_QUESTIONS: List[Dict[str, str]] = [
    # Q1 — Características de propiedades en barrio específico
    # (reemplaza: "rango de precios en Pocitos" — requería agregación)
    {
        "question": "¿Qué características tienen los apartamentos en venta en Pocitos con más de 2 dormitorios?",
        "reference": (
            "Los apartamentos en venta en Pocitos con más de 2 dormitorios suelen "
            "tener buenas superficies cubiertas, pisos intermedios o altos, y muchos "
            "incluyen amenities como piscina, gimnasio, parrillero y cochera. "
            "Es un barrio costero con acceso a la rambla, lo que se refleja en las "
            "características de las propiedades. Los edificios más nuevos cuentan con "
            "seguridad y amenities completos."
        ),
    },
    # Q2 — Disponibilidad por zona geográfica (kept — geographic context answerable)
    {
        "question": "¿Qué barrios de Montevideo tienen apartamentos disponibles cerca de la playa?",
        "reference": (
            "Los barrios con apartamentos disponibles cerca de la playa en Montevideo "
            "incluyen Pocitos, Punta Carretas, Buceo, Malvín y Carrasco, todos sobre "
            "la costa este de la ciudad. Las propiedades en estas zonas destacan por "
            "su proximidad a la rambla y las playas, con acceso a pie al mar. "
            "Los listings en estas zonas suelen mencionar la distancia a la playa "
            "y el acceso a espacios costeros como características del entorno."
        ),
    },
    # Q3 — Amenities en edificios de alquiler en una zona específica
    # (reemplaza: "cuánto cuesta alquilar 2 dorm en Pocitos" — requería precio promedio)
    {
        "question": "¿Qué amenities ofrecen los edificios de apartamentos en alquiler en Pocitos?",
        "reference": (
            "Los edificios de apartamentos en alquiler en Pocitos frecuentemente "
            "ofrecen amenities como piscina, gimnasio, parrillero y salón de fiestas. "
            "Los edificios más modernos suelen incluir seguridad las 24 horas, "
            "ascensor y cochera. Algunos cuentan con terraza común o rooftop. "
            "La disponibilidad de amenities varía según la antigüedad y categoría "
            "del edificio."
        ),
    },
    # Q4 — Amenities en un segmento específico (kept — retrievable from listings)
    {
        "question": "¿Qué amenities tienen los apartamentos disponibles en Carrasco?",
        "reference": (
            "Los apartamentos disponibles en Carrasco incluyen amenities como "
            "piscina, gimnasio, parrillero, seguridad y cochera. Es frecuente "
            "encontrar propiedades con áreas verdes, sala de fiestas y espacios "
            "exteriores. Carrasco es un barrio residencial premium donde las "
            "propiedades suelen tener superficies amplias y buenos acabados, "
            "reflejando el perfil del barrio."
        ),
    },
    # Q5 — Características del entorno de un barrio
    # (reemplaza: "diferencia precio/m² Pocitos vs Centro" — requería comparación agregada)
    {
        "question": "¿Cómo describen los listings las propiedades disponibles en el barrio Centro de Montevideo?",
        "reference": (
            "Los listings en el Centro de Montevideo describen propiedades con buena "
            "conectividad y acceso a servicios urbanos. Es una zona céntrica con "
            "transporte público, comercios y servicios cerca. Los apartamentos en "
            "el Centro varían en tamaño y antigüedad, y algunos edificios más nuevos "
            "ofrecen amenities básicos. La zona tiene alta densidad urbana y es "
            "conveniente para quienes priorizan ubicación central sobre amenities."
        ),
    },
    # Q6 — Tipo de propiedades disponibles en un barrio (kept — directly answerable)
    {
        "question": "¿Qué tipo de propiedades hay disponibles en venta en Punta Carretas?",
        "reference": (
            "En Punta Carretas hay disponibles principalmente apartamentos en venta "
            "de distintos tamaños y categorías. Es un barrio residencial premium "
            "cercano al shopping Punta Carretas y a la rambla. Las propiedades "
            "suelen contar con buenos amenities y están en edificios de mediana "
            "a buena categoría. Es una zona muy buscada por su combinación de "
            "servicios, seguridad y calidad de vida."
        ),
    },
    # Q7 — Disponibilidad por tipo de propiedad y característica (kept — answerable)
    {
        "question": "¿En qué barrios se pueden encontrar casas en alquiler con jardín en Montevideo?",
        "reference": (
            "Las casas en alquiler con jardín se encuentran principalmente en barrios "
            "residenciales de la zona este como Carrasco, Punta Gorda y Malvín. "
            "Estas zonas tienen mayor proporción de viviendas horizontales con "
            "espacios exteriores. Los listings de casas con jardín suelen destacar "
            "la superficie del terreno, la privacidad y la cercanía a colegios "
            "y espacios verdes, características valoradas por familias."
        ),
    },
    # Q8 — Pregunta con filtros específicos de características
    # (reemplaza: "barrios más accesibles" — requería ranking sobre todo el índice)
    {
        "question": "¿Hay apartamentos disponibles en Cordón con cochera y ascensor?",
        "reference": (
            "En Cordón se pueden encontrar apartamentos con cochera y ascensor, "
            "aunque no todos los edificios del barrio los ofrecen. Cordón es un "
            "barrio céntrico con buena oferta de apartamentos de distintos tamaños. "
            "Los edificios más modernos del barrio suelen incluir ascensor y algunos "
            "tienen cochera disponible. Es una zona valorada por su ubicación "
            "central y acceso a servicios y transporte."
        ),
    },
]


# ====================================================================
# PREGUNTAS PARA /recommend — Recomendaciones de propiedades
# ====================================================================
# Cada entrada incluye:
#   question  : solicitud del cliente en lenguaje natural
#   reference : descripción de qué debe contener una buena recomendación
#   filters   : PropertyFilters pre-construidos para retrieve_with_filters()
#
# NOTA: Los filtros se importan en conftest.py para evitar dependencias
# circulares. Esta estructura permite evaluar tanto el retrieval (¿se
# recuperaron los listings correctos?) como la generación (¿la respuesta
# es fiel y relevante?).
# ====================================================================

# ====================================================================
# PREGUNTAS PARA /recommend — Recomendaciones de propiedades
# ====================================================================
#
# DISTRIBUCIÓN DE MODOS — 2 preguntas por modos 1 y 2, 3 preguntas para modo 3 = 7 total
# ─────────────────────────────────────────────────────
# Mode 1 — Solo filtros estructurados (question vacía):
#   El router construye la query semántica desde los filtros via
#   _build_fallback_query(). PreferenceExtractionService no se invoca.
#   Evalúa que el pipeline funcione correctamente sin texto libre.
#
# Mode 2 — Solo texto libre (filter_kwargs vacío):
#   PreferenceExtractionService extrae todos los filtros desde el texto.
#   Evalúa la calidad de extracción semántica de preferencias.
#
# Mode 3 — Híbrido (texto + filtros estructurados):
#   Los filtros explícitos tienen precedencia. El LLM enriquece los gaps.
#   Evalúa la combinación de ambos canales.
#
# DATOS DE REFERENCIA (del CSV de listings):
#   - 3,377 listings: 1,963 alquiler (UYU), 1,473 venta (USD)
#   - Top barrios: POCITOS (430), CORDON (326), BUCEO (217),
#                  PUNTA CARRETAS (191), CENTRO (176), CARRASCO (95)
#   - Amenities más frecuentes: Ascensor (320), Parrillero (133),
#                               Gimnasio (69), Área verde (69)
#   - Barrios costeros (dist_playa < 500m): POCITOS (187), BUCEO (111)
#   - Casas en alquiler: CARRASCO (39), POCITOS (28), BUCEO (26)
# ====================================================================

RECOMMEND_QUESTIONS: List[Dict[str, Any]] = [

    # ── MODE 1 — Solo filtros estructurados ──────────────────────────────────

    {
        # Apartamentos en alquiler en Buceo con ascensor, 2 dormitorios.
        # Buceo: 217 listings, 111 con dist_playa < 500m. Ascensor: amenity
        # más frecuente en el índice (320 listings). Muy bien cubierto.
        "question": "",
        "reference": (
            "Una buena recomendación incluye apartamentos en alquiler en Buceo "
            "con ascensor y 2 dormitorios. Debe mencionar características como "
            "superficie cubierta, piso, precio en pesos uruguayos y amenities "
            "adicionales del edificio. Buceo es un barrio costero con acceso a "
            "la rambla y playas próximas, lo que puede estar reflejado en el "
            "contexto geográfico de los listings recuperados."
        ),
        "filter_kwargs": {
            "operation_type": "alquiler",
            "property_type":  "apartamentos",
            "barrio":         "BUCEO",
            "min_bedrooms":   2,
            "max_bedrooms":   2,
            "has_elevator":   True,
        },
    },
    {
        # Casas en alquiler en Carrasco con 3+ dormitorios y cochera.
        # Carrasco: 39 casas en alquiler confirmadas en el CSV.
        # Bien cubierto para probar filtros combinados.
        "question": "",
        "reference": (
            "Una buena recomendación incluye casas en alquiler en Carrasco "
            "con al menos 3 dormitorios y cochera disponible. Debe mencionar "
            "características físicas como superficie, número de baños y "
            "espacios exteriores si están disponibles. Los resultados deben "
            "corresponder exclusivamente al barrio Carrasco y al tipo de "
            "operación alquiler."
        ),
        "filter_kwargs": {
            "operation_type": "alquiler",
            "property_type":  "casas",
            "barrio":         "CARRASCO",
            "min_bedrooms":   3,
            "has_parking":    True,
        },
    },

    # ── MODE 2 — Solo texto libre ─────────────────────────────────────────────

    {
        # Query de inversión — sin filtros estructurales.
        # Requiere razonamiento semántico sobre rentabilidad y demanda.
        # PreferenceExtractionService debe extraer operation_type=venta
        # y property_type=apartamentos desde el texto.
        "question": (
            "Quiero invertir en un apartamento en Montevideo. "
            "Busco buena relación precio/m², en zona con demanda de alquiler, "
            "y que tenga amenities para atraer inquilinos."
        ),
        "reference": (
            "Una buena recomendación para inversión menciona apartamentos en venta "
            "con precio por m² competitivo en barrios con alta demanda de alquiler "
            "como Pocitos, Cordón o Punta Carretas. "
            "Debe incluir el precio por m², los amenities disponibles y una "
            "justificación de por qué la propiedad es atractiva como inversión. "
            "Priorizar opciones con ascensor y buen estado de conservación."
        ),
        "filter_kwargs": {},
    },
    {
        # Query de lifestyle — sin filtros estructurales.
        # Usa los amenities más frecuentes del índice (ascensor: 320,
        # parrillero: 133) y referencias a espacios verdes (presente en
        # n_plaza_800m y n_espacio_libre_800m de los listings).
        # Evalúa que el retriever encuentre listings relevantes sin barrio
        # ni operación definidos explícitamente.
        "question": (
            "Busco un apartamento con ascensor y parrillero, que esté en un "
            "barrio tranquilo con espacios verdes o plaza cerca. "
            "No tengo un barrio definido, priorizo la calidad de vida."
        ),
        "reference": (
            "Una buena recomendación incluye apartamentos que cuenten con ascensor "
            "y parrillero, ubicados en barrios con buena cantidad de espacios verdes "
            "o plazas en el entorno. Debe mencionar la disponibilidad de los amenities "
            "solicitados, el barrio de cada opción y las características del entorno "
            "como parques o plazas cercanas. La respuesta no debe limitarse a un "
            "único barrio — puede incluir opciones de distintas zonas."
        ),
        "filter_kwargs": {},
    },

    # ── MODE 3 — Híbrido (texto + filtros estructurados) ─────────────────────

    {
        # Pocitos venta 2 dorm piscina — el caso de uso híbrido más clásico.
        # Pocitos: 430 listings. Piscina: amenity presente en el índice.
        # Precio máximo 250,000 USD dentro del rango real (mediana venta ~159k).
        "question": (
            "Busco un apartamento en venta en Pocitos, 2 dormitorios, "
            "con piscina, hasta 250,000 dólares."
        ),
        "reference": (
            "Una buena recomendación incluye apartamentos de 2 dormitorios en venta "
            "en Pocitos con piscina disponible, dentro del presupuesto de 250,000 USD. "
            "Debe mencionar características clave como superficie cubierta, piso, "
            "precio en USD y otros amenities del edificio. Si no hay opciones que "
            "cumplan todos los criterios, debe indicarlo y sugerir la alternativa "
            "más cercana disponible."
        ),
        "filter_kwargs": {
            "operation_type": "venta",
            "property_type":  "apartamentos",
            "barrio":         "POCITOS",
            "max_price":      250000,
            "min_bedrooms":   2,
            "max_bedrooms":   2,
            "has_pool":       True,
        },
    },
    {
        # Cerca del mar 3 dorm cochera parrillero — barrios costeros confirmados
        # en el CSV: Pocitos (187), Buceo (111), Malvín (51), Punta Carretas (44).
        # max_dist_playa=800 filtra sobre el campo dist_playa del índice.
        "question": (
            "Busco apartamento en venta cerca del mar, mínimo 3 dormitorios, "
            "con cochera y parrillero. Presupuesto flexible."
        ),
        "reference": (
            "Una buena recomendación incluye apartamentos en venta en barrios "
            "costeros de Montevideo (Pocitos, Buceo, Malvín, Punta Carretas) "
            "con al menos 3 dormitorios, cochera y parrillero disponibles. "
            "Debe mencionar la distancia a la playa o rambla, precio en USD, "
            "superficie y amenities adicionales destacables. "
            "El orden de recomendación debe priorizar la cercanía al mar."
        ),
        "filter_kwargs": {
            "operation_type":  "venta",
            "property_type":   "apartamentos",
            "min_bedrooms":    3,
            "has_parking":     True,
            "has_parrillero":  True,
            "max_dist_playa":  800,
        },
    },
    {
        # Monoambientes y 1 dorm en alquiler en Centro o Cordón — multi-barrio.
        # Cobertura confirmada: 121 listings (CORDON: 70, CENTRO: 51).
        # Distribución: 35 monoambientes (0 dorm), 86 de 1 dormitorio.
        # Precio típico: 14,000–35,000 UYU, mediana ~26,000 UYU.
        # Prueba el filtro Union[str, List[str]] en PropertyFilters.barrio.
        "question": "Quiero alquilar un monoambiente o 1 dormitorio en el Centro o Cordón.",
        "reference": (
            "Una buena recomendación incluye apartamentos pequeños en alquiler "
            "(monoambientes o de 1 dormitorio) ubicados en el Centro o en Cordón. "
            "Debe mencionar la superficie cubierta, el piso, el precio en pesos "
            "uruguayos (rango típico 14,000–35,000 UYU) y los amenities disponibles "
            "como ascensor. Las opciones deben corresponder exclusivamente a los "
            "barrios indicados y a la operación alquiler. Una buena respuesta puede "
            "ofrecer opciones de ambos barrios para que el cliente compare, destacando "
            "las diferencias de precio, superficie o ubicación dentro del barrio."
        ),
        "filter_kwargs": {
            "operation_type": "alquiler",
            "property_type":  "apartamentos",
            "barrio":         ["CENTRO", "CORDON"],
            "max_bedrooms":   1,
        },
    }
]
