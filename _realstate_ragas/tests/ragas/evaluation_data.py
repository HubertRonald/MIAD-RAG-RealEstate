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

Flujo recomendado para mejorar la calidad de referencias:
  1. Correr una evaluación baseline con estas referencias.
  2. Revisar manualmente las respuestas del sistema.
  3. Para las preguntas bien respondidas, reemplazar la referencia genérica
     con la respuesta real del sistema (más específica y más justa para el recall).
"""

from typing import List, Dict, Any


# ====================================================================
# PREGUNTAS PARA /ask — Consultas de mercado inmobiliario
# ====================================================================
# Cubre los tipos de query más comunes:
#   - Preguntas de precio y rango
#   - Preguntas comparativas entre barrios
#   - Preguntas de amenities y características
#   - Preguntas de tipo de propiedad y disponibilidad
# ====================================================================

ASK_QUESTIONS: List[Dict[str, str]] = [
    {
        "question": "¿Cuál es el rango de precios de apartamentos en venta en Pocitos?",
        "reference": (
            "Los apartamentos en venta en Pocitos tienen precios que varían ampliamente "
            "según la superficie, piso, antigüedad y amenities. Los apartamentos de 1 a 2 "
            "dormitorios tienen precios más accesibles que los de 3 o más dormitorios. "
            "El precio por m² en Pocitos es uno de los más elevados de Montevideo, "
            "reflejando su ubicación costera y alta demanda residencial."
        ),
    },
    {
        "question": "¿Qué barrios de Montevideo tienen apartamentos disponibles cerca de la playa?",
        "reference": (
            "Los barrios más próximos a la playa en Montevideo incluyen Pocitos, "
            "Punta Carretas, Buceo, Malvín, Carrasco y Punta Gorda. Estos barrios "
            "tienen acceso directo a la rambla y las playas de la costa este. "
            "Pocitos y Punta Carretas son los de mayor densidad de edificios y "
            "oferta de apartamentos."
        ),
    },
    {
        "question": "¿Cuánto cuesta aproximadamente alquilar un apartamento de 2 dormitorios en Pocitos?",
        "reference": (
            "Los apartamentos de 2 dormitorios en alquiler en Pocitos varían en precio "
            "según la antigüedad del edificio, el piso, la superficie y los amenities. "
            "Los edificios más nuevos con amenities completos (piscina, gym, parrillero) "
            "tienen valores considerablemente más altos. Los precios pueden estar "
            "en dólares o pesos uruguayos dependiendo del acuerdo."
        ),
    },
    {
        "question": "¿Qué amenities son más comunes en los apartamentos de alta gama en Carrasco?",
        "reference": (
            "Los apartamentos de alta gama en Carrasco frecuentemente incluyen amenities "
            "como piscina, gimnasio, parrillero, seguridad las 24 horas, cochera, "
            "y en algunos casos rooftop o terrazas comunes. También es común encontrar "
            "áreas verdes, sala de fiestas y estacionamiento para visitas. "
            "El barrio se caracteriza por propiedades amplias y bien equipadas."
        ),
    },
    {
        "question": "¿Cuál es la diferencia de precio por m² entre apartamentos en Pocitos y en el Centro?",
        "reference": (
            "Pocitos tiene precios por m² significativamente más altos que el Centro "
            "de Montevideo, dada su ubicación costera, mayor demanda y perfil "
            "socioeconómico más alto. El Centro es más céntrico y con buena conectividad "
            "de transporte, pero sus precios por m² son más accesibles. "
            "La diferencia puede ser considerable dependiendo del segmento."
        ),
    },
    {
        "question": "¿Qué tipo de propiedades hay disponibles en venta en Punta Carretas?",
        "reference": (
            "Punta Carretas tiene una oferta de venta principalmente de apartamentos "
            "de distintos tamaños y categorías. Es un barrio residencial premium "
            "con buena oferta de servicios, próximo al shopping y a la rambla. "
            "Las propiedades suelen tener buenos amenities y precios en el segmento "
            "medio-alto a alto del mercado montevideano."
        ),
    },
    {
        "question": "¿En qué barrios se pueden encontrar casas en alquiler con jardín en Montevideo?",
        "reference": (
            "Las casas en alquiler con jardín se concentran principalmente en barrios "
            "residenciales como Carrasco, Punta Gorda, Malvín, Buceo y Parque Batlle. "
            "Estos barrios tienen mayor proporción de viviendas horizontales con espacios "
            "exteriores, a diferencia de zonas más céntricas donde predominan los edificios. "
            "Carrasco es el barrio con mayor oferta de casas de este tipo."
        ),
    },
    {
        "question": "¿Cuáles son los barrios más accesibles para comprar un apartamento en Montevideo?",
        "reference": (
            "Los barrios con precios por m² más accesibles para comprar apartamentos "
            "en Montevideo son zonas como el Centro, Cordón, Brazo Oriental, Unión "
            "y La Comercial. Estos barrios ofrecen precios considerablemente más bajos "
            "que las zonas costeras premium. Cordón es popular por su buena ubicación "
            "y precios moderados, con cercanía a servicios y transporte."
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

RECOMMEND_QUESTIONS: List[Dict[str, Any]] = [
    {
        "question": (
            "Busco un apartamento en venta en Pocitos, 2 dormitorios, "
            "con piscina, hasta 250,000 dólares."
        ),
        "reference": (
            "Una buena recomendación incluye apartamentos de 2 dormitorios en venta "
            "en Pocitos con piscina disponible, dentro del presupuesto de 250,000 USD. "
            "Debe mencionar características clave como superficie, piso, precio exacto "
            "y otros amenities del edificio. Si no hay opciones que cumplan todos los "
            "criterios, debe indicarlo y sugerir alternativas ajustando algún filtro."
        ),
        # PropertyFilters se construyen en conftest.py usando estos parámetros
        "filter_kwargs": {
            "operation_type": "venta",
            "property_type": "apartamentos",
            "barrio": "POCITOS",
            "max_price": 250000,
            "min_bedrooms": 2,
            "max_bedrooms": 2,
            "has_pool": True,
        },
    },
    {
        "question": (
            "Quiero alquilar un apartamento pequeño, monoambiente o 1 dormitorio, "
            "en el Centro o Cordón, precio razonable."
        ),
        "reference": (
            "Una buena recomendación incluye apartamentos de alquiler con 0 o 1 dormitorio "
            "en barrios céntricos. Debe mencionar precio (en pesos o USD), superficie "
            "aproximada, estado del edificio y características básicas. Si hay opciones "
            "tanto en Centro como en Cordón, debe diferenciarlas claramente."
        ),
        "filter_kwargs": {
            "operation_type": "alquiler",
            "property_type": "apartamentos",
            "max_bedrooms": 1,
        },
    },
    {
        "question": (
            "Necesito una casa en alquiler con jardín en Carrasco para una familia "
            "con niños pequeños. Que tenga buen acceso a escuelas."
        ),
        "reference": (
            "Una buena recomendación incluye casas en alquiler en Carrasco, "
            "mencionando la disponibilidad de jardín o espacios exteriores. "
            "Debe indicar número de dormitorios y baños, acceso a escuelas cercanas "
            "si hay datos disponibles, y características del entorno del barrio "
            "relevantes para familias con niños."
        ),
        "filter_kwargs": {
            "operation_type": "alquiler",
            "property_type": "casas",
            "barrio": "CARRASCO",
        },
    },
    {
        "question": (
            "Busco apartamento en venta cerca del mar, mínimo 3 dormitorios, "
            "con cochera y parrillero. Presupuesto flexible."
        ),
        "reference": (
            "Una buena recomendación incluye apartamentos en venta en barrios costeros "
            "(Pocitos, Punta Carretas, Buceo, Malvín, Carrasco) con al menos 3 dormitorios, "
            "cochera disponible y parrillero. Debe mencionar la distancia a la playa "
            "o rambla, precio, superficie y cualquier amenity adicional destacable. "
            "El orden de recomendación debe priorizar la cercanía al mar."
        ),
        "filter_kwargs": {
            "operation_type": "venta",
            "property_type": "apartamentos",
            "min_bedrooms": 3,
            "has_parking": True,
            "has_parrillero": True,
            "max_dist_playa": 800,
        },
    },
    {
        "question": (
            "Quiero invertir en un apartamento en Montevideo. "
            "Busco buena relación precio/m², en zona con demanda de alquiler, "
            "y que tenga amenities para atraer inquilinos."
        ),
        "reference": (
            "Una buena recomendación para inversión menciona apartamentos en venta "
            "con precio por m² competitivo en barrios con alta demanda de alquiler "
            "como Pocitos, Cordón, Parque Rodó o Punta Carretas. "
            "Debe incluir datos de precio por m², amenities disponibles y una "
            "justificación de por qué la propiedad es atractiva como inversión. "
            "Si hay opciones con elevator y buen estado de conservación, priorizarlas."
        ),
        "filter_kwargs": {
            "operation_type": "venta",
            "property_type": "apartamentos",
        },
    },
    # ── Mode 1 — structured filters only (empty question) ────────────────────
    # El router construye la query semántica desde los filtros via
    # _build_fallback_query(). PreferenceExtractionService no se invoca.
    # Evalúa que el pipeline funcione correctamente sin texto libre.
    {
        "question": "",
        "reference": (
            "Una buena recomendación incluye casas en alquiler en Carrasco "
            "con al menos 3 dormitorios y cochera disponible. Debe mencionar "
            "características físicas como superficie, número de baños y espacios "
            "exteriores si están disponibles. Los resultados deben corresponder "
            "exclusivamente al barrio Carrasco y al tipo de operación alquiler."
        ),
        "filter_kwargs": {
            "operation_type": "alquiler",
            "property_type":  "casas",
            "barrio":         "CARRASCO",
            "min_bedrooms":   3,
            "has_parking":    True,
        },
    },
]
