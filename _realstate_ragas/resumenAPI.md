# Documentación del API — Sistema RAG Inmobiliario Montevideo

El API REST expone **3 endpoints principales** para construir el índice y realizar consultas inteligentes sobre el mercado inmobiliario de Montevideo.

> **Nota**: Los endpoints de administración (`validate-load`, `health`) están disponibles pero no se documentan en Swagger para mantener la interfaz limpia.

---

## 1. Load from CSV

**Endpoint**: `POST /api/v1/documents/load-from-csv`

**Descripción**: Indexa el archivo `listings.csv` desde `docs/realstate_mvd/` y construye el índice vectorial FAISS. El procesamiento es **asíncrono** — el endpoint retorna inmediatamente con un `processing_id` para consultar el estado.

**Flujo**: CSV → `csv_document_service` (preprocesamiento + amenities + geo) → `chunking_service` → `embedding_service` (Gemini + checkpoint/resume) → FAISS Index

> Este proceso puede tomar entre 10 y 15 minutos para el corpus completo (~3.400 listings). Si se interrumpe, volver a ejecutar el endpoint — retomará desde el último checkpoint guardado automáticamente.

### Request Body

```json
{
  "collection_name": "realstate_mvd"
}
```

**Con filtros opcionales** (para indexar un segmento específico):

```json
{
  "collection_name": "realstate_mvd",
  "operation_type": "venta",
  "property_type": "apartamentos",
  "barrio": "POCITOS"
}
```

### Parámetros

| Parámetro          | Tipo   | Requerido | Default        | Descripción                               |
| ------------------- | ------ | --------- | -------------- | ------------------------------------------ |
| `collection_name` | string |  ✅ Sí     | `realstate_mvd` | Nombre de la colección / carpeta en `./docs/` |
| `operation_type`  | string | ❌ No     | —             | `"venta"` \| `"alquiler"`               |
| `property_type`   | string | ❌ No     | —             | `"apartamentos"` \| `"casas"`           |
| `barrio`          | string | ❌ No     | —             | Nombre exacto del barrio (ej: `"POCITOS"`) |

### Response

```json
{
  "success": true,
  "message": "Indexación iniciada en background",
  "processing_id": "csv_7a880482d6fa",
  "timestamp": "2026-05-01T18:25:49.834501",
  "collection": "realstate_mvd",
  "filters": null
}
```

| Campo             | Tipo    | Descripción                                          |
| ----------------- | ------- | ----------------------------------------------------- |
| `success`       | boolean | `true` si el proceso se encoló correctamente        |
| `message`       | string  | Mensaje descriptivo                                   |
| `processing_id` | string  | ID para consultar el estado del procesamiento         |
| `timestamp`     | string  | ISO 8601 — momento de inicio                         |
| `collection`    | string  | Nombre de la colección indexada                      |
| `filters`       | object  | Filtros aplicados, o `null` si se indexó todo       |

### Códigos de estado

- `200 OK` — Procesamiento encolado correctamente
- `404 Not Found` — La carpeta `./docs/{collection_name}/` no existe
- `422 Unprocessable Entity` — Valores inválidos en `operation_type` o `property_type`

---

## 2. Ask

**Endpoint**: `POST /api/v1/ask`

**Descripción**: Consulta de mercado en lenguaje natural. Recupera listings relevantes mediante búsqueda semántica en FAISS y genera una respuesta fundamentada **exclusivamente** en el contexto recuperado.

Soporta preguntas sobre precios, tendencias de mercado, características de barrios y comparaciones de zonas. Responde en el idioma de la pregunta (español / inglés, con tono rioplatense en español).

**Flujo**: pregunta → guardrail de scope → `RAGGraphService` (LangGraph) → retrieval → generación (Gemini)

### Request Body

**Mínimo requerido**:

```json
{
  "question": "¿Cuánto cuesta el m² en Pocitos?",
  "collection": "realstate_mvd"
}
```

**Con opciones avanzadas**:

```json
{
  "question": "¿Cuánto cuesta el m² en Pocitos?",
  "collection": "realstate_mvd",
  "use_reranking": false,
  "use_query_rewriting": true
}
```

### Parámetros

| Parámetro              | Tipo    | Requerido | Default   | Descripción                                                      |
| ----------------------- | ------- | --------- | --------- | ----------------------------------------------------------------- |
| `question`            | string  | ✅ Sí    | —        | Pregunta en lenguaje natural (español o inglés)                 |
| `collection`          | string  | ✅ Sí    | —        | Colección a consultar (usar `"realstate_mvd"`)                  |
| `use_reranking`       | boolean | ❌ No     | `false` | Activa reranking con Cross-Encoder (PENDIENTE de evaluación)    |
| `use_query_rewriting` | boolean | ❌ No     | `false` | Reescribe la consulta en modo few-shot para mejorar el retrieval |

### Response

```json
{
  "question": "¿Cuánto cuesta el m² en Pocitos?",
  "final_query": "¿Cuánto cuesta el m² en Pocitos?",
  "answer": "En Pocitos, el precio del m² varía entre USD 2.500 y USD 4.000 dependiendo...",
  "collection": "realstate_mvd",
  "files_consulted": ["listing_MLU123456", "listing_MLU789012"],
  "context_docs": [...],
  "reranker_used": false,
  "query_rewriting_used": false,
  "response_time_sec": 18.3
}
```

| Campo                    | Tipo          | Descripción                                                                |
| ------------------------ | ------------- | --------------------------------------------------------------------------- |
| `question`             | string        | Pregunta original                                                           |
| `final_query`          | string        | Consulta usada para retrieval (reescrita si `use_query_rewriting=true`)   |
| `answer`               | string        | Respuesta generada en el idioma de la pregunta                              |
| `collection`           | string        | Colección consultada                                                       |
| `files_consulted`      | array[string] | IDs de los listings consultados                                             |
| `context_docs`         | array[object] | Fragmentos usados como contexto (snippet + metadata)                        |
| `reranker_used`        | boolean       | Si se aplicó reranking                                                     |
| `query_rewriting_used` | boolean       | Si se aplicó reescritura de consulta                                        |
| `response_time_sec`    | number        | Tiempo total de procesamiento en segundos                                   |

### Códigos de estado

- `200 OK` — Consulta procesada correctamente
- `400 Bad Request` — Pregunta o colección vacía
- `500 Internal Server Error` — Error durante el procesamiento

### Notas

- Las preguntas fuera del dominio inmobiliario son rechazadas por el guardrail antes de llegar a retrieval.
- Si no hay contexto suficiente, el sistema indica claramente la falta de información sin inventar respuestas.
- `k=5` documentos recuperados por defecto (configurable en `app/routers/ask.py`).

---

## 3. Recommend

**Endpoint**: `POST /api/v1/recommend`

**Descripción**: Motor de recomendación de propiedades personalizado. Combina filtros estructurados explícitos con extracción de preferencias en lenguaje natural (vía LLM) para recuperar y recomendar los listings más relevantes.

Soporta tres modos de uso:

| Modo | Descripción | Ejemplo |
|------|-------------|---------|
| **Modo 1** — Filtros puros | Solo campos estructurados, sin texto libre | Buscar apartamentos en venta en Pocitos hasta USD 200k |
| **Modo 2** — Texto libre | Solo `question`, sin filtros estructurados | "Busco algo tranquilo cerca del mar con terraza" |
| **Modo 3** — Híbrido | Texto libre + filtros estructurados | Pregunta + `barrio`, `operation_type`, etc. explícitos |

**Flujo**: payload → guardrail → `PreferenceExtractionService` (Modos 2 y 3) → `RetrievalService` con `PropertyFilters` → `GenerationService` → respuesta estructurada

### Request Body

**Modo 1 — Solo filtros**:

```json
{
  "collection": "realstate_mvd",
  "operation_type": "venta",
  "property_type": "apartamentos",
  "barrio": "POCITOS",
  "max_price": 200000,
  "has_elevator": true,
  "max_recommendations": 3
}
```

**Modo 2 — Solo texto libre**:

```json
{
  "question": "Busco algo tranquilo cerca del mar con buena luz y terraza",
  "collection": "realstate_mvd",
  "max_recommendations": 5
}
```

**Modo 3 — Híbrido**:

```json
{
  "question": "que tenga ascensor y sea moderno, pensando en una familia con niños",
  "collection": "realstate_mvd",
  "operation_type": "venta",
  "property_type": "apartamentos",
  "barrio": "POCITOS",
  "max_recommendations": 3
}
```

### Parámetros

| Parámetro              | Tipo    | Requerido | Default        | Descripción                                       |
| ----------------------- | ------- | --------- | -------------- | -------------------------------------------------- |
| `question`            | string  | ❌ No     | —             | Texto libre con preferencias (Modos 2 y 3)        |
| `collection`          | string  | ✅ Sí    | —             | Colección a consultar (usar `"realstate_mvd"`)    |
| `operation_type`      | string  | ❌ No     | —             | `"venta"` \| `"alquiler"`                        |
| `property_type`       | string  | ❌ No     | —             | `"apartamentos"` \| `"casas"`                    |
| `barrio`              | string  | ❌ No     | —             | Nombre del barrio (ej: `"POCITOS"`, `"CARRASCO"`) |
| `min_price`           | number  | ❌ No     | —             | Precio mínimo en USD                              |
| `max_price`           | number  | ❌ No     | —             | Precio máximo en USD                              |
| `max_price_m2`        | number  | ❌ No     | —             | Precio máximo por m² en USD                      |
| `min_bedrooms`        | integer | ❌ No     | —             | Dormitorios mínimos (0 = monoambiente)            |
| `max_bedrooms`        | integer | ❌ No     | —             | Dormitorios máximos                               |
| `min_surface`         | number  | ❌ No     | —             | Superficie mínima en m²                          |
| `max_surface`         | number  | ❌ No     | —             | Superficie máxima en m²                          |
| `max_dist_plaza`      | number  | ❌ No     | —             | Distancia máxima a una plaza (metros)             |
| `max_dist_playa`      | number  | ❌ No     | —             | Distancia máxima a la playa (metros)              |
| `has_pool`            | boolean | ❌ No     | —             | Piscina                                           |
| `has_gym`             | boolean | ❌ No     | —             | Gimnasio                                          |
| `has_elevator`        | boolean | ❌ No     | —             | Ascensor                                          |
| `has_parrillero`      | boolean | ❌ No     | —             | Parrillero                                        |
| `has_terrace`         | boolean | ❌ No     | —             | Terraza                                           |
| `has_rooftop`         | boolean | ❌ No     | —             | Rooftop                                           |
| `has_security`        | boolean | ❌ No     | —             | Seguridad / portería                             |
| `has_storage`         | boolean | ❌ No     | —             | Depósito / baulera                               |
| `has_parking`         | boolean | ❌ No     | —             | Cochera                                           |
| `has_party_room`      | boolean | ❌ No     | —             | Salón de fiestas                                 |
| `has_green_area`      | boolean | ❌ No     | —             | Área verde / jardín                              |
| `has_playground`      | boolean | ❌ No     | —             | Área de juegos infantiles                        |
| `has_visitor_parking` | boolean | ❌ No     | —             | Estacionamiento para visitas                      |
| `max_recommendations` | integer | ❌ No     | `5`          | Máximo de propiedades a recomendar (1–10)        |

### Response

```json
{
  "question": "Busco algo tranquilo cerca del mar con buena luz y terraza",
  "answer": "**Recomendación 1: Pocitos — USD 289.000**\n- Características: ...",
  "collection": "realstate_mvd",
  "listings_used": [
    {
      "id": "MLU811943004",
      "barrio": "POCITOS",
      "barrio_confidence": "consistent",
      "operation_type": "venta",
      "is_dual_intent": false,
      "property_type": "apartamentos",
      "price_fixed": 289000,
      "currency_fixed": "USD",
      "price_m2": 2778.85,
      "bedrooms": 3,
      "bathrooms": 3,
      "surface_covered": 104,
      "surface_total": 109,
      "floor": 7,
      "age": 43,
      "garages": 1,
      "dist_plaza": 160.55,
      "dist_playa": 72.84,
      "n_escuelas_800m": 4,
      "source": "listing_MLU811943004",
      "semantic_score": 0.5856,
      "rerank_score": null,
      "match_score": 20,
      "rank": 1
    }
  ],
  "files_consulted": ["listing_MLU811943004"],
  "filters_applied": {
    "operation_type": "venta",
    "property_type": "apartamentos",
    "has_terrace": true,
    "max_dist_playa": 500
  },
  "response_time_sec": 18.4
}
```

#### Estructura de `listings_used`

| Campo               | Tipo           | Descripción                                                            |
| ------------------- | -------------- | ----------------------------------------------------------------------- |
| `id`              | string         | ID del listing (MercadoLibre)                                           |
| `barrio`          | string         | Barrio normalizado                                                      |
| `barrio_confidence` | string       | Confianza en la asignación del barrio: `consistent`, `marketing_inflation`, `low_confidence` |
| `is_dual_intent`  | boolean        | `true` si el listing está disponible tanto para venta como alquiler   |
| `operation_type`  | string         | `"venta"` \| `"alquiler"`                                            |
| `property_type`   | string         | `"apartamentos"` \| `"casas"`                                        |
| `price_fixed`     | number         | Precio en la moneda indicada en `currency_fixed`                       |
| `currency_fixed`  | string         | `"USD"` \| `"UYU"`                                                   |
| `price_m2`        | number \| null | Precio por m² en USD                                                  |
| `bedrooms`        | integer \| null | Número de dormitorios                                                 |
| `bathrooms`       | integer \| null | Número de baños                                                      |
| `surface_covered` | number \| null | Superficie cubierta en m²                                             |
| `surface_total`   | number \| null | Superficie total en m²                                                |
| `floor`           | integer \| null | Piso                                                                  |
| `age`             | integer \| null | Antigüedad en años                                                   |
| `garages`         | integer \| null | Número de cocheras                                                    |
| `dist_plaza`      | number \| null | Distancia a la plaza más cercana (metros)                             |
| `dist_playa`      | number \| null | Distancia a la playa más cercana (metros)                             |
| `n_escuelas_800m` | integer \| null | Número de escuelas en radio de 800m                                  |
| `semantic_score`  | number         | Score de similitud semántica FAISS (raw, 0–1)                         |
| `rerank_score`    | number \| null | Score del Cross-Encoder (`null` si reranking no activo)               |
| `match_score`     | integer \| null | Score de coincidencia para el usuario (0–100)                        |
| `rank`            | integer        | Posición en el ranking de recomendaciones                              |

### Códigos de estado

- `200 OK` — Recomendaciones generadas correctamente (o mensaje explicando por qué no hay resultados)
- `422 Unprocessable Entity` — Parámetros inválidos
- `500 Internal Server Error` — Error durante el procesamiento

### Notas

- Las preguntas fuera del dominio inmobiliario son rechazadas por el guardrail antes de llegar a retrieval.
- Los filtros explícitos del payload **siempre tienen precedencia** sobre los extraídos por el LLM desde `question`.
- Los amenities se combinan con lógica OR: se activan si el usuario los especifica en el payload **o** los menciona en la pregunta.
- Si no hay listings que coincidan con los filtros, el sistema responde indicando qué ajustar en la búsqueda.
- `barrio_confidence = "marketing_inflation"` indica que el barrio fue corregido por detección de inflación de marketing en el texto original del listing.

---

## Configuración del Sistema

### Modelos y Parámetros

| Componente                  | Modelo / Configuración                   | Ubicación                                        |
| --------------------------- | ----------------------------------------- | ------------------------------------------------- |
| **Embeddings**        | `models/gemini-embedding-001` (3072 dim) | `app/services/embedding_service.py`             |
| **Generation**        | `gemini-2.5-flash`, temperature=0.2, max_output_tokens=2000 | `app/services/generation_service.py` |
| **Preference Extraction** | `gemini-2.5-flash`, temperature=0.0   | `app/services/preference_extraction_service.py` |
| **Guardrail**         | `gemini-2.0-flash` (SDK directo)          | `app/routers/ask.py`                            |
| **Retrieval**         | top-k = 5                                 | `app/routers/ask.py`                            |
| **Reranking**         | `cross-encoder/ms-marco-MiniLM-L-6-v2`  | `app/services/reranking_service.py` (PENDIENTE) |
| **Batch size**        | 50 docs/batch, delay=5s entre batches     | `app/services/embedding_service.py`             |
| **Max desc chars**    | 5.000 chars (~1.562 tokens)               | `app/services/csv_document_service.py`          |

### Almacenamiento

| Tipo                    | Ubicación                            | Descripción                                 |
| ----------------------- | ------------------------------------- | -------------------------------------------- |
| **Documentos**    | `./docs/realstate_mvd/listings.csv` | CSV de listings limpio                       |
| **Índice FAISS** | `./faiss_index/realstate_mvd/`      | `index.faiss` (vectores) + `index.pkl` (metadata) |
| **Cloud Storage** | Bucket GCS (`CLOUD_STORAGE_BUCKET`) | Backup del índice FAISS en GCP (opcional)   |
| **Logs**          | `./logs/{processing_id}.json`       | Resultado del procesamiento asíncrono        |

---

## Flujo de Trabajo Típico

### Primera vez (construir el índice)

```bash
# 1. Verificar que el sistema está levantado
GET /api/v1/health

# 2. Iniciar indexación
POST /api/v1/documents/load-from-csv
Body: { "collection_name": "realstate_mvd" }
# → Retorna: { "processing_id": "csv_7a880482d6fa" }

# 3. Consultar estado (esperar ~10-15 min)
GET /api/v1/documents/validate-load/csv_7a880482d6fa
# → Verificar: "success": true, total_documents ~3.400
```

### Consultas

```bash
# Consulta de mercado
POST /api/v1/ask
Body: {
  "question": "¿Cuánto cuesta el m² en Pocitos vs Carrasco?",
  "collection": "realstate_mvd"
}

# Recomendación con filtros explícitos
POST /api/v1/recommend
Body: {
  "collection": "realstate_mvd",
  "operation_type": "alquiler",
  "max_price": 1200,
  "has_elevator": true,
  "max_recommendations": 5
}

# Recomendación con texto libre
POST /api/v1/recommend
Body: {
  "question": "Busco algo tranquilo para una familia con niños, cerca de plazas y escuelas",
  "collection": "realstate_mvd",
  "max_recommendations": 3
}
```
