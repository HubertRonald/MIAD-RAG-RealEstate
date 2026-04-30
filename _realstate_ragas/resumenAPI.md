# Documentación del API PENDIENTE ACTUALIZAR!!!

El API REST está compuesto por **4 endpoints principales** que permiten gestionar documentos y realizar consultas inteligentes mediante RAG (Retrieval-Augmented Generation).

---

## 1. Health Check

**Endpoint**: `GET /api/v1/health`

**Descripción**: Verificación del estado del sistema.

**Request**: Sin parámetros

**Response**:

```json
{
  "status": "healthy",
  "success": true,
  "timestamp": "2026-01-15T10:15:30.123456",
  "service": "API"
}
```

**Códigos de estado**:

- `200 OK`: Sistema funcionando correctamente

---

## 2. Load Documents

**Endpoint**: `POST /api/v1/documents/load-from-csv`

**Descripción**: Carga documentos desde XXXX y los procesa asíncronamente. El procesamiento incluye descarga, chunking (fragmentación) y generación de embeddings para crear un índice vectorial FAISS. 

### Request Body

**Mínimo requerido**:

```json
{
  "source_url": "https://drive.google.com/drive/folders/FOLDER_ID",
  "collection_name": "mi_coleccion",
  "chunking_strategy": "recursive_character"
}
```

**Con procesamiento multimodal**:

```json
{
  "source_url": "https://drive.google.com/drive/folders/FOLDER_ID",
  "collection_name": "mi_coleccion",
  "chunking_strategy": "recursive_character",
  "multimodal": true,
  "processing_options": {
    "file_extensions": ["pdf", "md"],
    "max_file_size_mb": 100,
    "timeout_per_file_seconds": 300
  }
}
```

### Parámetros

| Parámetro             | Tipo         | Requerido | Default   | Descripción                                                                 |
| ---------------------- | ------------ | --------- | --------- | ---------------------------------------------------------------------------- |
| `source_url`         | string (URL) | ✅ Sí    | -         | URL de Google Drive (carpeta o archivo individual)                           |
| `collection_name`    | string       | ✅ Sí    | -         | Nombre único para identificar esta colección de documentos                 |
| `chunking_strategy`  | string       | ✅ Sí    | -         | Estrategia de fragmentación (ver opciones abajo)                            |
| `multimodal`         | boolean      | ❌ No     | `false` | Activa procesamiento multimodal de PDFs (texto + imágenes VLM + tablas)    |
| `processing_options` | object       | ❌ No     | ver abajo | Configuración adicional de procesamiento                                    |

#### Estrategias de Chunking

| Estrategia                    | Valor                     | Descripción                                                      |
| ----------------------------- | ------------------------- | ----------------------------------------------------------------- |
| **Recursive Character** | `"recursive_character"` | División recursiva por caracteres (recomendado para uso general) |
| **Fixed Size**          | `"fixed_size"`          | Chunks de tamaño fijo con overlap configurable                   |
| **Semantic**            | `"semantic"`            | División basada en significado semántico                        |
| **Document Structure**  | `"document_structure"`  | División basada en estructura del documento (headers, párrafos) |

#### Processing Options (defaults)

| Campo                        | Tipo          | Default       | Descripción                               |
| ---------------------------- | ------------- | ------------- | ------------------------------------------ |
| `file_extensions`          | array[string] | `["pdf", "md"]` | Extensiones de archivo permitidas        |
| `max_file_size_mb`         | integer       | `100`       | Tamaño máximo por archivo en MB (1-1000) |
| `timeout_per_file_seconds` | integer       | `300`       | Timeout por archivo en segundos (30-3600) |


### Response

```json
{
  "success": true,
  "message": "Procesamiento iniciado en background",
  "processing_id": "proc_b8ae8dbda5f9",
  "timestamp": "2026-01-15T10:15:30.123456"
}
```

| Campo             | Tipo              | Descripción                                         |
| ----------------- | ----------------- | ---------------------------------------------------- |
| `success`       | boolean           | Indica si el procesamiento se inició correctamente  |
| `message`       | string            | Mensaje descriptivo                                  |
| `processing_id` | string            | ID único para consultar el estado del procesamiento |
| `timestamp`     | string (ISO 8601) | Momento en que se inició el procesamiento           |

### Códigos de estado

- `200 OK`: Procesamiento iniciado correctamente
- `400 Bad Request`: Parámetros inválidos
- `404 Not Found`: URL no accesible

### Notas importantes

- El procesamiento es **asíncrono**: el endpoint retorna inmediatamente con un `processing_id`
- Use el endpoint `GET /documents/load-from-url/{processing_id}` para consultar el estado
- Los parámetros de chunking (chunk_size, overlap, etc.) se configuran en `app/services/chunking_service.py`
- El modelo de embeddings se configura en `app/services/embedding_service.py`
- Los archivos Markdown nunca usan procesamiento multimodal independientemente del flag

---

## 3. Validate Load

**Endpoint**: `GET /api/v1/documents/load-from-url/{processing_id}`

**Descripción**: Consulta el estado y resultados de un procesamiento de documentos iniciado con el endpoint anterior.

### Parámetros

| Parámetro        | Ubicación | Tipo   | Descripción                     |
| ----------------- | ---------- | ------ | -------------------------------- |
| `processing_id` | path       | string | ID del procesamiento a consultar |

### Response

```json
{
  "success": true,
  "message": "Documentos descargados y procesados exitosamente con RAG",
  "data": {
    "processing_summary": {
      "rag_processing": true,
      "vector_store_created": true,
      "total_processing_time_sec": 45.23
    },
    "collection_info": {
      "name": "mi_coleccion",
      "documents_found": 5,
      "documents_processed_successfully": 5,
      "documents_failed": 0,
      "documents_count_before": 0,
      "documents_count_after": 5,
      "total_chunks_before": 0,
      "total_chunks_after": 150,
      "storage_size_mb": 12.5
    },
    "documents_processed": [
      {
        "filename": "documento1.pdf",
        "file_path": "./docs/mi_coleccion/documento1.pdf",
        "file_size_bytes": 524288,
        "download_url": "https://drive.google.com/...",
        "processing_time_seconds": 8.5
      }
    ],
    "failed_documents": [],
    "chunking_statistics": {
      "total_chunks": 150,
      "avg_chunk_size": 512,
      "min_chunk_size": 200,
      "max_chunk_size": 800,
      "strategy_used": "recursive_character"
    },
    "embedding_statistics": {
      "model_used": "models/gemini-embedding-001",
      "total_embeddings_generated": 150,
      "vector_store_created": true,
      "vector_store_path": "./faiss_index/mi_coleccion",
      "vector_store_type": "FAISS",
      "multimodal_processing": true,
      "multimodal_chunks": 12,
      "chunks_with_images": 8,
      "chunks_with_tables": 4,
      "total_vlm_calls": 23
    },
    "rag_status": {
      "vector_store_exists": true,
      "vector_store_ready": true,
      "rag_ready": true,
      "document_count": 150,
      "documents_in_collection": 5,
      "pdf_files": ["documento1.pdf", "documento2.pdf"]
    },
    "warnings": [],
    "processing_id": "proc_b8ae8dbda5f9",
    "timestamp": "2026-01-15T10:20:45.789123"
  }
}
```

### Estructura de la respuesta

#### `processing_summary`

- `rag_processing`: Indica si el procesamiento RAG se completó
- `vector_store_created`: Indica si se creó el índice vectorial
- `total_processing_time_sec`: Tiempo total de procesamiento

#### `collection_info`

- `name`: Nombre de la colección
- `documents_found`: Total de documentos encontrados en la URL
- `documents_processed_successfully`: Documentos procesados sin errores
- `documents_failed`: Documentos que fallaron
- `total_chunks_after`: Total de chunks generados
- `storage_size_mb`: Espacio ocupado en disco

#### `embedding_statistics` (campos multimodales, solo si `multimodal=true`)

| Campo                    | Tipo    | Descripción                                         |
| ------------------------ | ------- | ---------------------------------------------------- |
| `multimodal_processing` | boolean | Indica si se usó procesamiento multimodal           |
| `multimodal_chunks`     | integer | Chunks provenientes de páginas con contenido visual |
| `chunks_with_images`    | integer | Chunks que contienen descripciones `[FIGURA]`      |
| `chunks_with_tables`    | integer | Chunks que contienen tablas en Markdown             |
| `total_vlm_calls`       | integer | Total de llamadas al VLM de Gemini                  |

#### `rag_status`

- `vector_store_ready`: Indica si el índice vectorial está listo para consultas
- `rag_ready`: Indica si el sistema RAG está completamente operativo
- `document_count`: Número de embeddings en el índice FAISS

### Códigos de estado

- `200 OK`: Información recuperada correctamente
- `400 Bad Request`: processing_id vacío
- `404 Not Found`: processing_id no existe
- `422 Unprocessable Entity`: Archivo JSON corrupto
- `500 Internal Server Error`: Error interno del servidor

---

## 4. Ask Questions

**Endpoint**: `POST /api/v1/ask`

**Descripción**: Realiza consultas inteligentes al sistema RAG sobre documentos cargados. El sistema recupera fragmentos relevantes, aplica reranking adaptativo según el tipo de contenido, y genera una respuesta en el idioma de la pregunta (español o inglés).

### Request Body

**Mínimo requerido**:

```json
{
  "question": "¿Cuáles son los puntos principales del documento?",
  "collection": "mi_coleccion"
}
```

**Completo (con opciones avanzadas)**:

```json
{
  "question": "¿Cuáles son los puntos principales del documento?",
  "collection": "mi_coleccion",
  "use_reranking": true,
  "use_query_rewriting": true
}
```

### Parámetros

| Parámetro              | Tipo    | Requerido | Default   | Descripción                                                                              |
| ----------------------- | ------- | --------- | --------- | ----------------------------------------------------------------------------------------- |
| `question`            | string  | ✅ Sí    | -         | La pregunta a responder (español o inglés)                                              |
| `collection`          | string  | ✅ Sí    | -         | Nombre de la colección a consultar                                                       |
| `use_reranking`       | boolean | ❌ No     | `false` | Aplica reranking con Cross-Encoder. En colecciones multimodales, el reranking se aplica solo a chunks con contenido visual (`[FIGURA]` o tablas); los chunks de texto puro lo bypasean automáticamente. |
| `use_query_rewriting` | boolean | ❌ No     | `false` | Reescribe la consulta con estrategia `few_shot` para mejorar el retrieval              |

### Response

```json
{
  "question": "¿Cuáles son los puntos principales del documento?",
  "final_query": "¿Cuáles son los puntos principales del documento?",
  "answer": "Los puntos principales del documento incluyen: 1) Definición de objetivos...",
  "collection": "mi_coleccion",
  "files_consulted": ["documento1.pdf", "documento2.pdf"],
  "context_docs": [
    {
      "file_name": "documento1.pdf",
      "chunk_type": "text",
      "snippet": "Los objetivos principales son...",
      "content": "Los objetivos principales son definir las metas estratégicas...",
      "priority": "high",
      "rerank_score": null
    },
    {
      "file_name": "documento1.pdf",
      "chunk_type": "figure",
      "snippet": "[FIGURA] Diagrama de arquitectura del sistema...",
      "content": "[FIGURA] Diagrama de arquitectura del sistema mostrando los tres componentes principales...",
      "priority": "high",
      "rerank_score": 8.4231
    }
  ],
  "reranker_used": true,
  "query_rewriting_used": true,
  "response_time_sec": 2.45
}
```

### Estructura de la respuesta

| Campo                    | Tipo          | Descripción                                                                        |
| ------------------------ | ------------- | ----------------------------------------------------------------------------------- |
| `question`             | string        | Pregunta original del usuario                                                       |
| `final_query`          | string        | Consulta final usada para el retrieval (reescrita si `use_query_rewriting=true`)  |
| `answer`               | string        | Respuesta generada en el idioma de la pregunta                                      |
| `collection`           | string        | Colección consultada                                                               |
| `files_consulted`      | array[string] | Lista de archivos fuente consultados                                                |
| `context_docs`         | array[object] | Fragmentos utilizados como contexto (ver estructura abajo)                          |
| `reranker_used`        | boolean       | Indica si se aplicó reranking en algún chunk                                       |
| `query_rewriting_used` | boolean       | Indica si se aplicó reescritura de consulta                                         |
| `response_time_sec`    | number        | Tiempo total de procesamiento en segundos                                           |

#### Estructura de `context_docs`

| Campo            | Tipo              | Valores posibles                              | Descripción                                                         |
| ---------------- | ----------------- | --------------------------------------------- | -------------------------------------------------------------------- |
| `file_name`    | string            | -                                             | Nombre del archivo fuente                                            |
| `chunk_type`   | string            | `"text"`, `"figure"`, `"table"`, `"code"` | Tipo de contenido del chunk                                          |
| `snippet`      | string            | -                                             | Primeros 200 caracteres del fragmento                                |
| `content`      | string            | -                                             | Contenido completo del fragmento                                     |
| `priority`     | string            | `"high"`, `"medium"`                        | Prioridad del fragmento en el contexto                               |
| `rerank_score` | number \| null    | -                                             | Score del Cross-Encoder (`null` si el chunk fue omitido por el reranking) |

#### Tipos de chunk (`chunk_type`)

| Valor       | Descripción                                                            |
| ----------- | ----------------------------------------------------------------------- |
| `"text"`  | Texto extraído directamente del PDF                                    |
| `"figure"` | Descripción generada por VLM, marcada con `[FIGURA]`                 |
| `"table"` | Tabla extraída con pdfplumber, serializada en Markdown                 |
| `"code"`  | Bloque de código detectado por presencia de keywords (`def`, `class`) |

### Comportamiento del reranking adaptativo

Cuando `use_reranking=true`, el sistema aplica una estrategia de **reranking selectivo**:

- Chunks con `chunk_type` de `"figure"` o `"table"` (o que contengan `[FIGURA]`) → reranking con `ms-marco-MiniLM-L-6-v2`
- Chunks con `chunk_type` de `"text"` o `"code"` → bypass del reranking (documentos pasados directamente a generación)

Esta decisión se basa en los resultados de evaluación RAGAS: el reranking mejora la precisión para contenido visual pero no aporta beneficio para chunks de texto puro en este corpus.

### Códigos de estado

- `200 OK`: Consulta procesada correctamente
- `400 Bad Request`: Pregunta o colección vacía
- `500 Internal Server Error`: Error durante el procesamiento

### Notas importantes

- El idioma de la respuesta se detecta automáticamente desde la pregunta. Preguntas en español → respuesta en español; preguntas en inglés → respuesta en inglés.
- El parámetro `k` (número de documentos a recuperar) se configura en `app/routers/ask.py` al construir el `RetrievalService` (valor actual: `k=10`)
- `use_reranking` y `use_query_rewriting` están implementados vía `RAGGraphService` con los servicios `RerankingService` (cross-encoder `ms-marco-MiniLM-L-6-v2`) y `QueryRewritingService` (few-shot rewriting)
- Si no hay documentos relevantes, el sistema indica claramente la falta de información sin inventar respuestas

---

## Configuración del Sistema

### Modelos y Parámetros

| Componente            | Modelo / Configuración                    | Ubicación                                    |
| --------------------- | ------------------------------------------ | --------------------------------------------- |
| **Embeddings**  | `models/gemini-embedding-001`            | `app/services/embedding_service.py`         |
| **Chunking**    | Parámetros por estrategia                | `app/services/chunking_service.py`          |
| **Retrieval**   | top-k = 10                                 | `app/routers/ask.py` (al construir `RetrievalService`)  |
| **Reranking**   | `cross-encoder/ms-marco-MiniLM-L-6-v2`  | `app/services/reranking_service.py`         |
| **VLM**         | `gemini-2.5-flash` (imágenes ≥ 150px)  | `app/services/multimodal_document_service.py` |
| **Generation**  | `gemini-2.5-flash`, temperature=0.2      | `app/services/generation_service.py`        |
| **Normalización** | NFKC + ACCENT_MAP + hyphen regex        | `app/utils/text_utils.py`                   |

### Almacenamiento

| Tipo                    | Ubicación                           | Descripción                |
| ----------------------- | ------------------------------------ | --------------------------- |
| **Documentos**    | `./docs/{collection_name}/`        | PDFs y archivos descargados |
| **Índice FAISS** | `./faiss_index/{collection_name}/` | Vectorstore con embeddings (local) |
| **Cloud Storage** | Bucket GCS (`CLOUD_STORAGE_BUCKET`) | Backup persistente del índice FAISS en GCP |
| **Logs**          | `./logs/{processing_id}.json`      | Resultados de procesamiento |

---

## Flujo de Trabajo Típico

1. **Verificar salud del sistema**

   ```bash
   GET /api/v1/health
   ```

2. **Cargar documentos (con procesamiento multimodal)**

   ```bash
   POST /api/v1/documents/load-from-url
   Body: {
     "source_url": "https://drive.google.com/...",
     "collection_name": "mi_coleccion",
     "chunking_strategy": "recursive_character",
     "multimodal": true
   }
   # Retorna: { "processing_id": "proc_abc123" }
   ```

3. **Validar carga (esperar procesamiento)**

   ```bash
   GET /api/v1/documents/load-from-url/proc_abc123
   # Verificar: "rag_ready": true
   ```

4. **Realizar consultas**

   ```bash
   POST /api/v1/ask
   Body: {
     "question": "¿Qué es un transformer?",
     "collection": "mi_coleccion",
     "use_reranking": true,
     "use_query_rewriting": true
   }
   ```

---
