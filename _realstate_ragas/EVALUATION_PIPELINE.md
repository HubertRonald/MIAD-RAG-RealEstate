# Pipeline de Evaluación RAG — Sistema Inmobiliario Montevideo

Documentación del pipeline de evaluación con RAGAS + MLflow para el sistema RAG inmobiliario.

---

## Descripción General

El pipeline mide la calidad de dos endpoints:

| Endpoint | Descripción |
|---|---|
| `/ask` | Consultas de mercado orquestadas por LangGraph |
| `/recommend` | Recomendaciones de propiedades via `PropertyFilters` → `RetrievalService` → `GenerationService` |

### Métricas RAGAS

| Métrica | Componente evaluado | Criterio de aceptación | Criterio de rechazo |
|---|---|---|---|
| `context_precision` | Retriever | ≥ 0.75 | < 0.65 |
| `context_recall` | Retriever | ≥ 0.65 | < 0.50 |
| `faithfulness` | Generador | ≥ 0.85 | < 0.75 |
| `answer_relevancy` | Generador | ≥ 0.78 | < 0.65 |
| `answer_correctness` | RAG end-to-end | ≥ 0.60 | < 0.35 |
| `avg_cosine_similarity` | Embeddings + FAISS | ≥ 0.72 | — |

Cada métrica tiene tres zonas: **PASS** / **WARN** (zona gris entre umbrales) / **FAIL**. Los tests hacen `assert` solo sobre FAIL — un WARN pasa pytest pero queda visible en MLflow para revisión humana antes de un release.

> **Nota sobre avg_cosine_similarity:** El índice FAISS es `IndexFlatL2` (distancia L2, no coseno).
> LangChain normaliza los scores L2 via `1 / (1 + distance)` — los scores raw están en [0.60, 0.72].
> La función `l2_relevance_to_cosine()` en `retrieval_service.py` convierte estos scores a
> verdadera similitud coseno antes de loguear a MLflow. Los scores convertidos están en [0.78, 0.92].
> El umbral de aceptación 0.72 aplica sobre los scores **convertidos**.

### Tests Funcionales

| Test | Criterio |
|---|---|
| Latencia de respuesta | promedio ≤ 30s por query |
| Tasa de rechazo de queries fuera de dominio | ≥ 95% de queries inválidas rechazadas |
| F1 de extracción de filtros | F1 ≥ 0.85 (via `PreferenceExtractionService`) |
| Calidad de respuesta para queries inválidas | 100% de respuestas limpias, sin fuga del prompt ni errores técnicos |

---

## Estructura de Directorios

```
tests/ragas/
├── conftest.py                          # Fixtures de pytest, ciclo de vida MLflow, builders de datasets
├── evaluation_data.py                   # Preguntas y referencias para /ask y /recommend
├── thresholds.py                        # Fuente única de verdad para valores de aceptación/rechazo
├── mlflow_utils.py                      # Helpers de MLflow (experimentos, logging de métricas)
├── test_retrieval_metrics.py            # context_precision, context_recall (/ask)
├── test_generation_metrics.py           # faithfulness, answer_relevancy, answer_correctness (/ask)
├── test_recommend_metrics.py            # Las 5 métricas RAGAS (/recommend)
├── test_cosine_similarity.py            # avg_cosine_similarity (FAISS directo, sin evaluador LLM)
├── test_functional.py                   # Latencia, tasa de rechazo, calidad de respuesta
├── run_experiment.py                    # Script standalone para comparar múltiples configuraciones
├── cache/                               # Caches JSON de datasets generados automáticamente
│   ├── eval_ask_realstate_mvd_k3.json
│   └── eval_recommend_realstate_mvd_k3.json
├── ragas_metrics_report_ask.json        # Reporte de métricas para /ask (generado automáticamente)
└── ragas_metrics_report_recommend.json  # Reporte de métricas para /recommend (generado automáticamente)
```

---

## Dataset de Evaluación

### /ask — 8 preguntas

Todas las preguntas son respondibles directamente desde listings individuales. Se excluyeron
preguntas de agregación (rangos de precio, comparativas entre barrios, rankings) ya que el
índice FAISS contiene listings individuales, no datos de mercado agregados.

| # | Pregunta | Tipo |
|---|---|---|
| Q1 | Características de apts +2 dorm en venta en Pocitos | Características por barrio |
| Q2 | Barrios con apartamentos cerca de la playa | Disponibilidad geográfica |
| Q3 | Amenities en edificios de alquiler en Pocitos | Amenities por zona |
| Q4 | Amenities en apartamentos disponibles en Carrasco | Amenities por barrio |
| Q5 | Cómo describen los listings las propiedades en el Centro | Contexto del entorno |
| Q6 | Tipo de propiedades en venta en Punta Carretas | Disponibilidad por operación |
| Q7 | Casas en alquiler con jardín en Montevideo | Tipo de propiedad + característica |
| Q8 | Apartamentos en Cordón con cochera y ascensor | Filtros específicos de características |

### /recommend — 7 preguntas (2 / 2 / 3 por modo)

| Modo | Descripción | Preguntas |
|---|---|---|
| **Mode 1** — Solo filtros estructurados | `question` vacío, `_build_fallback_query()` construye la query semántica | Buceo alquiler 2 dorm ascensor / Carrasco casas alquiler 3 dorm cochera |
| **Mode 2** — Solo texto libre | `filter_kwargs` vacío, `PreferenceExtractionService` extrae los filtros | Inversión en Montevideo / Apartamento con ascensor y parrillero cerca de plaza |
| **Mode 3** — Híbrido | Texto + filtros estructurados combinados | Pocitos venta 2 dorm piscina ≤ 250k / Cerca del mar 3 dorm cochera parrillero / Monoambiente–1 dorm alquiler Centro–Cordón (multi-barrio) |

---

## Configuración Inicial

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

La única dependencia específica del pipeline de evaluación que no estaba en el stack de la app es `mlflow`. Todo lo demás (ragas, pandas, langchain-google-genai) ya está presente.

### 2. Variables de entorno

Crear un archivo `.env` en la raíz del repositorio si no existe:

```bash
# Requerido — API key de Google AI (la misma que usa la app)
GOOGLE_API_KEY=tu_api_key_aqui

# MLflow — estos son los valores por defecto, solo setear si se quiere cambiar
MLFLOW_EXPERIMENT_NAME=realstate_rag_evaluation
MLFLOW_TRACKING_URI=mlruns

# Configuración de evaluación — todos opcionales, los defaults corresponden a la config de producción
EVAL_K=3
EVAL_FETCH_K=60
EVAL_MAX_RECOMMENDATIONS=5
EVAL_LLM_MODEL=gemini-2.5-flash
EVAL_EMBEDDING_MODEL=models/gemini-embedding-001
EVAL_TEMPERATURE=0.0
EVAL_COLLECTION=realstate_mvd
EVAL_FAISS_PATH=./faiss_index/realstate_mvd
EVAL_ENDPOINT=ask
```

> **Nota sobre temperatura:** La evaluación siempre corre con `temperature=0.0` para garantizar
> reproducibilidad. La producción usa `0.2`. No modificar `EVAL_TEMPERATURE` en evaluaciones.

### 3. Exportar la API key antes de correr los tests

El archivo `.env` lo carga la app al iniciar, pero pytest necesita la key exportada en la terminal:

```bash
export GOOGLE_API_KEY="tu_api_key_aqui"

# Alternativa — cargar desde .env con source (más seguro que xargs para keys con caracteres especiales):
set -a && source .env && set +a
```

### 4. Crear directorios necesarios

```bash
mkdir -p tests/ragas/cache
```

`mlruns/` se crea automáticamente por MLflow en el primer run.

### 5. `pytest.ini` — requerido para importar módulos de la app

Verificar que `pytest.ini` en la raíz del repositorio incluye `pythonpath = .`:

```ini
[pytest]
pythonpath = .
testpaths = tests
# ... resto de la configuración
```

Sin esto, `from app.services...` en `conftest.py` lanza `ModuleNotFoundError`.

---

## Configurar MLflow Localmente

Abrir una terminal dedicada e iniciar el servidor de tracking desde la raíz del repositorio:

```bash
cd /ruta/a/realstate_ragas
mlflow ui --port 5000
```

Dejar esa terminal abierta. Abrir `http://127.0.0.1:5000` en el browser.

> MLflow almacena todo en `mlruns/` en la raíz del repositorio. No requiere base de datos
> ni configuración adicional para uso local. Ver la sección Cloud más abajo para setups remotos.

### Nueva línea base después de re-indexar

Cuando el índice FAISS es reconstruido (por ejemplo tras limpiar el dataset de listings),
los resultados anteriores no son comparables. Para empezar una nueva línea base limpia:

```bash
# Opción A — nuevo experimento (recomendado, preserva historial anterior)
MLFLOW_EXPERIMENT_NAME=realstate_rag_v2 pytest tests/ragas/ -v -s

# Opción B — borrar mlruns completamente (borra todo el historial)
rm -rf mlruns/
pytest tests/ragas/ -v -s
```

Siempre borrar los caches antes de una nueva línea base:

```bash
rm -f tests/ragas/cache/eval_ask_realstate_mvd_k3.json
rm -f tests/ragas/cache/eval_recommend_realstate_mvd_k3.json
```

---

## Ejecutar los Tests

Siempre correr desde la **raíz del repositorio**.

### Secuencia incremental recomendada (primera vez o tras re-indexar)

**Paso 1 — Verificar FAISS y retrieval (sin llamadas a la API):**
```bash
pytest tests/ragas/test_cosine_similarity.py -v -s
```
Confirma que el índice FAISS carga, `retrieve_with_scores()` funciona, la conversión
L2→coseno produce scores en [0.78, 0.92], y aparece un run en `http://127.0.0.1:5000`.

**Paso 2 — Una métrica RAGAS para confirmar el pipeline completo:**
```bash
pytest tests/ragas/test_retrieval_metrics.py::test_context_precision_above_threshold -v -s
```

**Paso 3 — Tests funcionales (usan el LLM pero no el evaluador RAGAS):**
```bash
pytest tests/ragas/test_functional.py -v -s
```

**Paso 4 — Suite completa:**
```bash
pytest tests/ragas/ -v -s
```

El flag `-s` es importante — permite que la tabla de resumen del run de MLflow se imprima en la terminal al finalizar la sesión.

### Correr archivos de tests específicos

```bash
# Solo métricas de retrieval (/ask)
pytest tests/ragas/test_retrieval_metrics.py -v -s

# Solo métricas de generación (/ask)
pytest tests/ragas/test_generation_metrics.py -v -s

# Todas las métricas de /recommend
pytest tests/ragas/test_recommend_metrics.py -v -s

# Solo tests funcionales
pytest tests/ragas/test_functional.py -v -s

# Una métrica específica
pytest tests/ragas/test_retrieval_metrics.py::test_context_precision_above_threshold -v -s
```

### Cambiar la configuración sin modificar código

```bash
# Evaluar con k=5 en lugar de k=3
EVAL_K=5 EVAL_FETCH_K=100 pytest tests/ragas/ -v -s

# Evaluar solo el endpoint /recommend
EVAL_ENDPOINT=recommend pytest tests/ragas/test_recommend_metrics.py -v -s
```

Cada combinación diferente de `k` / `collection` genera su propio archivo de cache.

---

## Cache de Datasets

Construir el dataset de evaluación requiere llamar a Gemini para cada pregunta (14 llamadas
en total: 8 para /ask + 6 para /recommend). Para evitar repetir esto en cada ejecución,
los datasets se cachean como archivos JSON en `tests/ragas/cache/`.

Los nombres de los archivos incluyen la colección y el valor de k:
```
tests/ragas/cache/eval_ask_realstate_mvd_k3.json
tests/ragas/cache/eval_recommend_realstate_mvd_k3.json
```

**Para forzar regeneración** (después de actualizar el índice FAISS, cambiar las preguntas,
o al iniciar una nueva línea base):
```bash
rm -f tests/ragas/cache/eval_ask_realstate_mvd_k3.json
rm -f tests/ragas/cache/eval_recommend_realstate_mvd_k3.json
```

---

## Scores de Relevancia y match_score

### Conversión L2 → Coseno

El índice FAISS es `IndexFlatL2`. LangChain normaliza las distancias L2 via
`relevance_score = 1 / (1 + L2_distance)`, produciendo scores en [0.60, 0.72] para este corpus.

La función `l2_relevance_to_cosine()` en `retrieval_service.py` convierte estos scores a
verdadera similitud coseno (scores convertidos: [0.78, 0.92]):

```python
def l2_relevance_to_cosine(relevance_score: float) -> float:
    l2_distance = (1.0 / relevance_score) - 1.0
    cosine = 1.0 - (l2_distance ** 2) / 2.0
    return max(0.0, min(1.0, cosine))
```

Esta función se usa en:
- `test_cosine_similarity.py` — para calcular `avg_cosine_similarity`
- `routers/ask.py` — antes de llamar a `_cosine_to_match_score()`

### match_score (1–100)

El campo `match_score` en `ListingInfo` convierte el score coseno a un entero 1–100
para uso directo en el frontend. Los límites en `routers/ask.py` están calibrados con
los scores observados en este corpus:

```python
_SCORE_LOW  = 0.78   # mínimo observado (Q8: 0.788)
_SCORE_HIGH = 0.92   # máximo observado (Q3: 0.917)
```

> **Recalibrar después de re-indexar:** Los scores coseno pueden cambiar si el corpus cambia.
> Correr `test_cosine_similarity.py`, revisar los scores por query en MLflow
> (`functional_avg_cosine_q01`...`q08`), y actualizar `_SCORE_LOW` / `_SCORE_HIGH`
> en `routers/ask.py` según los nuevos valores observados.

---

## Campos de Metadata en FAISS

Tras la limpieza del dataset, los documentos indexados incluyen los siguientes campos
nuevos además de los originales:

| Campo | Tipo | Descripción |
|---|---|---|
| `barrio_fixed` | `str` | Barrio corregido con ground truth geoespacial. Reemplaza `barrio`. |
| `barrio_confidence` | `str` | Calidad del barrio: `consistent` \| `no_barrio_in_text` \| `genuine_ambiguity` \| `marketing_inflation` |
| `is_dual_intent` | `bool` | `True` si el listing está disponible para venta Y alquiler simultáneamente |

El filtro `PropertyFilters` usa `barrio_fixed` como clave de metadata (no `barrio`).
El campo `barrio` en `ListingInfo` (modelo de respuesta) mapea desde `barrio_fixed`.

### Multi-barrio filtering

`PropertyFilters.barrio` acepta string o lista de strings:

```python
# Un barrio
filters = PropertyFilters(barrio="POCITOS")

# Múltiples barrios
filters = PropertyFilters(barrio=["CENTRO", "CORDON"])
```

---

## Comparar Configuraciones con `run_experiment.py`

```bash
# Correr todas las configuraciones definidas en EXPERIMENT_CONFIGS
python tests/ragas/run_experiment.py

# Correr una configuración específica por nombre
python tests/ragas/run_experiment.py --config baseline_k3

# Evaluar solo /ask
python tests/ragas/run_experiment.py --endpoint ask

# Evaluar solo /recommend
python tests/ragas/run_experiment.py --endpoint recommend
```

### Agregar una nueva configuración

Editar `EXPERIMENT_CONFIGS` en `run_experiment.py`:

```python
{
    "name":                   "mi_experimento",
    "k":                      7,
    "fetch_k":                140,
    "max_recommendations":    5,
    "prompt_variant":         "default",
    "description_truncation": "full",
    "llm_model":              "gemini-2.5-flash",
    "embedding_model":        "models/gemini-embedding-001",
    "temperature":            0.0,
    "collection":             "realstate_mvd",
    "faiss_path":             "./faiss_index/realstate_mvd",
},
```

---

## Interpretar Resultados en MLflow

Abrir `http://127.0.0.1:5000` y seleccionar el experimento `realstate_rag_evaluation`.

### Métricas clave para comparar entre runs

| Clave en MLflow | Significado |
|---|---|
| `ask_context_precision_avg` | Precisión de contexto promedio para /ask |
| `ask_context_recall_avg` | Recall de contexto promedio para /ask |
| `ask_faithfulness_avg` | Fidelidad promedio para /ask |
| `ask_answer_relevancy_avg` | Relevancia de respuesta promedio para /ask |
| `ask_answer_correctness_avg` | Correctitud de respuesta promedio para /ask |
| `recommend_context_precision_avg` | Precisión de contexto promedio para /recommend |
| `functional_avg_cosine_similarity` | Similitud coseno promedio (convertida de L2) |
| `functional_rejection_rate` | Tasa de rechazo de queries fuera de dominio |
| `functional_filter_extraction_f1` | F1 de extracción de filtros |

### Codificación de estado

Cada métrica tiene una clave `_status` correspondiente logueada como float:

| Valor | Significado |
|---|---|
| `1.0` | PASS — score ≥ umbral de aceptación |
| `0.5` | WARN — score en la zona gris entre umbrales |
| `0.0` | FAIL — score < umbral de rechazo |

### Columna Dataset en MLflow

Cada run loguea el dataset de evaluación como artefacto via `mlflow.log_input()`.
Aparece en la pestaña "Inputs" del run con el nombre `eval_ask_realstate_mvd_k3`
o `eval_recommend_realstate_mvd_k3`. Contiene las preguntas y referencias usadas.

### Columna Model en MLflow

Cada run loguea un modelo pyfunc `rag_model` con la configuración completa del sistema
(LLM, embeddings, k, fetch_k, temperatura, variante de prompt). Aparece en la pestaña
"Artifacts" del run. Para registrar en el Model Registry y comparar versiones, cambiar
`registered_model_name = None` a `registered_model_name = "realstate_rag"` en
`mlflow_utils.log_model_artifact()`.

---

## Actualizar Umbrales

Todos los valores de aceptación/rechazo viven en un único lugar:

```
tests/ragas/thresholds.py
```

Editar `RAGAS_THRESHOLDS` o `FUNCTIONAL_THRESHOLDS` ahí. No se necesitan cambios en
los archivos de tests — todos importan desde `thresholds.py`.

---

## Límites del Tier Gratuito de Gemini

| Límite | Valor |
|---|---|
| Requests por minuto (RPM) | 5 |
| Requests por día (RPD) | 20 |
| Tokens por minuto (TPM) | 250,000 |

El `RunConfig(max_workers=1)` en todos los archivos de tests respeta el límite de RPM.
Para el límite de RPD, usar el cache de datasets y correr una o dos métricas por día
hasta que la línea base esté establecida. Con el dataset actual (15 preguntas total: 8 /ask + 7 /recommend),
construir ambos datasets consume 15 RPD — dejar al menos 5 RPD para las evaluaciones RAGAS.

---

## Migrar a Cloud

El único cambio necesario es una variable de entorno — sin modificaciones en el código:

```bash
# Local (default)
MLFLOW_TRACKING_URI=mlruns

# Servidor MLflow self-hosted
MLFLOW_TRACKING_URI=http://tu-servidor-mlflow:5000

# MLflow gestionado en Databricks
MLFLOW_TRACKING_URI=databricks
DATABRICKS_HOST=https://tu-workspace.azuredatabricks.net
DATABRICKS_TOKEN=tu_token

# Artifact store respaldado en AWS S3
MLFLOW_TRACKING_URI=http://tu-servidor-mlflow:5000
MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
AWS_ACCESS_KEY_ID=tu_key
AWS_SECRET_ACCESS_KEY=tu_secret
```

---

## Solución de Problemas

**`DefaultCredentialsError` al correr los tests:**
`GOOGLE_API_KEY` no está exportada en la terminal actual. Ejecutar:
```bash
export GOOGLE_API_KEY="tu_api_key_aqui"
```

**`ModuleNotFoundError: No module named 'app'`:**
Falta `pythonpath = .` en `pytest.ini`. Ver sección Configuración Inicial paso 5.

**`FileNotFoundError` para el índice FAISS:**
El índice aún no fue construido, o `EVAL_FAISS_PATH` apunta a una ubicación incorrecta.
Llamar primero al endpoint `/load-from-csv` para construir la colección y luego re-correr.

**Los tests pasan pero no aparece nada en la UI de MLflow:**
Verificar que `mlflow ui` estaba corriendo antes de iniciar pytest, y que ambos fueron
lanzados desde la misma raíz del repositorio para compartir la misma carpeta `mlruns/`.

**El archivo de cache genera resultados desactualizados:**
Eliminar el archivo correspondiente en `tests/ragas/cache/` para forzar regeneración.
Siempre borrar el cache después de re-indexar el FAISS con un dataset limpiado.

**avg_cosine_similarity baja inesperadamente tras re-indexar:**
Los scores coseno convertidos deberían estar en [0.78, 0.92] para este corpus con
`IndexFlatL2`. Si caen por debajo de 0.72, revisar que `l2_relevance_to_cosine()` se
está aplicando en `test_cosine_similarity.py` y que `_SCORE_LOW` / `_SCORE_HIGH` en
`routers/ask.py` están calibrados con los nuevos scores observados.

**La evaluación RAGAS es lenta:**
Es esperado — `max_workers=1` es intencional para respetar el límite de 5 RPM.
Cada métrica toma aproximadamente 2–3 minutos para el dataset de 8 preguntas.
