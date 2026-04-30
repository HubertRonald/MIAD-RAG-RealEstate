# Pipeline de Evaluación RAG — Sistema de Recomendación Inmobiliario Montevideo

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

### Tests Funcionales

| Test | Criterio |
|---|---|
| Latencia de respuesta | promedio ≤ 30s por query |
| Tasa de rechazo de queries fuera de dominio | ≥ 95% de queries inválidas rechazadas |
| Accuracy de intención + F1 de extracción de filtros | Accuracy ≥ 90%, F1 ≥ 85%  |
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

# O cargar todo el .env de una vez:
export $(cat .env | xargs) && pytest ...
```

### 4. Crear directorios necesarios

```bash
mkdir -p tests/ragas/cache
```

`mlruns/` se crea automáticamente por MLflow en el primer run.

---

## Configurar MLflow Localmente

Abrir una terminal dedicada e iniciar el servidor de tracking desde la raíz del repositorio:

```bash
cd /ruta/a/realstate_ragas
mlflow ui --port 5000
```

Dejar esa terminal abierta. Abrir `http://localhost:5000` en el browser — se verá la página de experimentos. Cada ejecución de pytest o de `run_experiment.py` crea un nuevo run bajo el experimento `realstate_rag_evaluation`.

> MLflow almacena todo en `mlruns/` en la raíz del repositorio. No requiere base de datos
> ni configuración adicional para uso local. Ver la sección Cloud más abajo para setups remotos.

---

## Ejecutar los Tests

Siempre correr desde la **raíz del repositorio** para que los paths relativos de `mlruns/` y `tests/ragas/cache/` resuelvan correctamente.

### Secuencia incremental recomendada (primera vez)

Correr las 5 métricas RAGAS de una sola vez en el primer intento puede agotar la cuota gratuita de Gemini (20 RPD) antes de confirmar que el pipeline funciona. Correr en este orden:

**Paso 1 — Verificar FAISS y retrieval (sin llamadas a la API):**
```bash
pytest tests/ragas/test_cosine_similarity.py -v -s
```
Confirma que el índice FAISS carga correctamente, que `retrieve_with_scores()` funciona, y que aparece un run en `http://localhost:5000`.

**Paso 2 — Una métrica RAGAS para confirmar el pipeline completo:**
```bash
pytest tests/ragas/test_retrieval_metrics.py::test_context_precision_above_threshold -v -s
```

**Paso 3 — Tests funcionales (usan el LLM pero no el evaluador RAGAS):**
```bash
pytest tests/ragas/test_functional.py -v -s
```

**Paso 4 — Suite completa una vez confirmado el funcionamiento:**
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
```

### Cambiar la configuración sin modificar código

```bash
# Evaluar con k=5 en lugar de k=3
EVAL_K=5 EVAL_FETCH_K=100 pytest tests/ragas/ -v -s

# Evaluar solo el endpoint /recommend
EVAL_ENDPOINT=recommend pytest tests/ragas/test_recommend_metrics.py -v -s
```

Cada combinación diferente de `k` / `collection` genera su propio archivo de cache, por lo que las configuraciones no se sobreescriben entre sí.

---

## Cache de Datasets

Construir el dataset de evaluación requiere llamar a Gemini para cada pregunta (13 llamadas en total para el dataset por defecto). Para evitar repetir esto en cada ejecución, los datasets se cachean como archivos JSON en `tests/ragas/cache/`.

Los nombres de los archivos incluyen la colección y el valor de k:
```
tests/ragas/cache/eval_ask_realstate_mvd_k3.json
tests/ragas/cache/eval_recommend_realstate_mvd_k3.json
```

**Para forzar regeneración** (después de actualizar el índice FAISS o cambiar las preguntas):
```bash
rm tests/ragas/cache/eval_ask_realstate_mvd_k3.json
pytest tests/ragas/ -v -s
```

---

## Comparar Configuraciones con `run_experiment.py`

Para comparación sistemática de configuraciones de retrieval (diferentes `k`, `fetch_k`, variantes de prompt), usar el script standalone en lugar de pytest. Crea un run de MLflow por configuración e imprime una tabla comparativa al final.

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

Abrir `http://localhost:5000` y seleccionar el experimento `realstate_rag_evaluation`.

### Métricas clave para comparar entre runs

| Clave en MLflow | Significado |
|---|---|
| `ask_context_precision_avg` | Precisión de contexto promedio para /ask |
| `ask_faithfulness_avg` | Fidelidad promedio para /ask |
| `recommend_context_precision_avg` | Precisión de contexto promedio para /recommend |
| `functional_avg_cosine_similarity` | Score de relevancia FAISS promedio |
| `functional_rejection_rate` | Tasa de rechazo de queries fuera de dominio |

### Codificación de estado

Cada métrica tiene una clave `_status` correspondiente logueada como float:

| Valor | Significado |
|---|---|
| `1.0` | PASS — score ≥ umbral de aceptación |
| `0.5` | WARN — score en la zona gris entre umbrales |
| `0.0` | FAIL — score < umbral de rechazo |

Filtrar runs por `ask_faithfulness_status = 1.0` en la UI de MLflow para encontrar rápidamente los runs donde todas las métricas pasaron.

### Scores por muestra

Los scores individuales por pregunta se loguean como `ask_context_precision_q01`, `ask_context_precision_q02`, etc. Usarlos para identificar qué preguntas específicas están bajando los promedios.

---

## Actualizar Umbrales

Todos los valores de aceptación/rechazo viven en un único lugar:

```
tests/ragas/thresholds.py
```

Editar `RAGAS_THRESHOLDS` o `FUNCTIONAL_THRESHOLDS` ahí. No se necesitan cambios en los archivos de tests — todos importan desde `thresholds.py`.

## Calibrar `match_score` (1–100) -COMPLETADO-

Después del primer run de evaluación, revisar los scores coseno por query logueados en MLflow (`functional_avg_cosine_q01`, `functional_avg_cosine_q02`, etc.) para ver la distribución real del corpus. Luego actualizar los límites en `routers/ask.py`:

```python
_SCORE_LOW  = 0.50   # actualizar con el mínimo observado para matches reales
_SCORE_HIGH = 0.95   # actualizar con el máximo observado
```

---

## Límites del Tier Gratuito de Gemini

| Límite | Valor |
|---|---|
| Requests por minuto (RPM) | 5 |
| Requests por día (RPD) | 20 |
| Tokens por minuto (TPM) | 250,000 |

El `RunConfig(max_workers=1)` en todos los archivos de tests respeta el límite de RPM. Para el límite de RPD, usar el cache de datasets y correr una o dos métricas por día hasta que la línea base esté establecida.

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

**`FileNotFoundError` para el índice FAISS:**
El índice aún no fue construido, o `EVAL_FAISS_PATH` apunta a una ubicación incorrecta.
Llamar primero al endpoint `/load-from-csv` para construir la colección y luego re-correr.

**Los tests pasan pero no aparece nada en la UI de MLflow:**
Verificar que `mlflow ui` estaba corriendo antes de iniciar pytest, y que ambos fueron lanzados desde la misma raíz del repositorio para compartir la misma carpeta `mlruns/`.

**El archivo de cache genera resultados desactualizados:**
Eliminar el archivo correspondiente en `tests/ragas/cache/` para forzar la regeneración del dataset.

**La evaluación RAGAS es lenta:**
Es esperado — `max_workers=1` es intencional para respetar el límite de 5 RPM.
Cada evaluación de métrica toma aproximadamente 2–3 minutos para el dataset de 8 preguntas por defecto.
