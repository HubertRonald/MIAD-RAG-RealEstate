# Observability Latency Runbook — Medición histórica post-pruebas

## MIAD RAG Real Estate — Ventana 11 a 13 de mayo de 2026

Este documento describe cómo medir los tiempos disponibles **después de cerradas las pruebas con usuarios**, usando únicamente la evidencia que ya quedó registrada en **Cloud Logging / Cloud Run**.

La medición solicitada corresponde a:

| Zona horaria | Inicio inclusivo | Fin inclusivo |
|---|---:|---:|
| America/Bogota | `2026-05-11 00:00:00` | `2026-05-13 23:59:59` |

Equivalente recomendado en UTC:

```sql
timestamp >= TIMESTAMP("2026-05-11T05:00:00Z")
AND timestamp < TIMESTAMP("2026-05-14T05:00:00Z")
```

Se usa fin exclusivo (`< 2026-05-14T05:00:00Z`) para cubrir hasta el último instante del 13 de mayo en hora Colombia sin problemas de precisión.

---

## 1. Aclaración importante sobre medición histórica

Como las pruebas con usuarios ya cerraron, **no es posible reconstruir retroactivamente tiempos internos que no fueron registrados en su momento**.

Esto significa:

| Medición | ¿Se puede medir históricamente? | Fuente |
|---|---:|---|
| Latencia HTTP del frontend Cloud Run | Sí | Request logs nativos de Cloud Run |
| Latencia HTTP del backend Cloud Run | Sí | Request logs nativos de Cloud Run |
| Status HTTP, errores 4xx/5xx | Sí | Request logs nativos de Cloud Run |
| IP vista por Cloud Run | Sí, con cuidado | `httpRequest.remoteIp` |
| User agent | Sí | `httpRequest.userAgent` |
| Trace de Cloud Logging | Parcial | `trace` |
| Tiempo exacto Gemini embedding | No, salvo que ya existan logs propios | Requiere instrumentación previa |
| Tiempo exacto FAISS retrieval | No, salvo que ya existan logs propios | Requiere instrumentación previa |
| Tiempo exacto BigQuery enrichment por request | No, salvo que ya existan logs propios o job labels | Requiere instrumentación previa |
| Tiempo exacto Gemini generation | No, salvo que ya existan logs propios | Requiere instrumentación previa |
| Tiempo de permanencia por sesión | No confiable si no se registraron eventos de sesión | Requiere instrumentación previa |

Conclusión práctica:

> Para la entrega histórica del 11 al 13 de mayo de 2026, la medición defendible debe concentrarse en latencias HTTP reales de Cloud Run para frontend y backend. Los tiempos internos del RAG solo pueden declararse como una limitación si no estaban instrumentados durante las pruebas.

---

## 2. Qué archivos sí sirven y cuáles no

### 2.1 Archivos que sí sirven con lo que ya existe

Estos archivos funcionan usando logs nativos de Cloud Run, siempre que los logs del rango consultado aún estén retenidos:

```text
queries/
└── observability/
    ├── cloudrun_request_detail.sql
    ├── cloudrun_latency_by_endpoint.sql
    ├── cloudrun_latency_timeseries.sql
    └── cloudrun_backend_app_logs_inventory.sql

scripts/
└── observability/
    ├── export_cloudrun_request_logs.sh
    └── run_log_query_to_csv.sh
```

### 2.2 Archivos que no sirven para la medición histórica si no hubo instrumentación

Estos archivos **no devolverán datos útiles** para la ventana histórica si en las pruebas no se emitieron logs JSON con `event_type = "rag_timing"` o `event_type = "frontend_session_event"`:

```text
queries/
└── observability/
    ├── rag_timeline_detail.sql
    ├── rag_timeline_building_blocks.sql
    └── frontend_session_duration.sql
```

Estos archivos pueden conservarse solo como propuesta futura, pero **no deberían usarse como evidencia de las pruebas históricas**.

---

## 3. Ubicaciones correctas: Cloud Run, Log Bucket y BigQuery

Este punto es clave para evitar errores de ubicación y costos innecesarios.

| Elemento | Valor correcto en este caso | Comentario |
|---|---|---|
| Región Cloud Run | `us-east4` | Se usa para filtrar los logs del servicio. |
| Log bucket `_Default` | `global` | Confirmado con `gcloud logging buckets list`. |
| Linked dataset | `logging_miad_rag` | Debe crearse contra el bucket `_Default` en `global`. |
| BigQuery job location para `bq query` | `US` | No usar `global`. No usar `us-east4` para este linked dataset. |

Resumen práctico:

```text
Cloud Run region      = us-east4
Log bucket location   = global
BigQuery job location = US
```

> Aunque Cloud Run esté desplegado en `us-east4`, el bucket de logs `_Default` está en `global`. Por tanto, el linked dataset se crea sobre `location=global` y las consultas BigQuery se ejecutan con `BQ_LOCATION=US`. No usar `BQ_LOCATION=us-east4` para este linked dataset.

---

## 4. Estructura recomendada en el repo

Para la medición histórica, dejaría la estructura así:

```text
MIAD-RAG-RealEstate/
├── docs/
│   └── runbooks/
│       └── observability-latency.md
├── queries/
│   └── observability/
│       ├── cloudrun_request_detail.sql
│       ├── cloudrun_latency_by_endpoint.sql
│       ├── cloudrun_latency_timeseries.sql
│       └── cloudrun_backend_app_logs_inventory.sql
└── scripts/
    └── observability/
        ├── export_cloudrun_request_logs.sh
        └── run_log_query_to_csv.sh
```

Agregar a `.gitignore`:

```gitignore
observability_exports/
*.local.csv
*.local.json
```

No subir al repo los CSV o JSON exportados, porque pueden contener IP, user agent, trazas o URLs.

---

## 5. Validar bucket de logs y linked dataset

### 5.1 Verificar ubicación del bucket de logs

```bash
PROJECT_ID="miad-paad-rs-dev"

gcloud logging buckets list \
  --project="${PROJECT_ID}"
```

Salida esperada para este caso:

```text
LOCATION: global
BUCKET_ID: _Default
RETENTION_DAYS: 30
LIFECYCLE_STATE: ACTIVE
```

### 5.2 Verificar si ya existe un linked dataset

```bash
PROJECT_ID="miad-paad-rs-dev"

gcloud logging links list \
  --project="${PROJECT_ID}" \
  --bucket="_Default" \
  --location="global"
```

Si devuelve:

```text
Listed 0 items.
```

entonces todavía **no existe** el linked dataset y `bq query` fallará con un error similar a:

```text
Not found: Dataset miad-paad-rs-dev:logging_miad_rag was not found in location US
```

---

## 6. Crear el linked dataset para consultar con BigQuery

Este paso solo es necesario si se quiere consultar desde BigQuery CLI (`bq query`) o exportar resultados a CSV usando el script `run_log_query_to_csv.sh`.

### 6.1 Habilitar Analytics en el bucket `_Default`

Si el bucket aún no está habilitado para Analytics:

```bash
PROJECT_ID="miad-paad-rs-dev"

gcloud logging buckets update "_Default" \
  --project="${PROJECT_ID}" \
  --location="global" \
  --enable-analytics
```

### 6.2 Crear el linked dataset

```bash
PROJECT_ID="miad-paad-rs-dev"
LINKED_DATASET="logging_miad_rag"

gcloud logging links create "${LINKED_DATASET}" \
  --project="${PROJECT_ID}" \
  --bucket="_Default" \
  --location="global"
```

### 6.3 Validar que el link quedó creado

```bash
PROJECT_ID="miad-paad-rs-dev"

gcloud logging links list \
  --project="${PROJECT_ID}" \
  --bucket="_Default" \
  --location="global"
```

Debería aparecer un link asociado a `logging_miad_rag`.

---

## 7. Script para exportar request logs nativos de Cloud Run

Este script **no depende de BigQuery ni del linked dataset**. Usa `gcloud logging read` directamente contra Cloud Logging.

Archivo:

```text
scripts/observability/export_cloudrun_request_logs.sh
```

> Importante: se dejan los timestamps UTC fijos para evitar errores de conversión con `date`. Para Colombia, el rango correcto es `2026-05-11T05:00:00Z` a `2026-05-14T05:00:00Z` exclusivo.

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-miad-paad-rs-dev}"
REGION="${REGION:-us-east4}"
FRONTEND_SERVICE="${FRONTEND_SERVICE:-miad-rag-frontend}"
BACKEND_SERVICE="${BACKEND_SERVICE:-miad-rag-backend}"

# Ventana histórica solicitada, convertida manualmente de America/Bogota a UTC.
START_TS="${START_TS:-2026-05-11T05:00:00Z}"
END_TS="${END_TS:-2026-05-14T05:00:00Z}"

# Debe cubrir al menos la antigüedad de la ventana consultada.
FRESHNESS="${FRESHNESS:-30d}"
LIMIT="${LIMIT:-50000}"
OUT_DIR="${OUT_DIR:-./observability_exports}"

mkdir -p "${OUT_DIR}"

STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
RANGE_LABEL="20260511_000000_to_20260513_235959_COT"

RAW_JSON="${OUT_DIR}/cloudrun_requests_${RANGE_LABEL}_${STAMP}.json"
CSV="${OUT_DIR}/cloudrun_requests_${RANGE_LABEL}_${STAMP}.csv"

FILTER=$(cat <<EOF
resource.type="cloud_run_revision"
resource.labels.location="${REGION}"
(resource.labels.service_name="${FRONTEND_SERVICE}" OR resource.labels.service_name="${BACKEND_SERVICE}")
logName="projects/${PROJECT_ID}/logs/run.googleapis.com%2Frequests"
timestamp >= "${START_TS}"
timestamp < "${END_TS}"
EOF
)

echo "Project   : ${PROJECT_ID}"
echo "Region    : ${REGION}"
echo "Start UTC : ${START_TS}"
echo "End UTC   : ${END_TS} exclusive"
echo "Output    : ${OUT_DIR}"

gcloud logging read "${FILTER}" \
  --project="${PROJECT_ID}" \
  --format=json \
  --freshness="${FRESHNESS}" \
  --order=asc \
  --limit="${LIMIT}" \
  > "${RAW_JSON}"

jq -r '
def latency_to_ms:
  if . == null then null
  else
    (capture("(?<seconds>[0-9]+)(\\.(?<fraction>[0-9]+))?s")? // null) as $m
    | if $m == null then null
      else ((( $m.seconds | tonumber) + (("0." + ($m.fraction // "0")) | tonumber)) * 1000)
      end
  end;

(["timestamp",
  "service",
  "revision",
  "method",
  "url",
  "status",
  "latency_ms",
  "remote_ip",
  "remote_ip_base64",
  "user_agent",
  "trace",
  "insert_id"] | @csv),
(.[] | [
  .timestamp,
  .resource.labels.service_name,
  .resource.labels.revision_name,
  .httpRequest.requestMethod,
  .httpRequest.requestUrl,
  .httpRequest.status,
  (.httpRequest.latency | latency_to_ms),
  .httpRequest.remoteIp,
  (.httpRequest.remoteIp // "" | @base64),
  .httpRequest.userAgent,
  .trace,
  .insertId
] | @csv)
' "${RAW_JSON}" > "${CSV}"

echo "Generado JSON: ${RAW_JSON}"
echo "Generado CSV : ${CSV}"
```

Ejecución:

```bash
chmod +x scripts/observability/export_cloudrun_request_logs.sh

PROJECT_ID="miad-paad-rs-dev" \
REGION="us-east4" \
START_TS="2026-05-11T05:00:00Z" \
END_TS="2026-05-14T05:00:00Z" \
FRESHNESS="30d" \
scripts/observability/export_cloudrun_request_logs.sh
```

---

## 8. Script para ejecutar queries SQL y exportar CSV

Este script aplica cuando ya existe un **linked dataset** de Log Analytics hacia BigQuery.

Archivo:

```text
scripts/observability/run_log_query_to_csv.sh
```

Características de esta versión:

- usa `BQ_LOCATION="US"` por defecto;
- no guarda errores como si fueran CSV válidos;
- guarda errores en `.err` solo cuando falla;
- reemplaza automáticamente la referencia de Observability Analytics por la del linked dataset.

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-miad-paad-rs-dev}"
LINKED_DATASET="${LINKED_DATASET:-logging_miad_rag}"
BQ_LOCATION="${BQ_LOCATION:-US}"
SQL_FILE="${1:?Uso: $0 queries/observability/archivo.sql}"
OUT_DIR="${OUT_DIR:-./observability_exports}"

mkdir -p "${OUT_DIR}"

BASENAME="$(basename "${SQL_FILE}" .sql)"
STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
TMP_SQL="${OUT_DIR}/${BASENAME}_${STAMP}.bq.sql"
OUT_CSV="${OUT_DIR}/${BASENAME}_${STAMP}.csv"
TMP_CSV="${OUT_CSV}.tmp"
ERR_FILE="${OUT_CSV}.err"

sed "s|miad-paad-rs-dev.global._Default._AllLogs|${PROJECT_ID}.${LINKED_DATASET}._AllLogs|g" \
  "${SQL_FILE}" > "${TMP_SQL}"

echo "Project        : ${PROJECT_ID}"
echo "Linked dataset : ${LINKED_DATASET}"
echo "BQ location    : ${BQ_LOCATION}"
echo "SQL file       : ${SQL_FILE}"
echo "SQL usado      : ${TMP_SQL}"

if bq query \
  --project_id="${PROJECT_ID}" \
  --location="${BQ_LOCATION}" \
  --use_legacy_sql=false \
  --format=csv \
  < "${TMP_SQL}" > "${TMP_CSV}" 2> "${ERR_FILE}"; then

  mv "${TMP_CSV}" "${OUT_CSV}"
  rm -f "${ERR_FILE}"
  echo "CSV generado   : ${OUT_CSV}"

else
  echo "ERROR ejecutando query."
  echo
  echo "STDERR:"
  cat "${ERR_FILE}" || true

  rm -f "${TMP_CSV}"
  exit 1
fi
```

Ejecución de una query:

```bash
chmod +x scripts/observability/run_log_query_to_csv.sh

PROJECT_ID="miad-paad-rs-dev" \
LINKED_DATASET="logging_miad_rag" \
BQ_LOCATION="US" \
scripts/observability/run_log_query_to_csv.sh queries/observability/cloudrun_latency_by_endpoint.sql
```

---

## 9. Ejecutar las cuatro queries históricas

El script `run_log_query_to_csv.sh` ejecuta **una query por vez**. Para ejecutar todas:

```bash
PROJECT_ID="miad-paad-rs-dev"
LINKED_DATASET="logging_miad_rag"
BQ_LOCATION="US"

for SQL_FILE in \
  queries/observability/cloudrun_request_detail.sql \
  queries/observability/cloudrun_latency_by_endpoint.sql \
  queries/observability/cloudrun_latency_timeseries.sql \
  queries/observability/cloudrun_backend_app_logs_inventory.sql
do
  PROJECT_ID="${PROJECT_ID}" \
  LINKED_DATASET="${LINKED_DATASET}" \
  BQ_LOCATION="${BQ_LOCATION}" \
  scripts/observability/run_log_query_to_csv.sh "${SQL_FILE}"
done
```

> No usar `BQ_LOCATION="global"`. BigQuery no ejecuta jobs en `global`. Para este linked dataset asociado al bucket global, usar `BQ_LOCATION="US"`.

---

## 10. Limpieza de CSV fallidos

Si antes se generó un CSV que realmente contiene el error de BigQuery, eliminarlo:

```bash
rm -f observability_exports/cloudrun_latency_by_endpoint_20260516T032543Z.csv
rm -f observability_exports/cloudrun_latency_by_endpoint_20260516T032543Z.bq.sql
```

Con el nuevo script, los errores ya no deberían quedar guardados como CSV válidos.

---

# 11. Queries SQL históricas

Todas las queries están fijadas a:

```sql
timestamp >= TIMESTAMP("2026-05-11T05:00:00Z")
AND timestamp < TIMESTAMP("2026-05-14T05:00:00Z")
```

En Observability Analytics puede usarse:

```sql
FROM `miad-paad-rs-dev.global._Default._AllLogs`
```

En BigQuery mediante linked dataset puede usarse:

```sql
FROM `miad-paad-rs-dev.logging_miad_rag._AllLogs`
```

El script `run_log_query_to_csv.sh` reemplaza automáticamente la primera referencia por la segunda.

---

## 11.1 Detalle request por request

Archivo:

```text
queries/observability/cloudrun_request_detail.sql
```

```sql
SELECT
  timestamp,
  trace,
  JSON_VALUE(resource.labels.service_name) AS service_name,
  JSON_VALUE(resource.labels.revision_name) AS revision_name,
  http_request.request_method AS method,
  REGEXP_EXTRACT(http_request.request_url, r'https?://[^/]+([^?]*)') AS path,
  http_request.request_url AS full_url,
  http_request.status AS status,
  (
    COALESCE(http_request.latency.seconds, 0) * 1000
    + COALESCE(http_request.latency.nanos, 0) / 1000000
  ) AS latency_ms,
  http_request.remote_ip AS remote_ip,
  TO_HEX(SHA256(COALESCE(http_request.remote_ip, ""))) AS remote_ip_hash,
  http_request.user_agent AS user_agent,
  insert_id
FROM
  `miad-paad-rs-dev.global._Default._AllLogs`
WHERE
  timestamp >= TIMESTAMP("2026-05-11T05:00:00Z")
  AND timestamp < TIMESTAMP("2026-05-14T05:00:00Z")
  AND resource.type = "cloud_run_revision"
  AND JSON_VALUE(resource.labels.location) = "us-east4"
  AND JSON_VALUE(resource.labels.service_name) IN ("miad-rag-frontend", "miad-rag-backend")
  AND log_id = "run.googleapis.com/requests"
  AND http_request.request_method IS NOT NULL
ORDER BY
  timestamp DESC;
```

---

## 11.2 Latencia agregada por servicio y endpoint

Archivo:

```text
queries/observability/cloudrun_latency_by_endpoint.sql
```

```sql
WITH requests AS (
  SELECT
    timestamp,
    JSON_VALUE(resource.labels.service_name) AS service_name,
    http_request.request_method AS method,
    REGEXP_EXTRACT(http_request.request_url, r'https?://[^/]+([^?]*)') AS path,
    http_request.status AS status,
    (
      COALESCE(http_request.latency.seconds, 0) * 1000
      + COALESCE(http_request.latency.nanos, 0) / 1000000
    ) AS latency_ms
  FROM
    `miad-paad-rs-dev.global._Default._AllLogs`
  WHERE
    timestamp >= TIMESTAMP("2026-05-11T05:00:00Z")
    AND timestamp < TIMESTAMP("2026-05-14T05:00:00Z")
    AND resource.type = "cloud_run_revision"
    AND JSON_VALUE(resource.labels.location) = "us-east4"
    AND JSON_VALUE(resource.labels.service_name) IN ("miad-rag-frontend", "miad-rag-backend")
    AND log_id = "run.googleapis.com/requests"
    AND http_request.request_method IS NOT NULL
)
SELECT
  service_name,
  method,
  path,
  COUNT(*) AS requests,
  COUNTIF(status >= 400 AND status < 500) AS errors_4xx,
  COUNTIF(status >= 500) AS errors_5xx,
  ROUND(AVG(latency_ms), 2) AS avg_ms,
  ROUND(MIN(latency_ms), 2) AS min_ms,
  ROUND(MAX(latency_ms), 2) AS max_ms,
  ROUND(APPROX_QUANTILES(latency_ms, 100)[OFFSET(50)], 2) AS p50_ms,
  ROUND(APPROX_QUANTILES(latency_ms, 100)[OFFSET(90)], 2) AS p90_ms,
  ROUND(APPROX_QUANTILES(latency_ms, 100)[OFFSET(95)], 2) AS p95_ms,
  ROUND(APPROX_QUANTILES(latency_ms, 100)[OFFSET(99)], 2) AS p99_ms
FROM requests
GROUP BY
  service_name,
  method,
  path
ORDER BY
  p95_ms DESC;
```

---

## 11.3 Serie temporal de latencias por minuto

Archivo:

```text
queries/observability/cloudrun_latency_timeseries.sql
```

```sql
WITH requests AS (
  SELECT
    TIMESTAMP_TRUNC(timestamp, MINUTE) AS minute,
    JSON_VALUE(resource.labels.service_name) AS service_name,
    http_request.status AS status,
    (
      COALESCE(http_request.latency.seconds, 0) * 1000
      + COALESCE(http_request.latency.nanos, 0) / 1000000
    ) AS latency_ms
  FROM
    `miad-paad-rs-dev.global._Default._AllLogs`
  WHERE
    timestamp >= TIMESTAMP("2026-05-11T05:00:00Z")
    AND timestamp < TIMESTAMP("2026-05-14T05:00:00Z")
    AND resource.type = "cloud_run_revision"
    AND JSON_VALUE(resource.labels.location) = "us-east4"
    AND JSON_VALUE(resource.labels.service_name) IN ("miad-rag-frontend", "miad-rag-backend")
    AND log_id = "run.googleapis.com/requests"
    AND http_request.request_method IS NOT NULL
)
SELECT
  minute,
  service_name,
  COUNT(*) AS requests,
  COUNTIF(status >= 500) AS errors_5xx,
  ROUND(AVG(latency_ms), 2) AS avg_ms,
  ROUND(APPROX_QUANTILES(latency_ms, 100)[OFFSET(50)], 2) AS p50_ms,
  ROUND(APPROX_QUANTILES(latency_ms, 100)[OFFSET(95)], 2) AS p95_ms,
  ROUND(APPROX_QUANTILES(latency_ms, 100)[OFFSET(99)], 2) AS p99_ms
FROM requests
GROUP BY
  minute,
  service_name
ORDER BY
  minute ASC,
  service_name;
```

---

## 11.4 Inventario exploratorio de logs de aplicación backend

Esta query no mide latencias internas por sí sola, pero ayuda a revisar si durante las pruebas quedaron logs propios del backend mencionando `bigquery`, `gemini`, `faiss`, `gcs`, `retrieval`, `generation`, `recommend` o errores asociados.

Archivo:

```text
queries/observability/cloudrun_backend_app_logs_inventory.sql
```

```sql
SELECT
  timestamp,
  severity,
  log_id,
  trace,
  JSON_VALUE(resource.labels.service_name) AS service_name,
  JSON_VALUE(resource.labels.revision_name) AS revision_name,
  text_payload,
  json_payload,
  proto_payload
FROM
  `miad-paad-rs-dev.global._Default._AllLogs`
WHERE
  timestamp >= TIMESTAMP("2026-05-11T05:00:00Z")
  AND timestamp < TIMESTAMP("2026-05-14T05:00:00Z")
  AND resource.type = "cloud_run_revision"
  AND JSON_VALUE(resource.labels.location) = "us-east4"
  AND JSON_VALUE(resource.labels.service_name) = "miad-rag-backend"
  AND log_id != "run.googleapis.com/requests"
  AND (
    LOWER(COALESCE(text_payload, "")) LIKE "%bigquery%"
    OR LOWER(COALESCE(text_payload, "")) LIKE "%gemini%"
    OR LOWER(COALESCE(text_payload, "")) LIKE "%faiss%"
    OR LOWER(COALESCE(text_payload, "")) LIKE "%gcs%"
    OR LOWER(COALESCE(text_payload, "")) LIKE "%retrieval%"
    OR LOWER(COALESCE(text_payload, "")) LIKE "%generation%"
    OR LOWER(COALESCE(text_payload, "")) LIKE "%recommend%"
    OR LOWER(TO_JSON_STRING(json_payload)) LIKE "%bigquery%"
    OR LOWER(TO_JSON_STRING(json_payload)) LIKE "%gemini%"
    OR LOWER(TO_JSON_STRING(json_payload)) LIKE "%faiss%"
    OR LOWER(TO_JSON_STRING(json_payload)) LIKE "%gcs%"
    OR LOWER(TO_JSON_STRING(json_payload)) LIKE "%retrieval%"
    OR LOWER(TO_JSON_STRING(json_payload)) LIKE "%generation%"
    OR LOWER(TO_JSON_STRING(json_payload)) LIKE "%recommend%"
  )
ORDER BY
  timestamp ASC;
```

---

# 12. Cómo interpretar lo que sí se obtiene

## 12.1 Con `cloudrun_latency_by_endpoint.sql`

Resultado esperado:

| Campo | Uso |
|---|---|
| `service_name` | Diferencia frontend vs backend |
| `path` | Identifica `/`, `/api/v1/recommend`, `/api/v1/ask`, healthchecks, etc. |
| `requests` | Volumen de llamadas |
| `errors_4xx` | Problemas de autenticación, permisos o cliente |
| `errors_5xx` | Problemas de backend o runtime |
| `avg_ms` | Promedio general |
| `p50_ms` | Experiencia típica |
| `p95_ms` | Experiencia lenta |
| `p99_ms` | Casos extremos |

## 12.2 Con `cloudrun_request_detail.sql`

Permite revisar request por request:

- timestamp;
- servicio;
- endpoint;
- status;
- latencia;
- IP;
- user agent;
- trace.

Sirve para seleccionar casos extremos y revisar si se concentran en una hora, endpoint, revisión o usuario/IP.

## 12.3 Con `cloudrun_latency_timeseries.sql`

Permite ver comportamiento por minuto:

- picos de tráfico;
- picos de p95/p99;
- errores por minuto;
- diferencias frontend/backend.

## 12.4 Con `cloudrun_backend_app_logs_inventory.sql`

Permite revisar si quedó evidencia textual adicional de ejecución interna.

Si no hay logs con duraciones internas, esta query no permite inferir tiempos exactos de BigQuery, GCS, Gemini o FAISS. Solo ayuda a documentar qué mensajes existían.

---

# 13. Relación con el KPI de eficiencia del proceso de búsqueda

KPI declarado:

```text
Eficiencia del proceso de búsqueda <= 3 minutos en demo reportado por usuario
```

Con estos logs históricos se puede medir:

```text
Tiempo técnico de respuesta del frontend/backend Cloud Run
Tiempo técnico de respuesta de /api/v1/recommend y /api/v1/ask, si aparecen en logs
Percentiles p50, p95 y p99 de latencia técnica
Errores HTTP durante la ventana de pruebas
```

Pero no se puede reconstruir completamente:

```text
Tiempo humano de lectura de interfaz
Tiempo de parametrización de filtros
Tiempo de escritura de preferencias
Tiempo de revisión de resultados
Tiempo total real de permanencia por sesión
```

Texto sugerido:

> El indicador de eficiencia del proceso de búsqueda menor o igual a 3 minutos se valida principalmente con evidencia observacional o reporte de usuario durante la demo. Los logs históricos de Cloud Run permiten complementar esta validación midiendo la latencia técnica de respuesta del frontend y backend durante la ventana de pruebas. Sin embargo, al no existir instrumentación de eventos de sesión en frontend durante la demo, los logs no permiten reconstruir de forma exacta el tiempo total humano desde el inicio de la parametrización hasta la revisión del resultado.

---

# 14. Limitaciones para el informe

Texto sugerido para el informe:

> Dado que la ventana de pruebas con usuarios ya se encontraba cerrada al momento del análisis, la medición se realizó con logs históricos disponibles en Cloud Logging. Esto permite calcular latencias HTTP reales de los servicios Cloud Run de frontend y backend para el periodo comprendido entre el 11 de mayo de 2026 a las 00:00:00 y el 13 de mayo de 2026 a las 23:59:59 hora Colombia. Sin embargo, no es posible reconstruir retroactivamente los tiempos internos por etapa del pipeline RAG —por ejemplo, embeddings Gemini, recuperación FAISS, enriquecimiento BigQuery o generación LLM— si dichos eventos no fueron instrumentados y registrados durante la ejecución de las pruebas. Por tanto, el análisis histórico se limita a tiempos observables desde la infraestructura y deja la instrumentación fina como recomendación para pruebas futuras.

---

# 15. Checklist mínimo para la medición histórica

- [ ] Verificar que los logs del 11 al 13 de mayo de 2026 aún estén retenidos.
- [ ] Confirmar que el bucket `_Default` está en `global`.
- [ ] Ejecutar `export_cloudrun_request_logs.sh` con `START_TS="2026-05-11T05:00:00Z"` y `END_TS="2026-05-14T05:00:00Z"`.
- [ ] Revisar si el CSV trae registros de `miad-rag-frontend` y `miad-rag-backend`.
- [ ] Si se requiere `bq query`, validar o crear el linked dataset `logging_miad_rag`.
- [ ] Ejecutar `bq query` con `BQ_LOCATION="US"`.
- [ ] Exportar `cloudrun_request_detail.csv`.
- [ ] Exportar `cloudrun_latency_by_endpoint.csv`.
- [ ] Exportar `cloudrun_latency_timeseries.csv`.
- [ ] Revisar logs de aplicación con `cloudrun_backend_app_logs_inventory.sql`.
- [ ] Documentar limitación de medición interna no retroactiva.
- [ ] Preparar análisis de p50, p95, p99 y errores.

---

# 16. Orden recomendado de ejecución

Primero, extraer logs nativos sin BigQuery:

```bash
PROJECT_ID="miad-paad-rs-dev" \
REGION="us-east4" \
START_TS="2026-05-11T05:00:00Z" \
END_TS="2026-05-14T05:00:00Z" \
FRESHNESS="30d" \
scripts/observability/export_cloudrun_request_logs.sh
```

Luego, validar el linked dataset:

```bash
PROJECT_ID="miad-paad-rs-dev"

gcloud logging links list \
  --project="${PROJECT_ID}" \
  --bucket="_Default" \
  --location="global"
```

Si no existe, crearlo:

```bash
PROJECT_ID="miad-paad-rs-dev"
LINKED_DATASET="logging_miad_rag"

gcloud logging buckets update "_Default" \
  --project="${PROJECT_ID}" \
  --location="global" \
  --enable-analytics

gcloud logging links create "${LINKED_DATASET}" \
  --project="${PROJECT_ID}" \
  --bucket="_Default" \
  --location="global"
```

Finalmente, ejecutar las queries SQL:

```bash
PROJECT_ID="miad-paad-rs-dev"
LINKED_DATASET="logging_miad_rag"
BQ_LOCATION="US"

for SQL_FILE in \
  queries/observability/cloudrun_request_detail.sql \
  queries/observability/cloudrun_latency_by_endpoint.sql \
  queries/observability/cloudrun_latency_timeseries.sql \
  queries/observability/cloudrun_backend_app_logs_inventory.sql
do
  PROJECT_ID="${PROJECT_ID}" \
  LINKED_DATASET="${LINKED_DATASET}" \
  BQ_LOCATION="${BQ_LOCATION}" \
  scripts/observability/run_log_query_to_csv.sh "${SQL_FILE}"
done
```


---

# 17. Instrumentación sugerida en backend

Futura implementación

## 17.1 Helper de timing

Archivo sugerido:

```text
apps/backend/app/observability/timing.py
```

```python
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger("rag_timing")


def log_timing(
    *,
    request_id: str,
    stage: str,
    stage_order: int,
    duration_ms: float,
    component: str = "backend",
    session_id_hash: Optional[str] = None,
    status: str = "ok",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "event_type": "rag_timing",
        "request_id": request_id,
        "session_id_hash": session_id_hash,
        "component": component,
        "stage_order": stage_order,
        "stage": stage,
        "duration_ms": round(duration_ms, 2),
        "status": status,
    }

    if extra:
        payload.update(extra)

    logger.info(payload)


@contextmanager
def timed_stage(
    *,
    request_id: str,
    stage: str,
    stage_order: int,
    component: str = "backend",
    session_id_hash: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    start = time.perf_counter()
    status = "ok"

    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        log_timing(
            request_id=request_id,
            session_id_hash=session_id_hash,
            component=component,
            stage=stage,
            stage_order=stage_order,
            duration_ms=duration_ms,
            status=status,
            extra=extra,
        )
```

## 17.2 Uso conceptual en endpoint `/api/v1/recommend`

```python
import uuid
from app.observability.timing import timed_stage

request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
session_id_hash = request.headers.get("x-session-id-hash")

with timed_stage(
    request_id=request_id,
    session_id_hash=session_id_hash,
    stage="backend_request_total",
    stage_order=30,
):
    with timed_stage(
        request_id=request_id,
        session_id_hash=session_id_hash,
        stage="gemini_embedding",
        stage_order=50,
    ):
        # llamada a embedding
        pass

    with timed_stage(
        request_id=request_id,
        session_id_hash=session_id_hash,
        stage="faiss_retrieval",
        stage_order=60,
    ):
        # búsqueda FAISS
        pass

    with timed_stage(
        request_id=request_id,
        session_id_hash=session_id_hash,
        stage="bigquery_enrichment",
        stage_order=70,
    ):
        # enriquecimiento BigQuery
        pass

    with timed_stage(
        request_id=request_id,
        session_id_hash=session_id_hash,
        stage="gemini_generation",
        stage_order=80,
    ):
        # generación LLM
        pass
```

---

# 18. Instrumentación sugerida en frontend

## 18.1 Generar sesión y request id

```python
import hashlib
import logging
import time
import uuid
import streamlit as st

logger = logging.getLogger("frontend_observability")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

session_id_hash = hashlib.sha256(
    st.session_state["session_id"].encode("utf-8")
).hexdigest()


def log_frontend_event(event_name: str, request_id: str | None = None, **extra):
    payload = {
        "event_type": "frontend_session_event",
        "event_name": event_name,
        "session_id_hash": session_id_hash,
        "request_id": request_id,
        **extra,
    }

    logger.info(payload)


def log_frontend_timing(
    *,
    request_id: str,
    stage: str,
    stage_order: int,
    duration_ms: float,
    status: str = "ok",
    **extra,
):
    payload = {
        "event_type": "rag_timing",
        "request_id": request_id,
        "session_id_hash": session_id_hash,
        "component": "frontend",
        "stage_order": stage_order,
        "stage": stage,
        "duration_ms": round(duration_ms, 2),
        "status": status,
        **extra,
    }

    logger.info(payload)
```

## 18.2 Medir llamada frontend → backend

```python
request_id = str(uuid.uuid4())

log_frontend_event("search_submitted", request_id=request_id)

headers = {
    "X-Request-ID": request_id,
    "X-Session-ID-Hash": session_id_hash,
}

start = time.perf_counter()

try:
    response = requests.post(
        backend_url,
        json=payload,
        headers=headers,
        timeout=120,
    )
    response.raise_for_status()
    status = "ok"
except Exception:
    status = "error"
    raise
finally:
    duration_ms = (time.perf_counter() - start) * 1000
    log_frontend_timing(
        request_id=request_id,
        stage="frontend_backend_call",
        stage_order=20,
        duration_ms=duration_ms,
        status=status,
    )

log_frontend_event("backend_response_received", request_id=request_id)

# Luego de renderizar resultado:
log_frontend_event("response_rendered", request_id=request_id)
```


---

# 19. Recomendación práctica final

Actualmente usar solo:

```text
cloudrun_request_detail.sql
cloudrun_latency_by_endpoint.sql
cloudrun_latency_timeseries.sql
cloudrun_backend_app_logs_inventory.sql
export_cloudrun_request_logs.sh
run_log_query_to_csv.sh
```

No usar como evidencia histórica:

```text
rag_timeline_detail.sql
rag_timeline_building_blocks.sql
frontend_session_duration.sql
```

Estos últimos solo tendrían sentido si el frontend/backend emitieran logs estructurados durante las pruebas.

Resumen operativo:

```text
Filtrar Cloud Run por region=us-east4.
Consultar Cloud Logging bucket en location=global.
Crear linked dataset contra bucket _Default en global.
Ejecutar BigQuery con BQ_LOCATION=US.
No usar BQ_LOCATION=us-east4 para el linked dataset de logs global.
Usar timestamps UTC fijos: 2026-05-11T05:00:00Z a 2026-05-14T05:00:00Z.
```
