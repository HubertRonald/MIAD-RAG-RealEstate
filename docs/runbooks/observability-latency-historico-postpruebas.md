# Observability Latency Runbook — Medición histórica post-pruebas

## MIAD RAG Real Estate — Ventana 11 a 13 de mayo de 2026

Este documento describe cómo medir los tiempos disponibles **después de cerradas las pruebas con usuarios**, usando únicamente la evidencia que ya quedó registrada en Cloud Logging / Cloud Run.

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

Estos archivos **no devolverán datos útiles** ya que en las pruebas no se emitieron logs JSON con `event_type = "rag_timing"` o `event_type = "frontend_session_event"`:

```text
queries/
└── observability/
    ├── rag_timeline_detail.sql
    ├── rag_timeline_building_blocks.sql
    └── frontend_session_duration.sql
```

Estos archivos pueden conservarse solo como propuesta futura, pero **no deberían usarse como evidencia de las pruebas históricas**.

---

## 3. Estructura recomendada en el repo

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

## 4. Script para exportar request logs nativos de Cloud Run

Archivo:

```text
scripts/observability/export_cloudrun_request_logs.sh
```

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-miad-paad-rs-dev}"
REGION="${REGION:-us-east4}"
FRONTEND_SERVICE="${FRONTEND_SERVICE:-miad-rag-frontend}"
BACKEND_SERVICE="${BACKEND_SERVICE:-miad-rag-backend}"

# Ventana solicitada en hora Colombia.
LOCAL_TZ="${LOCAL_TZ:-America/Bogota}"
START_LOCAL="${START_LOCAL:-2026-05-11 00:00:00}"
END_LOCAL_EXCLUSIVE="${END_LOCAL_EXCLUSIVE:-2026-05-14 00:00:00}"

# Cloud Logging consulta en UTC.
START_TS="$(TZ="${LOCAL_TZ}" date -d "${START_LOCAL}" -u +"%Y-%m-%dT%H:%M:%SZ")"
END_TS="$(TZ="${LOCAL_TZ}" date -d "${END_LOCAL_EXCLUSIVE}" -u +"%Y-%m-%dT%H:%M:%SZ")"

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

echo "Project    : ${PROJECT_ID}"
echo "Region     : ${REGION}"
echo "Local TZ   : ${LOCAL_TZ}"
echo "Start local: ${START_LOCAL}"
echo "End local  : 2026-05-13 23:59:59"
echo "Start UTC  : ${START_TS}"
echo "End UTC    : ${END_TS} exclusive"
echo "Output     : ${OUT_DIR}"

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
      else ((($m.seconds | tonumber) + (("0." + ($m.fraction // "0")) | tonumber)) * 1000)
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
LOCAL_TZ="America/Bogota" \
START_LOCAL="2026-05-11 00:00:00" \
END_LOCAL_EXCLUSIVE="2026-05-14 00:00:00" \
FRESHNESS="30d" \
scripts/observability/export_cloudrun_request_logs.sh
```

---

## 5. Script para ejecutar queries SQL y exportar CSV

Este script aplica cuando ya existe un **linked dataset** de Log Analytics hacia BigQuery.

Archivo:

```text
scripts/observability/run_log_query_to_csv.sh
```

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-miad-paad-rs-dev}"
LINKED_DATASET="${LINKED_DATASET:-logging_miad_rag}"
SQL_FILE="${1:?Uso: $0 queries/observability/archivo.sql}"
OUT_DIR="${OUT_DIR:-./observability_exports}"

mkdir -p "${OUT_DIR}"

BASENAME="$(basename "${SQL_FILE}" .sql)"
STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
TMP_SQL="${OUT_DIR}/${BASENAME}_${STAMP}.bq.sql"
OUT_CSV="${OUT_DIR}/${BASENAME}_${STAMP}.csv"

sed "s|miad-paad-rs-dev.global._Default._AllLogs|${PROJECT_ID}.${LINKED_DATASET}._AllLogs|g" \
  "${SQL_FILE}" > "${TMP_SQL}"

bq query \
  --project_id="${PROJECT_ID}" \
  --use_legacy_sql=false \
  --format=csv \
  < "${TMP_SQL}" > "${OUT_CSV}"

echo "SQL usado : ${TMP_SQL}"
echo "CSV      : ${OUT_CSV}"
```

Ejecución:

```bash
chmod +x scripts/observability/run_log_query_to_csv.sh

PROJECT_ID="miad-paad-rs-dev" \
LINKED_DATASET="logging_miad_rag" \
scripts/observability/run_log_query_to_csv.sh queries/observability/cloudrun_latency_by_endpoint.sql
```

---

# 6. Queries SQL históricas

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

## 6.1 Detalle request por request

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

## 6.2 Latencia agregada por servicio y endpoint

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

## 6.3 Serie temporal de latencias por minuto

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

## 6.4 Inventario exploratorio de logs de aplicación backend

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

# 7. Cómo interpretar lo que sí se obtiene

## 7.1 Con `cloudrun_latency_by_endpoint.sql`

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

## 7.2 Con `cloudrun_request_detail.sql`

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

## 7.3 Con `cloudrun_latency_timeseries.sql`

Permite ver comportamiento por minuto:

- picos de tráfico;
- picos de p95/p99;
- errores por minuto;
- diferencias frontend/backend.

## 7.4 Con `cloudrun_backend_app_logs_inventory.sql`

Permite revisar si quedó evidencia textual adicional de ejecución interna.

Si no hay logs con duraciones internas, esta query no permite inferir tiempos exactos de BigQuery, GCS, Gemini o FAISS. Solo ayuda a documentar qué mensajes existían.

---

# 8. Limitaciones para el informe

Texto sugerido para el informe:

> Dado que la ventana de pruebas con usuarios ya se encontraba cerrada al momento del análisis, la medición se realizó con logs históricos disponibles en Cloud Logging. Esto permite calcular latencias HTTP reales de los servicios Cloud Run de frontend y backend para el periodo comprendido entre el 11 de mayo de 2026 a las 00:00:00 y el 13 de mayo de 2026 a las 23:59:59 hora Colombia. Sin embargo, no es posible reconstruir retroactivamente los tiempos internos por etapa del pipeline RAG —por ejemplo, embeddings Gemini, recuperación FAISS, enriquecimiento BigQuery o generación LLM— si dichos eventos no fueron instrumentados y registrados durante la ejecución de las pruebas. Por tanto, el análisis histórico se limita a tiempos observables desde la infraestructura y deja la instrumentación fina como recomendación para pruebas futuras.

---

# 9. Checklist mínimo para la medición histórica

- [ ] Verificar que los logs del 11 al 13 de mayo de 2026 aún estén retenidos.
- [ ] Ejecutar `export_cloudrun_request_logs.sh`.
- [ ] Revisar si el CSV trae registros de `miad-rag-frontend` y `miad-rag-backend`.
- [ ] Si hay Observability Analytics, correr `cloudrun_latency_by_endpoint.sql`.
- [ ] Exportar `cloudrun_request_detail.csv`.
- [ ] Exportar `cloudrun_latency_by_endpoint.csv`.
- [ ] Exportar `cloudrun_latency_timeseries.csv`.
- [ ] Revisar logs de aplicación con `cloudrun_backend_app_logs_inventory.sql`.
- [ ] Documentar limitación de medición interna no retroactiva.
- [ ] Preparar análisis de p50, p95, p99 y errores.

---

# 10. Recomendación práctica

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

Estos últimos solo tendrían sentido si el frontend/backend emiten logs estructurados durante las pruebas.

---

# 11. Instrumentación sugerida en backend

Futura implementación

## 11.1 Helper de timing

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

## 11.2 Uso conceptual en endpoint `/api/v1/recommend`

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

# 12. Instrumentación sugerida en frontend

## 12.1 Generar sesión y request id

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

## 12.2 Medir llamada frontend → backend

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
