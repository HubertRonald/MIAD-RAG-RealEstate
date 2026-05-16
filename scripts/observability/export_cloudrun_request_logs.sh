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