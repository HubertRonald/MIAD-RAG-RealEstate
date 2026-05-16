#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-miad-paad-rs-dev}"
REGION="${REGION:-us-east4}"
FRONTEND_SERVICE="${FRONTEND_SERVICE:-miad-rag-frontend}"
BACKEND_SERVICE="${BACKEND_SERVICE:-miad-rag-backend}"
SINCE_HOURS="${SINCE_HOURS:-24}"
LIMIT="${LIMIT:-20000}"
OUT_DIR="${OUT_DIR:-./observability_exports}"

mkdir -p "${OUT_DIR}"

START_TS="$(date -u -d "${SINCE_HOURS} hours ago" +"%Y-%m-%dT%H:%M:%SZ")"
STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"

RAW_JSON="${OUT_DIR}/cloudrun_requests_${STAMP}.json"
CSV="${OUT_DIR}/cloudrun_requests_${STAMP}.csv"

FILTER=$(cat <<EOF
resource.type="cloud_run_revision"
resource.labels.location="${REGION}"
(resource.labels.service_name="${FRONTEND_SERVICE}" OR resource.labels.service_name="${BACKEND_SERVICE}")
logName="projects/${PROJECT_ID}/logs/run.googleapis.com%2Frequests"
timestamp >= "${START_TS}"
EOF
)

echo "Project : ${PROJECT_ID}"
echo "Region  : ${REGION}"
echo "Since   : ${START_TS}"
echo "Output  : ${OUT_DIR}"

gcloud logging read "${FILTER}" \
  --project="${PROJECT_ID}" \
  --format=json \
  --freshness="${SINCE_HOURS}h" \
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
  .httpRequest.userAgent,
  .trace,
  .insertId
] | @csv)
' "${RAW_JSON}" > "${CSV}"

echo "Generado JSON: ${RAW_JSON}"
echo "Generado CSV : ${CSV}"