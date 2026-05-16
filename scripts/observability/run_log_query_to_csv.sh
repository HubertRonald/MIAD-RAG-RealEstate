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