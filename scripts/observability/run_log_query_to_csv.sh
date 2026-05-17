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
  echo

  echo "STDOUT:"
  cat "${TMP_CSV}" || true
  echo

  rm -f "${TMP_CSV}"
  exit 1
fi