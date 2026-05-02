#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-miad-paad-rs-dev}"
REGION="${REGION:-us-east4}"
BUCKET_NAME="${BUCKET_NAME:-miad-paad-rs-staging-dev}"
DATASET_ID="${DATASET_ID:-ds_miad_rag_rs}"
TABLE_ID="${TABLE_ID:-real_estate_listings}"

LOCAL_CSV_PATH="${1:-../samples/real_estate_listings.csv}"
GCS_OBJECT_NAME="${2:-real_estate_listings.csv}"
GCS_URI="gs://${BUCKET_NAME}/real_estate_listings/${GCS_OBJECT_NAME}"

SCHEMA_PATH="${SCHEMA_PATH:-../schemas/real_estate_listings_schema.json}"

echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Source CSV: ${LOCAL_CSV_PATH}"
echo "Target GCS: ${GCS_URI}"
echo "Target BigQuery: ${PROJECT_ID}.${DATASET_ID}.${TABLE_ID}"

if [ ! -f "${LOCAL_CSV_PATH}" ]; then
  echo "ERROR: CSV file not found: ${LOCAL_CSV_PATH}"
  exit 1
fi

if [ ! -f "${SCHEMA_PATH}" ]; then
  echo "ERROR: BigQuery schema file not found: ${SCHEMA_PATH}"
  exit 1
fi

echo "Uploading CSV to Cloud Storage..."
gcloud storage cp "${LOCAL_CSV_PATH}" "${GCS_URI}" \
  --project="${PROJECT_ID}"

echo "Loading CSV into BigQuery..."
bq --project_id="${PROJECT_ID}" load \
  --location="${REGION}" \
  --source_format=CSV \
  --skip_leading_rows=1 \
  --field_delimiter="," \
  --quote='"' \
  --allow_quoted_newlines \
  --replace \
  "${PROJECT_ID}:${DATASET_ID}.${TABLE_ID}" \
  "${GCS_URI}" \
  "${SCHEMA_PATH}"

echo "Load completed successfully."