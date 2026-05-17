SELECT
  timestamp,
  severity,
  log_id,
  trace,
  JSON_VALUE(resource.labels.service_name) AS service_name,
  JSON_VALUE(resource.labels.revision_name) AS revision_name,
  text_payload,
  TO_JSON_STRING(json_payload) AS json_payload_str,
  TO_JSON_STRING(proto_payload) AS proto_payload_str
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
    OR LOWER(COALESCE(TO_JSON_STRING(json_payload), "")) LIKE "%bigquery%"
    OR LOWER(COALESCE(TO_JSON_STRING(json_payload), "")) LIKE "%gemini%"
    OR LOWER(COALESCE(TO_JSON_STRING(json_payload), "")) LIKE "%faiss%"
    OR LOWER(COALESCE(TO_JSON_STRING(json_payload), "")) LIKE "%gcs%"
    OR LOWER(COALESCE(TO_JSON_STRING(json_payload), "")) LIKE "%retrieval%"
    OR LOWER(COALESCE(TO_JSON_STRING(json_payload), "")) LIKE "%generation%"
    OR LOWER(COALESCE(TO_JSON_STRING(json_payload), "")) LIKE "%recommend%"
  )
ORDER BY
  timestamp ASC;