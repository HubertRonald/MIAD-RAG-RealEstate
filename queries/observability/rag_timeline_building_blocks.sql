WITH timing AS (
  SELECT
    JSON_VALUE(json_payload.request_id) AS request_id,
    JSON_VALUE(json_payload.session_id_hash) AS session_id_hash,
    JSON_VALUE(json_payload.stage) AS stage,
    CAST(JSON_VALUE(json_payload.duration_ms) AS FLOAT64) AS duration_ms,
    timestamp
  FROM
    `miad-paad-rs-dev.global._Default._AllLogs`
  WHERE
    timestamp >= TIMESTAMP("2026-05-11T05:00:00Z")
    AND timestamp < TIMESTAMP("2026-05-14T05:00:00Z")
    AND resource.type = "cloud_run_revision"
    AND JSON_VALUE(json_payload.event_type) = "rag_timing"
)
SELECT
  request_id,
  ANY_VALUE(session_id_hash) AS session_id_hash,
  MIN(timestamp) AS first_event_ts,
  MAX(timestamp) AS last_event_ts,
  TIMESTAMP_DIFF(MAX(timestamp), MIN(timestamp), MILLISECOND) AS observed_e2e_ms,

  MAX(IF(stage = "frontend_backend_call", duration_ms, NULL)) AS frontend_backend_call_ms,
  MAX(IF(stage = "query_understanding", duration_ms, NULL)) AS query_understanding_ms,
  MAX(IF(stage = "gemini_embedding", duration_ms, NULL)) AS gemini_embedding_ms,
  MAX(IF(stage = "faiss_retrieval", duration_ms, NULL)) AS faiss_retrieval_ms,
  MAX(IF(stage = "bigquery_enrichment", duration_ms, NULL)) AS bigquery_enrichment_ms,
  MAX(IF(stage = "gemini_generation", duration_ms, NULL)) AS gemini_generation_ms,
  MAX(IF(stage = "response_serialization", duration_ms, NULL)) AS response_serialization_ms,
  MAX(IF(stage = "frontend_response_rendered", duration_ms, NULL)) AS frontend_response_rendered_ms,

  SUM(duration_ms) AS total_instrumented_ms
FROM timing
GROUP BY
  request_id
ORDER BY
  first_event_ts DESC;