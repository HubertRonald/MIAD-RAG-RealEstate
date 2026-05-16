SELECT
  timestamp,
  trace,
  JSON_VALUE(json_payload.request_id) AS request_id,
  JSON_VALUE(json_payload.session_id_hash) AS session_id_hash,
  JSON_VALUE(json_payload.component) AS component,
  CAST(JSON_VALUE(json_payload.stage_order) AS INT64) AS stage_order,
  JSON_VALUE(json_payload.stage) AS stage,
  CAST(JSON_VALUE(json_payload.duration_ms) AS FLOAT64) AS duration_ms,
  JSON_VALUE(json_payload.status) AS status,
  json_payload
FROM
  `miad-paad-rs-dev.global._Default._AllLogs`
WHERE
  timestamp >= TIMESTAMP("2026-05-11T05:00:00Z")
  AND timestamp < TIMESTAMP("2026-05-14T05:00:00Z")
  AND resource.type = "cloud_run_revision"
  AND JSON_VALUE(json_payload.event_type) = "rag_timing"
ORDER BY
  request_id,
  stage_order,
  timestamp;