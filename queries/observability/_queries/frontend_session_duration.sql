WITH events AS (
  SELECT
    JSON_VALUE(json_payload.session_id_hash) AS session_id_hash,
    JSON_VALUE(json_payload.event_name) AS event_name,
    timestamp
  FROM
    `miad-paad-rs-dev.global._Default._AllLogs`
  WHERE
    timestamp >= TIMESTAMP("2026-05-11T05:00:00Z")
    AND timestamp < TIMESTAMP("2026-05-14T05:00:00Z")
    AND resource.type = "cloud_run_revision"
    AND JSON_VALUE(resource.labels.service_name) = "miad-rag-frontend"
    AND JSON_VALUE(json_payload.event_type) = "frontend_session_event"
)
SELECT
  session_id_hash,
  MIN(timestamp) AS first_seen,
  MAX(timestamp) AS last_seen,
  TIMESTAMP_DIFF(MAX(timestamp), MIN(timestamp), SECOND) AS active_seconds,
  COUNTIF(event_name = "search_submitted") AS searches_submitted,
  COUNTIF(event_name = "response_rendered") AS responses_rendered
FROM events
GROUP BY
  session_id_hash
ORDER BY
  last_seen DESC;