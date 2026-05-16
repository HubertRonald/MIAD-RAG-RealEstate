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