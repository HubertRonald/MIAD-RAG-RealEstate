WITH requests AS (
  SELECT
    TIMESTAMP_TRUNC(timestamp, MINUTE) AS minute,
    JSON_VALUE(resource.labels.service_name) AS service_name,
    http_request.status AS status,
    (
      COALESCE(http_request.latency.seconds, 0) * 1000
      + COALESCE(http_request.latency.nanos, 0) / 1000000
    ) AS latency_ms
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
)
SELECT
  minute,
  service_name,
  COUNT(*) AS requests,
  COUNTIF(status >= 500) AS errors_5xx,
  ROUND(AVG(latency_ms), 2) AS avg_ms,
  ROUND(APPROX_QUANTILES(latency_ms, 100)[OFFSET(50)], 2) AS p50_ms,
  ROUND(APPROX_QUANTILES(latency_ms, 100)[OFFSET(95)], 2) AS p95_ms,
  ROUND(APPROX_QUANTILES(latency_ms, 100)[OFFSET(99)], 2) AS p99_ms
FROM requests
GROUP BY
  minute,
  service_name
ORDER BY
  minute ASC,
  service_name;