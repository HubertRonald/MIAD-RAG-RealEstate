# -------------------------------------------------------------------
# BigQuery Job permissions
# -------------------------------------------------------------------
# These permissions are required because both backend and indexer run
# BigQuery queries using the BigQuery Jobs API.
#
# - backend:
#     enriches FAISS-retrieved listing IDs with SELECT * from BigQuery.
#
# - indexer:
#     reads real_estate_listings from BigQuery to build the FAISS index.
#
# roles/bigquery.dataViewer allows reading tables.
# roles/bigquery.jobUser allows creating query jobs.
# -------------------------------------------------------------------

resource "google_project_iam_member" "backend_bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.backend.email}"

  depends_on = [
    google_project_service.services,
    google_service_account.backend,
  ]
}

resource "google_project_iam_member" "indexer_bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.indexer.email}"

  depends_on = [
    google_project_service.services,
    google_service_account.indexer,
  ]
}