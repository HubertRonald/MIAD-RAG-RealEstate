# -------------------------------------------------------------------
# BigQuery runtime permissions
# -------------------------------------------------------------------
# Existing permissions in main.tf already cover:
# - roles/bigquery.dataViewer for reading BigQuery tables.
# - roles/storage.objectViewer for backend reads from the FAISS index bucket.
# - roles/storage.objectAdmin for indexer writes to the FAISS index bucket.
#
# This file adds:
# - roles/bigquery.jobUser:
#     required for bigquery.jobs.create when running queries.
#
# - roles/bigquery.readSessionUser:
#     required when pandas/to_dataframe uses the BigQuery Storage Read API.
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

resource "google_project_iam_member" "backend_bq_read_session_user" {
  project = var.project_id
  role    = "roles/bigquery.readSessionUser"
  member  = "serviceAccount:${google_service_account.backend.email}"

  depends_on = [
    google_project_service.services,
    google_service_account.backend,
  ]
}

resource "google_project_iam_member" "indexer_bq_read_session_user" {
  project = var.project_id
  role    = "roles/bigquery.readSessionUser"
  member  = "serviceAccount:${google_service_account.indexer.email}"

  depends_on = [
    google_project_service.services,
    google_service_account.indexer,
  ]
}