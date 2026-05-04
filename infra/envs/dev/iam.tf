# -------------------------------------------------------------------
# BigQuery Job permissions
# -------------------------------------------------------------------
# Required because both backend and indexer execute BigQuery queries.
#
# Existing permissions in main.tf already cover:
# - roles/bigquery.dataViewer for reading BigQuery data.
# - roles/storage.objectViewer for backend index reads.
# - roles/storage.objectAdmin for indexer index writes.
#
# This file only adds roles/bigquery.jobUser, which grants
# bigquery.jobs.create so query jobs can be created.
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