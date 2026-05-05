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

# -------------------------------------------------------------------
# IAP service identity
# -------------------------------------------------------------------
# Creates/ensures the IAP service agent exists:
# service-PROJECT_NUMBER@gcp-sa-iap.iam.gserviceaccount.com
# This service agent needs roles/run.invoker on the frontend Cloud Run
# service when IAP protects the frontend.
# -------------------------------------------------------------------

resource "google_project_service_identity" "iap" {
  provider = google-beta

  project = var.project_id
  service = "iap.googleapis.com"

  depends_on = [
    google_project_service.services
  ]
}

resource "google_cloud_run_v2_service_iam_member" "frontend_invoker_iap_service_agent" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.frontend.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_project_service_identity.iap.email}"

  depends_on = [
    google_cloud_run_v2_service.frontend,
    google_project_service_identity.iap
  ]
}

resource "google_iap_web_cloud_run_service_iam_member" "frontend_iap_access" {
  for_each = toset(var.frontend_iap_members)

  project                = var.project_id
  location               = var.region
  cloud_run_service_name = google_cloud_run_v2_service.frontend.name
  role                   = "roles/iap.httpsResourceAccessor"
  member                 = each.value

  depends_on = [
    google_cloud_run_v2_service.frontend
  ]
}
