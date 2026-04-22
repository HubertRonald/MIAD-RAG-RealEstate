output "artifact_registry_repo" {
  value = google_artifact_registry_repository.repo.name
}

output "frontend_url" {
  value = google_cloud_run_v2_service.frontend.uri
}

output "backend_url" {
  value = google_cloud_run_v2_service.backend.uri
}

output "staging_bucket" {
  value = google_storage_bucket.staging.name
}

output "index_bucket" {
  value = google_storage_bucket.index.name
}

output "bigquery_dataset" {
  value = google_bigquery_dataset.rag.dataset_id
}

output "frontend_service_account" {
  value = google_service_account.frontend.email
}

output "backend_service_account" {
  value = google_service_account.backend.email
}

output "indexer_service_account" {
  value = google_service_account.indexer.email
}