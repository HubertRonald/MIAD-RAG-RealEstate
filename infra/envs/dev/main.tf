locals {
  required_services = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "bigquery.googleapis.com",
    "bigquerystorage.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "logging.googleapis.com"
  ]
}

resource "google_project_service" "services" {
  for_each = toset(local.required_services)
  project  = var.project_id
  service  = each.value

  disable_on_destroy = false
}

/*
resource "google_artifact_registry_repository" "repo" {
  provider      = google-beta
  project       = var.project_id
  location      = var.region
  repository_id = var.artifact_repo_name
  format        = "DOCKER"
  description   = "Artifact Registry repo for MIAD RAG Real Estate"
  labels        = var.labels

  depends_on = [google_project_service.services]
}
*/

resource "google_storage_bucket" "staging" {
  name                        = var.staging_bucket_name
  location                    = var.region
  project                     = var.project_id
  uniform_bucket_level_access = true
  force_destroy               = false
  labels                      = var.labels
}

resource "google_storage_bucket" "index" {
  name                        = var.index_bucket_name
  location                    = var.region
  project                     = var.project_id
  uniform_bucket_level_access = true
  force_destroy               = false
  labels                      = var.labels
}

resource "google_service_account" "frontend" {
  project      = var.project_id
  account_id   = var.sa_frontend_id
  display_name = "Frontend service account"
}

resource "google_service_account" "backend" {
  project      = var.project_id
  account_id   = var.sa_backend_id
  display_name = "Backend service account"
}

resource "google_service_account" "indexer" {
  project      = var.project_id
  account_id   = var.sa_indexer_id
  display_name = "Indexer job service account"
}

/*
resource "google_service_account" "github_deployer" {
  project      = var.project_id
  account_id   = var.sa_github_deployer_id
  display_name = "GitHub deployer service account"
}
*/

resource "google_project_iam_member" "frontend_bq_viewer" {
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.frontend.email}"
}

resource "google_project_iam_member" "frontend_run_invoker_backend_token" {
  project = var.project_id
  role    = "roles/iam.serviceAccountTokenCreator"
  member  = "serviceAccount:${google_service_account.frontend.email}"
}

resource "google_project_iam_member" "backend_bq_viewer" {
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_project_iam_member" "backend_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_project_iam_member" "indexer_bq_viewer" {
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.indexer.email}"
}

resource "google_project_iam_member" "indexer_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.indexer.email}"
}

resource "google_storage_bucket_iam_member" "backend_index_bucket_reader" {
  bucket = google_storage_bucket.index.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_storage_bucket_iam_member" "indexer_index_bucket_admin" {
  bucket = google_storage_bucket.index.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.indexer.email}"
}

resource "google_storage_bucket_iam_member" "indexer_staging_bucket_reader" {
  bucket = google_storage_bucket.staging.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.indexer.email}"
}

resource "google_storage_bucket_iam_member" "frontend_staging_bucket_reader" {
  bucket = google_storage_bucket.staging.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.frontend.email}"
}

resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "gemini-api-key"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "streamlit_auth_secret" {
  secret_id = "streamlit-auth-secret"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "ragas_api_key" {
  secret_id = "ragas-api-key"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "service_config" {
  secret_id = "service-config"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_bigquery_dataset" "rag" {
  project                    = var.project_id
  dataset_id                 = var.bigquery_dataset_id
  location                   = var.region
  delete_contents_on_destroy = false
  labels                     = var.labels
}

resource "google_bigquery_dataset_iam_member" "dashboard_data_viewer" {
  for_each = toset(var.bq_dashboard_viewer_members)

  project    = var.project_id
  dataset_id = google_bigquery_dataset.rag.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = each.value
}

resource "google_project_iam_member" "dashboard_job_user" {
  for_each = toset(var.bq_dashboard_viewer_members)

  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = each.value
}

resource "google_bigquery_table" "real_estate_listings" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.rag.dataset_id
  table_id            = var.bigquery_main_table_id
  deletion_protection = false
  schema              = file("${path.module}/../../../data/schemas/real_estate_listings_schema.json")
  labels              = var.labels
}

resource "google_bigquery_table" "rag_eval_results" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.rag.dataset_id
  table_id            = var.bigquery_eval_table_id
  deletion_protection = false
  schema              = file("${path.module}/../../../data/schemas/rag_eval_results_schema.json")
  labels              = var.labels
}

resource "google_cloud_run_v2_service" "frontend" {
  name                = var.frontend_service_name
  location            = var.region
  project             = var.project_id
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.frontend.email

    containers {
      image = var.frontend_image

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      env {
        name  = "BACKEND_URL"
        value = google_cloud_run_v2_service.backend.uri
      }
    }
  }

  depends_on = [google_project_service.services]
}

resource "google_cloud_run_v2_service_iam_member" "frontend_invoker" {
  for_each = toset(var.frontend_allowed_members)

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.frontend.name
  role     = "roles/run.invoker"
  member   = each.value
}

resource "google_cloud_run_v2_service" "backend" {
  name                = var.backend_service_name
  location            = var.region
  project             = var.project_id
  ingress             = "INGRESS_TRAFFIC_INTERNAL_ONLY"
  deletion_protection = false

  template {
    service_account = google_service_account.backend.email

    containers {
      image = var.backend_image

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      env {
        name  = "BQ_DATASET"
        value = var.bigquery_dataset_id
      }

      env {
        name  = "BQ_TABLE"
        value = var.bigquery_main_table_id
      }

      env {
        name  = "INDEX_BUCKET"
        value = google_storage_bucket.index.name
      }

      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [google_project_service.services]
}

resource "google_cloud_run_v2_service_iam_member" "backend_invoker_frontend_sa" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.frontend.email}"
}

resource "google_cloud_run_v2_job" "indexer" {
  provider            = google-beta
  name                = var.indexer_job_name
  location            = var.region
  project             = var.project_id
  deletion_protection = false

  template {
    template {
      service_account = google_service_account.indexer.email

      containers {
        image = var.job_image

        resources {
          limits = {
            cpu    = "1"
            memory = "1Gi"
          }
        }

        env {
          name  = "BQ_DATASET"
          value = var.bigquery_dataset_id
        }

        env {
          name  = "BQ_TABLE"
          value = var.bigquery_main_table_id
        }

        env {
          name  = "INDEX_BUCKET"
          value = google_storage_bucket.index.name
        }

        env {
          name = "GEMINI_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.gemini_api_key.secret_id
              version = "latest"
            }
          }
        }
      }

      timeout = "3600s"
    }
  }

  depends_on = [google_project_service.services]
}