locals {
  required_services = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "bigquery.googleapis.com",
    "bigquerystorage.googleapis.com",
    "generativelanguage.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "logging.googleapis.com",
    "iap.googleapis.com"
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
  iap_enabled         = true
  deletion_protection = false

  template {
    service_account = google_service_account.frontend.email

    # Streamlit puede atender varias sesiones, pero no conviene ponerlo altísimo
    # porque mantiene estado de sesión y puede consumir memoria.
    max_instance_request_concurrency = 10
    timeout                          = "300s"

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = var.frontend_image

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }

        startup_cpu_boost = true
      }

      env {
        name  = "ENV"
        value = "dev"
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

resource "google_cloud_run_v2_service_iam_member" "backend_invoker_user_dev" {
  for_each = toset(var.backend_invoker_members_dev)

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  member   = each.value
}

resource "google_cloud_run_v2_service" "backend" {
  name                = var.backend_service_name
  location            = var.region
  project             = var.project_id
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.backend.email

    # Backend RAG: mejor concurrencia baja mientras medimos memoria/latencia.
    max_instance_request_concurrency = 3
    timeout                          = "600s"

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = var.backend_image

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }

        startup_cpu_boost = true
      }

      # -----------------------------------------------------------------
      # App/runtime
      # -----------------------------------------------------------------
      env {
        name  = "APP_NAME"
        value = "miad-rag-backend"
      }

      env {
        name  = "APP_VERSION"
        value = "0.1.0"
      }

      env {
        name  = "ENV"
        value = "dev"
      }

      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }

      env {
        name  = "JSON_LOGS"
        value = "true"
      }

      env {
        name  = "CORS_ALLOW_ORIGINS"
        value = "*"
      }

      # -----------------------------------------------------------------
      # GCP
      # -----------------------------------------------------------------
      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "GCP_LOCATION"
        value = var.region
      }

      # -----------------------------------------------------------------
      # BigQuery
      # Nombres alineados con RuntimeSettings del backend.
      # Mantengo también BQ_DATASET/BQ_TABLE por compatibilidad.
      # -----------------------------------------------------------------
      env {
        name  = "BQ_PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "BQ_DATASET_ID"
        value = var.bigquery_dataset_id
      }

      env {
        name  = "BQ_LISTINGS_TABLE"
        value = var.bigquery_main_table_id
      }

      env {
        name  = "BQ_LOCATION"
        value = var.region
      }

      env {
        name  = "BQ_DATASET"
        value = var.bigquery_dataset_id
      }

      env {
        name  = "BQ_TABLE"
        value = var.bigquery_main_table_id
      }

      # -----------------------------------------------------------------
      # GCS / FAISS
      # -----------------------------------------------------------------
      env {
        name  = "INDEX_BUCKET"
        value = google_storage_bucket.index.name
      }

      env {
        name  = "INDEX_PREFIX"
        value = "faiss"
      }

      env {
        name  = "INDEX_LOCAL_ROOT"
        value = "/tmp/faiss_index"
      }

      env {
        name  = "DEFAULT_COLLECTION"
        value = "realstate_mvd"
      }

      # -----------------------------------------------------------------
      # Retrieval
      # -----------------------------------------------------------------
      env {
        name  = "RETRIEVAL_K"
        value = "10"
      }

      env {
        name  = "RETRIEVAL_FETCH_K"
        value = "200"
      }

      # -----------------------------------------------------------------
      # Gemini
      # -----------------------------------------------------------------
      env {
        name  = "GEMINI_EMBEDDING_MODEL"
        value = "models/gemini-embedding-001"
      }

      env {
        name  = "GEMINI_GENERATION_MODEL"
        value = "gemini-2.5-flash"
      }

      env {
        name  = "GEMINI_TEMPERATURE"
        value = "0.2"
      }

      env {
        name  = "GEMINI_MAX_OUTPUT_TOKENS"
        value = "5000"
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

      env {
        name = "GOOGLE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }

      # -----------------------------------------------------------------
      # Cost estimation
      # -----------------------------------------------------------------
      env {
        name  = "EMBEDDING_PRICE_PER_M_TOKENS"
        value = "0.025"
      }

      env {
        name  = "CHARS_PER_TOKEN"
        value = "3.2"
      }

      # -----------------------------------------------------------------
      # Reranking
      # Mantener apagado mientras no subamos memoria o validemos latencia.
      # -----------------------------------------------------------------
      env {
        name  = "ENABLE_RERANKING_MODEL"
        value = "false"
      }

      env {
        name  = "RERANKING_MODEL"
        value = "cross-encoder/ms-marco-MiniLM-L-6-v2"
      }

      env {
        name  = "RERANKING_TOP_K"
        value = "3"
      }

      # -----------------------------------------------------------------
      # Frontend scoring
      # -----------------------------------------------------------------
      env {
        name  = "SCORE_LOW"
        value = "0.78"
      }

      env {
        name  = "SCORE_HIGH"
        value = "0.92"
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
      timeout         = "3600s"
      max_retries     = 0

      containers {
        image = var.job_image

        resources {
          limits = {
            cpu    = "4"
            memory = "8Gi"
          }
        }

        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }

        env {
          name  = "GCP_LOCATION"
          value = var.region
        }

        env {
          name  = "BQ_PROJECT_ID"
          value = var.project_id
        }

        env {
          name  = "BQ_DATASET_ID"
          value = var.bigquery_dataset_id
        }

        env {
          name  = "BQ_LISTINGS_TABLE"
          value = var.bigquery_main_table_id
        }

        env {
          name  = "BQ_LOCATION"
          value = var.region
        }

        env {
          name  = "COLLECTION"
          value = "realstate_mvd"
        }

        env {
          name  = "INDEX_BUCKET"
          value = google_storage_bucket.index.name
        }

        env {
          name  = "INDEX_PREFIX"
          value = "faiss"
        }

        env {
          name  = "DRY_RUN"
          value = "false"
        }

        dynamic "env" {
          for_each = var.indexer_bq_limit == null ? [] : [var.indexer_bq_limit]
          iterator = bq_limit

          content {
            name  = "BQ_LIMIT"
            value = tostring(bq_limit.value)
          }
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

        env {
          name = "GOOGLE_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.gemini_api_key.secret_id
              version = "latest"
            }
          }
        }
      }
    }
  }

  depends_on = [google_project_service.services]
}