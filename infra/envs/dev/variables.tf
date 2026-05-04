variable "project_id" {
  type        = string
  description = "GCP Project ID"
  default     = "miad-paad-rs-dev"
}

variable "region" {
  type        = string
  description = "GCP region"
  default     = "us-east4"
}

variable "artifact_repo_name" {
  type    = string
  default = "miad-rag-repo"
}

variable "frontend_service_name" {
  type    = string
  default = "miad-rag-frontend"
}

variable "backend_service_name" {
  type    = string
  default = "miad-rag-backend"
}

variable "indexer_job_name" {
  type    = string
  default = "miad-rag-indexer-job"
}

variable "staging_bucket_name" {
  type    = string
  default = "miad-paad-rs-staging-dev"
}

variable "index_bucket_name" {
  type    = string
  default = "miad-paad-rs-index-dev"
}

variable "bigquery_dataset_id" {
  type    = string
  default = "ds_miad_rag_rs"
}

variable "bigquery_main_table_id" {
  type    = string
  default = "real_estate_listings"
}

variable "bigquery_eval_table_id" {
  type    = string
  default = "rag_eval_results"
}

variable "sa_frontend_id" {
  type    = string
  default = "sa-rag-frontend"
}

variable "sa_backend_id" {
  type    = string
  default = "sa-rag-backend"
}

variable "sa_indexer_id" {
  type    = string
  default = "sa-rag-indexer"
}

variable "sa_github_deployer_id" {
  type    = string
  default = "sa-github-deployer"
}

variable "frontend_image" {
  type        = string
  description = "Artifact Registry image URL for frontend"
}

variable "backend_image" {
  type        = string
  description = "Artifact Registry image URL for backend"
}

variable "job_image" {
  type        = string
  description = "Artifact Registry image URL for indexer job"
}

variable "indexr_bq_limit" {
  type        = number
  description = "Optional BigQuery LIMIT for indexer runs. Use null for full"
  default     = 500
  nullable    = true
}

variable "frontend_allowed_members" {
  type        = list(string)
  description = "IAM principals allowed to invoke frontend Cloud Run"
  default     = []
}

variable "bq_dashboard_viewer_members" {
  type        = list(string)
  description = "Users or groups allowed to query the BigQuery dataset from Looker Studio"
  default     = []
}

variable "labels" {
  type = map(string)
  default = {
    app     = "miad-rag-realestate"
    env     = "dev"
    course  = "paad"
    program = "miad"
  }
}