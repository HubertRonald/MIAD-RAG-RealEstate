project_id      = "miad-paad-rs-dev"
region          = "us-east4"

frontend_image  = "us-east4-docker.pkg.dev/miad-paad-rs-dev/miad-rag-repo/miad-rag-frontend:latest"
backend_image   = "us-east4-docker.pkg.dev/miad-paad-rs-dev/miad-rag-repo/miad-rag-backend:latest"
job_image       = "us-east4-docker.pkg.dev/miad-paad-rs-dev/miad-rag-repo/miad-rag-indexer-job:latest"

frontend_allowed_members = [
  "user:hubert.ronald@gmail.com"
]