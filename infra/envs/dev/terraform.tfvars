project_id = "miad-paad-rs-dev"
region     = "us-east4"

frontend_image = "us-east4-docker.pkg.dev/miad-paad-rs-dev/miad-rag-repo/miad-rag-frontend:latest"
backend_image  = "us-east4-docker.pkg.dev/miad-paad-rs-dev/miad-rag-repo/miad-rag-backend:latest"
job_image      = "us-east4-docker.pkg.dev/miad-paad-rs-dev/miad-rag-repo/miad-rag-indexer-job:latest"

# NO público
frontend_allowed_members = []

# Usuarios autorizados por IAP
frontend_iap_members = [
  "user:hubert.ronald@gmail.com",
  "user:hrmcanales@gmail.com",
  "user:paulina.luissi@gmail.com",
  "user:alebarbosac@gmail.com",
  "user:jmmarin1@gmail.com"
]

backend_invoker_members_dev = [
  "user:hubert.ronald@gmail.com",
  "user:hrmcanales@gmail.com"
]

bq_dashboard_viewer_members = [
  "user:hubert.ronald@gmail.com",
  "user:hrmcanales@gmail.com",
  "user:paulina.luissi@gmail.com",
  "user:alebarbosac@gmail.com",
  "user:jmmarin1@gmail.com"
]