#!/bin/bash
# ======================
# GITHUB
# ======================
mkdir -p .github/workflows
touch .github/workflows/{backend-ci.yml,frontend-ci.yml,job-ci.yml,terraform.yml}

# ======================
# INFRA
# ======================
mkdir -p infra/bootstrap
touch infra/bootstrap/{backend.tf,providers.tf,versions.tf}

mkdir -p infra/envs/dev
touch infra/envs/dev/{main.tf,variables.tf,outputs.tf,terraform.tfvars}

mkdir -p infra/modules/{artifact_registry,cloud_run_service,cloud_run_job,bigquery,gcs,iam,secrets}

# ======================
# APPS
# ======================
# Backend
mkdir -p apps/backend/app/{routers,services,models,utils}
mkdir -p apps/backend/tests
touch apps/backend/{main.py,requirements.txt,Dockerfile}

# Frontend
mkdir -p apps/frontend/app/{pages,components}
touch apps/frontend/{requirements.txt,Dockerfile}
touch apps/frontend/app/main.py

# Job Indexer
mkdir -p apps/job-indexer/app/services
mkdir -p apps/job-indexer/app/utils
mkdir -p apps/job-indexer/tests
touch apps/job-indexer/app/build_index.py
touch apps/job-indexer/{requirements.txt,Dockerfile}

# ======================
# SHARED
# ======================
mkdir -p shared/python/miad_rag_common/{config,schemas,logging,gcp}
mkdir -p shared/contracts/{openapi,jsonschemas,examples}

# ======================
# DATA
# ======================
mkdir -p data/{schemas,samples,dictionaries}
touch data/schemas/bigquery_schema.json
touch data/dictionaries/data_dictionary.md

# ======================
# EVAL
# ======================
mkdir -p eval/ragas/{datasets,notebooks,scripts}


echo "Estructura creada correctamente 🚀"