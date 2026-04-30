<p align="center">
    <img src="../../figs/Banner_PAAD_01.jpg" width="980" />
</p>

<p align="left">
  <a href="https://cloud.google.com" target="_blank">
    <img src="https://img.shields.io/badge/Google%20Cloud%20Platform-4285F4?style=flat-square&logo=googlecloud&logoColor=white" />
  </a>
  <a href="https://github.com/features/actions" target="_blank">
    <img src="https://img.shields.io/badge/GitHub%20Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white" />
  </a>
  <a href="https://registry.terraform.io/providers/hashicorp/google/latest/docs" target="_blank">
    <img src="https://img.shields.io/badge/Terraform-7B42BC?style=flat-square&logo=terraform&logoColor=white" />
  </a>
  <a href="https://yaml.org/" target="_blank">
    <img src="https://img.shields.io/badge/YAML-CB171E?style=flat-square&logo=yaml&logoColor=white" />
  </a>
</p>

# GitHub Actions + GCP Workload Identity Federation

Este documento describe la integración entre GitHub Actions y Google Cloud Platform para el proyecto `MIAD-RAG-RealEstate`.

## Objetivo

Permitir que GitHub Actions despliegue infraestructura y servicios en GCP sin utilizar llaves JSON de service accounts.

## Proyecto GCP

| Recurso | Valor |
|---|---|
| Project ID | `miad-paad-rs-dev` |
| Región | `us-east4` |
| Artifact Registry | `miad-rag-repo` |
| Frontend | `miad-rag-frontend` |
| Backend | `miad-rag-backend` |
| Indexer Job | `miad-rag-indexer-job` |

## Ramas

| Rama | Uso |
|---|---|
| `feature/rony` | Desarrollo y validación inicial |
| `dev` | Integración y validación |
| `main` | Despliegue a GCP |

## Terraform State

El estado remoto de Terraform se almacena en: `gs://miad-paad-rs-tfstate-dev/terraform/dev`

| Service Account | Uso |
|---|---|
| `miad-paad-rs-tfstate-dev` | Bucket para almacenar el estado remoto de Terraform |

## Service Account

## Workload Identity Federation

Se utiliza Workload Identity Federation para evitar almacenar credenciales persistentes en GitHub.

Variables requeridas en GitHub:

```text
GCP_PROJECT_ID
GCP_REGION
GCP_ARTIFACT_REPO
GCP_WORKLOAD_IDENTITY_PROVIDER
GCP_SERVICE_ACCOUNT
```

## Workflows

```text
.github/workflows/terraform.yml
.github/workflows/backend-ci.yml
.github/workflows/frontend-ci.yml
.github/workflows/job-ci.yml
```

## Deployment Rule

Solo la rama main despliega infraestructura y servicios en GCP.

## GCP Cloud shell

Validar que tengas permisos de owner/editor o permisos suficientes de IAM en miad-paad-rs-dev.

```bash
export PROJECT_ID="miad-paad-rs-dev"
export PROJECT_NUMBER="$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')"
export REGION="us-east4"

export GITHUB_ORG="HubertRonald"
export GITHUB_REPO="MIAD-RAG-RealEstate"
export GITHUB_REPOSITORY="${GITHUB_ORG}/${GITHUB_REPO}"

export POOL_ID="github-pool"
export PROVIDER_ID="github-provider"
export DEPLOYER_SA_ID="sa-github-deployer"
export DEPLOYER_SA_EMAIL="${DEPLOYER_SA_ID}@${PROJECT_ID}.iam.gserviceaccount.com"

export TF_STATE_BUCKET="miad-paad-rs-tfstate-dev"

gcloud config set project "${PROJECT_ID}"

gcloud services enable \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  cloudresourcemanager.googleapis.com \
  serviceusage.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  bigquery.googleapis.com \
  logging.googleapis.com
```

### Crear service account de GitHub Actions

```bash
gcloud iam service-accounts create "${DEPLOYER_SA_ID}" \
  --project="${PROJECT_ID}" \
  --display-name="GitHub Actions Deployer"
```

### Dar permisos al deployer
Para dev y prototipo MIAD, esto es práctico. Luego se pueden reducir roles.

```bash
for ROLE in \
  roles/run.admin \
  roles/artifactregistry.admin \
  roles/storage.admin \
  roles/bigquery.admin \
  roles/secretmanager.admin \
  roles/iam.serviceAccountAdmin \
  roles/iam.serviceAccountUser \
  roles/resourcemanager.projectIamAdmin \
  roles/serviceusage.serviceUsageAdmin
do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${DEPLOYER_SA_EMAIL}" \
    --role="${ROLE}"
done
```

### Crear Workload Identity Pool

```bash
gcloud iam workload-identity-pools create "${POOL_ID}" \
  --project="${PROJECT_ID}" \
  --location="global" \
  --display-name="GitHub Actions Pool"
```

### Crear Provider OIDC para GitHub

```bash
gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_ID}" \
  --project="${PROJECT_ID}" \
  --location="global" \
  --workload-identity-pool="${POOL_ID}" \
  --display-name="GitHub OIDC Provider" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository=='${GITHUB_REPOSITORY}'"
```


### Permitir que ese repo use la service account
```bash
gcloud iam service-accounts add-iam-policy-binding "${DEPLOYER_SA_EMAIL}" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/${GITHUB_REPOSITORY}"
```

### Valor que se usará en GitHub Actions
```bash
echo "WORKLOAD_IDENTITY_PROVIDER=projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}"
echo "SERVICE_ACCOUNT=${DEPLOYER_SA_EMAIL}"
```

## Variables de GitHub
En Cloud Shell se puede instalar gh si no está disponible. El comando gh variable set es oficial para crear variables de GitHub Actions.

```bash
gh --version || sudo apt update && sudo apt install gh -y
gh auth login
```

En el repo, configura como Repository variables:

```bash
export GITHUB_ORG="HubertRonald"
REPO="${GITHUB_ORG}/MIAD-RAG-RealEstate"

gh variable set GCP_PROJECT_ID --body "miad-paad-rs-dev" --repo "$REPO"
gh variable set GCP_REGION --body "us-east4" --repo "$REPO"
gh variable set GCP_ARTIFACT_REPO --body "miad-rag-repo" --repo "$REPO"
gh variable set GCP_WORKLOAD_IDENTITY_PROVIDER --body "projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}" --repo "$REPO"
gh variable set GCP_SERVICE_ACCOUNT --body "${DEPLOYER_SA_EMAIL}" --repo "$REPO"
```

También se puede hacer manualmente en GitHub:

```text
Settings → Secrets and variables → Actions → Variables
```

## Bucket para Terraform State
Antes de terraform init, crea el bucket remoto de estado (Promocionar a rama `main`) con el gcp cloud shell:

```bash
export PROJECT_ID="miad-paad-rs-dev"
export REGION="us-east4"
export TF_STATE_BUCKET="miad-paad-rs-tfstate-dev"

gcloud config set project "${PROJECT_ID}"

gcloud storage buckets create "gs://${TF_STATE_BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --uniform-bucket-level-access

gcloud storage buckets update "gs://${TF_STATE_BUCKET}" \
  --versioning
```

## Repository "miad-rag-repo"

```bash
gcloud artifacts repositories create miad-rag-repo \
  --repository-format=docker \
  --location=us-east4 \
  --project=miad-paad-rs-dev \
  --description="Docker repository for MIAD RAG Real Estate"
  ```

## Cloud Run services and job bootstrap

Los workflows de GitHub Actions están diseñados para crear o actualizar los servicios de Cloud Run.

Recursos esperados:

```text
miad-rag-frontend
miad-rag-backend
miad-rag-indexer-job
```

Durante el primer despliegue, si el recurso no existe, el workflow lo crea.
En despliegues posteriores, el workflow actualiza la imagen del recurso existente.


## Sobre Terraform vs GitHub Actions

Lo más limpio sería que Terraform cree:

```text
Artifact Registry
Cloud Run services
Cloud Run job
Buckets
BigQuery
Secrets
IAM
```

y que GitHub Actions solo actualice imágenes.

Pero como ahora se está en fase bootstrap, este enfoque create-or-update en GitHub Actions desbloquea rápido.

Más adelante se puede volver al patrón estricto:

```text
Terraform crea recursos
GitHub Actions actualiza imágenes
```

## Consideraciones de Infraestructura

### Artifact Registry (pre-requisito)
Antes de ejecutar Terraform, el repositorio de imágenes debe existir en Artifact Registry:

```bash
gcloud artifacts repositories create miad-rag-repo \
  --repository-format=docker \
  --location=us-east4
```

> Lanzar lo anterior desde la cloud shell cli de GCP

### Configuración de Cloud Run

Los servicios Cloud Run son definidos mediante Terraform, incluyendo:

- Memoria y CPU
- Variables de entorno
- Acceso mediante Service Accounts
- Restricciones de ingreso

Los cambios en recursos (CPU, RAM, timeout) generan nuevas revisiones del servicio sin eliminar el código desplegado.


### Escalabilidad

Cloud Run permite configurar:

- min_instances
- max_instances

Esto se define en Terraform y no afecta la lógica de la aplicación, solo su comportamiento en ejecución.

### Terraform local

Para validar la infraestructura localmente:

```bash
terraform fmt -recursive
terraform validate
terraform plan
```

---

## Orden correcto de despliegue (CRÍTICO)

```md
El orden de despliegue es:

1. Crear Artifact Registry
2. Ejecutar CI/CD para generar imágenes
3. Ejecutar Terraform (apply=true)
4. Ejecutar job de indexación