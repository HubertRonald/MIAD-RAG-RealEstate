# Makefile para despliegue automatizado en Cloud Run
# Optimizado para Linux/Mac (bash)
# =====================================================

# Usar bash como shell
SHELL := /bin/bash

include .env
export

# Variables derivadas
ifeq ($(GCP_REGISTRY_TYPE),artifact-registry)
	IMAGE_URL := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GCP_ARTIFACT_REPO)/$(GCP_SERVICE_NAME):latest
	IMAGE_TAG := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GCP_ARTIFACT_REPO)/$(GCP_SERVICE_NAME):$(shell date +%Y%m%d-%H%M%S)
else
	IMAGE_URL := gcr.io/$(GCP_PROJECT_ID)/$(GCP_SERVICE_NAME):latest
	IMAGE_TAG := gcr.io/$(GCP_PROJECT_ID)/$(GCP_SERVICE_NAME):$(shell date +%Y%m%d-%H%M%S)
endif

# =====================================================
# COMANDOS DE AYUDA
# =====================================================

.PHONY: help
help: ## Mostrar esta ayuda
	@echo "Comandos disponibles para despliegue en Cloud Run:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Flujo recomendado:"
	@echo "  1. make setup          (primera vez)"
	@echo "  2. make setup-secrets  (primera vez)"
	@echo "  3. make deploy         (cada vez que quieras desplegar)"

.PHONY: info
info: ## Mostrar configuración actual
	@echo "============================================="
	@echo "CONFIGURACION ACTUAL"
	@echo "============================================="
	@echo "Proyecto GCP:     $(GCP_PROJECT_ID)"
	@echo "Region:           $(GCP_REGION)"
	@echo "Servicio:         $(GCP_SERVICE_NAME)"
	@echo "Tipo de registro: $(GCP_REGISTRY_TYPE)"
	@echo "Imagen:           $(IMAGE_URL)"
	@echo ""
	@echo "Recursos Cloud Run:"
	@echo "  Memoria:        $(CLOUD_RUN_MEMORY)"
	@echo "  CPU:            $(CLOUD_RUN_CPU)"
	@echo "  Timeout:        $(CLOUD_RUN_TIMEOUT)s"
	@echo "  Min Instances:  $(CLOUD_RUN_MIN_INSTANCES)"
	@echo "  Max Instances:  $(CLOUD_RUN_MAX_INSTANCES)"
	@echo "============================================="

# =====================================================
# CONFIGURACION INICIAL (SOLO PRIMERA VEZ)
# =====================================================

.PHONY: setup
setup: ## Configurar proyecto GCP (primera vez)
	@echo "🔧 Configurando proyecto GCP..."
	@echo ""
	gcloud config set project $(GCP_PROJECT_ID)
	@echo ""
	@echo "Habilitando APIs necesarias..."
	gcloud services enable run.googleapis.com
	gcloud services enable cloudbuild.googleapis.com
ifeq ($(GCP_REGISTRY_TYPE),artifact-registry)
	gcloud services enable artifactregistry.googleapis.com
	@echo ""
	@echo "Creando repositorio de Artifact Registry..."
	gcloud artifacts repositories create $(GCP_ARTIFACT_REPO) \
		--repository-format=docker \
		--location=$(GCP_REGION) \
		--description="Repositorio para $(GCP_SERVICE_NAME)" || echo "Repositorio ya existe, continuando..."
else
	gcloud services enable containerregistry.googleapis.com
endif
	@echo ""
	@echo " Configuración completada exitosamente"

.PHONY: setup-secrets
setup-secrets: ## Configurar secretos en Secret Manager
	@echo " Configurando secretos en Secret Manager..."
	@echo ""
	gcloud services enable secretmanager.googleapis.com
	@echo ""
	@echo "Creando/actualizando secreto google-api-key..."
	@echo -n "$(GOOGLE_API_KEY)" | gcloud secrets create google-api-key --data-file=- || \
		echo -n "$(GOOGLE_API_KEY)" | gcloud secrets versions add google-api-key --data-file=-
	@echo ""
	@echo "Creando/actualizando secreto langsmith-api-key..."
	@if [ "$(LANGSMITH_API_KEY)" != "tu_clave_langsmith_aqui" ] && [ -n "$(LANGSMITH_API_KEY)" ]; then \
		echo -n "$(LANGSMITH_API_KEY)" | gcloud secrets create langsmith-api-key --data-file=- || \
		echo -n "$(LANGSMITH_API_KEY)" | gcloud secrets versions add langsmith-api-key --data-file=-; \
		echo "  ✓ LangSmith API key configurada"; \
	else \
		echo "  ⊘ LangSmith API key no configurada (opcional)"; \
	fi
	@echo ""
	@echo " Secretos configurados exitosamente"

# =====================================================
# CONSTRUCCION Y DESPLIEGUE
# =====================================================

.PHONY: build
build: ## Construir imagen Docker localmente
	@echo "  Construyendo imagen Docker localmente..."
	@echo ""
	docker build -t $(GCP_SERVICE_NAME):local .
	@echo ""
	@echo " Imagen construida: $(GCP_SERVICE_NAME):local"

.PHONY: run-local
run-local: ## Ejecutar contenedor localmente en puerto 8000
	@echo " Ejecutando contenedor localmente..."
	@echo "Servidor disponible en: http://localhost:8000"
	@echo "Health check: http://localhost:8000/health"
	@echo ""
	docker run -p 8000:8000 --env-file .env $(GCP_SERVICE_NAME):local

.PHONY: build-cloud
build-cloud: ## Construir imagen en Cloud Build
	@echo "  Construyendo imagen en Cloud Build..."
	@echo "Imagen destino: $(IMAGE_URL)"
	@echo ""
	gcloud builds submit --tag $(IMAGE_URL)
	@echo ""
	@echo " Imagen construida en la nube: $(IMAGE_URL)"

.PHONY: deploy
deploy: build-cloud ##  Construir y desplegar en Cloud Run (comando principal)
	@echo ""
	@echo " Desplegando en Cloud Run..."
	@echo ""
	@if [ "$(LANGSMITH_API_KEY)" != "tu_clave_langsmith_aqui" ] && [ -n "$(LANGSMITH_API_KEY)" ]; then \
		echo "  Configurando con LangSmith habilitado..."; \
		gcloud run deploy $(GCP_SERVICE_NAME) \
			--image $(IMAGE_URL) \
			--platform managed \
			--region $(GCP_REGION) \
			--allow-unauthenticated \
			--port 8000 \
			--memory $(CLOUD_RUN_MEMORY) \
			--cpu $(CLOUD_RUN_CPU) \
			--timeout $(CLOUD_RUN_TIMEOUT) \
			--max-instances $(CLOUD_RUN_MAX_INSTANCES) \
			--min-instances $(CLOUD_RUN_MIN_INSTANCES) \
			--set-secrets GOOGLE_API_KEY=google-api-key:latest,LANGSMITH_API_KEY=langsmith-api-key:latest,/secrets/apikey.json=google-drive-credentials:latest \
			--set-env-vars LANGCHAIN_TRACING_V2=$(LANGCHAIN_TRACING_V2),LANGSMITH_TRACING=$(LANGSMITH_TRACING),LANGSMITH_ENDPOINT=$(LANGSMITH_ENDPOINT),LANGSMITH_PROJECT=$(LANGSMITH_PROJECT),GOOGLE_DRIVE_CREDENTIALS=/secrets/apikey.json,CLOUD_STORAGE_BUCKET=cloud_faiss_index; \
	else \
		echo "  Configurando sin LangSmith..."; \
		gcloud run deploy $(GCP_SERVICE_NAME) \
			--image $(IMAGE_URL) \
			--platform managed \
			--region $(GCP_REGION) \
			--allow-unauthenticated \
			--port 8000 \
			--memory $(CLOUD_RUN_MEMORY) \
			--cpu $(CLOUD_RUN_CPU) \
			--timeout $(CLOUD_RUN_TIMEOUT) \
			--max-instances $(CLOUD_RUN_MAX_INSTANCES) \
			--min-instances $(CLOUD_RUN_MIN_INSTANCES) \
			--set-secrets GOOGLE_API_KEY=google-api-key:latest,/secrets/apikey.json=google-drive-credentials:latest \
			--set-env-vars GOOGLE_DRIVE_CREDENTIALS=/secrets/apikey.json,CLOUD_STORAGE_BUCKET=cloud_faiss_index; \
	fi
	@echo ""
	@echo " Despliegue completado exitosamente"
	@echo ""
	@make url

.PHONY: deploy-fast
deploy-fast: ## Desplegar sin reconstruir (usa última imagen)
	@echo "⚡ Desplegando última imagen sin reconstruir..."
	@echo ""
	gcloud run deploy $(GCP_SERVICE_NAME) \
		--image $(IMAGE_URL) \
		--platform managed \
		--region $(GCP_REGION)
	@echo ""
	@echo " Despliegue rápido completado"

.PHONY: update-config
update-config: ## Actualizar solo configuración (env vars, memoria, CPU)
	@echo "  Actualizando configuración de Cloud Run..."
	@echo ""
	gcloud run services update $(GCP_SERVICE_NAME) \
		--region $(GCP_REGION) \
		--memory $(CLOUD_RUN_MEMORY) \
		--cpu $(CLOUD_RUN_CPU) \
		--timeout $(CLOUD_RUN_TIMEOUT) \
		--max-instances $(CLOUD_RUN_MAX_INSTANCES) \
		--min-instances $(CLOUD_RUN_MIN_INSTANCES)
	@echo ""
	@echo " Configuración actualizada"

# =====================================================
# MONITOREO Y LOGS
# =====================================================

.PHONY: logs
logs: ## Ver logs en tiempo real
	@echo "📋 Mostrando logs en tiempo real (Ctrl+C para salir)..."
	@echo ""
	gcloud run logs tail $(GCP_SERVICE_NAME) --region $(GCP_REGION)

.PHONY: logs-recent
logs-recent: ## Ver últimos 100 logs
	@echo "📋 Últimos 100 logs:"
	@echo ""
	gcloud run logs read $(GCP_SERVICE_NAME) --region $(GCP_REGION) --limit 100

.PHONY: status
status: ## Ver estado del servicio
	@echo " Estado del servicio $(GCP_SERVICE_NAME):"
	@echo ""
	gcloud run services describe $(GCP_SERVICE_NAME) --region $(GCP_REGION)

# =====================================================
# PRUEBAS Y VALIDACION
# =====================================================

.PHONY: url
url: ## Obtener URL del servicio
	@echo " URL del servicio en Cloud Run:"
	@gcloud run services describe $(GCP_SERVICE_NAME) --region $(GCP_REGION) --format='value(status.url)'

.PHONY: test-health
test-health: ## Probar endpoint de health
	@echo " Probando endpoint /health..."
	@echo ""
	@URL=$$(gcloud run services describe $(GCP_SERVICE_NAME) --region $(GCP_REGION) --format='value(status.url)') && \
		echo "URL: $$URL/health" && \
		curl -s $$URL/health | python -m json.tool

.PHONY: open
open: ## Abrir servicio en el navegador
	@URL=$$(gcloud run services describe $(GCP_SERVICE_NAME) --region $(GCP_REGION) --format='value(status.url)') && \
		if [[ "$$OSTYPE" == "darwin"* ]]; then \
			open $$URL; \
		else \
			xdg-open $$URL 2>/dev/null || echo "Abre manualmente: $$URL"; \
		fi

# =====================================================
# GESTION DE VERSIONES
# =====================================================

.PHONY: revisions
revisions: ## Listar todas las revisiones
	@echo " Revisiones del servicio:"
	@echo ""
	gcloud run revisions list --service=$(GCP_SERVICE_NAME) --region=$(GCP_REGION)

.PHONY: rollback
rollback: ## Revertir al despliegue anterior
	@echo " Revirtiendo al despliegue anterior..."
	@echo ""
	gcloud run services update-traffic $(GCP_SERVICE_NAME) \
		--region $(GCP_REGION) \
		--to-revisions LATEST=0,PREVIOUS=100
	@echo ""
	@echo " Rollback completado"

# =====================================================
# LIMPIEZA
# =====================================================

.PHONY: delete
delete: ## Eliminar servicio de Cloud Run
	@echo "  ¿Estás seguro de eliminar el servicio $(GCP_SERVICE_NAME)?"
	@echo "Presiona Ctrl+C para cancelar o Enter para continuar..."
	@read dummy
	gcloud run services delete $(GCP_SERVICE_NAME) --region $(GCP_REGION) --quiet
	@echo " Servicio eliminado"

.PHONY: clean
clean: ## Limpiar imágenes Docker locales
	@echo " Limpiando imágenes Docker locales..."
	docker rmi $(GCP_SERVICE_NAME):local || echo "No hay imágenes locales para limpiar"
	@echo " Limpieza completada"

# =====================================================
# DESARROLLO LOCAL
# =====================================================

.PHONY: dev
dev: ## Ejecutar servidor en modo desarrollo (con reload)
	@echo " Iniciando servidor en modo desarrollo..."
	uvicorn main:app --reload --host 0.0.0.0 --port 8000 --timeout-keep-alive 300

.PHONY: test
test: ## Ejecutar tests
	@echo " Ejecutando tests..."
	pytest -v

# Por defecto, mostrar ayuda
.DEFAULT_GOAL := help
