# Frontend Streamlit - MIAD RAG Real Estate

Prototipo inicial de frontend para `apps/frontend`.

## Estructura

```text
apps/frontend/
├── Dockerfile
├── requirements.txt
└── app/
    ├── main.py
    ├── .streamlit/config.toml
    ├── services/backend_client.py
    ├── components/search_panel.py
    ├── components/property_cards.py
    ├── components/map_view.py
    ├── components/debug_panel.py
    └── utils/formatting.py
```

## Ejecución local

La aplicación Streamlit puede ejecutarse localmente desde un devcontainer y consumir el backend desplegado en Cloud Run.

Cuando el backend de Cloud Run es privado, se requiere un `identity token` válido. Para pruebas locales, se recomienda generar el token mediante impersonación de una service account con permiso `roles/run.invoker` sobre el backend.

---

## 1. Configuración inicial en Cloud Shell CLI de GCP

Estos pasos se ejecutan una sola vez desde **Cloud Shell CLI** o desde una terminal con permisos suficientes sobre el proyecto de GCP.

```bash
export PROJECT_ID="miad-paad-rs-dev"
export REGION="us-east4"
export BACKEND_SERVICE="miad-rag-backend"
export CALLER_SA="streamlit-local-dev@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "${PROJECT_ID}"
```

Habilitar la API requerida para generar credenciales por impersonación:

```bash
gcloud services enable iamcredentials.googleapis.com \
  --project="${PROJECT_ID}"
```

Crear la service account si todavía no existe:

```bash
gcloud iam service-accounts describe "${CALLER_SA}" \
  --project="${PROJECT_ID}" \
  || gcloud iam service-accounts create streamlit-local-dev \
      --project="${PROJECT_ID}" \
      --display-name="Streamlit local dev Cloud Run invoker"
```

> Nota: si el comando muestra inicialmente `NOT_FOUND` y luego `Created service account`, es normal. Significa que la service account no existía y fue creada correctamente.

Dar permiso a la service account para invocar el backend privado de Cloud Run:

```bash
gcloud run services add-iam-policy-binding "${BACKEND_SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --member="serviceAccount:${CALLER_SA}" \
  --role="roles/run.invoker"
```

Dar permiso al usuario activo para impersonar la service account:

```bash
export USER_EMAIL="$(gcloud config get-value account)"

gcloud iam service-accounts add-iam-policy-binding "${CALLER_SA}" \
  --project="${PROJECT_ID}" \
  --member="user:${USER_EMAIL}" \
  --role="roles/iam.serviceAccountTokenCreator"
```

Obtener la URL real del backend:

```bash
gcloud run services describe "${BACKEND_SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format='value(status.url)'
```

Ejemplo de salida:

```text
https://miad-rag-backend-cpaoxwrjxq-uk.a.run.app
```

Usar siempre la URL que retorna `gcloud run services describe`. No se debe asumir que la URL tendrá el número del proyecto en el dominio.

---

## 2. Ejecución desde el devcontainer

Estos pasos se ejecutan dentro del **devcontainer**, ubicado en la raíz del repositorio.

```bash
cd apps/frontend
```

Crear y activar el entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar dependencias:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary=:all: pyarrow==14.0.2
python -m pip install -r requirements.txt
```

Configurar variables de entorno:

```bash
export PROJECT_ID="miad-paad-rs-dev"
export REGION="us-east4"
export BACKEND_SERVICE="miad-rag-backend"
export CALLER_SA="streamlit-local-dev@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "${PROJECT_ID}"
```

Obtener dinámicamente la URL real del backend:

```bash
export BACKEND_URL="$(gcloud run services describe ${BACKEND_SERVICE} \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --format='value(status.url)')"

echo "${BACKEND_URL}"
```

Configurar el modo de autenticación:

```bash
export BACKEND_AUTH_MODE="auto"
```

Generar el token para consumir el backend privado:

```bash
export BACKEND_AUTH_TOKEN="$(gcloud auth print-identity-token \
  --impersonate-service-account="${CALLER_SA}" \
  --audiences="${BACKEND_URL}")"
```

> Importante: no usar `gcloud auth print-identity-token --audiences=${BACKEND_URL}` directamente con un usuario humano, porque puede fallar con el error:
>
> ```text
> Invalid account type for `--audiences`. Requires valid service account.
> ```
>
> Para este flujo local se usa impersonación de service account.

Probar conectividad contra el backend:

```bash
curl -i "${BACKEND_URL}/health" \
  -H "Authorization: Bearer ${BACKEND_AUTH_TOKEN}"
```

Si el backend usa otro endpoint de salud, ajustar la ruta según corresponda, por ejemplo:

```bash
curl -i "${BACKEND_URL}/api/v1/health" \
  -H "Authorization: Bearer ${BACKEND_AUTH_TOKEN}"
```

Lanzar la aplicación Streamlit:

```bash
streamlit run app/main.py --server.address=0.0.0.0 --server.port=8080
```

---

## Resumen del flujo

```text
Cloud Shell CLI
    ├── Crear service account
    ├── Dar roles/run.invoker sobre el backend Cloud Run
    └── Dar permiso al usuario para impersonar la service account

Devcontainer
    ├── Crear entorno virtual
    ├── Instalar dependencias
    ├── Obtener BACKEND_URL
    ├── Generar BACKEND_AUTH_TOKEN impersonando la service account
    └── Ejecutar Streamlit en el puerto 8080
```

---

## Variables principales

| Variable             | Descripción                                           |
| -------------------- | ----------------------------------------------------- |
| `PROJECT_ID`         | ID del proyecto GCP donde está desplegado el backend  |
| `REGION`             | Región de Cloud Run                                   |
| `BACKEND_SERVICE`    | Nombre del servicio backend en Cloud Run              |
| `CALLER_SA`          | Service account usada para invocar el backend privado |
| `BACKEND_URL`        | URL real del backend obtenida desde Cloud Run         |
| `BACKEND_AUTH_MODE`  | Modo de autenticación usado por el frontend           |
| `BACKEND_AUTH_TOKEN` | Identity token usado para consumir el backend privado |

---

## Comando rápido para sesiones posteriores

Una vez configurados los permisos iniciales en Cloud Shell, en futuras sesiones dentro del devcontainer basta con ejecutar:

```bash
cd apps/frontend
source .venv/bin/activate

export PROJECT_ID="miad-paad-rs-dev"
export REGION="us-east4"
export BACKEND_SERVICE="miad-rag-backend"
export CALLER_SA="streamlit-local-dev@${PROJECT_ID}.iam.gserviceaccount.com"

export BACKEND_URL="$(gcloud run services describe ${BACKEND_SERVICE} \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --format='value(status.url)')"

export BACKEND_AUTH_MODE="auto"

export BACKEND_AUTH_TOKEN="$(gcloud auth print-identity-token \
  --impersonate-service-account="${CALLER_SA}" \
  --audiences="${BACKEND_URL}")"

streamlit run app/main.py --server.address=0.0.0.0 --server.port=8080
```

---

## Renovación del token de autenticación

Cuando el backend de Cloud Run es privado, el frontend local debe enviar un `identity token` válido en cada petición. Estos tokens son de corta duración y pueden expirar durante una sesión de desarrollo.

Si el frontend muestra un error similar a:

```text
Backend respondió HTTP 401:
Your client does not have permission to the requested URL /api/v1/recommend.
```

renovar el token dentro del devcontainer:

```bash
cd apps/frontend
source .venv/bin/activate

export PROJECT_ID="miad-paad-rs-dev"
export REGION="us-east4"
export BACKEND_SERVICE="miad-rag-backend"
export CALLER_SA="streamlit-local-dev@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "${PROJECT_ID}"

export BACKEND_URL="$(gcloud run services describe "${BACKEND_SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format='value(status.url)')"

export BACKEND_AUTH_MODE="auto"

export BACKEND_AUTH_TOKEN="$(gcloud auth print-identity-token \
  --impersonate-service-account="${CALLER_SA}" \
  --audiences="${BACKEND_URL}")"

echo "${BACKEND_URL}"
echo "Token actualizado. Longitud: ${#BACKEND_AUTH_TOKEN}"
```

Probar conectividad contra el backend:

```bash
curl -i "${BACKEND_URL}/health" \
  -H "Authorization: Bearer ${BACKEND_AUTH_TOKEN}"
```

Si el backend no tiene `/health`, usar el endpoint de salud disponible:

```bash
curl -i "${BACKEND_URL}/api/v1/health" \
  -H "Authorization: Bearer ${BACKEND_AUTH_TOKEN}"
```

Luego reiniciar Streamlit para que tome las variables de entorno actualizadas:

```bash
streamlit run app/main.py --server.address=0.0.0.0 --server.port=8080
```

> Nota: si Streamlit ya estaba ejecutándose, detenerlo con `Ctrl + C` y lanzarlo nuevamente. Las variables de entorno exportadas después del inicio del proceso no siempre se reflejan en la aplicación ya levantada.


---

## Checklist si sigue el 401

Ejecuta esto:

```bash
echo "BACKEND_URL=${BACKEND_URL}"
echo "CALLER_SA=${CALLER_SA}"
echo "TOKEN_LENGTH=${#BACKEND_AUTH_TOKEN}"
````

Luego valida que la service account tenga permiso sobre el backend:

```bash
gcloud run services get-iam-policy "${BACKEND_SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format="table(bindings.role, bindings.members)"
```

Debe aparecer algo como:

```text
roles/run.invoker    serviceAccount:streamlit-local-dev@miad-paad-rs-dev.iam.gserviceaccount.com
```

Y recuerda: `BACKEND_URL` debe ser exactamente la URL que retorna:

```bash
gcloud run services describe "${BACKEND_SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format='value(status.url)'
```

