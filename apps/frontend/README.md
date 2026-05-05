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

```bash
cd apps/frontend
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary=:all: pyarrow==14.0.2
python -m pip install -r requirements.txt

export BACKEND_URL="https://miad-rag-backend-xxxxxxxxxxxx.us-east4.run.app"
export BACKEND_AUTH_MODE="auto"
# Opcional si se prueba contra backend privado desde el cloud shell cli de Google Cloud Platform:
# export BACKEND_AUTH_TOKEN="$(gcloud auth print-identity-token --audiences=${BACKEND_URL})"

streamlit run app/main.py --server.port=8080
```