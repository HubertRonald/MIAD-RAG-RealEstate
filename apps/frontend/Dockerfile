FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY apps/frontend/requirements.txt ./apps/frontend/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r ./apps/frontend/requirements.txt

COPY apps/frontend ./apps/frontend

EXPOSE 8080

CMD ["sh", "-c", "streamlit run app/main.py --server.address=0.0.0.0 --server.port=${PORT:-8080}"]
