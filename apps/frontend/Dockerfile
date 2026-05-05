FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY apps/frontend/requirements.txt ./apps/frontend/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r ./apps/frontend/requirements.txt

COPY apps/frontend ./apps/frontend

# Copiar config de Streamlit al working directory efectivo
RUN mkdir -p ./.streamlit \
    && cp ./apps/frontend/app/.streamlit/config.toml ./.streamlit/config.toml

EXPOSE 8080

CMD ["sh", "-c", "streamlit run apps/frontend/app/main.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]