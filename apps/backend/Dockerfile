FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY shared/python ./shared/python
COPY apps/backend/requirements.txt ./apps/backend/requirements.txt

RUN pip install --upgrade pip && pip install --no-cache-dir -r ./apps/backend/requirements.txt
COPY apps/backend ./apps/backend

ENV PYTHONPATH="/app/apps/backend:/app/shared/python"

WORKDIR /app/apps/backend
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
