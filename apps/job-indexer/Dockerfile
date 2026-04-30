FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY shared/python ./shared/python
COPY apps/job-indexer/requirements.txt ./apps/job-indexer/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r ./apps/job-indexer/requirements.txt

COPY apps/job-indexer ./apps/job-indexer

ENV PYTHONPATH="/app/apps/job-indexer:/app/shared/python"

CMD ["python", "apps/job-indexer/app/build_index.py"]