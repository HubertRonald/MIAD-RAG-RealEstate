from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class IndexerSettings(BaseSettings):
    """
    Configuración runtime del Cloud Run Job encargado de construir FAISS.

    Responsabilidades:
      - Leer listings desde BigQuery.
      - Convertir filas en Documents.
      - Generar embeddings masivos.
      - Construir índice FAISS.
      - Publicar index.faiss, index.pkl, manifest.json y listing_ids.json en GCS.
      - Opcionalmente registrar métricas en MLflow.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # App
    APP_NAME: str = "miad-rag-indexer-job"
    APP_VERSION: str = "0.1.0"
    ENV: str = "dev"

    LOG_LEVEL: str = "INFO"
    JSON_LOGS: bool = True

    # GCP
    PROJECT_ID: str = "miad-paad-rs-dev"
    GCP_LOCATION: str = "us-east4"

    # BigQuery source
    BQ_PROJECT_ID: str = "miad-paad-rs-dev"
    BQ_DATASET_ID: str = "ds_miad_rag_rs"
    BQ_LISTINGS_TABLE: str = "real_estate_listings"
    BQ_LOCATION: Optional[str] = "us-east4"

    # Optional query controls
    BQ_LIMIT: Optional[int] = None
    BQ_WHERE_CLAUSE: Optional[str] = None

    # FAISS / collection
    COLLECTION: str = "realstate_mvd"
    LOCAL_WORKDIR: str = "/tmp/miad-rag-indexer"
    LOCAL_INDEX_DIRNAME: str = "faiss_index"

    # GCS output
    INDEX_BUCKET: str = "miad-paad-rs-index-dev"
    INDEX_PREFIX: str = "faiss"

    # Gemini embeddings
    GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"

    # Bulk embedding controls.
    # These are only needed by job-indexer, not by backend.
    EMBEDDING_BATCH_SIZE: int = 50
    EMBEDDING_MAX_BATCH_SIZE: int = 100
    EMBEDDING_REQUEST_DELAY_SECONDS: int = 15

    # Cost estimation for bulk embeddings.
    # Operational estimate only. Verify current pricing in official Gemini docs.
    EMBEDDING_PRICE_PER_M_TOKENS: float = 0.025
    CHARS_PER_TOKEN: float = 3.2

    # MLflow
    # For now, keep disabled in Cloud Run Job.
    # Formal RAGAS/MLflow experiments can continue running locally.
    ENABLE_MLFLOW: bool = False
    MLFLOW_TRACKING_URI: Optional[str] = None
    MLFLOW_EXPERIMENT_NAME: str = "miad-rag-realestate-indexer"

    # Safety / execution
    DRY_RUN: bool = False
    UPLOAD_LATEST: bool = True
    UPLOAD_VERSIONED: bool = True

    @property
    def source_table_fqn(self) -> str:
        return f"{self.BQ_PROJECT_ID}.{self.BQ_DATASET_ID}.{self.BQ_LISTINGS_TABLE}"

    @property
    def gcs_latest_prefix(self) -> str:
        return f"{self.INDEX_PREFIX.strip('/')}/{self.COLLECTION}/latest"

    def gcs_version_prefix(self, version: str) -> str:
        return f"{self.INDEX_PREFIX.strip('/')}/{self.COLLECTION}/versions/{version}"


@lru_cache
def get_settings() -> IndexerSettings:
    return IndexerSettings()
