from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    """
    Configuración runtime del backend FastAPI.

    Responsabilidades:
      - Servir API FastAPI en Cloud Run.
      - Cargar índice FAISS desde GCS hacia /tmp.
      - Atender /ask y /recommend.
      - Consultar BigQuery para enriquecer listings.
      - Usar Gemini para generación de respuesta.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # App
    APP_NAME: str = "miad-rag-backend"
    APP_VERSION: str = "0.1.0"
    ENV: str = "dev"
    PORT: int = 8080

    LOG_LEVEL: str = "INFO"
    JSON_LOGS: bool = True

    # CORS
    CORS_ALLOW_ORIGINS: str = "*"

    # GCP
    PROJECT_ID: str = "miad-paad-rs-dev"
    GCP_LOCATION: str = "us-east4"

    # GCS - FAISS index
    INDEX_BUCKET: str = "miad-paad-rs-index-dev"
    INDEX_PREFIX: str = "faiss"
    INDEX_LOCAL_ROOT: str = "/tmp/faiss_index"
    DEFAULT_COLLECTION: str = "realstate_mvd"

    # BigQuery
    BQ_PROJECT_ID: str = "miad-paad-rs-dev"
    BQ_DATASET_ID: str = "ds_miad_rag_rs"
    BQ_LISTINGS_TABLE: str = "real_estate_listings"
    BQ_LISTING_ID_COLUMN: str = "id"
    BQ_LOCATION: Optional[str] = "us-east4"

    # Retrieval
    RETRIEVAL_K: int = 10
    RETRIEVAL_FETCH_K: int = 200

    # Gemini / LangChain
    GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    GEMINI_GENERATION_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.2
    GEMINI_MAX_OUTPUT_TOKENS: int = 5000

    # Cost estimation for online query embeddings.
    # Operational estimate only. Verify current pricing in official Gemini docs.
    EMBEDDING_REQUEST_TIMEOUT_SECONDS: int = 120
    EMBEDDING_PRICE_PER_M_TOKENS: float = 0.025
    CHARS_PER_TOKEN: float = 3.2

    # Reranking
    ENABLE_RERANKING_MODEL: bool = False
    RERANKING_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKING_TOP_K: int = 3

    # Scores para frontend
    SCORE_LOW: float = 0.78
    SCORE_HIGH: float = 0.92

    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    @property
    def google_genai_api_key(self) -> Optional[str]:
        return self.GOOGLE_API_KEY or self.GEMINI_API_KEY

    @property
    def cors_origins_list(self) -> list[str]:
        if self.CORS_ALLOW_ORIGINS.strip() == "*":
            return ["*"]

        return [
            origin.strip()
            for origin in self.CORS_ALLOW_ORIGINS.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> RuntimeSettings:
    return RuntimeSettings()
