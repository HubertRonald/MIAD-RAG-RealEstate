from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config.runtime import get_settings
from miad_rag_common.logging.structured_logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class EmbeddingService:
    """
    Servicio de embeddings para el backend FastAPI.

    Responsabilidades:
      - Cargar un índice FAISS ya construido.
      - Exponer el modelo de embeddings al RetrievalService.
      - Estimar costo de embeddings online por consulta, si se quiere loggear.

    No construye índices.
    No genera embeddings masivos.
    No guarda FAISS.
    No sube ni descarga desde Cloud Storage.
    """

    def __init__(self, model_name: str = "models/gemini-embedding-001") -> None:
        self.model_name = model_name
        self.price_per_m_tokens = getattr(
            settings,
            "EMBEDDING_PRICE_PER_M_TOKENS",
            0.025,
        )
        self.chars_per_token = getattr(
            settings,
            "CHARS_PER_TOKEN",
            3.2,
        )

        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model=model_name,
        )

        self.vectorstore: Optional[FAISS] = None

    # =========================================================================
    # CARGA DE FAISS
    # =========================================================================

    def load_vectorstore(self, persist_path: str | Path) -> FAISS:
        """
        Carga un índice FAISS previamente descargado a disco.

        En Cloud Run, el flujo esperado es:
          GCSIndexService.ensure_local_index()
          → /tmp/faiss_index/{collection}/latest
          → EmbeddingService.load_vectorstore(...)
        """
        index_path = Path(persist_path)

        if not index_path.exists():
            raise FileNotFoundError(
                f"El índice no existe en la ruta: {index_path}"
            )

        index_file = index_path / "index.faiss"
        pkl_file = index_path / "index.pkl"

        if not index_file.exists() or not pkl_file.exists():
            raise FileNotFoundError(
                "Índice FAISS incompleto. "
                f"Se esperaban index.faiss e index.pkl en {index_path}"
            )

        logger.info(
            "loading_faiss_vectorstore",
            extra={
                "index_path": str(index_path),
                "model": self.model_name,
            },
        )

        self.vectorstore = FAISS.load_local(
            str(index_path),
            self.embeddings_model,
            allow_dangerous_deserialization=True,
        )

        logger.info(
            "faiss_vectorstore_loaded",
            extra={
                "index_path": str(index_path),
                "model": self.model_name,
            },
        )

        return self.vectorstore

    # =========================================================================
    # COSTO ONLINE ESTIMADO
    # =========================================================================

    def estimate_texts_cost(self, texts: list[str]) -> dict[str, Any]:
        """
        Estima costo de embeddings para consultas online.

        Importante:
          RetrievalService invoca internamente el embedding de la query
          cuando llama a FAISS similarity_search. Por eso esta función es
          solo una estimación auxiliar para logging, no medición exacta.
        """
        total_chars = sum(len(text or "") for text in texts)
        estimated_tokens = int(total_chars / self.chars_per_token)

        embedding_cost_usd = (
            estimated_tokens / 1_000_000
        ) * self.price_per_m_tokens

        return {
            "total_texts": len(texts),
            "total_chars": total_chars,
            "estimated_tokens": estimated_tokens,
            "embedding_cost_usd": round(embedding_cost_usd, 8),
            "model": self.model_name,
            "price_per_m_tokens": self.price_per_m_tokens,
            "chars_per_token": self.chars_per_token,
        }

    def estimate_query_cost(self, query: str) -> dict[str, Any]:
        """
        Estima costo de embedding para una única query del usuario.
        """
        return self.estimate_texts_cost([query])

    # =========================================================================
    # GETTERS
    # =========================================================================

    def get_vectorstore(self) -> Optional[FAISS]:
        return self.vectorstore

    def get_embeddings_model(self) -> GoogleGenerativeAIEmbeddings:
        return self.embeddings_model
