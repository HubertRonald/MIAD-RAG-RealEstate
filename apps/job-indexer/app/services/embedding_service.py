from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from google.api_core import exceptions as google_exceptions
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config.runtime import get_settings
from miad_rag_common.logging.structured_logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class EmbeddingService:
    """
    Servicio de embeddings para el Cloud Run Job.

    Responsabilidades:
      - Generar embeddings masivos para listings.
      - Aplicar batch processing.
      - Aplicar retry/backoff ante rate limits y timeouts.
      - Guardar checkpoint para retomar si el job falla.
      - Construir FAISS desde embeddings pregenerados.
      - Calcular estadísticas estimadas de costo.

    No sube nada a Cloud Storage.
    La publicación del índice vive en GCSIndexPublisher.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self.max_batch_size = settings.EMBEDDING_MAX_BATCH_SIZE
        self.request_delay = settings.EMBEDDING_REQUEST_DELAY_SECONDS

        self.price_per_m_tokens = settings.EMBEDDING_PRICE_PER_M_TOKENS
        self.chars_per_token = settings.CHARS_PER_TOKEN

        self.request_timeout_seconds = getattr(
            settings,
            "EMBEDDING_REQUEST_TIMEOUT_SECONDS",
            120,
        )

        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=settings.google_genai_api_key,
            request_options={"timeout": self.request_timeout_seconds},
        )

        self.vectorstore: Optional[FAISS] = None
        self._cost_stats: dict[str, Any] = {}

    # =========================================================================
    # EMBEDDING CON RETRY
    # =========================================================================

    def _embed_with_retry(
        self,
        batch: list[str],
        max_retries: int = 5,
    ) -> list[list[float]]:
        """
        Genera embeddings para un batch con retry.

        Maneja:
          - ResourceExhausted: rate limit / cuota.
          - DeadlineExceeded: timeout del servicio.

        Args:
            batch:
                Lista de textos.
            max_retries:
                Número máximo de reintentos.

        Returns:
            Lista de vectores de embeddings.
        """
        rate_limit_delay = 30

        for attempt in range(max_retries):
            try:
                return self.embeddings_model.embed_documents(batch)

            except google_exceptions.ResourceExhausted:
                if attempt == max_retries - 1:
                    raise

                logger.warning(
                    "embedding_rate_limit_retry",
                    extra={
                        "delay_seconds": rate_limit_delay,
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "batch_size": len(batch),
                    },
                )

                time.sleep(rate_limit_delay)
                rate_limit_delay *= 2

            except google_exceptions.DeadlineExceeded:
                if attempt == max_retries - 1:
                    raise

                wait_seconds = 30 * (attempt + 1)

                logger.warning(
                    "embedding_deadline_exceeded_retry",
                    extra={
                        "delay_seconds": wait_seconds,
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "batch_size": len(batch),
                    },
                )

                time.sleep(wait_seconds)

        return []

    # =========================================================================
    # CHECKPOINT / RESUME
    # =========================================================================

    def _default_checkpoint_dir(self) -> Path:
        return (
            Path(settings.LOCAL_WORKDIR)
            / "_embedding_checkpoints"
            / settings.COLLECTION
        )

    def _checkpoint_path(self, persist_path: str | Path | None = None) -> Path:
        base_path = (
            Path(persist_path)
            if persist_path
            else self._default_checkpoint_dir()
        )
        return base_path / "_embedding_checkpoint.json"

    def _load_checkpoint(
        self,
        persist_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Carga checkpoint si existe.

        Returns:
            {
              "completed_batches": int,
              "embeddings": list[list[float]]
            }
        """
        checkpoint_path = self._checkpoint_path(persist_path)

        if not checkpoint_path.exists():
            return {
                "completed_batches": 0,
                "embeddings": [],
            }

        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))

            completed_batches = int(data.get("completed_batches", 0))
            embeddings = data.get("embeddings", [])

            logger.info(
                "embedding_checkpoint_loaded",
                extra={
                    "checkpoint_path": str(checkpoint_path),
                    "completed_batches": completed_batches,
                    "embeddings_loaded": len(embeddings),
                },
            )

            return {
                "completed_batches": completed_batches,
                "embeddings": embeddings,
            }

        except Exception as exc:
            logger.warning(
                "embedding_checkpoint_read_failed_starting_from_zero",
                extra={
                    "checkpoint_path": str(checkpoint_path),
                    "error": str(exc),
                },
            )

            return {
                "completed_batches": 0,
                "embeddings": [],
            }

    def _save_checkpoint(
        self,
        persist_path: str | Path | None,
        completed_batches: int,
        embeddings: list[list[float]],
    ) -> None:
        checkpoint_path = self._checkpoint_path(persist_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_path.write_text(
            json.dumps(
                {
                    "completed_batches": completed_batches,
                    "embeddings": embeddings,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        logger.info(
            "embedding_checkpoint_saved",
            extra={
                "checkpoint_path": str(checkpoint_path),
                "completed_batches": completed_batches,
                "embeddings_count": len(embeddings),
            },
        )

    def _clear_checkpoint(
        self,
        persist_path: str | Path | None = None,
    ) -> None:
        checkpoint_path = self._checkpoint_path(persist_path)

        if checkpoint_path.exists():
            checkpoint_path.unlink()

            logger.info(
                "embedding_checkpoint_removed",
                extra={"checkpoint_path": str(checkpoint_path)},
            )

    # =========================================================================
    # VALIDACIÓN / COSTOS
    # =========================================================================

    @staticmethod
    def _validate_texts(texts: list[str]) -> None:
        if not texts:
            raise ValueError("La lista de textos está vacía.")

        for index, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(
                    f"Texto en posición {index} está vacío o es inválido."
                )

    def estimate_texts_cost(self, texts: list[str]) -> dict[str, Any]:
        """
        Estima tokens y costo USD a partir de caracteres.

        Es una estimación operacional, no facturación exacta.
        """
        total_chars = sum(len(text or "") for text in texts)
        estimated_tokens = int(total_chars / self.chars_per_token)

        embedding_cost_usd = (
            estimated_tokens / 1_000_000
        ) * self.price_per_m_tokens

        return {
            "total_texts": len(texts),
            "total_documents": len(texts),
            "total_chunks": len(texts),  # compatibilidad con servicio local
            "total_chars": total_chars,
            "estimated_tokens": estimated_tokens,
            "embedding_cost_usd": round(embedding_cost_usd, 6),
            "model": self.model_name,
            "price_per_m_tokens": self.price_per_m_tokens,
            "chars_per_token": self.chars_per_token,
        }

    # =========================================================================
    # GENERACIÓN DE EMBEDDINGS
    # =========================================================================

    def create_embeddings(
        self,
        texts: list[str],
        persist_path: str | Path | None = None,
    ) -> list[list[float]]:
        """
        Genera embeddings para una lista de textos.

        Si persist_path se proporciona, guarda checkpoint después de cada batch.
        Si el job falla, al relanzar retoma desde el último batch completado.
        """
        self._validate_texts(texts)

        if len(texts) <= self.max_batch_size:
            logger.info(
                "embedding_single_batch_started",
                extra={
                    "texts_count": len(texts),
                    "model": self.model_name,
                },
            )

            embeddings = self._embed_with_retry(texts)

            logger.info(
                "embedding_single_batch_completed",
                extra={
                    "texts_count": len(texts),
                    "embeddings_count": len(embeddings),
                },
            )

            return embeddings

        checkpoint = self._load_checkpoint(persist_path)
        start_batch = checkpoint["completed_batches"]
        all_embeddings: list[list[float]] = checkpoint["embeddings"]

        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        remaining_batches = total_batches - start_batch

        logger.info(
            "embedding_batch_generation_started",
            extra={
                "texts_count": len(texts),
                "batch_size": self.batch_size,
                "total_batches": total_batches,
                "start_batch": start_batch + 1,
                "remaining_batches": remaining_batches,
                "request_delay_seconds": self.request_delay,
                "model": self.model_name,
            },
        )

        for batch_index in range(start_batch, total_batches):
            batch_number = batch_index + 1
            start_index = batch_index * self.batch_size
            batch = texts[start_index:start_index + self.batch_size]

            batch_started_at = time.time()

            logger.info(
                "embedding_batch_started",
                extra={
                    "batch_number": batch_number,
                    "total_batches": total_batches,
                    "batch_size": len(batch),
                },
            )

            batch_embeddings = self._embed_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

            batch_elapsed = round(time.time() - batch_started_at, 3)

            logger.info(
                "embedding_batch_completed",
                extra={
                    "batch_number": batch_number,
                    "total_batches": total_batches,
                    "batch_size": len(batch),
                    "elapsed_sec": batch_elapsed,
                    "embeddings_total": len(all_embeddings),
                },
            )

            self._save_checkpoint(
                persist_path=persist_path,
                completed_batches=batch_number,
                embeddings=all_embeddings,
            )

            if batch_number < total_batches:
                logger.info(
                    "embedding_batch_delay",
                    extra={
                        "delay_seconds": self.request_delay,
                        "next_batch": batch_number + 1,
                        "total_batches": total_batches,
                    },
                )
                time.sleep(self.request_delay)

        logger.info(
            "embedding_generation_completed",
            extra={
                "texts_count": len(texts),
                "embeddings_count": len(all_embeddings),
                "total_batches": total_batches,
            },
        )

        return all_embeddings

    # =========================================================================
    # CONSTRUCCIÓN DE FAISS
    # =========================================================================

    def build_vectorstore(
        self,
        documents: list[Document],
        persist_path: str | Path | None = None,
    ) -> FAISS:
        """
        Construye un índice FAISS desde Documents.

        A diferencia del backend:
          - aquí sí se generan embeddings;
          - aquí sí se calculan estadísticas de costo;
          - aquí se usa checkpoint/resume.

        Nota:
          Este método NO guarda el índice en GCS.
          El guardado local final puede hacerlo FAISSBuilder.save_vectorstore().
        """
        if not documents:
            raise ValueError("La lista de documents está vacía.")

        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]

        cost_estimate = self.estimate_texts_cost(texts)

        logger.info(
            "faiss_embedding_build_started",
            extra={
                "documents_count": len(documents),
                "estimated_tokens": cost_estimate["estimated_tokens"],
                "estimated_embedding_cost_usd": cost_estimate["embedding_cost_usd"],
                "model": self.model_name,
            },
        )

        embeddings = self.create_embeddings(
            texts=texts,
            persist_path=persist_path,
        )

        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Cantidad de embeddings no coincide con cantidad de textos. "
                f"texts={len(texts)}, embeddings={len(embeddings)}"
            )

        text_embedding_pairs = list(zip(texts, embeddings))

        vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=self.embeddings_model,
            metadatas=metadatas,
        )

        self.vectorstore = vectorstore
        self._cost_stats = cost_estimate

        self._clear_checkpoint(persist_path)

        logger.info(
            "faiss_embedding_build_completed",
            extra=self._cost_stats,
        )

        return vectorstore

    # =========================================================================
    # GETTERS
    # =========================================================================

    def get_embeddings_model(self) -> GoogleGenerativeAIEmbeddings:
        return self.embeddings_model

    def get_vectorstore(self) -> Optional[FAISS]:
        return self.vectorstore

    def get_cost_stats(self) -> dict[str, Any]:
        return self._cost_stats

    @property
    def cost_stats(self) -> dict[str, Any]:
        return self._cost_stats
