from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from app.config.runtime import get_settings
from app.services.embedding_service import EmbeddingService
from miad_rag_common.logging.structured_logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class FAISSBuilder:
    """
    Construye y guarda el índice FAISS localmente.

    Responsabilidad de esta clase:
      - Orquestar la construcción local del vectorstore FAISS.
      - Guardar index.faiss e index.pkl en disco local.
      - Escribir manifest.json y listing_ids.json.

    Importante:
      - NO sube nada a Cloud Storage.
      - La publicación en GCS vive en GCSIndexPublisher.
      - La generación de embeddings y cálculo de costos vive en EmbeddingService.

    Output local esperado:
      index.faiss
      index.pkl
      manifest.json
      listing_ids.json
    """

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    def build_vectorstore(self, documents: list[Document]) -> FAISS:
        """
        Construye un vectorstore FAISS desde Documents.

        Este método delega la generación de embeddings al EmbeddingService,
        para que se activen:
          - batch processing,
          - retry/backoff,
          - checkpoint/resume,
          - cálculo de estadísticas de costo.

        No usa FAISS.from_documents directamente, porque eso generaría
        embeddings por dentro de LangChain y no poblaría cost_stats.
        """
        if not documents:
            raise ValueError("No hay documentos para construir FAISS.")

        checkpoint_dir = self._checkpoint_dir(
            collection=settings.COLLECTION,
        )

        logger.info(
            "building_faiss_vectorstore",
            extra={
                "documents_count": len(documents),
                "collection": settings.COLLECTION,
                "checkpoint_dir": str(checkpoint_dir),
            },
        )

        vectorstore = self.embedding_service.build_vectorstore(
            documents=documents,
            persist_path=checkpoint_dir,
        )

        logger.info(
            "faiss_vectorstore_built",
            extra={
                "documents_count": len(documents),
                "collection": settings.COLLECTION,
                "embedding_statistics": self._get_embedding_stats(),
            },
        )

        return vectorstore

    def save_vectorstore(
        self,
        vectorstore: FAISS,
        collection: str,
        version: str,
    ) -> Path:
        """
        Guarda el índice FAISS en disco local.

        Args:
            vectorstore:
                Vectorstore FAISS ya construido.
            collection:
                Nombre lógico de la colección, por ejemplo realstate_mvd.
            version:
                Versión temporal del índice, por ejemplo 20260501T030000Z.

        Returns:
            Path del directorio local donde quedó guardado el índice.
        """
        output_dir = self._local_index_dir(
            collection=collection,
            version=version,
        )

        if output_dir.exists():
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        vectorstore.save_local(str(output_dir))

        index_file = output_dir / "index.faiss"
        pkl_file = output_dir / "index.pkl"

        if not index_file.exists() or not pkl_file.exists():
            raise RuntimeError(
                f"FAISS no generó los archivos esperados en {output_dir}. "
                "Se esperaban index.faiss e index.pkl."
            )

        logger.info(
            "faiss_vectorstore_saved",
            extra={
                "collection": collection,
                "version": version,
                "output_dir": str(output_dir),
                "index_file": str(index_file),
                "pkl_file": str(pkl_file),
            },
        )

        return output_dir

    def write_auxiliary_files(
        self,
        output_dir: Path,
        documents: list[Document],
        version: str,
        started_at: str,
        finished_at: str,
    ) -> dict[str, Any]:
        """
        Escribe archivos auxiliares del índice:

          - listing_ids.json
          - manifest.json

        El manifest incluye estadísticas de costo si EmbeddingService las generó.
        """
        if not output_dir.exists():
            raise FileNotFoundError(
                f"No existe el directorio local del índice: {output_dir}"
            )

        listing_ids = self._extract_listing_ids(documents)

        listing_ids_path = output_dir / "listing_ids.json"
        listing_ids_path.write_text(
            json.dumps(
                listing_ids,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        embedding_statistics = self._get_embedding_stats()

        manifest = {
            "collection": settings.COLLECTION,
            "version": version,
            "created_at": finished_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "source_table": settings.source_table_fqn,
            "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
            "index_type": "faiss",
            "documents_count": len(documents),
            "listing_ids_count": len(listing_ids),
            "schema_version": "v1",
            "environment": settings.ENV,
            "embedding_statistics": embedding_statistics,
            "files": {
                "index": "index.faiss",
                "metadata": "index.pkl",
                "manifest": "manifest.json",
                "listing_ids": "listing_ids.json",
            },
        }

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info(
            "faiss_auxiliary_files_written",
            extra={
                "collection": settings.COLLECTION,
                "version": version,
                "output_dir": str(output_dir),
                "manifest_path": str(manifest_path),
                "listing_ids_path": str(listing_ids_path),
                "documents_count": len(documents),
                "listing_ids_count": len(listing_ids),
                "embedding_statistics": embedding_statistics,
            },
        )

        return manifest

    def _local_index_dir(
        self,
        collection: str,
        version: str,
    ) -> Path:
        """
        Ruta local final del índice FAISS.

        Ejemplo:
          /tmp/miad-rag-indexer/faiss_index/realstate_mvd/20260501T030000Z
        """
        return (
            Path(settings.LOCAL_WORKDIR)
            / settings.LOCAL_INDEX_DIRNAME
            / collection
            / version
        )

    def _checkpoint_dir(
        self,
        collection: str,
    ) -> Path:
        """
        Ruta local para checkpoint de embeddings.

        Es separada del output final para que:
          - no se mezcle _embedding_checkpoint.json con index.faiss/index.pkl;
          - el output final quede limpio para subir a GCS;
          - si el job falla, pueda retomarse durante la misma ejecución/localidad.
        """
        return (
            Path(settings.LOCAL_WORKDIR)
            / "_embedding_checkpoints"
            / collection
        )

    def _get_embedding_stats(self) -> dict[str, Any]:
        """
        Lee estadísticas de costo desde EmbeddingService.

        Compatible con dos estilos:
          - get_cost_stats()
          - propiedad cost_stats
        """
        if hasattr(self.embedding_service, "get_cost_stats"):
            stats = self.embedding_service.get_cost_stats()
            if isinstance(stats, dict):
                return stats

        stats = getattr(self.embedding_service, "cost_stats", None)
        if isinstance(stats, dict):
            return stats

        return {}

    @staticmethod
    def _extract_listing_ids(documents: list[Document]) -> list[str]:
        """
        Extrae IDs únicos conservando el orden original de recuperación/indexación.
        """
        result: list[str] = []
        seen: set[str] = set()

        for doc in documents:
            metadata = doc.metadata or {}

            listing_id = (
                metadata.get("id")
                or metadata.get("listing_id")
                or metadata.get("property_id")
            )

            if listing_id is None:
                continue

            listing_id_str = str(listing_id)

            if listing_id_str not in seen:
                seen.add(listing_id_str)
                result.append(listing_id_str)

        return result


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
