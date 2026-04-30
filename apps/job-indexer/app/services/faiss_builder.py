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

    Output esperado:
      index.faiss
      index.pkl
      manifest.json
      listing_ids.json
    """

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    def build_vectorstore(self, documents: list[Document]) -> FAISS:
        if not documents:
            raise ValueError("No hay documentos para construir FAISS.")

        logger.info(
            "building_faiss_vectorstore",
            extra={"documents_count": len(documents)},
        )

        return FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_service.get_embeddings_model(),
        )

    def save_vectorstore(
        self,
        vectorstore: FAISS,
        collection: str,
        version: str,
    ) -> Path:
        output_dir = self._local_index_dir(collection, version)

        if output_dir.exists():
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        vectorstore.save_local(str(output_dir))

        index_file = output_dir / "index.faiss"
        pkl_file = output_dir / "index.pkl"

        if not index_file.exists() or not pkl_file.exists():
            raise RuntimeError(
                f"FAISS no generó los archivos esperados en {output_dir}"
            )

        logger.info(
            "faiss_vectorstore_saved",
            extra={
                "collection": collection,
                "version": version,
                "output_dir": str(output_dir),
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
        listing_ids = self._extract_listing_ids(documents)

        listing_ids_path = output_dir / "listing_ids.json"
        listing_ids_path.write_text(
            json.dumps(listing_ids, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
        }

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return manifest

    def _local_index_dir(self, collection: str, version: str) -> Path:
        return (
            Path(settings.LOCAL_WORKDIR)
            / settings.LOCAL_INDEX_DIRNAME
            / collection
            / version
        )

    @staticmethod
    def _extract_listing_ids(documents: list[Document]) -> list[str]:
        result = []
        seen = set()

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
