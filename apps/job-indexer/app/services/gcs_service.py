from __future__ import annotations

from pathlib import Path

from app.config.runtime import get_settings
from miad_rag_common.gcp.gcs_client import GCSClient
from miad_rag_common.logging.structured_logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class GCSIndexPublisher:
    """
    Publica el índice FAISS generado localmente en Cloud Storage.

    Sube dos rutas:

      1. Ruta versionada:
         gs://{INDEX_BUCKET}/{INDEX_PREFIX}/{COLLECTION}/versions/{version}/

      2. Ruta latest:
         gs://{INDEX_BUCKET}/{INDEX_PREFIX}/{COLLECTION}/latest/

    El backend descarga siempre desde latest.
    La ruta versionada permite trazabilidad y rollback manual.
    """

    def __init__(self) -> None:
        self.gcs = GCSClient(
            bucket_name=settings.INDEX_BUCKET,
            project_id=settings.PROJECT_ID,
        )

    def publish_index(self, local_index_dir: Path, version: str) -> dict:
        """
        Publica el directorio local del índice en GCS.

        Args:
            local_index_dir:
                Directorio local que contiene:
                  - index.faiss
                  - index.pkl
                  - manifest.json
                  - listing_ids.json

            version:
                Versión temporal del índice, por ejemplo 20260430T210000Z.

        Returns:
            Diccionario con las URIs subidas a latest y versions.
        """
        if not local_index_dir.exists():
            raise FileNotFoundError(
                f"No existe local_index_dir: {local_index_dir}"
            )

        required_files = [
            "index.faiss",
            "index.pkl",
            "manifest.json",
            "listing_ids.json",
        ]

        missing_files = [
            file_name
            for file_name in required_files
            if not (local_index_dir / file_name).exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                "El índice local está incompleto. "
                f"Faltan archivos: {missing_files}. "
                f"Directorio: {local_index_dir}"
            )

        uploaded = {
            "versioned": [],
            "latest": [],
        }

        if settings.UPLOAD_VERSIONED:
            version_prefix = settings.gcs_version_prefix(version)

            uploaded["versioned"] = self.gcs.upload_directory(
                source_dir=local_index_dir,
                prefix=version_prefix,
            )

            logger.info(
                "faiss_index_uploaded_versioned",
                extra={
                    "bucket": settings.INDEX_BUCKET,
                    "prefix": version_prefix,
                    "files_count": len(uploaded["versioned"]),
                },
            )

        if settings.UPLOAD_LATEST:
            latest_prefix = settings.gcs_latest_prefix

            uploaded["latest"] = self.gcs.upload_directory(
                source_dir=local_index_dir,
                prefix=latest_prefix,
            )

            logger.info(
                "faiss_index_uploaded_latest",
                extra={
                    "bucket": settings.INDEX_BUCKET,
                    "prefix": latest_prefix,
                    "files_count": len(uploaded["latest"]),
                },
            )

        return uploaded
