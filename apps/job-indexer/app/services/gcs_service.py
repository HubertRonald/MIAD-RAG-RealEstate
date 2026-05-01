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
      - versions/{version}/
      - latest/
    """

    def __init__(self) -> None:
        self.gcs = GCSClient(
            bucket_name=settings.INDEX_BUCKET,
            project_id=settings.PROJECT_ID,
        )

    def publish_index(self, local_index_dir: Path, version: str) -> dict:
        if not local_index_dir.exists():
            raise FileNotFoundError(f"No existe local_index_dir: {local_index_dir}")

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
