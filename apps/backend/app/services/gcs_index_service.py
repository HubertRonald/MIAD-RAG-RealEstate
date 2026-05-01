from __future__ import annotations

from pathlib import Path

from app.config.runtime import get_settings
from miad_rag_common.gcp.gcs_client import GCSClient
from miad_rag_common.logging.structured_logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class GCSIndexService:
    """
    Descarga el índice FAISS desde GCS hacia /tmp en Cloud Run.

    Estructura esperada:
      gs://{INDEX_BUCKET}/{INDEX_PREFIX}/{collection}/latest/index.faiss
      gs://{INDEX_BUCKET}/{INDEX_PREFIX}/{collection}/latest/index.pkl
      gs://{INDEX_BUCKET}/{INDEX_PREFIX}/{collection}/latest/manifest.json
    """

    def __init__(self) -> None:
        self.gcs = GCSClient(
            bucket_name=settings.INDEX_BUCKET,
            project_id=settings.PROJECT_ID,
        )

    def _remote_prefix(self, collection: str, version: str = "latest") -> str:
        return f"{settings.INDEX_PREFIX.strip('/')}/{collection}/{version}/"

    def _local_path(self, collection: str, version: str = "latest") -> Path:
        return Path(settings.INDEX_LOCAL_ROOT) / collection / version

    def ensure_local_index(self, collection: str, version: str = "latest") -> Path:
        """
        Retorna ruta local del índice. Si no existe, lo descarga desde GCS.
        """
        local_path = self._local_path(collection, version)
        index_file = local_path / "index.faiss"
        pkl_file = local_path / "index.pkl"

        if index_file.exists() and pkl_file.exists():
            return local_path

        remote_prefix = self._remote_prefix(collection, version)

        logger.info(
            "downloading_faiss_index",
            extra={
                "collection": collection,
                "bucket": settings.INDEX_BUCKET,
                "prefix": remote_prefix,
                "local_path": str(local_path),
            },
        )

        downloaded = self.gcs.download_prefix(
            prefix=remote_prefix,
            destination_dir=local_path,
            strip_prefix=True,
        )

        if not downloaded:
            raise FileNotFoundError(
                f"No se encontraron objetos en gs://{settings.INDEX_BUCKET}/{remote_prefix}"
            )

        if not index_file.exists() or not pkl_file.exists():
            raise FileNotFoundError(
                "Índice FAISS incompleto. Se esperaban index.faiss e index.pkl "
                f"en gs://{settings.INDEX_BUCKET}/{remote_prefix}"
            )

        logger.info(
            "faiss_index_ready",
            extra={
                "collection": collection,
                "local_path": str(local_path),
                "files": [str(path) for path in downloaded],
            },
        )

        return local_path

    def read_manifest(self, collection: str, version: str = "latest") -> dict:
        manifest_blob = f"{self._remote_prefix(collection, version)}manifest.json"
        return self.gcs.read_json(manifest_blob)
