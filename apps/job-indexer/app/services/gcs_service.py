from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.config.runtime import get_settings
from miad_rag_common.gcp.gcs_client import GCSClient
from miad_rag_common.logging.structured_logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class GCSIndexPublisher:
    """
    Publica el índice FAISS generado localmente en Cloud Storage.

    Configuración esperada desde runtime.py:

      INDEX_BUCKET = "miad-paad-rs-index-dev"
      INDEX_PREFIX = "faiss"
      COLLECTION = "realstate_mvd"

    Rutas de salida:

      latest:
        gs://miad-paad-rs-index-dev/faiss/realstate_mvd/latest/

      versionada:
        gs://miad-paad-rs-index-dev/faiss/realstate_mvd/versions/<version>/

    Archivos esperados:
      - index.faiss
      - index.pkl
      - manifest.json
      - listing_ids.json

    Nota:
      Este servicio NO construye FAISS.
      Solo publica en GCS lo que FAISSBuilder ya dejó en disco local.
    """

    REQUIRED_FILES = [
        "index.faiss",
        "index.pkl",
        "manifest.json",
        "listing_ids.json",
    ]

    def __init__(self) -> None:
        self.gcs = GCSClient(
            bucket_name=settings.INDEX_BUCKET,
            project_id=settings.PROJECT_ID,
        )

    def _resolve_collection(self, collection: Optional[str] = None) -> str:
        """
        Usa collection explícita o settings.COLLECTION.

        En el job-indexer normalmente no se pasa collection porque la colección
        viene del runtime.py.
        """
        resolved = (collection or settings.COLLECTION or "").strip()

        if not resolved:
            raise ValueError(
                "No se recibió collection y settings.COLLECTION está vacío."
            )

        return resolved

    def _latest_prefix(self, collection: Optional[str] = None) -> str:
        """
        Retorna el prefijo latest dentro del bucket.

        Ejemplo:
          faiss/realstate_mvd/latest

        URI completo equivalente:
          gs://miad-paad-rs-index-dev/faiss/realstate_mvd/latest/
        """
        resolved_collection = self._resolve_collection(collection)
        return f"{settings.INDEX_PREFIX.strip('/')}/{resolved_collection}/latest"

    def _version_prefix(
        self,
        version: str,
        collection: Optional[str] = None,
    ) -> str:
        """
        Retorna el prefijo versionado dentro del bucket.

        Ejemplo:
          faiss/realstate_mvd/versions/20260430T210000Z
        """
        if not version or not version.strip():
            raise ValueError("version es requerida para publicar ruta versionada.")

        if version == "latest":
            raise ValueError(
                "version no puede ser 'latest' para ruta versionada."
            )

        resolved_collection = self._resolve_collection(collection)

        return (
            f"{settings.INDEX_PREFIX.strip('/')}/"
            f"{resolved_collection}/"
            f"versions/{version}"
        )

    def _gcs_uri(self, prefix: str) -> str:
        """
        Retorna URI completo para logs/debug.

        upload_directory() recibe solo prefix, pero para trazabilidad conviene
        loggear el gs:// completo.
        """
        return f"gs://{settings.INDEX_BUCKET}/{prefix.rstrip('/')}/"

    def _validate_local_index(self, local_index_dir: Path) -> None:
        """
        Valida que el índice local tenga todos los archivos requeridos.
        """
        if not local_index_dir.exists():
            raise FileNotFoundError(
                f"No existe local_index_dir: {local_index_dir}"
            )

        if not local_index_dir.is_dir():
            raise NotADirectoryError(
                f"local_index_dir no es un directorio: {local_index_dir}"
            )

        missing_files = [
            file_name
            for file_name in self.REQUIRED_FILES
            if not (local_index_dir / file_name).exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                "El índice local está incompleto. "
                f"Faltan archivos: {missing_files}. "
                f"Directorio: {local_index_dir}"
            )

    def publish_index(
        self,
        local_index_dir: Path,
        version: str,
        collection: Optional[str] = None,
    ) -> dict[str, Any]:
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

            collection:
                Colección FAISS. Si es None, usa settings.COLLECTION.

        Returns:
            Diccionario con URIs subidas a latest y versioned.
        """
        resolved_collection = self._resolve_collection(collection)
        self._validate_local_index(local_index_dir)

        uploaded: dict[str, list[str]] = {
            "versioned": [],
            "latest": [],
        }

        if not settings.UPLOAD_VERSIONED and not settings.UPLOAD_LATEST:
            logger.warning(
                "faiss_index_upload_skipped",
                extra={
                    "reason": "UPLOAD_VERSIONED and UPLOAD_LATEST are both false",
                    "collection": resolved_collection,
                    "version": version,
                    "local_index_dir": str(local_index_dir),
                },
            )
            return uploaded

        if settings.UPLOAD_VERSIONED:
            version_prefix = self._version_prefix(
                version=version,
                collection=resolved_collection,
            )

            uploaded["versioned"] = self.gcs.upload_directory(
                source_dir=local_index_dir,
                prefix=version_prefix,
            )

            logger.info(
                "faiss_index_uploaded_versioned",
                extra={
                    "collection": resolved_collection,
                    "version": version,
                    "bucket": settings.INDEX_BUCKET,
                    "prefix": version_prefix,
                    "gcs_uri": self._gcs_uri(version_prefix),
                    "files_count": len(uploaded["versioned"]),
                    "files": uploaded["versioned"],
                },
            )

        if settings.UPLOAD_LATEST:
            latest_prefix = self._latest_prefix(
                collection=resolved_collection,
            )

            uploaded["latest"] = self.gcs.upload_directory(
                source_dir=local_index_dir,
                prefix=latest_prefix,
            )

            logger.info(
                "faiss_index_uploaded_latest",
                extra={
                    "collection": resolved_collection,
                    "version": version,
                    "bucket": settings.INDEX_BUCKET,
                    "prefix": latest_prefix,
                    "gcs_uri": self._gcs_uri(latest_prefix),
                    "files_count": len(uploaded["latest"]),
                    "files": uploaded["latest"],
                },
            )

        return uploaded
