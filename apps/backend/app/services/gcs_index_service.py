from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from app.config.runtime import get_settings
from miad_rag_common.gcp.gcs_client import GCSClient
from miad_rag_common.logging.structured_logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class GCSIndexService:
    """
    Descarga el índice FAISS desde GCS hacia /tmp en Cloud Run.

    Configuración esperada desde runtime.py:

      INDEX_BUCKET = "miad-paad-rs-index-dev"
      INDEX_PREFIX = "faiss"
      DEFAULT_COLLECTION = "realstate_mvd"
      INDEX_LOCAL_ROOT = "/tmp/faiss_index"

    Rutas esperadas en GCS:

      latest:
        gs://miad-paad-rs-index-dev/faiss/realstate_mvd/latest/

      versionada:
        gs://miad-paad-rs-index-dev/faiss/realstate_mvd/versions/<version>/

    Archivos esperados:
      - index.faiss
      - index.pkl
      - manifest.json
      - listing_ids.json
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
        Usa la colección recibida o DEFAULT_COLLECTION si no se proporciona.
        """
        resolved = (collection or settings.DEFAULT_COLLECTION or "").strip()

        if not resolved:
            raise ValueError(
                "No se recibió collection y DEFAULT_COLLECTION está vacío."
            )

        return resolved

    def _remote_prefix(
        self,
        collection: Optional[str] = None,
        version: str = "latest",
    ) -> str:
        """
        Retorna el prefijo dentro del bucket, NO el gs:// completo.

        Ejemplo:
          faiss/realstate_mvd/latest/

        URI completo equivalente:
          gs://miad-paad-rs-index-dev/faiss/realstate_mvd/latest/
        """
        resolved_collection = self._resolve_collection(collection)
        base = f"{settings.INDEX_PREFIX.strip('/')}/{resolved_collection}"

        if version == "latest":
            return f"{base}/latest/"

        return f"{base}/versions/{version}/"

    def _remote_uri(
        self,
        collection: Optional[str] = None,
        version: str = "latest",
    ) -> str:
        """
        Retorna el URI completo para logs/debug.
        """
        return f"gs://{settings.INDEX_BUCKET}/{self._remote_prefix(collection, version)}"

    def _local_path(
        self,
        collection: Optional[str] = None,
        version: str = "latest",
    ) -> Path:
        """
        Ruta local donde se descarga el índice en Cloud Run.

        Ejemplo:
          /tmp/faiss_index/realstate_mvd/latest
        """
        resolved_collection = self._resolve_collection(collection)
        return Path(settings.INDEX_LOCAL_ROOT) / resolved_collection / version

    def _has_required_local_files(self, local_path: Path) -> bool:
        return all(
            (local_path / file_name).exists()
            for file_name in self.REQUIRED_FILES
        )

    def ensure_local_index(
        self,
        collection: Optional[str] = None,
        version: str = "latest",
        force_download: bool = False,
    ) -> Path:
        """
        Retorna la ruta local del índice FAISS.

        Si el índice ya existe en /tmp y force_download=False, reutiliza cache local.
        Si no existe, descarga desde GCS.

        Args:
            collection:
                Colección FAISS. Si es None, usa settings.DEFAULT_COLLECTION.
            version:
                "latest" o una versión específica, por ejemplo 20260430T210000Z.
            force_download:
                Si True, borra y descarga de nuevo desde GCS.

        Returns:
            Path local del índice.
        """
        resolved_collection = self._resolve_collection(collection)
        local_path = self._local_path(resolved_collection, version)

        if not force_download and self._has_required_local_files(local_path):
            logger.info(
                "faiss_index_cached_locally",
                extra={
                    "collection": resolved_collection,
                    "version": version,
                    "local_path": str(local_path),
                    "remote_uri": self._remote_uri(resolved_collection, version),
                },
            )
            return local_path

        remote_prefix = self._remote_prefix(resolved_collection, version)
        remote_uri = self._remote_uri(resolved_collection, version)

        logger.info(
            "downloading_faiss_index",
            extra={
                "collection": resolved_collection,
                "version": version,
                "bucket": settings.INDEX_BUCKET,
                "prefix": remote_prefix,
                "remote_uri": remote_uri,
                "local_path": str(local_path),
                "force_download": force_download,
            },
        )

        tmp_path = local_path.parent / f".{local_path.name}.download"

        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        tmp_path.mkdir(parents=True, exist_ok=True)

        downloaded = self.gcs.download_prefix(
            prefix=remote_prefix,
            destination_dir=tmp_path,
            strip_prefix=True,
        )

        if not downloaded:
            raise FileNotFoundError(
                f"No se encontraron objetos en {remote_uri}"
            )

        missing = [
            file_name
            for file_name in self.REQUIRED_FILES
            if not (tmp_path / file_name).exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Índice FAISS incompleto en GCS. "
                f"Faltan archivos: {missing}. "
                f"Ruta: {remote_uri}"
            )

        if local_path.exists():
            shutil.rmtree(local_path)

        tmp_path.rename(local_path)

        logger.info(
            "faiss_index_ready",
            extra={
                "collection": resolved_collection,
                "version": version,
                "remote_uri": remote_uri,
                "local_path": str(local_path),
                "files_count": len(downloaded),
            },
        )

        return local_path

    def read_remote_manifest(
        self,
        collection: Optional[str] = None,
        version: str = "latest",
    ) -> dict[str, Any]:
        """
        Lee manifest.json directamente desde GCS.
        """
        resolved_collection = self._resolve_collection(collection)
        manifest_blob = (
            f"{self._remote_prefix(resolved_collection, version)}manifest.json"
        )
        return self.gcs.read_json(manifest_blob)

    def read_local_manifest(
        self,
        collection: Optional[str] = None,
        version: str = "latest",
    ) -> dict[str, Any]:
        """
        Lee manifest.json desde /tmp.
        """
        resolved_collection = self._resolve_collection(collection)
        local_manifest = (
            self._local_path(resolved_collection, version) / "manifest.json"
        )

        if not local_manifest.exists():
            raise FileNotFoundError(
                f"No existe manifest local: {local_manifest}"
            )

        return json.loads(local_manifest.read_text(encoding="utf-8"))

    def refresh_if_remote_changed(
        self,
        collection: Optional[str] = None,
        version: str = "latest",
    ) -> Path:
        """
        Compara manifest local vs manifest remoto.

        Si cambia la versión del manifest remoto, fuerza descarga.

        Esto es útil principalmente para version='latest', porque una instancia
        viva de Cloud Run puede tener un índice viejo en /tmp mientras el
        job-indexer ya publicó un latest nuevo.
        """
        resolved_collection = self._resolve_collection(collection)
        local_path = self._local_path(resolved_collection, version)

        if not self._has_required_local_files(local_path):
            return self.ensure_local_index(
                collection=resolved_collection,
                version=version,
                force_download=True,
            )

        try:
            local_manifest = self.read_local_manifest(
                collection=resolved_collection,
                version=version,
            )
            remote_manifest = self.read_remote_manifest(
                collection=resolved_collection,
                version=version,
            )

            local_version = local_manifest.get("version")
            remote_version = remote_manifest.get("version")

            if local_version != remote_version:
                logger.info(
                    "faiss_remote_manifest_changed",
                    extra={
                        "collection": resolved_collection,
                        "version_alias": version,
                        "local_version": local_version,
                        "remote_version": remote_version,
                        "remote_uri": self._remote_uri(
                            resolved_collection,
                            version,
                        ),
                    },
                )

                return self.ensure_local_index(
                    collection=resolved_collection,
                    version=version,
                    force_download=True,
                )

        except Exception as exc:
            logger.warning(
                "faiss_manifest_compare_failed_using_local_cache",
                extra={
                    "collection": resolved_collection,
                    "version": version,
                    "error": str(exc),
                },
            )

        return local_path

    # Compatibilidad con nombre anterior
    def read_manifest(
        self,
        collection: Optional[str] = None,
        version: str = "latest",
    ) -> dict[str, Any]:
        """
        Alias compatible. Lee manifest remoto desde GCS.
        """
        return self.read_remote_manifest(collection, version)
