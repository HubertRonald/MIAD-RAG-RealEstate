from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from google.cloud import storage


class GCSClient:
    """
    Cliente liviano de Google Cloud Storage.

    Sirve para:
    - backend: descargar índice FAISS desde GCS a /tmp.
    - job-indexer: subir índice FAISS y manifest.json.
    - cualquier app: leer/escribir artefactos pequeños.
    """

    def __init__(
        self,
        bucket_name: str,
        project_id: Optional[str] = None,
        client: Optional[storage.Client] = None,
    ) -> None:
        if not bucket_name:
            raise ValueError("bucket_name es requerido.")

        self.bucket_name = bucket_name
        self.client = client or storage.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)

    def blob_exists(self, blob_name: str) -> bool:
        return self.bucket.blob(blob_name).exists()

    def list_blobs(self, prefix: str) -> list[str]:
        return [blob.name for blob in self.client.list_blobs(self.bucket, prefix=prefix)]

    def download_blob(self, blob_name: str, destination_path: str | Path) -> Path:
        destination = Path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        blob = self.bucket.blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(
                f"No existe el objeto gs://{self.bucket_name}/{blob_name}"
            )

        blob.download_to_filename(str(destination))
        return destination

    def upload_file(
        self,
        source_path: str | Path,
        blob_name: str,
        content_type: Optional[str] = None,
    ) -> str:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"No existe el archivo local: {source}")

        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(str(source), content_type=content_type)

        return f"gs://{self.bucket_name}/{blob_name}"

    def download_prefix(
        self,
        prefix: str,
        destination_dir: str | Path,
        strip_prefix: bool = True,
    ) -> list[Path]:
        """
        Descarga todos los objetos bajo un prefijo.

        Ejemplo:
          prefix = "faiss/realstate_mvd/latest/"
          destination_dir = "/tmp/faiss_index/realstate_mvd/latest"
        """
        destination_root = Path(destination_dir)
        destination_root.mkdir(parents=True, exist_ok=True)

        downloaded: list[Path] = []

        for blob in self.client.list_blobs(self.bucket, prefix=prefix):
            if blob.name.endswith("/"):
                continue

            if strip_prefix:
                relative_name = blob.name.removeprefix(prefix).lstrip("/")
            else:
                relative_name = blob.name

            destination = destination_root / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(destination))
            downloaded.append(destination)

        return downloaded

    def upload_directory(
        self,
        source_dir: str | Path,
        prefix: str,
    ) -> list[str]:
        """
        Sube recursivamente un directorio local a un prefijo GCS.
        """
        source_root = Path(source_dir)
        if not source_root.exists() or not source_root.is_dir():
            raise NotADirectoryError(f"No existe el directorio local: {source_root}")

        uploaded: list[str] = []

        for path in source_root.rglob("*"):
            if path.is_dir():
                continue

            relative = path.relative_to(source_root).as_posix()
            blob_name = f"{prefix.rstrip('/')}/{relative}"
            uploaded.append(self.upload_file(path, blob_name))

        return uploaded

    def read_text(self, blob_name: str, encoding: str = "utf-8") -> str:
        blob = self.bucket.blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(
                f"No existe el objeto gs://{self.bucket_name}/{blob_name}"
            )
        return blob.download_as_text(encoding=encoding)

    def write_text(
        self,
        blob_name: str,
        content: str,
        content_type: str = "text/plain",
    ) -> str:
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(content, content_type=content_type)
        return f"gs://{self.bucket_name}/{blob_name}"

    def read_json(self, blob_name: str) -> dict[str, Any]:
        return json.loads(self.read_text(blob_name))

    def write_json(self, blob_name: str, payload: dict[str, Any]) -> str:
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        return self.write_text(
            blob_name=blob_name,
            content=content,
            content_type="application/json",
        )