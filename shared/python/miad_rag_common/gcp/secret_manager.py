from __future__ import annotations

import os
from typing import Optional

from google.cloud import secretmanager


def build_secret_version_path(
    project_id: str,
    secret_id: str,
    version: str = "latest",
) -> str:
    if not project_id or not secret_id:
        raise ValueError("project_id y secret_id son requeridos.")
    return f"projects/{project_id}/secrets/{secret_id}/versions/{version}"


class SecretManagerClient:
    """
    Cliente liviano para Secret Manager.

    Uso típico:
      gemini_key = secrets.get_secret_or_env(
          project_id="miad-paad-rs-dev",
          secret_id="gemini-api-key",
          env_var="GOOGLE_API_KEY",
      )
    """

    def __init__(self, client: Optional[secretmanager.SecretManagerServiceClient] = None):
        self.client = client or secretmanager.SecretManagerServiceClient()

    def get_secret(
        self,
        project_id: str,
        secret_id: str,
        version: str = "latest",
    ) -> str:
        name = build_secret_version_path(project_id, secret_id, version)
        response = self.client.access_secret_version(request={"name": name})
        return response.payload.data.decode("utf-8")

    def get_secret_or_env(
        self,
        project_id: str,
        secret_id: str,
        env_var: str,
        version: str = "latest",
        required: bool = True,
    ) -> Optional[str]:
        """
        Primero busca variable de entorno, luego Secret Manager.

        Esto permite:
        - desarrollo local con .env,
        - Cloud Run con Secret Manager.
        """
        value = os.getenv(env_var)
        if value:
            return value

        try:
            return self.get_secret(project_id, secret_id, version)
        except Exception:
            if required:
                raise
            return None
