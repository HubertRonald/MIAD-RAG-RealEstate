from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token


class BackendClientError(RuntimeError):
    """Raised when the backend request fails or returns an invalid response."""


@dataclass(frozen=True)
class BackendClientConfig:
    backend_url: str
    timeout_sec: int = 90
    auth_mode: str = "auto"  # auto | none | iam | token


class BackendClient:
    def __init__(self, config: BackendClientConfig):
        if not config.backend_url:
            raise BackendClientError("BACKEND_URL no está configurado.")
        self.config = config
        self.base_url = config.backend_url.rstrip("/")

    @classmethod
    def from_env(cls) -> "BackendClient":
        return cls(
            BackendClientConfig(
                backend_url=os.getenv("BACKEND_URL", "").strip(),
                timeout_sec=int(os.getenv("BACKEND_TIMEOUT_SEC", "90")),
                auth_mode=os.getenv("BACKEND_AUTH_MODE", "auto").strip().lower(),
            )
        )

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/ask", json_payload=payload)

    def recommend(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/recommend", json_payload=payload)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = self._resolve_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _resolve_token(self) -> str | None:
        explicit_token = os.getenv("BACKEND_AUTH_TOKEN", "").strip()
        if explicit_token:
            return explicit_token

        mode = self.config.auth_mode
        if mode == "none":
            return None
        if mode not in {"auto", "iam", "token"}:
            raise BackendClientError(f"BACKEND_AUTH_MODE inválido: {mode}")
        if mode == "token":
            raise BackendClientError("BACKEND_AUTH_MODE=token requiere definir BACKEND_AUTH_TOKEN.")

        try:
            request = GoogleAuthRequest()
            return id_token.fetch_id_token(request, self.base_url)
        except Exception as exc:  # noqa: BLE001 - queremos fallback explícito en modo auto
            if mode == "iam":
                raise BackendClientError(
                    "No fue posible obtener identity token para Cloud Run. "
                    "Valida la service account del frontend o usa BACKEND_AUTH_TOKEN en local."
                ) from exc
            return None

    def _request(self, method: str, path: str, json_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self._headers(),
                json=json_payload,
                timeout=self.config.timeout_sec,
            )
        except requests.RequestException as exc:
            raise BackendClientError(f"No fue posible conectar con el backend: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text[:1200]
            raise BackendClientError(f"Backend respondió HTTP {response.status_code}: {detail}")

        try:
            return response.json()
        except ValueError as exc:
            raise BackendClientError(f"Respuesta no JSON desde {url}: {response.text[:500]}") from exc
