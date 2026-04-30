from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config.runtime import get_settings
from miad_rag_common.logging.structured_logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class MLflowService:
    """
    Registro opcional de experimentos del indexer.

    Este servicio no debe romper el job si MLflow no está habilitado.
    """

    def __init__(self) -> None:
        self.enabled = settings.ENABLE_MLFLOW

        if not self.enabled:
            return

        try:
            import mlflow

            self.mlflow = mlflow

            if settings.MLFLOW_TRACKING_URI:
                self.mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

            self.mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        except Exception as exc:
            logger.warning(
                "mlflow_disabled_due_to_error",
                extra={"error": str(exc)},
            )
            self.enabled = False
            self.mlflow = None

    def log_index_build(
        self,
        manifest: dict[str, Any],
        local_index_dir: Path,
        metrics: dict[str, float | int],
        params: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return

        try:
            with self.mlflow.start_run(run_name=f"index-{manifest['version']}"):
                for key, value in params.items():
                    if value is not None:
                        self.mlflow.log_param(key, value)

                for key, value in metrics.items():
                    if value is not None:
                        self.mlflow.log_metric(key, value)

                self.mlflow.log_dict(manifest, "manifest.json")

                if local_index_dir.exists():
                    self.mlflow.log_artifacts(
                        str(local_index_dir),
                        artifact_path="faiss_index",
                    )

        except Exception as exc:
            logger.warning(
                "mlflow_log_failed",
                extra={"error": str(exc)},
            )
