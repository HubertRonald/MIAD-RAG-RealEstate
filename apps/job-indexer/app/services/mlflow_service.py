from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.config.runtime import get_settings
from miad_rag_common.logging.structured_logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class MLflowService:
    """
    Servicio opcional para registrar experimentos del job-indexer.

    Estado actual:
      - En Cloud Run Job puede quedar deshabilitado con ENABLE_MLFLOW=false.
      - No rompe el job si MLflow no se usa.
      - Permite habilitar tracking remoto en el futuro con MLFLOW_TRACKING_URI.

    Nota:
      Para esta fase, los experimentos formales pueden seguir corriendo localmente
      con RAGAS + MLflow, usando el flujo existente de _realstate_ragas.
    """

    def __init__(self) -> None:
        self.enabled = settings.ENABLE_MLFLOW
        self.mlflow = None

        if not self.enabled:
            logger.info("mlflow_disabled")
            return

        try:
            import mlflow

            self.mlflow = mlflow

            if settings.MLFLOW_TRACKING_URI:
                self.mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

            self.mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

            logger.info(
                "mlflow_enabled",
                extra={
                    "tracking_uri": settings.MLFLOW_TRACKING_URI,
                    "experiment_name": settings.MLFLOW_EXPERIMENT_NAME,
                },
            )

        except Exception as exc:
            # Para implementación futura: no tumbamos el job por MLflow.
            # Si en el futuro quieres hacerlo obligatorio, aquí se puede cambiar
            # este comportamiento por "raise".
            logger.warning(
                "mlflow_initialization_failed_disabling_mlflow",
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
        tags: Optional[dict[str, str]] = None,
        uploaded: Optional[dict[str, list[str]]] = None,
        log_full_faiss_artifacts: bool = False,
    ) -> None:
        """
        Registra una corrida de construcción del índice si MLflow está habilitado.

        Si ENABLE_MLFLOW=false, este método no hace nada.
        """
        if not self.enabled or self.mlflow is None:
            logger.info(
                "mlflow_log_skipped",
                extra={
                    "reason": "disabled",
                    "collection": manifest.get("collection"),
                    "version": manifest.get("version"),
                },
            )
            return

        try:
            run_name = f"index-{manifest.get('collection')}-{manifest.get('version')}"

            with self.mlflow.start_run(run_name=run_name):
                self.mlflow.set_tag("component", "job-indexer")
                self.mlflow.set_tag("collection", str(manifest.get("collection")))
                self.mlflow.set_tag("index_version", str(manifest.get("version")))
                self.mlflow.set_tag("environment", settings.ENV)

                for key, value in (tags or {}).items():
                    self.mlflow.set_tag(key, value)

                for key, value in params.items():
                    if value is not None:
                        self.mlflow.log_param(key, value)

                for key, value in metrics.items():
                    if value is not None:
                        self.mlflow.log_metric(key, float(value))

                self.mlflow.log_dict(manifest, "manifest.json")

                if uploaded:
                    self.mlflow.log_dict(uploaded, "uploaded_gcs_uris.json")

                manifest_path = local_index_dir / "manifest.json"
                listing_ids_path = local_index_dir / "listing_ids.json"

                if manifest_path.exists():
                    self.mlflow.log_artifact(
                        str(manifest_path),
                        artifact_path="index_metadata",
                    )

                if listing_ids_path.exists():
                    self.mlflow.log_artifact(
                        str(listing_ids_path),
                        artifact_path="index_metadata",
                    )

                if log_full_faiss_artifacts:
                    index_file = local_index_dir / "index.faiss"
                    pkl_file = local_index_dir / "index.pkl"

                    if index_file.exists():
                        self.mlflow.log_artifact(
                            str(index_file),
                            artifact_path="faiss_index",
                        )

                    if pkl_file.exists():
                        self.mlflow.log_artifact(
                            str(pkl_file),
                            artifact_path="faiss_index",
                        )

            logger.info(
                "mlflow_index_build_logged",
                extra={
                    "run_name": run_name,
                    "experiment_name": settings.MLFLOW_EXPERIMENT_NAME,
                },
            )

        except Exception as exc:
            # No se tumba el job por un fallo de MLflow.
            logger.warning(
                "mlflow_log_failed",
                extra={
                    "error": str(exc),
                    "collection": manifest.get("collection"),
                    "version": manifest.get("version"),
                },
            )
        