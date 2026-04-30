"""
MLflow Utilities — Evaluación RAG Inmobiliario
===============================================

Helpers para integrar métricas RAGAS con MLflow.

Responsabilidades:
  - Crear/obtener experimentos MLflow por nombre.
  - Loguear parámetros de configuración del experimento.
  - Loguear métricas RAGAS (promedio + por muestra + estado PASS/WARN/FAIL).
  - Construir nombres de runs descriptivos.
  - Evaluar el estado de cada métrica contra los thresholds de tres zonas.
"""

import mlflow
from datetime import datetime
from typing import Dict, Any, List, Optional

from tests.ragas.thresholds import MetricStatus, get_threshold


# ====================================================================
# EXPERIMENTO
# ====================================================================

def setup_mlflow_experiment(
    experiment_name: str = "realstate_rag_evaluation",
    tracking_uri: str = "mlruns",
) -> str:
    """
    Crea o recupera el experimento MLflow por nombre.

    Args:
        experiment_name : Nombre del experimento (aparece en la UI de MLflow).
        tracking_uri    : Ruta local del tracking store (default: "mlruns" en cwd).
                          Para cloud, usar "http://mlflow-server:5000" o URI de S3/GCS.

    Returns:
        experiment_id: ID del experimento creado o existente.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"[MLflow] Created experiment '{experiment_name}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"[MLflow] Using experiment '{experiment_name}' (id={experiment_id})")
    return experiment_id


def build_run_name(config: Dict[str, Any]) -> str:
    """
    Construye un nombre descriptivo para el run de MLflow.

    Formato: {collection}_k{k}_fk{fetch_k}_{variant}_{timestamp}
    Ejemplo: realstate_mvd_k5_fk100_default_1430
    """
    collection = config.get("collection", "unk")
    k          = config.get("k", "?")
    fetch_k    = config.get("fetch_k", "?")
    variant    = config.get("prompt_variant", "default")
    ts         = datetime.now().strftime("%H%M")
    return f"{collection}_k{k}_fk{fetch_k}_{variant}_{ts}"


# ====================================================================
# PARÁMETROS
# ====================================================================

def log_experiment_params(config: Dict[str, Any]) -> None:
    """
    Loguea los parámetros de configuración al run activo de MLflow.
    Convierte todos los valores a string para compatibilidad.
    """
    if mlflow.active_run() is None:
        print("[MLflow] Warning: log_experiment_params called with no active run.")
        return
    mlflow.log_params({k: str(v) for k, v in config.items()})


# ====================================================================
# MÉTRICAS RAGAS — CON SISTEMA DE TRES ZONAS
# ====================================================================

def log_ragas_metrics(
    metric_name: str,
    average_score: float,
    individual_scores: List[float],
    endpoint: str = "ask",
) -> MetricStatus:
    """
    Loguea métricas RAGAS al run activo de MLflow y evalúa el estado.

    Loguea las siguientes keys a MLflow:
      - {endpoint}_{metric_name}_avg     : score promedio (float)
      - {endpoint}_{metric_name}_q{N}    : score por muestra (N=01,02...)
      - {endpoint}_{metric_name}_status  : 1.0=PASS, 0.5=WARN, 0.0=FAIL

    Returns:
        MetricStatus: Estado de la métrica según los thresholds configurados.
    """
    if mlflow.active_run() is None:
        print(f"[MLflow] Warning: log_ragas_metrics called with no active run.")
        return MetricStatus.WARN

    prefix = f"{endpoint}_{metric_name}"

    mlflow.log_metric(f"{prefix}_avg", round(average_score, 4))
    for i, score in enumerate(individual_scores):
        mlflow.log_metric(f"{prefix}_q{i+1:02d}", round(score, 4))

    threshold_cfg = get_threshold(metric_name)
    status = MetricStatus.WARN
    if threshold_cfg is not None:
        status = threshold_cfg.evaluate(average_score)
        status_value = {"PASS": 1.0, "WARN": 0.5, "FAIL": 0.0}[status.value]
        mlflow.log_metric(f"{prefix}_status", status_value)
        mlflow.set_tag(f"{prefix}_status_label", status.value)
        print(f"  [MLflow] {threshold_cfg.summary(average_score)} (endpoint={endpoint})")

    return status


def log_functional_metric(
    metric_name: str,
    value: float,
    endpoint: str = "functional",
) -> MetricStatus:
    """
    Loguea una métrica funcional (no RAGAS) al run activo de MLflow.

    Args:
        metric_name : Nombre de la métrica (ej: "rejection_rate").
        value       : Valor de la métrica.
        endpoint    : Prefijo del grupo (default: "functional").

    Returns:
        MetricStatus del threshold correspondiente.
    """
    if mlflow.active_run() is None:
        return MetricStatus.WARN

    prefix = f"{endpoint}_{metric_name}"
    mlflow.log_metric(prefix, round(value, 4))

    threshold_cfg = get_threshold(metric_name)
    status = MetricStatus.WARN
    if threshold_cfg is not None:
        status = threshold_cfg.evaluate(value)
        status_value = {"PASS": 1.0, "WARN": 0.5, "FAIL": 0.0}[status.value]
        mlflow.log_metric(f"{prefix}_status", status_value)
        mlflow.set_tag(f"{prefix}_status_label", status.value)
        print(f"  [MLflow] {threshold_cfg.summary(value)} (endpoint={endpoint})")

    return status


def log_dataset_info(
    n_samples: int,
    endpoint: str = "ask",
    from_cache: bool = False,
) -> None:
    """Loguea metadatos del dataset de evaluación al run activo."""
    if mlflow.active_run() is None:
        return
    mlflow.log_param(f"{endpoint}_dataset_samples", str(n_samples))
    mlflow.log_param(f"{endpoint}_dataset_from_cache", str(from_cache))


# ====================================================================
# RESUMEN DE RUN
# ====================================================================

def print_run_summary(run_id: str) -> None:
    """
    Imprime una tabla con todas las métricas del run y su estado PASS/WARN/FAIL.

    Llamar al final de run_experiment.py o al cerrar el run de pytest.
    """
    client  = mlflow.tracking.MlflowClient()
    run     = client.get_run(run_id)
    metrics = run.data.metrics

    avg_metrics = {k: v for k, v in metrics.items() if k.endswith("_avg")}
    if not avg_metrics:
        return

    print(f"\n{'═'*72}")
    print(f"  RUN SUMMARY  —  {run_id[:8]}...")
    print(f"{'═'*72}")
    print(f"  {'Metric':<42} {'Score':>8}  Status")
    print(f"  {'─'*42} {'─'*8}  {'─'*8}")

    all_statuses = []
    for key, value in sorted(avg_metrics.items()):
        parts      = key.rsplit("_avg", 1)[0]
        status_val = metrics.get(f"{parts}_status", 0.5)
        status_str = {1.0: "PASS ✓", 0.5: "WARN ⚠", 0.0: "FAIL ✗"}.get(status_val, "?")
        all_statuses.append(status_val)
        print(f"  {key:<42} {value:>8.4f}  {status_str}")

    overall = (
        "PASS ✓" if all(s == 1.0 for s in all_statuses) else
        "FAIL ✗" if any(s == 0.0 for s in all_statuses) else
        "WARN ⚠"
    )
    print(f"  {'─'*62}")
    print(f"  Overall: {overall}")
    print(f"{'═'*72}\n")
