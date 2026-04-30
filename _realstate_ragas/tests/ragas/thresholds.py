"""
Thresholds de Evaluación — Sistema RAG Inmobiliario Montevideo
==============================================================

Define los criterios de aceptación y rechazo para cada métrica.

SISTEMA DE TRES ZONAS
─────────────────────
  PASS  : score >= accept_threshold  → métrica OK para producción
  WARN  : reject_threshold <= score < accept_threshold  → zona gris, requiere revisión
  FAIL  : score < reject_threshold   → bloquea el despliegue

La zona WARN entre reject y accept es intencional: permite que el sistema
pase tests sin ser promovido automáticamente, forzando revisión humana
antes de un release.

FUENTE DE LOS THRESHOLDS
─────────────────────────
Los valores provienen del documento de criterios de evaluación del proyecto:
  - answer_correctness : accept >= 0.60, reject < 0.35
  - context_precision  : accept >= 0.75, reject < 0.65
  - context_recall     : accept >= 0.65, reject < 0.50
  - faithfulness       : accept >= 0.85, reject < 0.75
  - answer_relevancy   : accept >= 0.78, reject < 0.65
  - avg_cosine_similarity : accept >= 0.72, reject: N/A (no hay criterio de rechazo)

Los thresholds funcionales están en FUNCTIONAL_THRESHOLDS:
  - rejection_rate     : accept >= 0.95, reject <= 0.90
  - intent_accuracy    : accept >= 0.90, reject < 0.85
  - filter_extraction_f1 : accept >= 0.85, reject < 0.80
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ====================================================================
# TIPOS
# ====================================================================

class MetricStatus(str, Enum):
    """Estado de una métrica según el sistema de tres zonas."""
    PASS = "PASS"   # score >= accept_threshold
    WARN = "WARN"   # reject_threshold <= score < accept_threshold
    FAIL = "FAIL"   # score < reject_threshold


@dataclass(frozen=True)
class ThresholdConfig:
    """
    Configuración de umbrales para una métrica.

    Attributes:
        metric_name       : Nombre de la métrica (ej: "context_precision").
        accept_threshold  : Score mínimo para PASS.
        reject_threshold  : Score por debajo del cual es FAIL.
                            None = sin criterio de rechazo (solo PASS/WARN).
        higher_is_better  : True para métricas de calidad (todas las actuales).
    """
    metric_name:       str
    accept_threshold:  float
    reject_threshold:  Optional[float]
    higher_is_better:  bool = True

    def evaluate(self, score: float) -> MetricStatus:
        """
        Evalúa el score y retorna el estado de la métrica.

        Args:
            score: Valor de la métrica (float en [0, 1]).

        Returns:
            MetricStatus.PASS, WARN o FAIL.
        """
        if self.higher_is_better:
            if score >= self.accept_threshold:
                return MetricStatus.PASS
            if self.reject_threshold is not None and score < self.reject_threshold:
                return MetricStatus.FAIL
            return MetricStatus.WARN
        else:
            # Para métricas donde menor es mejor (ej: latencia)
            if score <= self.accept_threshold:
                return MetricStatus.PASS
            if self.reject_threshold is not None and score > self.reject_threshold:
                return MetricStatus.FAIL
            return MetricStatus.WARN

    def summary(self, score: float) -> str:
        """Retorna un string legible con el resultado de la evaluación."""
        status = self.evaluate(score)
        zone_desc = {
            MetricStatus.PASS: f">= {self.accept_threshold} ✓",
            MetricStatus.WARN: f"entre {self.reject_threshold} y {self.accept_threshold} ⚠",
            MetricStatus.FAIL: f"< {self.reject_threshold} ✗",
        }
        return f"{self.metric_name}: {score:.4f} [{status.value}] ({zone_desc[status]})"


# ====================================================================
# RAGAS METRIC THRESHOLDS
# ====================================================================

RAGAS_THRESHOLDS: dict[str, ThresholdConfig] = {
    "answer_correctness": ThresholdConfig(
        metric_name="answer_correctness",
        accept_threshold=0.60,
        reject_threshold=0.35,
    ),
    "context_precision": ThresholdConfig(
        metric_name="context_precision",
        accept_threshold=0.75,
        reject_threshold=0.65,
    ),
    "context_recall": ThresholdConfig(
        metric_name="context_recall",
        accept_threshold=0.65,
        reject_threshold=0.50,
    ),
    "faithfulness": ThresholdConfig(
        metric_name="faithfulness",
        accept_threshold=0.85,
        reject_threshold=0.75,
    ),
    "answer_relevancy": ThresholdConfig(
        metric_name="answer_relevancy",
        accept_threshold=0.78,
        reject_threshold=0.65,
    ),
    "avg_cosine_similarity": ThresholdConfig(
        metric_name="avg_cosine_similarity",
        accept_threshold=0.72,
        reject_threshold=None,  # No hay criterio de rechazo definido
    ),
}


# ====================================================================
# FUNCTIONAL TEST THRESHOLDS
# ====================================================================

FUNCTIONAL_THRESHOLDS: dict[str, ThresholdConfig] = {
    "rejection_rate": ThresholdConfig(
        metric_name="rejection_rate",
        accept_threshold=0.95,   # >= 95% de queries fuera de dominio correctamente rechazadas
        reject_threshold=0.90,   # <= 90% = FAIL
    ),
    "intent_accuracy": ThresholdConfig(
        metric_name="intent_accuracy",
        accept_threshold=0.90,
        reject_threshold=0.85,
    ),
    "filter_extraction_f1": ThresholdConfig(
        metric_name="filter_extraction_f1",
        accept_threshold=0.85,
        reject_threshold=0.80,
    ),
    "response_time_sec": ThresholdConfig(
        metric_name="response_time_sec",
        accept_threshold=30.0,   # <= 30s por query en entorno automatizado
        reject_threshold=60.0,   # > 60s = FAIL (demo: <= 3 min total no aplica aquí)
        higher_is_better=False,
    ),
    "out_of_domain_clear_response_rate": ThresholdConfig(
        metric_name="out_of_domain_clear_response_rate",
        accept_threshold=1.0,    # 100% de respuestas claras, sin errores técnicos
        reject_threshold=None,
    ),
}


# ====================================================================
# HELPER
# ====================================================================

def get_threshold(metric_name: str) -> Optional[ThresholdConfig]:
    """
    Retorna la configuración de umbrales para una métrica.

    Busca en RAGAS_THRESHOLDS y FUNCTIONAL_THRESHOLDS.

    Returns:
        ThresholdConfig o None si la métrica no está configurada.
    """
    return RAGAS_THRESHOLDS.get(metric_name) or FUNCTIONAL_THRESHOLDS.get(metric_name)
