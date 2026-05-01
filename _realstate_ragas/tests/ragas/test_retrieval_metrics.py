"""
Tests de Métricas de Recuperación — /ask endpoint
==================================================

Evalúa la calidad del RetrievalService usando métricas RAGAS de retrieval.

MÉTRICAS:
  - context_precision : ¿Los documentos recuperados son relevantes para la pregunta?
  - context_recall    : ¿Se recuperó toda la información relevante disponible?

MLFLOW:
  El logging a MLflow es transparente — gestionado por el fixture mlflow_run
  (autouse=True en conftest.py) y save_metric_to_report().
  No se necesita importar mlflow en este archivo.

DATASET:
  Usa evaluation_dataset (fixture de conftest.py):
  8 preguntas sobre el mercado inmobiliario de Montevideo (/ask endpoint).
  El dataset se cachea en tests/ragas/cache/ para evitar reconstrucción.

UMBRALES:
  Los thresholds (0.1) son intencionalmentne conservadores para esta primera
  iteración. Una vez establecida la línea base, ajustar a valores más altos
  (0.5–0.7 para precision, 0.4–0.6 para recall).
"""

import pytest
from ragas.metrics import context_precision, context_recall
from ragas import evaluate
from ragas.run_config import RunConfig

from tests.ragas.conftest import save_metric_to_report
from tests.ragas.thresholds import MetricStatus, RAGAS_THRESHOLDS


# ====================================================================
# CONTEXT PRECISION — ¿Son relevantes los documentos recuperados?
# ====================================================================

def test_context_precision_above_threshold(
    evaluation_dataset,
    ragas_llm,
    ragas_embeddings,
):
    """
    Verifica que el sistema recupere documentos relevantes con alta precisión.

    Context Precision mide qué fracción de los documentos recuperados
    son realmente útiles para responder la pregunta (signal vs. noise).

    Criterios (ver thresholds.py):
      PASS  : >= 0.75
      WARN  : [0.65, 0.75)  — zona gris, requiere revisión manual
      FAIL  : < 0.65

    Un score bajo indica que retrieve_documents() o retrieve_with_filters()
    están devolviendo documentos no pertinentes — revisar k, fetch_k y
    la calidad del texto indexado.
    """
    run_config = RunConfig(timeout=180, max_workers=1, max_retries=2)

    metric_results = evaluate(
        evaluation_dataset,
        metrics=[context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    df            = metric_results.to_pandas()
    scores        = df["context_precision"].tolist()
    avg_precision = float(df["context_precision"].mean())

    cfg    = RAGAS_THRESHOLDS["context_precision"]
    status = save_metric_to_report(
        metric_name="context_precision",
        metric_category="retrieval",
        average_score=avg_precision,
        individual_scores=scores,
        threshold=cfg.accept_threshold,
        passed=(avg_precision >= cfg.accept_threshold),
        report_filename="ragas_metrics_report_ask.json",
        endpoint="ask",
    )

    assert status != MetricStatus.FAIL, (
        f"Context precision FAIL: {avg_precision:.3f} < {cfg.reject_threshold} "
        f"(reject threshold). Individual scores: {[round(s, 3) for s in scores]}"
    )


# ====================================================================
# CONTEXT RECALL — ¿Se recuperó toda la información relevante?
# ====================================================================

def test_context_recall_above_threshold(
    evaluation_dataset,
    ragas_llm,
    ragas_embeddings,
):
    """
    Verifica que el sistema recupere suficiente información relevante.

    Criterios (ver thresholds.py):
      PASS  : >= 0.65
      WARN  : [0.50, 0.65)
      FAIL  : < 0.50

    Un score bajo indica que k es demasiado pequeño o que los embeddings
    no capturan correctamente la semántica. Probar aumentar k/fetch_k
    vía EVAL_K / EVAL_FETCH_K env vars.
    """
    run_config = RunConfig(timeout=180, max_workers=1, max_retries=2)

    metric_results = evaluate(
        evaluation_dataset,
        metrics=[context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    df         = metric_results.to_pandas()
    scores     = df["context_recall"].tolist()
    avg_recall = float(df["context_recall"].mean())

    cfg    = RAGAS_THRESHOLDS["context_recall"]
    status = save_metric_to_report(
        metric_name="context_recall",
        metric_category="retrieval",
        average_score=avg_recall,
        individual_scores=scores,
        threshold=cfg.accept_threshold,
        passed=(avg_recall >= cfg.accept_threshold),
        report_filename="ragas_metrics_report_ask.json",
        endpoint="ask",
    )

    assert status != MetricStatus.FAIL, (
        f"Context recall FAIL: {avg_recall:.3f} < {cfg.reject_threshold} "
        f"(reject threshold). Individual scores: {[round(s, 3) for s in scores]}"
    )
