"""
Tests de Métricas de Generación — /ask endpoint
================================================

Evalúa la calidad del GenerationService usando métricas RAGAS basadas en LLM.

MÉTRICAS:
  - faithfulness       : ¿La respuesta se basa solo en el contexto sin alucinar?
  - answer_relevancy   : ¿La respuesta es relevante y enfocada en la pregunta?
  - answer_correctness : ¿La respuesta es correcta según el ground truth?

MLFLOW:
  El logging a MLflow es transparente — gestionado por el fixture mlflow_run
  (autouse=True en conftest.py) y save_metric_to_report().
  No se necesita importar mlflow en este archivo.

DATASET:
  Usa evaluation_dataset (fixture de conftest.py):
  8 preguntas sobre el mercado inmobiliario de Montevideo (/ask endpoint).

UMBRALES:
  Los thresholds (0.1) son conservadores para la primera iteración.
  Faithfulness debería ser alto (0.7+) una vez el sistema esté calibrado
  — valores bajos indican alucinación que hay que priorizar.
"""

import pytest
from ragas.metrics import answer_correctness, answer_relevancy, faithfulness
from ragas import evaluate
from ragas.run_config import RunConfig

from tests.ragas.conftest import save_metric_to_report
from tests.ragas.thresholds import MetricStatus, RAGAS_THRESHOLDS


# ====================================================================
# FAITHFULNESS — ¿La respuesta se basa solo en el contexto?
# ====================================================================

def test_faithfulness_no_hallucination(
    evaluation_dataset,
    ragas_llm,
    ragas_embeddings,
):
    """
    Verifica que el sistema NO alucine (se base solo en el contexto recuperado).

    Criterios (ver thresholds.py):
      PASS  : >= 0.85
      WARN  : [0.75, 0.85)
      FAIL  : < 0.75

    Para el sistema inmobiliario, la alucinación es especialmente peligrosa:
    inventar precios, amenities o características puede afectar decisiones reales.
    Un FAIL en faithfulness debe bloquear el despliegue.
    """
    run_config = RunConfig(timeout=180, max_workers=1, max_retries=2)

    metric_results = evaluate(
        evaluation_dataset,
        metrics=[faithfulness],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    df               = metric_results.to_pandas()
    scores           = df["faithfulness"].tolist()
    avg_faithfulness = float(df["faithfulness"].mean())

    cfg    = RAGAS_THRESHOLDS["faithfulness"]
    status = save_metric_to_report(
        metric_name="faithfulness",
        metric_category="generation",
        average_score=avg_faithfulness,
        individual_scores=scores,
        threshold=cfg.accept_threshold,
        passed=(avg_faithfulness >= cfg.accept_threshold),
        report_filename="ragas_metrics_report_ask.json",
        endpoint="ask",
    )

    assert status != MetricStatus.FAIL, (
        f"Faithfulness FAIL: {avg_faithfulness:.3f} < {cfg.reject_threshold}. "
        f"Individual scores: {[round(s, 3) for s in scores]}"
    )


# ====================================================================
# ANSWER RELEVANCY — ¿La respuesta es relevante para la pregunta?
# ====================================================================

def test_answer_relevancy_focused_responses(
    evaluation_dataset,
    ragas_llm,
    ragas_embeddings,
):
    """
    Verifica que las respuestas sean relevantes y enfocadas en la pregunta.

    Criterios (ver thresholds.py):
      PASS  : >= 0.78
      WARN  : [0.65, 0.78)
      FAIL  : < 0.65

    Score bajo puede indicar que las instrucciones de seguridad del prompt
    están siendo demasiado restrictivas al punto de no responder directamente.
    """
    run_config = RunConfig(timeout=180, max_workers=1, max_retries=2)

    metric_results = evaluate(
        evaluation_dataset,
        metrics=[answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    df            = metric_results.to_pandas()
    scores        = df["answer_relevancy"].tolist()
    avg_relevancy = float(df["answer_relevancy"].mean())

    cfg    = RAGAS_THRESHOLDS["answer_relevancy"]
    status = save_metric_to_report(
        metric_name="answer_relevancy",
        metric_category="generation",
        average_score=avg_relevancy,
        individual_scores=scores,
        threshold=cfg.accept_threshold,
        passed=(avg_relevancy >= cfg.accept_threshold),
        report_filename="ragas_metrics_report_ask.json",
        endpoint="ask",
    )

    assert status != MetricStatus.FAIL, (
        f"Answer relevancy FAIL: {avg_relevancy:.3f} < {cfg.reject_threshold}. "
        f"Individual scores: {[round(s, 3) for s in scores]}"
    )


# ====================================================================
# ANSWER CORRECTNESS — ¿La respuesta es correcta según ground truth?
# ====================================================================

def test_answer_correctness_accuracy(
    evaluation_dataset,
    ragas_llm,
    ragas_embeddings,
):
    """
    Verifica que las respuestas sean correctas según el ground truth.

    Criterios (ver thresholds.py):
      PASS  : >= 0.60
      WARN  : [0.35, 0.60)
      FAIL  : < 0.35

    Las referencias en evaluation_data.py son de nivel alto (no verbatim),
    lo que puede producir scores moderados. Afinar las referencias con
    respuestas reales del sistema para evaluaciones más precisas.
    """
    run_config = RunConfig(timeout=180, max_workers=1, max_retries=2)

    metric_results = evaluate(
        evaluation_dataset,
        metrics=[answer_correctness],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    df              = metric_results.to_pandas()
    scores          = df["answer_correctness"].tolist()
    avg_correctness = float(df["answer_correctness"].mean())

    cfg    = RAGAS_THRESHOLDS["answer_correctness"]
    status = save_metric_to_report(
        metric_name="answer_correctness",
        metric_category="generation",
        average_score=avg_correctness,
        individual_scores=scores,
        threshold=cfg.accept_threshold,
        passed=(avg_correctness >= cfg.accept_threshold),
        report_filename="ragas_metrics_report_ask.json",
        endpoint="ask",
    )

    assert status != MetricStatus.FAIL, (
        f"Answer correctness FAIL: {avg_correctness:.3f} < {cfg.reject_threshold}. "
        f"Individual scores: {[round(s, 3) for s in scores]}"
    )
