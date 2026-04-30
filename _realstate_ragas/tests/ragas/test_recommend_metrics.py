"""
Tests de Métricas — /recommend endpoint
========================================

Evalúa la calidad del pipeline de recomendaciones:
  PropertyFilters → RetrievalService → GenerationService

MÉTRICAS (mismas que /ask, dataset distinto):
  - context_precision  : ¿Los listings recuperados son relevantes para la solicitud?
  - context_recall     : ¿Se recuperaron todos los listings relevantes disponibles?
  - faithfulness       : ¿La recomendación se basa solo en los listings recuperados?
  - answer_relevancy   : ¿La recomendación responde a lo que pidió el cliente?
  - answer_correctness : ¿La recomendación es correcta según la referencia esperada?

DATASET:
  Usa evaluation_dataset_recommend (fixture de conftest.py):
  5 solicitudes de recomendación con PropertyFilters pre-definidos.

MLFLOW:
  Métricas logueadas con prefijo "recommend_" (vs "ask_" en los otros tests).
  Permite comparar ambos endpoints dentro del mismo run de MLflow.

REPORTE:
  Separado en ragas_metrics_report_recommend.json para no mezclar con /ask.
"""

import pytest
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,
)
from ragas import evaluate
from ragas.run_config import RunConfig

from tests.ragas.conftest import save_metric_to_report


# Configuración común para todos los tests de este módulo
_RUN_CONFIG = RunConfig(
    timeout=180,
    max_workers=1,   # MUST be 1 to stay under 5 RPM
    max_retries=2,
)

_THRESHOLD      = 0.1          # Línea base conservadora
_REPORT         = "ragas_metrics_report_recommend.json"
_ENDPOINT       = "recommend"


# ====================================================================
# RETRIEVAL — Context Precision
# ====================================================================

def test_recommend_context_precision(
    evaluation_dataset_recommend,
    ragas_llm,
    ragas_embeddings,
):
    """
    ¿Los listings recuperados son relevantes para la solicitud del cliente?

    Context Precision es especialmente importante para /recommend:
    un listing irrelevante (barrio equivocado, tipo incorrecto, precio
    fuera de rango) en el top-k reduce la calidad de la recomendación.
    Si el score es bajo, revisar que PropertyFilters esté filtrando
    correctamente o aumentar fetch_k para ampliar el espacio de búsqueda.
    """
    results  = evaluate(
        evaluation_dataset_recommend,
        metrics=[context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=_RUN_CONFIG,
    )
    df     = results.to_pandas()
    scores = df["context_precision"].tolist()
    avg    = float(df["context_precision"].mean())
    passed = avg >= _THRESHOLD

    save_metric_to_report(
        metric_name="context_precision",
        metric_category="retrieval",
        average_score=avg,
        individual_scores=scores,
        threshold=_THRESHOLD,
        passed=passed,
        report_filename=_REPORT,
        endpoint=_ENDPOINT,
    )
    assert avg >= _THRESHOLD, (
        f"[recommend] Context precision: {avg:.3f} < {_THRESHOLD}. "
        f"Scores: {[round(s, 3) for s in scores]}"
    )


# ====================================================================
# RETRIEVAL — Context Recall
# ====================================================================

def test_recommend_context_recall(
    evaluation_dataset_recommend,
    ragas_llm,
    ragas_embeddings,
):
    """
    ¿Se recuperaron todos los listings relevantes para la solicitud?

    Un recall bajo puede indicar que los filtros en PropertyFilters son
    demasiado restrictivos, o que k es pequeño para este tipo de búsqueda.
    Probar incrementar k vía EVAL_K o relajar algún filtro de amenities.
    """
    results  = evaluate(
        evaluation_dataset_recommend,
        metrics=[context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=_RUN_CONFIG,
    )
    df     = results.to_pandas()
    scores = df["context_recall"].tolist()
    avg    = float(df["context_recall"].mean())
    passed = avg >= _THRESHOLD

    save_metric_to_report(
        metric_name="context_recall",
        metric_category="retrieval",
        average_score=avg,
        individual_scores=scores,
        threshold=_THRESHOLD,
        passed=passed,
        report_filename=_REPORT,
        endpoint=_ENDPOINT,
    )
    assert avg >= _THRESHOLD, (
        f"[recommend] Context recall: {avg:.3f} < {_THRESHOLD}. "
        f"Scores: {[round(s, 3) for s in scores]}"
    )


# ====================================================================
# GENERATION — Faithfulness
# ====================================================================

def test_recommend_faithfulness(
    evaluation_dataset_recommend,
    ragas_llm,
    ragas_embeddings,
):
    """
    ¿Las recomendaciones se basan solo en los listings proporcionados?

    Faithfulness es crítico para /recommend: el modelo no debe inventar
    amenities, precios ni características que no están en los listings
    recuperados. Un score bajo es una señal de alerta inmediata.
    """
    results  = evaluate(
        evaluation_dataset_recommend,
        metrics=[faithfulness],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=_RUN_CONFIG,
    )
    df     = results.to_pandas()
    scores = df["faithfulness"].tolist()
    avg    = float(df["faithfulness"].mean())
    passed = avg >= _THRESHOLD

    save_metric_to_report(
        metric_name="faithfulness",
        metric_category="generation",
        average_score=avg,
        individual_scores=scores,
        threshold=_THRESHOLD,
        passed=passed,
        report_filename=_REPORT,
        endpoint=_ENDPOINT,
    )
    assert avg >= _THRESHOLD, (
        f"[recommend] Faithfulness: {avg:.3f} < {_THRESHOLD}. "
        f"Scores: {[round(s, 3) for s in scores]}"
    )


# ====================================================================
# GENERATION — Answer Relevancy
# ====================================================================

def test_recommend_answer_relevancy(
    evaluation_dataset_recommend,
    ragas_llm,
    ragas_embeddings,
):
    """
    ¿Las recomendaciones son relevantes para lo que pidió el cliente?

    Answer Relevancy evalúa si la respuesta aborda directamente la solicitud.
    Para /recommend, score bajo puede indicar que el modelo está describiendo
    listings pero no conectando con los criterios específicos del cliente.
    """
    results  = evaluate(
        evaluation_dataset_recommend,
        metrics=[answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=_RUN_CONFIG,
    )
    df     = results.to_pandas()
    scores = df["answer_relevancy"].tolist()
    avg    = float(df["answer_relevancy"].mean())
    passed = avg >= _THRESHOLD

    save_metric_to_report(
        metric_name="answer_relevancy",
        metric_category="generation",
        average_score=avg,
        individual_scores=scores,
        threshold=_THRESHOLD,
        passed=passed,
        report_filename=_REPORT,
        endpoint=_ENDPOINT,
    )
    assert avg >= _THRESHOLD, (
        f"[recommend] Answer relevancy: {avg:.3f} < {_THRESHOLD}. "
        f"Scores: {[round(s, 3) for s in scores]}"
    )


# ====================================================================
# GENERATION — Answer Correctness
# ====================================================================

def test_recommend_answer_correctness(
    evaluation_dataset_recommend,
    ragas_llm,
    ragas_embeddings,
):
    """
    ¿Las recomendaciones son correctas según las referencias esperadas?

    Las referencias en evaluation_data.py describen qué debe contener
    una buena recomendación (no listings específicos). Scores moderados
    (0.3–0.5) son aceptables con estas referencias de alto nivel.
    """
    results  = evaluate(
        evaluation_dataset_recommend,
        metrics=[answer_correctness],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=_RUN_CONFIG,
    )
    df     = results.to_pandas()
    scores = df["answer_correctness"].tolist()
    avg    = float(df["answer_correctness"].mean())
    passed = avg >= _THRESHOLD

    save_metric_to_report(
        metric_name="answer_correctness",
        metric_category="generation",
        average_score=avg,
        individual_scores=scores,
        threshold=_THRESHOLD,
        passed=passed,
        report_filename=_REPORT,
        endpoint=_ENDPOINT,
    )
    assert avg >= _THRESHOLD, (
        f"[recommend] Answer correctness: {avg:.3f} < {_THRESHOLD}. "
        f"Scores: {[round(s, 3) for s in scores]}"
    )
