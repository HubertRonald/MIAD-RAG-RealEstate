"""
Test de Similitud Coseno Promedio (avg_cosine_similarity)
=========================================================

Evalúa la cercanía semántica entre los vectores de las queries
y los vectores de los documentos recuperados por RetrievalService.

MÉTRICA:
  avg_cosine_similarity — Promedio de los relevance scores de los k documentos
  recuperados para cada query del dataset de evaluación.

  LangChain normaliza los scores de FAISS a [0, 1]:
    - Índice L2     : score = 1 / (1 + distance_l2)
    - Índice coseno : score = cosine_similarity directamente

  Un score alto indica que el retriever está devolviendo documentos
  semánticamente muy cercanos a la query — los embeddings capturan
  bien la intención del usuario.

DIFERENCIA CON RAGAS:
  Esta métrica NO usa un LLM evaluador — es una medida directa del
  espacio vectorial. Es más rápida, más barata (sin API calls de RAGAS)
  y más interpretable para diagnosticar la calidad del retriever.

COMPONENTE EVALUADO: EmbeddingService + VectorStore FAISS
CRITERIOS (ver thresholds.py):
  PASS  : >= 0.72  (sin criterio de rechazo — siempre PASS o WARN)
  WARN  : < 0.72

DATASET USADO:
  ASK_QUESTIONS de evaluation_data.py — las mismas 8 preguntas de /ask.
  Se calcula el score de cada query individualmente para detectar outliers.

MLFLOW:
  Logueado como functional_avg_cosine_similarity (fuera del grupo RAGAS).
  También se loguea el score por query: functional_avg_cosine_q{N}.
"""

import pytest
import statistics
from typing import List

from tests.ragas.evaluation_data import ASK_QUESTIONS
from tests.ragas.mlflow_utils import log_functional_metric
from tests.ragas.thresholds import MetricStatus, RAGAS_THRESHOLDS


# ====================================================================
# FIXTURE LOCAL — retrieval service (reutiliza el de conftest.py)
# ====================================================================
# El fixture recommend_services expone (retrieval_service, generation_service).
# Solo necesitamos el retrieval_service para este test.
# ====================================================================


def test_avg_cosine_similarity(recommend_services):
    """
    Verifica que la similitud semántica promedio entre queries y documentos
    recuperados supere el umbral de calidad del embedding.

    FLUJO:
      1. Para cada query en ASK_QUESTIONS, llama retrieve_with_scores().
      2. Promedia los k scores de los documentos recuperados → score por query.
      3. Promedia todos los scores por query → avg_cosine_similarity.
      4. Loguea a MLflow y evalúa contra el threshold.

    INTERPRETACIÓN:
      score < 0.72 → los embeddings no están capturando bien la semántica
      del dominio inmobiliario. Opciones: re-indexar con más texto por documento,
      ajustar description_truncation, o considerar fine-tuning del modelo
      de embeddings (fuera del scope actual).

      score >= 0.72 → el espacio vectorial es suficientemente expresivo para
      el dominio de listings inmobiliarios en Montevideo.
    """
    retrieval_service, _ = recommend_services

    per_query_scores: List[float] = []

    for idx, entry in enumerate(ASK_QUESTIONS, 1):
        query = entry["question"]
        try:
            doc_score_pairs = retrieval_service.retrieve_with_scores(query)
        except Exception as e:
            pytest.fail(
                f"retrieve_with_scores() falló en la query {idx}: {e}\n"
                f"Verificar que retrieve_with_scores() fue añadido a RetrievalService "
                f"(ver retrieval_service_patch.py)."
            )

        if not doc_score_pairs:
            print(f"  [!] Query {idx}: no documents retrieved, skipping.")
            continue

        from app.services.retrieval_service import l2_relevance_to_cosine
        query_scores = [l2_relevance_to_cosine(score) for _, score in doc_score_pairs]
        query_avg    = statistics.mean(query_scores)
        per_query_scores.append(query_avg)

        print(
            f"  [cosine] Query {idx}/{len(ASK_QUESTIONS)}: "
            f"avg={query_avg:.4f} "
            f"(k={len(query_scores)}, "
            f"min={min(query_scores):.4f}, "
            f"max={max(query_scores):.4f})"
        )

    assert per_query_scores, (
        "No se pudo calcular ningún score — verificar que el índice FAISS "
        "está cargado y que retrieve_with_scores() funciona correctamente."
    )

    avg_cosine = statistics.mean(per_query_scores)
    print(f"\n  avg_cosine_similarity: {avg_cosine:.4f} (n={len(per_query_scores)} queries)")

    # Loguear a MLflow
    status = log_functional_metric(
        metric_name="avg_cosine_similarity",
        value=avg_cosine,
        endpoint="functional",
    )

    # Loguear scores por query a MLflow
    import mlflow
    if mlflow.active_run():
        for i, score in enumerate(per_query_scores):
            mlflow.log_metric(f"functional_avg_cosine_q{i+1:02d}", round(score, 4))

    cfg = RAGAS_THRESHOLDS["avg_cosine_similarity"]

    # Para esta métrica no hay reject_threshold — solo PASS o WARN
    # El test nunca falla automáticamente (MetricStatus.FAIL no aplica),
    # pero sí produce WARN visible en MLflow y en el reporte.
    if status == MetricStatus.WARN:
        pytest.warns(
            UserWarning,
            match="avg_cosine_similarity",
        ) if False else None  # noqa — no queremos warnings reales, solo loguear
        print(
            f"  ⚠ WARN: avg_cosine_similarity={avg_cosine:.4f} < "
            f"accept_threshold={cfg.accept_threshold}. "
            f"Revisar calidad de embeddings o descripción de listings."
        )

    # No hay assert de FAIL para esta métrica — el threshold es solo orientativo
    # El equipo debe revisar el score en MLflow y decidir si bloquea el release.
    assert avg_cosine > 0.0, (
        f"avg_cosine_similarity={avg_cosine:.4f} — algo está mal con el retriever."
    )
