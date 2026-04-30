"""
run_experiment.py — Comparador de Configuraciones RAG con MLflow
================================================================

Script standalone (sin pytest) para ejecutar múltiples configuraciones
del sistema RAG, medir métricas RAGAS y comparar runs en MLflow.

USO:
    # Correr todas las configuraciones definidas en EXPERIMENT_CONFIGS:
    python tests/ragas/run_experiment.py

    # Correr solo una configuración específica:
    python tests/ragas/run_experiment.py --config baseline_k3

    # Correr solo el endpoint /ask o /recommend:
    python tests/ragas/run_experiment.py --endpoint ask
    python tests/ragas/run_experiment.py --endpoint recommend

    # Ver resultados en la UI de MLflow:
    mlflow ui --port 5000

DISEÑO:
    - Cada configuración en EXPERIMENT_CONFIGS genera un run separado en MLflow.
    - Dentro de cada run se evalúan todas las métricas RAGAS configuradas.
    - Los datasets se regeneran por cada configuración (no se usa caché de pytest)
      para que k, fetch_k y temperature se reflejen correctamente.
    - Al final imprime una tabla comparativa de todos los runs del experimento.

AGREGA NUEVAS CONFIGURACIONES:
    Añadir una entrada a EXPERIMENT_CONFIGS con los parámetros que quieras comparar.
    El nombre del run en MLflow lo construye build_run_name().
"""

import sys
import os
import time
import json
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Asegurar que el repo root está en el path antes de importar módulos de la app
_REPO_ROOT = Path(__file__).resolve().parents[2]  # tests/ragas/ → repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mlflow
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,
)
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService, PropertyFilters
from app.services.generation_service import GenerationService

from tests.ragas.evaluation_data import ASK_QUESTIONS, RECOMMEND_QUESTIONS
from tests.ragas.mlflow_utils import (
    setup_mlflow_experiment,
    build_run_name,
    log_experiment_params,
    log_ragas_metrics,
    log_dataset_info,
)

# RAGGraphService opcional
try:
    from app.services.rag_graph_service import RAGGraphService
    _HAS_RAG_GRAPH = True
except ImportError:
    _HAS_RAG_GRAPH = False


# ====================================================================
# CONFIGURACIONES DE EXPERIMENTO
# ====================================================================
# Añadir o modificar configuraciones aquí para comparar distintos setups.
# Cada entrada genera un run separado en MLflow.
# ====================================================================

EXPERIMENT_CONFIGS: List[Dict[str, Any]] = [
    {
        "name":                   "baseline_k3",
        "k":                      3,
        "fetch_k":                60,
        "max_recommendations":    5,
        "prompt_variant":         "default",
        "description_truncation": "full",
        "llm_model":              "gemini-2.5-flash",
        "embedding_model":        "models/gemini-embedding-001",
        "temperature":            0.0,
        "collection":             "realstate_mvd",
        "faiss_path":             "./faiss_index/realstate_mvd",
    },
    {
        "name":                   "wider_k5",
        "k":                      5,
        "fetch_k":                100,
        "max_recommendations":    5,
        "prompt_variant":         "default",
        "description_truncation": "full",
        "llm_model":              "gemini-2.5-flash",
        "embedding_model":        "models/gemini-embedding-001",
        "temperature":            0.0,
        "collection":             "realstate_mvd",
        "faiss_path":             "./faiss_index/realstate_mvd",
    },
    {
        "name":                   "wider_k10",
        "k":                      10,
        "fetch_k":                200,
        "max_recommendations":    5,
        "prompt_variant":         "default",
        "description_truncation": "full",
        "llm_model":              "gemini-2.5-flash",
        "embedding_model":        "models/gemini-embedding-001",
        "temperature":            0.0,
        "collection":             "realstate_mvd",
        "faiss_path":             "./faiss_index/realstate_mvd",
    },
]

# Métricas RAGAS a evaluar en todos los runs
RAGAS_METRICS = [
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,
]

RAGAS_RUN_CONFIG = RunConfig(
    timeout=180,
    max_workers=1,   # MUST be 1 — respetar 5 RPM de Gemini
    max_retries=2,
)


# ====================================================================
# INICIALIZACIÓN DE SERVICIOS
# ====================================================================

def _init_services(
    config: Dict[str, Any],
) -> Tuple[Optional[Any], RetrievalService, GenerationService]:
    """
    Inicializa los servicios RAG para una configuración dada.

    Returns:
        (ask_rag_system_or_None, retrieval_service, generation_service)
    """
    embedding_service = EmbeddingService()
    faiss_path = config["faiss_path"]
    print(f"  Loading FAISS index from {faiss_path}...")
    embedding_service.load_vectorstore(faiss_path)

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        k=config["k"],
        fetch_k=config["fetch_k"],
    )
    generation_service = GenerationService(
        model=config["llm_model"],
        temperature=config["temperature"],
    )

    ask_system = None
    if _HAS_RAG_GRAPH:
        ask_system = RAGGraphService(
            retrieval_service=retrieval_service,
            generation_service=generation_service,
            use_query_rewriting=False,
            use_reranking=False,
        )

    return ask_system, retrieval_service, generation_service


# ====================================================================
# CONSTRUCCIÓN DE DATASETS
# ====================================================================

def _build_ask_dataset(
    ask_system,
    retrieval_service: RetrievalService,
    generation_service: GenerationService,
) -> EvaluationDataset:
    """
    Construye el EvaluationDataset para /ask ejecutando las preguntas
    contra el sistema RAG.

    Si RAGGraphService no está disponible, cae back al pipeline simple
    retrieval + generation directamente.
    """
    print(f"  Building /ask dataset ({len(ASK_QUESTIONS)} questions)...")
    samples = []

    for idx, entry in enumerate(ASK_QUESTIONS, 1):
        print(f"    [{idx}/{len(ASK_QUESTIONS)}] {entry['question'][:60]}...")
        try:
            if ask_system is not None:
                result = ask_system.process_question(entry["question"])
                contexts = result["context"]
                answer   = result["answer"]
            else:
                # Fallback: pipeline directo sin LangGraph
                docs     = retrieval_service.retrieve_documents(entry["question"])
                result   = generation_service.generate_response(entry["question"], docs)
                contexts = result["context"]
                answer   = result["answer"]

            samples.append(SingleTurnSample(
                user_input         = entry["question"],
                retrieved_contexts = contexts,
                response           = answer,
                reference          = entry["reference"],
            ))
        except Exception as e:
            print(f"    [!] Error: {e}")

        if idx < len(ASK_QUESTIONS):
            time.sleep(7)  # 5 RPM rate limit

    return EvaluationDataset(samples=samples)


def _build_recommend_dataset(
    retrieval_service: RetrievalService,
    generation_service: GenerationService,
    max_recommendations: int,
) -> EvaluationDataset:
    """
    Construye el EvaluationDataset para /recommend ejecutando las solicitudes
    con PropertyFilters.
    """
    print(f"  Building /recommend dataset ({len(RECOMMEND_QUESTIONS)} questions)...")
    samples = []

    for idx, entry in enumerate(RECOMMEND_QUESTIONS, 1):
        print(f"    [{idx}/{len(RECOMMEND_QUESTIONS)}] {entry['question'][:60]}...")
        try:
            filters = PropertyFilters(**entry.get("filter_kwargs", {}))
            docs    = retrieval_service.retrieve_with_filters(entry["question"], filters)
            result  = generation_service.generate_recommendations(
                entry["question"], docs, max_recommendations=max_recommendations,
            )
            samples.append(SingleTurnSample(
                user_input         = entry["question"],
                retrieved_contexts = result["context"],
                response           = result["answer"],
                reference          = entry["reference"],
            ))
        except Exception as e:
            print(f"    [!] Error: {e}")

        if idx < len(RECOMMEND_QUESTIONS):
            time.sleep(7)

    return EvaluationDataset(samples=samples)


# ====================================================================
# EVALUACIÓN RAGAS
# ====================================================================

def _run_ragas_evaluation(
    dataset: EvaluationDataset,
    ragas_llm: LangchainLLMWrapper,
    ragas_embeddings: Any,
    endpoint: str,
) -> Dict[str, float]:
    """
    Corre RAGAS evaluate() sobre el dataset y retorna scores promedio.

    Returns:
        Dict {metric_name: avg_score}
    """
    print(f"  Running RAGAS evaluation for /{endpoint}...")
    metric_results = evaluate(
        dataset,
        metrics=RAGAS_METRICS,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RAGAS_RUN_CONFIG,
    )

    df = metric_results.to_pandas()
    metric_names = ["context_precision", "context_recall",
                    "faithfulness", "answer_relevancy", "answer_correctness"]

    scores_avg = {}
    for name in metric_names:
        if name in df.columns:
            avg = float(df[name].mean())
            individual = df[name].tolist()
            scores_avg[name] = avg
            # Loguear a MLflow
            log_ragas_metrics(name, avg, individual, endpoint=endpoint)
            print(f"    {endpoint}/{name}: {avg:.4f}")

    return scores_avg


# ====================================================================
# RUNNER PRINCIPAL
# ====================================================================

def run_all_experiments(
    configs: List[Dict[str, Any]],
    endpoint: str = "both",
    experiment_name: str = "realstate_rag_evaluation",
) -> None:
    """
    Ejecuta todos los experimentos y loguea resultados a MLflow.

    Args:
        configs         : Lista de configuraciones a comparar.
        endpoint        : "ask" | "recommend" | "both"
        experiment_name : Nombre del experimento MLflow.
    """
    experiment_id = setup_mlflow_experiment(experiment_name)

    all_run_results = []

    for config in configs:
        config_name = config.get("name", "unnamed")
        print(f"\n{'='*60}")
        print(f"  Config: {config_name}  (k={config['k']}, fetch_k={config['fetch_k']})")
        print(f"{'='*60}")

        run_name = build_run_name(config)

        # Inicializar modelos de evaluación (temperatura 0.0 siempre en eval)
        ragas_llm_raw = ChatGoogleGenerativeAI(
            model=config["llm_model"],
            temperature=0.0,
            request_timeout=120,
            max_retries=5,
        )
        ragas_llm  = LangchainLLMWrapper(langchain_llm=ragas_llm_raw)
        ragas_emb  = GoogleGenerativeAIEmbeddings(model=config["embedding_model"])

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            log_experiment_params(config)
            run_scores = {"run_id": run.info.run_id, "name": config_name}

            # Inicializar servicios RAG
            ask_system, retrieval_service, generation_service = _init_services(config)

            # /ask
            if endpoint in ("ask", "both"):
                ask_dataset = _build_ask_dataset(ask_system, retrieval_service, generation_service)
                log_dataset_info(len(ask_dataset.samples), endpoint="ask")
                ask_scores  = _run_ragas_evaluation(ask_dataset, ragas_llm, ragas_emb, endpoint="ask")
                run_scores.update({f"ask_{k}": v for k, v in ask_scores.items()})

            # /recommend
            if endpoint in ("recommend", "both"):
                rec_dataset = _build_recommend_dataset(
                    retrieval_service, generation_service, config["max_recommendations"]
                )
                log_dataset_info(len(rec_dataset.samples), endpoint="recommend")
                rec_scores  = _run_ragas_evaluation(rec_dataset, ragas_llm, ragas_emb, endpoint="recommend")
                run_scores.update({f"recommend_{k}": v for k, v in rec_scores.items()})

            all_run_results.append(run_scores)
            print(f"  Run {run.info.run_id} completed.")

    # ── Tabla comparativa ────────────────────────────────────────────────────
    _print_comparison_table(all_run_results, endpoint)

    # ── Guardar JSON de comparación ──────────────────────────────────────────
    comparison_path = Path("tests/ragas/experiment_comparison.json")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": experiment_name,
            "timestamp":  datetime.now().isoformat(),
            "endpoint":   endpoint,
            "runs":       all_run_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nComparison saved to {comparison_path}")
    print(f"View in MLflow UI: mlflow ui --port 5000")


def _print_comparison_table(
    results: List[Dict[str, Any]],
    endpoint: str,
) -> None:
    """
    Imprime una tabla de comparación de métricas entre configuraciones.
    """
    metric_keys = [
        "ask_context_precision", "ask_context_recall",
        "ask_faithfulness", "ask_answer_relevancy", "ask_answer_correctness",
        "recommend_context_precision", "recommend_context_recall",
        "recommend_faithfulness", "recommend_answer_relevancy", "recommend_answer_correctness",
    ]
    # Filtrar por endpoint
    if endpoint == "ask":
        metric_keys = [k for k in metric_keys if k.startswith("ask_")]
    elif endpoint == "recommend":
        metric_keys = [k for k in metric_keys if k.startswith("recommend_")]

    # Solo columnas que tienen datos
    available = [k for k in metric_keys if any(k in r for r in results)]

    print(f"\n{'─'*80}")
    print("EXPERIMENT COMPARISON")
    print(f"{'─'*80}")

    # Header
    header = f"{'Config':<20}" + "".join(f"{k.split('_', 1)[-1][:14]:>15}" for k in available)
    print(header)
    print("─" * len(header))

    # Rows
    for r in results:
        row = f"{r['name']:<20}"
        for k in available:
            val = r.get(k)
            row += f"{val:>15.4f}" if val is not None else f"{'N/A':>15}"
        print(row)

    print(f"{'─'*80}")


# ====================================================================
# ENTRYPOINT
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Corre experimentos de evaluación RAG y loguea métricas a MLflow."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Nombre de la configuración a correr (ej: 'baseline_k3'). "
             "Por defecto corre todas.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        choices=["ask", "recommend", "both"],
        default="both",
        help="Endpoint a evaluar (default: both).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="realstate_rag_evaluation",
        help="Nombre del experimento MLflow (default: realstate_rag_evaluation).",
    )
    args = parser.parse_args()

    # Filtrar configuraciones si se especificó una
    configs_to_run = EXPERIMENT_CONFIGS
    if args.config:
        configs_to_run = [c for c in EXPERIMENT_CONFIGS if c.get("name") == args.config]
        if not configs_to_run:
            print(f"Error: No se encontró la configuración '{args.config}'.")
            print(f"Configuraciones disponibles: {[c['name'] for c in EXPERIMENT_CONFIGS]}")
            sys.exit(1)

    run_all_experiments(
        configs=configs_to_run,
        endpoint=args.endpoint,
        experiment_name=args.experiment,
    )
