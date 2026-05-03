"""
Fixtures de Evaluación — Sistema RAG Inmobiliario Montevideo
============================================================

Reemplaza el conftest.py original (orientado a PDF/Markdown) con fixtures
adaptadas al sistema de listings inmobiliarios.

FIXTURES PRINCIPALES
────────────────────
  experiment_config         — Parámetros del experimento (env vars o defaults).
  mlflow_run                — Ciclo de vida del run MLflow (autouse, session).
  ragas_llm                 — LLM RAGAS (gemini-2.5-flash, temperature=0.0).
  ragas_embeddings          — Embeddings RAGAS (gemini-embedding-001).
  ask_rag_system            — RAGGraphService para /ask.
  recommend_services        — (RetrievalService, GenerationService) para /recommend.
  evaluation_dataset        — Dataset RAGAS para /ask (caché JSON).
  evaluation_dataset_recommend — Dataset RAGAS para /recommend (caché JSON).

MLFLOW — DISEÑO
────────────────
  mlflow_run tiene autouse=True y scope="session":
    - Arranca el run MLflow al inicio de la sesión de pytest.
    - Lo cierra al final (yield fixture con context manager).
    - Los tests NO necesitan pedir este fixture explícitamente.
    - save_metric_to_report() detecta el run activo via mlflow.active_run()
      y loguea automáticamente — los test files no necesitan cambios.

PARÁMETROS DEL EXPERIMENTO (env vars)
──────────────────────────────────────
  EVAL_K                    — Documentos a recuperar (default: 3)
  EVAL_FETCH_K              — Candidatos vectoriales pre-filtrado (default: 60)
  EVAL_MAX_RECOMMENDATIONS  — Máx. recomendaciones /recommend (default: 5)
  EVAL_PROMPT_VARIANT       — Variante del prompt (default: "default")
  EVAL_DESCRIPTION_TRUNCATION — Truncado de descripciones (default: "full")
  EVAL_LLM_MODEL            — Modelo de generación (default: "gemini-2.5-flash")
  EVAL_EMBEDDING_MODEL      — Modelo de embeddings (default: "models/gemini-embedding-001")
  EVAL_TEMPERATURE          — Temperatura de evaluación (default: 0.0)
  EVAL_COLLECTION           — Nombre de la colección FAISS (default: "realstate_mvd")
  EVAL_FAISS_PATH           — Ruta al índice FAISS (default: "./faiss_index/realstate_mvd")
  EVAL_ENDPOINT             — Endpoint a evaluar: "ask" | "recommend" | "both" (default: "ask")
  MLFLOW_EXPERIMENT_NAME    — Nombre del experimento MLflow (default: "realstate_rag_evaluation")
  MLFLOW_TRACKING_URI       — URI del tracking server (default: "mlruns")
"""

import pytest
import warnings
import json
import os
import time
import mlflow

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()   

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message=".*position_ids.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

from ragas import SingleTurnSample, EvaluationDataset
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
    log_evaluation_dataset,
    log_model_info,
    log_model_artifact,
    print_run_summary,
)
from tests.ragas.thresholds import (
    MetricStatus,
    get_threshold,
    RAGAS_THRESHOLDS,
)

# Importaciones opcionales — servicios PENDIENTE en el diagrama de arquitectura
try:
    from app.services.rag_graph_service import RAGGraphService
    _HAS_RAG_GRAPH = True
except ImportError:
    _HAS_RAG_GRAPH = False

try:
    from app.services.query_rewriting_service import QueryRewritingService
    _HAS_QUERY_REWRITING = True
except ImportError:
    _HAS_QUERY_REWRITING = False

try:
    from app.services.reranking_service import RerankingService
    _HAS_RERANKING = True
except ImportError:
    _HAS_RERANKING = False


# ====================================================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ====================================================================

@pytest.fixture(scope="session")
def experiment_config() -> Dict[str, Any]:
    """
    Parámetros del experimento leídos desde variables de entorno.

    Permite ejecutar la misma suite de tests con distintas configuraciones
    sin modificar código:
        EVAL_K=5 EVAL_FETCH_K=100 pytest tests/ragas/

    Returns:
        Diccionario con todos los parámetros del experimento.
        Se loguea íntegro a MLflow al inicio del run.
    """
    return {
        "k":                     int(os.getenv("EVAL_K", "3")),
        "fetch_k":               int(os.getenv("EVAL_FETCH_K", "60")),
        "max_recommendations":   int(os.getenv("EVAL_MAX_RECOMMENDATIONS", "5")),
        "prompt_variant":        os.getenv("EVAL_PROMPT_VARIANT", "default"),
        "description_truncation": os.getenv("EVAL_DESCRIPTION_TRUNCATION", "full"),
        "llm_model":             os.getenv("EVAL_LLM_MODEL", "gemini-2.5-flash"),
        "embedding_model":       os.getenv("EVAL_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        "temperature":           float(os.getenv("EVAL_TEMPERATURE", "0.0")),
        "collection":            os.getenv("EVAL_COLLECTION", "realstate_mvd"),
        "faiss_path":            os.getenv("EVAL_FAISS_PATH", "./faiss_index/realstate_mvd"),
        "endpoint":              os.getenv("EVAL_ENDPOINT", "ask"),
    }


# ====================================================================
# MLFLOW RUN — autouse, gestiona el ciclo de vida completo
# ====================================================================

@pytest.fixture(scope="session", autouse=True)
def mlflow_run(experiment_config):
    """
    Gestiona el ciclo de vida del run MLflow para toda la sesión de pytest.

    autouse=True → se activa automáticamente sin que los tests lo pidan.
    scope="session" → un único run por ejecución de pytest.

    Flujo:
      1. Crea/recupera el experimento MLflow.
      2. Inicia el run con nombre descriptivo.
      3. Loguea todos los parámetros del experimento.
      4. yield → los tests corren con el run activo.
      5. Al finalizar la sesión, cierra el run (éxito o fallo).

    Los tests no necesitan conocer MLflow — save_metric_to_report()
    detecta el run activo vía mlflow.active_run() y loguea automáticamente.
    """
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "realstate_rag_evaluation")
    tracking_uri    = os.getenv("MLFLOW_TRACKING_URI", "mlruns")

    experiment_id = setup_mlflow_experiment(experiment_name, tracking_uri)
    run_name      = build_run_name(experiment_config)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        log_experiment_params(experiment_config)
        log_model_info(experiment_config)
        log_model_artifact(experiment_config)
        print(f"\n[MLflow] Run started: {run.info.run_id}")
        print(f"[MLflow] Run name: {run_name}")
        print(f"[MLflow] Config: {experiment_config}")
        yield run
    print_run_summary(run.info.run_id)
    print(f"\n[MLflow] Run finished: {run.info.run_id}")


# ====================================================================
# MODELOS DE EVALUACIÓN
# ====================================================================

@pytest.fixture(scope="session")
def ragas_llm(experiment_config):
    """
    LLM configurado para evaluación con RAGAS.

    IMPORTANTE: Usa temperature=0.0 (desde experiment_config) para
    garantizar reproducibilidad en evaluaciones. La producción usa 0.2.

    Límites de la API gratuita de Gemini: 5 RPM, 250k TPM, 20 RPD.
    El RunConfig en los tests ya usa max_workers=1 para respetar el RPM.

    Returns:
        LangchainLLMWrapper: LLM listo para evaluación RAGAS.
    """
    langchain_llm = ChatGoogleGenerativeAI(
        model=experiment_config["llm_model"],
        temperature=experiment_config["temperature"],
        request_timeout=120,
        max_retries=5,
    )
    return LangchainLLMWrapper(langchain_llm=langchain_llm)


@pytest.fixture(scope="session")
def ragas_embeddings(experiment_config):
    """
    Modelo de embeddings para evaluación con RAGAS.

    DEBE ser el mismo modelo usado para indexar documentos en FAISS.
    Cambiar este modelo invalida la comparabilidad con el índice existente.

    Returns:
        GoogleGenerativeAIEmbeddings: Modelo configurado para evaluación.
    """
    return GoogleGenerativeAIEmbeddings(model=experiment_config["embedding_model"])


# ====================================================================
# SERVICIOS RAG
# ====================================================================

@pytest.fixture(scope="session")
def _embedding_service(experiment_config):
    """
    EmbeddingService con el índice FAISS cargado.

    Fixture privada (prefijo _) — compartida entre ask_rag_system y
    recommend_services para no cargar el índice dos veces.

    Raises:
        pytest.skip: Si el índice FAISS no existe en la ruta configurada.
    """
    service = EmbeddingService()
    faiss_path = experiment_config["faiss_path"]
    try:
        service.load_vectorstore(faiss_path)
    except FileNotFoundError:
        pytest.skip(
            f"Índice FAISS no encontrado en '{faiss_path}'. "
            f"Ejecutar primero el endpoint /load-from-csv para construir la colección "
            f"'{experiment_config['collection']}'."
        )
    return service


@pytest.fixture(scope="session")
def ask_rag_system(experiment_config, _embedding_service):
    """
    RAGGraphService configurado para evaluación del endpoint /ask.

    Usa temperature=0.0 desde experiment_config para reproducibilidad.
    QueryRewritingService y RerankingService se inyectan si están disponibles;
    si no, se instancia un RAGGraphService básico (solo retrieval + generation).

    Returns:
        RAGGraphService listo para llamar process_question().

    Raises:
        pytest.skip: Si RAGGraphService no está disponible.
    """
    if not _HAS_RAG_GRAPH:
        pytest.skip("RAGGraphService no disponible. Verificar app/services/rag_graph_service.py.")

    retrieval_service = RetrievalService(
        embedding_service=_embedding_service,
        k=experiment_config["k"],
        fetch_k=experiment_config["fetch_k"],
    )
    generation_service = GenerationService(
        model=experiment_config["llm_model"],
        temperature=experiment_config["temperature"],  # 0.0 para evaluación
    )

    kwargs = dict(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        use_query_rewriting=False,
        use_reranking=False,
    )

    if _HAS_QUERY_REWRITING:
        kwargs["query_rewriting_service"] = QueryRewritingService()
        kwargs["rewriting_strategy"] = "few_shot"


    # To evaluate with reranking active: EVAL_USE_RERANKING=true pytest tests/ragas/
    use_reranking = os.getenv("EVAL_USE_RERANKING", "false").lower() == "true"
    if _HAS_RERANKING and use_reranking:
        kwargs["reranking_service"] = RerankingService(top_k=3)
        kwargs["use_reranking"] = True

    return RAGGraphService(**kwargs)


@pytest.fixture(scope="session")
def recommend_services(
    experiment_config,
    _embedding_service,
) -> Tuple[RetrievalService, GenerationService]:
    """
    Servicios de retrieval y generación para evaluación del endpoint /recommend.

    Returns:
        Tuple (RetrievalService, GenerationService) listo para usar.
    """
    retrieval_service = RetrievalService(
        embedding_service=_embedding_service,
        k=experiment_config["k"],
        fetch_k=experiment_config["fetch_k"],
    )
    generation_service = GenerationService(
        model=experiment_config["llm_model"],
        temperature=experiment_config["temperature"],  # 0.0 para evaluación
    )
    return retrieval_service, generation_service


# ====================================================================
# DATASETS DE EVALUACIÓN — /ask
# ====================================================================

@pytest.fixture(scope="session")
def evaluation_dataset(ask_rag_system, experiment_config) -> EvaluationDataset:
    """
    Dataset RAGAS para evaluar el endpoint /ask.

    Construye SingleTurnSamples ejecutando cada pregunta de ASK_QUESTIONS
    contra el sistema RAG y capturando: pregunta, contextos recuperados,
    respuesta generada y referencia esperada.

    CACHÉ JSON: Guarda el dataset después de construirlo. En ejecuciones
    posteriores, carga desde caché para evitar llamadas a la API y consumir
    cuota innecesariamente. Eliminar el archivo de caché para forzar
    regeneración (útil al cambiar preguntas o al actualizar el índice FAISS).

    NOMBRE DE CACHÉ: Incluye k y collection para que distintas configuraciones
    tengan su propio caché y no se mezclen resultados.

    Rate limiting: sleep de 7s entre preguntas (5 RPM = 12s/req, pero la
    generación tarda varios segundos, por lo que 7s de sleep es suficiente).

    Returns:
        EvaluationDataset con todas las muestras de /ask.
    """
    k          = experiment_config["k"]
    collection = experiment_config["collection"]
    cache_file = Path(f"tests/ragas/cache/eval_ask_{collection}_k{k}.json")

    # ── intentar cargar desde caché ──────────────────────────────────────────
    if cache_file.exists():
        print(f"\n[eval_ask] Loading from cache: {cache_file}")
        try:
            samples = _load_samples_from_cache(cache_file)
            print(f"[eval_ask] Loaded {len(samples)} samples from cache.")
            log_dataset_info(len(samples), endpoint="ask", from_cache=True)
            log_evaluation_dataset(samples, endpoint="ask", collection=collection, k=k)
            return EvaluationDataset(samples=samples)
        except Exception as e:
            print(f"[eval_ask] Cache load failed ({e}). Regenerating...")

    # ── construir dataset desde cero ─────────────────────────────────────────
    print(f"\n[eval_ask] Building dataset: {len(ASK_QUESTIONS)} questions...")
    samples = []

    for idx, entry in enumerate(ASK_QUESTIONS, 1):
        print(f"  [{idx}/{len(ASK_QUESTIONS)}] {entry['question'][:70]}...")
        try:
            result = ask_rag_system.process_question(entry["question"])
            samples.append(SingleTurnSample(
                user_input         = result["question"],
                retrieved_contexts = result["context"],
                response           = result["answer"],
                reference          = entry["reference"],
            ))
        except Exception as e:
            print(f"  [!] Error en pregunta {idx}: {e}")
            continue

        if idx < len(ASK_QUESTIONS):
            time.sleep(7)

    _save_samples_to_cache(samples, cache_file)
    log_dataset_info(len(samples), endpoint="ask", from_cache=False)
    log_evaluation_dataset(
        samples,
        endpoint="ask",
        collection=collection,
        k=k,
    )
    print(f"[eval_ask] Dataset built: {len(samples)} samples.")
    return EvaluationDataset(samples=samples)


# ====================================================================
# HELPERS DE QUERY — fallback para mode 1 (question vacía)
# ====================================================================

def _build_fallback_query(filter_kwargs: Dict[str, Any]) -> str:
    """
    Construye una query semántica en español desde filter_kwargs cuando
    entry["question"] == "" (mode 1 — solo filtros estructurados).

    Necesario porque retrieve_with_filters("", filters) embeds una cadena
    vacía, produciendo un vector sin señal semántica. Con una query descriptiva
    el ranking FAISS dentro del conjunto filtrado es más relevante, y RAGAS
    answer_relevancy requiere un user_input no vacío para evaluar correctamente.
    """
    parts: List[str] = []
    prop_raw = filter_kwargs.get("property_type", "propiedad")
    prop = prop_raw.rstrip("s") if prop_raw.endswith("s") else prop_raw
    op_type = filter_kwargs.get("operation_type", "")
    barrio  = filter_kwargs.get("barrio", "")
    if isinstance(barrio, list):
        barrio_str = " o ".join(barrio)
    else:
        barrio_str = barrio
    base = prop
    if op_type:
        base += f" en {op_type}"
    if barrio_str:
        base += f" en {barrio_str}"
    parts.append(base)

    min_bed = filter_kwargs.get("min_bedrooms")
    max_bed = filter_kwargs.get("max_bedrooms")
    if min_bed is not None and max_bed is not None and min_bed == max_bed:
        parts.append("monoambiente" if min_bed == 0 else f"{min_bed} dormitorios")
    elif min_bed is not None and max_bed is not None:
        parts.append(f"{min_bed} a {max_bed} dormitorios")
    elif min_bed is not None:
        parts.append(f"mínimo {min_bed} dormitorios")
    elif max_bed is not None:
        parts.append("monoambiente" if max_bed == 0 else f"hasta {max_bed} dormitorios")

    amenity_map = {
        "has_elevator":   "ascensor",
        "has_parking":    "cochera",
        "has_pool":       "piscina",
        "has_parrillero": "parrillero",
        "has_gym":        "gimnasio",
        "has_terrace":    "terraza",
    }
    amenities = [label for key, label in amenity_map.items() if filter_kwargs.get(key)]
    if amenities:
        parts.append("con " + " y ".join(amenities))

    max_price = filter_kwargs.get("max_price")
    if max_price:
        parts.append(f"hasta {max_price:,} USD")

    return ", ".join(parts)


# ====================================================================
# DATASETS DE EVALUACIÓN — /recommend
# ====================================================================

@pytest.fixture(scope="session")
def evaluation_dataset_recommend(
    recommend_services,
    experiment_config,
) -> EvaluationDataset:
    """
    Dataset RAGAS para evaluar el endpoint /recommend.

    Construye SingleTurnSamples ejecutando cada solicitud de
    RECOMMEND_QUESTIONS contra la pipeline retrieve_with_filters()
    + generate_recommendations().

    Los PropertyFilters se construyen dinámicamente desde filter_kwargs
    en RECOMMEND_QUESTIONS, evitando importar PropertyFilters en
    evaluation_data.py (que es un módulo de datos puro).

    Returns:
        EvaluationDataset con todas las muestras de /recommend.
    """
    retrieval_service, generation_service = recommend_services
    k          = experiment_config["k"]
    collection = experiment_config["collection"]
    max_rec    = experiment_config["max_recommendations"]
    cache_file = Path(f"tests/ragas/cache/eval_recommend_{collection}_k{k}.json")

    # ── intentar cargar desde caché ──────────────────────────────────────────
    if cache_file.exists():
        print(f"\n[eval_recommend] Loading from cache: {cache_file}")
        try:
            samples = _load_samples_from_cache(cache_file)
            print(f"[eval_recommend] Loaded {len(samples)} samples from cache.")
            log_dataset_info(len(samples), endpoint="recommend", from_cache=True)
            log_evaluation_dataset(samples, endpoint="recommend", collection=collection, k=k)
            return EvaluationDataset(samples=samples)
        except Exception as e:
            print(f"[eval_recommend] Cache load failed ({e}). Regenerating...")

    # ── construir dataset desde cero ─────────────────────────────────────────
    print(f"\n[eval_recommend] Building dataset: {len(RECOMMEND_QUESTIONS)} questions...")
    samples = []

    for idx, entry in enumerate(RECOMMEND_QUESTIONS, 1):
        raw_question = entry["question"]
        filter_kwargs = entry.get("filter_kwargs", {})

        # Mode 1 — question vacía: construir query semántica desde los filtros.
        # Necesario para (a) retrieval con señal, (b) prompt de generación no vacío,
        # (c) user_input válido para RAGAS answer_relevancy.
        if not raw_question.strip():
            semantic_query = _build_fallback_query(filter_kwargs)
            print(f"  [{idx}/{len(RECOMMEND_QUESTIONS)}] [mode1-fallback] {semantic_query[:70]}...")
        else:
            semantic_query = raw_question
            print(f"  [{idx}/{len(RECOMMEND_QUESTIONS)}] {raw_question[:70]}...")

        try:
            filters = PropertyFilters(**filter_kwargs)
            docs    = retrieval_service.retrieve_with_filters(semantic_query, filters)
            result  = generation_service.generate_recommendations(
                semantic_query,
                docs,
                max_recommendations=max_rec,
            )
            samples.append(SingleTurnSample(
                user_input         = semantic_query,
                retrieved_contexts = result["context"],
                response           = result["answer"],
                reference          = entry["reference"],
            ))
        except Exception as e:
            print(f"  [!] Error en solicitud {idx}: {e}")
            continue

        if idx < len(RECOMMEND_QUESTIONS):
            time.sleep(7)

    _save_samples_to_cache(samples, cache_file)
    log_dataset_info(len(samples), endpoint="recommend", from_cache=False)
    log_evaluation_dataset(
        samples,
        endpoint="recommend",
        collection=collection,
        k=k,
    )
    print(f"[eval_recommend] Dataset built: {len(samples)} samples.")
    return EvaluationDataset(samples=samples)


# ====================================================================
# HELPERS DE CACHÉ
# ====================================================================

def _load_samples_from_cache(cache_file: Path) -> List[SingleTurnSample]:
    with open(cache_file, "r", encoding="utf-8") as f:
        cached_data = json.load(f)
    return [
        SingleTurnSample(
            user_input         = item["user_input"],
            retrieved_contexts = item["retrieved_contexts"],
            response           = item["response"],
            reference          = item["reference"],
        )
        for item in cached_data
    ]


def _save_samples_to_cache(
    samples: List[SingleTurnSample],
    cache_file: Path,
) -> None:
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_data = [
            {
                "user_input":          s.user_input,
                "retrieved_contexts":  s.retrieved_contexts,
                "response":            s.response,
                "reference":           s.reference,
            }
            for s in samples
        ]
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        print(f"  [cache] Saved to {cache_file}")
    except Exception as e:
        print(f"  [cache] Failed to save ({e})")


# ====================================================================
# REPORTE DE MÉTRICAS — JSON + MLflow
# ====================================================================

def save_metric_to_report(
    metric_name: str,
    metric_category: str,
    average_score: float,
    individual_scores: List[float],
    threshold: float,          # kept for backward-compat but ignored — use RAGAS_THRESHOLDS
    passed: bool,              # kept for backward-compat — status is re-evaluated here
    report_filename: str = "ragas_metrics_report_ask.json",
    endpoint: str = "ask",
) -> MetricStatus:
    """
    Persiste una métrica RAGAS en el reporte JSON Y la loguea a MLflow.

    La evaluación de PASS/WARN/FAIL se hace contra RAGAS_THRESHOLDS, no
    contra el argumento `threshold` legacy. El argumento se mantiene por
    compatibilidad con las firmas originales de los test files.

    Returns:
        MetricStatus: Estado de la métrica (PASS, WARN o FAIL).
    """
    # ── Log a MLflow (incluye evaluación de tres zonas) ──────────────────────
    status = log_ragas_metrics(
        metric_name=metric_name,
        average_score=average_score,
        individual_scores=individual_scores,
        endpoint=endpoint,
    )

    # ── Persistir en JSON ────────────────────────────────────────────────────
    report_file = Path(f"tests/ragas/{report_filename}")

    threshold_cfg = get_threshold(metric_name)
    accept_val = threshold_cfg.accept_threshold if threshold_cfg else threshold
    reject_val = threshold_cfg.reject_threshold if threshold_cfg else None

    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            report_data = json.load(f)
    else:
        report_data = {
            "metadata": {
                "first_execution":  datetime.now().isoformat(),
                "last_updated":     None,
                "total_samples":    len(individual_scores),
                "mlflow_run_id":    mlflow.active_run().info.run_id if mlflow.active_run() else None,
            },
            "retrieval_metrics":  {},
            "generation_metrics": {},
            "summary": {
                "total_metrics":     6,   # 5 RAGAS + cosine similarity
                "completed_metrics": 0,
                "passed_metrics":    0,
                "warned_metrics":    0,
                "failed_metrics":    0,
                "overall_status":    "incomplete",
            },
        }

    report_data["metadata"]["last_updated"] = datetime.now().isoformat()

    metric_data = {
        "average":           round(average_score, 4),
        "scores":            [round(s, 4) for s in individual_scores],
        "accept_threshold":  accept_val,
        "reject_threshold":  reject_val,
        "status":            status.value,
        "executed_at":       datetime.now().isoformat(),
    }

    if metric_category == "retrieval":
        report_data["retrieval_metrics"][metric_name] = metric_data
    elif metric_category == "generation":
        report_data["generation_metrics"][metric_name] = metric_data

    # Actualizar summary
    total_completed = (
        len(report_data["retrieval_metrics"]) +
        len(report_data["generation_metrics"])
    )
    report_data["summary"]["completed_metrics"] = total_completed

    pass_count  = 0
    warn_count  = 0
    fail_count  = 0
    for metrics_dict in [report_data["retrieval_metrics"], report_data["generation_metrics"]]:
        for m in metrics_dict.values():
            s = m.get("status", "WARN")
            if s == "PASS":   pass_count += 1
            elif s == "WARN": warn_count += 1
            elif s == "FAIL": fail_count += 1

    report_data["summary"]["passed_metrics"] = pass_count
    report_data["summary"]["warned_metrics"] = warn_count
    report_data["summary"]["failed_metrics"] = fail_count

    total_metrics = report_data["summary"]["total_metrics"]
    if total_completed >= total_metrics:
        if fail_count > 0:
            report_data["summary"]["overall_status"] = "complete_fail"
        elif warn_count > 0:
            report_data["summary"]["overall_status"] = "complete_warn"
        else:
            report_data["summary"]["overall_status"] = "complete_pass"
    else:
        report_data["summary"]["overall_status"] = f"incomplete ({total_completed}/{total_metrics})"

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    return status