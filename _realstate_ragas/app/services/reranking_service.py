"""
Servicio de Reranking para el Sistema RAG
==========================================

Este módulo implementa reranking de documentos usando un modelo Cross-Encoder
de sentence-transformers para mejorar la precisión del retrieval en Tutor-IA.

"""

import logging
from typing import List, Optional

from langchain.schema import Document
from langsmith import traceable
from sentence_transformers import CrossEncoder


logger = logging.getLogger(__name__)


# ===========================================================================
# CONFIGURACIÓN DEL SERVICIO
# ===========================================================================

# Modelo Cross-Encoder multilingüe: mejor rendimiento en contenido español y
# descripciones VLM multimodales que el modelo original ms-marco inglés.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Número de documentos a retornar después del reranking.
DEFAULT_TOP_K = 3


# ===========================================================================
# CLASE PRINCIPAL
# ===========================================================================

class RerankingService:
    """
    Servicio de reranking de documentos con Cross-Encoder para el RAG de Tutor-IA.

    Recibe los documentos recuperados por FAISS y los reordena usando un modelo
    Cross-Encoder que evalúa cada par (query, documento) conjuntamente, produciendo
    scores de relevancia más precisos que la similitud coseno bi-encoder.

    El modelo se descarga automáticamente desde HuggingFace en la primera ejecución.
    Las ejecuciones posteriores usan la caché local (~/.cache/huggingface/).
    """

    def __init__(
        self,
        model_name: str = CROSS_ENCODER_MODEL,
        top_k: int = DEFAULT_TOP_K,
    ):
        """
        Inicializa el servicio cargando el modelo Cross-Encoder.

        La carga del modelo ocurre una sola vez en la inicialización (no lazy),
        de modo que el primer request no sufra la latencia de descarga.

        Args:
            model_name: Nombre del modelo Cross-Encoder en HuggingFace Hub.
                        Default: "BAAI/bge-reranker-v2-m3"
            top_k:      Número de documentos a retornar después del reranking.
                        Debe ser menor que el k del retriever FAISS inicial.
                        Default: 3
        """
        self.model_name  = model_name
        self.top_k       = top_k

        logger.info(f"[RerankingService] Cargando modelo Cross-Encoder: {model_name} ...")
        self.cross_encoder = CrossEncoder(model_name)
        logger.info(f"[RerankingService] Modelo cargado | top_k={top_k}")

    # -----------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # -----------------------------------------------------------------------

    @traceable(name="rerank_documents", run_type="llm", metadata={"strategy": "cross-encoder"})
    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3,
    ) -> List[Document]:
        """
        Reordena documentos por relevancia usando el Cross-Encoder.

        Crea pares (query, documento) y los evalúa conjuntamente con el modelo,
        produciendo un score por par. Los documentos se ordenan de mayor a menor
        score y se retornan los top-k más relevantes.

        La query usada aquí debe ser la pregunta ORIGINAL del usuario.

        Args:
            query:     Pregunta original del usuario.
            documents: Lista de documentos recuperados por FAISS (List[Document]).
            top_k:     Número de documentos a retornar. Si es None, usa el valor
                       configurado en __init__. 

        Returns:
            Lista de objetos Document ordenados por relevancia (mayor score primero),
            limitada a top_k elementos.

        Raises:
            ValueError: Si query está vacío o documents está vacío.
        """
        if not query or not query.strip():
            raise ValueError("La pregunta no puede estar vacía.")

        if not documents:
            raise ValueError("La lista de documentos no puede estar vacía.")

        effective_top_k = top_k if top_k is not None else self.top_k

        # No retornar más documentos de los que entraron
        effective_top_k = min(effective_top_k, len(documents))

        logger.info(
            f"[rerank_documents] Reranking {len(documents)} documentos → top {effective_top_k} | "
            f"query='{query[:60]}{'...' if len(query) > 60 else ''}'"
        )

        # Construir pares (query, contenido_documento) para el Cross-Encoder
        pairs = [(query, doc.page_content) for doc in documents]

        # Calcular scores de relevancia para todos los pares de una vez (batch)
        scores = self.cross_encoder.predict(pairs)

        # Asociar cada documento con su score y posición original
        scored_documents = [
            (score, original_rank, doc)
            for original_rank, (score, doc) in enumerate(zip(scores, documents))
        ]

        # Ordenar por score descendente (mayor relevancia primero)
        scored_documents.sort(key=lambda x: x[0], reverse=True)

        # Escribir rerank_score en metadata de cada documento y extraer top-k
        reranked = []
        for score, _, doc in scored_documents[:effective_top_k]:
            doc.metadata["rerank_score"] = round(float(score), 4)
            reranked.append(doc)

        logger.info(
            f"[rerank_documents] Reranking completado | "
            f"scores top-{effective_top_k}: {[round(float(s), 4) for s, _, _ in scored_documents[:effective_top_k]]}"
        )

        return reranked

    # -----------------------------------------------------------------------
    # MÉTODO OPCIONAL: RERANKING CON METADATA
    # -----------------------------------------------------------------------

    @traceable(name="rerank_with_metadata", run_type="llm", metadata={"strategy": "cross-encoder"})
    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[dict]:
        """
        Reordena documentos e incluye información detallada para análisis y debugging.

        Útil durante el desarrollo para entender qué documentos sube o baja el
        reranker respecto al orden original del retriever FAISS, y con qué scores.

        Args:
            query:     Pregunta original del usuario.
            documents: Lista de documentos recuperados por FAISS.
            top_k:     Número de documentos a retornar. None = usa self.top_k.

        Returns:
            Lista de dicts con los siguientes campos por documento:
                document:       Objeto Document original.
                score:          Score de relevancia del Cross-Encoder (float).
                original_rank:  Posición en la lista de FAISS (0-indexed).
                reranked_rank:  Nueva posición tras el reranking (0-indexed).
                source_file:    Nombre del archivo fuente (de doc.metadata).
                content_preview: Primeros 120 caracteres del contenido.

        Raises:
            ValueError: Si query o documents están vacíos.
        """
        if not query or not query.strip():
            raise ValueError("La query no puede estar vacía.")

        if not documents:
            raise ValueError("La lista de documentos no puede estar vacía.")

        effective_top_k = top_k if top_k is not None else self.top_k
        effective_top_k = min(effective_top_k, len(documents))

        logger.info(
            f"[rerank_with_metadata] Reranking {len(documents)} documentos "
            f"con metadata | top_k={effective_top_k}"
        )

        # Construir pares y calcular scores
        pairs  = [(query, doc.page_content) for doc in documents]
        scores = self.cross_encoder.predict(pairs)

        # Construir registros completos con posición original
        records = [
            {
                "document":       doc,
                "score":          float(score),
                "original_rank":  rank,
                "source_file":    doc.metadata.get("source_file", "unknown"),
                "content_preview": doc.page_content[:120].replace("\n", " "),
            }
            for rank, (score, doc) in enumerate(zip(scores, documents))
        ]

        # Ordenar por score descendente y añadir reranked_rank
        records.sort(key=lambda x: x["score"], reverse=True)
        for new_rank, record in enumerate(records):
            record["reranked_rank"] = new_rank

        top_records = records[:effective_top_k]

        logger.info(
            f"[rerank_with_metadata] Completado | "
            f"top-{effective_top_k} scores: {[round(r['score'], 4) for r in top_records]}"
        )

        return top_records