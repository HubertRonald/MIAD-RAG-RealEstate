"""
Servicio de Embeddings para el Sistema RAG
==========================================

Este módulo implementa la funcionalidad de generar embeddings vectoriales
a partir de chunks de texto y construir una base de datos vectorial con FAISS.

"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import shutil
import os
import time

from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.api_core import exceptions as google_exceptions

from app.services.cloud_storage_service import CloudStorageService


# ===========================
# CONFIGURACIÓN DE EMBEDDINGS
# ===========================
"""
IMPORTANTE: Estas variables definen el modelo de embeddings usado en el servicio.

LÍMITES DE GOOGLE AI FREE TIER:
- RPM (Requests Per Minute): 100
- TPM (Tokens Per Minute): 30,000
- RPD (Requests Per Day): 1,000
"""
EMBEDDINGS_MODEL = "models/gemini-embedding-001"

BATCH_SIZE     = 50    # ~69 batches para 3,436 docs
MAX_BATCH_SIZE = 100
REQUEST_DELAY  = 15     # segundos entre batches — margen seguro bajo 100 RPM

# =============================================================================
# PRECIOS EMBEDDINGS (verificar en https://ai.google.dev/pricing)
# gemini-embedding-001 — precio por millón de tokens (USD)
# =============================================================================
EMBEDDING_PRICE_PER_M_TOKENS = 0.025   # $0.025 / 1M tokens
CHARS_PER_TOKEN              = 3.2       # Estimación estándar: 1 token ≈ 4 caracteres, 3.2 obtenido en sample (200) de listings.csv


class EmbeddingService:
    """
    Servicio para generar embeddings y construir base de datos vectorial.
    """

    def __init__(self):
        """
        Inicializa el servicio de embeddings.

        Raises:
            ValueError: Si no se puede inicializar el modelo de embeddings.
        """
        self.model              = EMBEDDINGS_MODEL
        self.batch_size         = BATCH_SIZE
        self.max_batch_size     = MAX_BATCH_SIZE
        self.request_delay      = REQUEST_DELAY
        self.embeddings_model   = GoogleGenerativeAIEmbeddings(model=self.model)
        self.vectorstore: Optional[FAISS] = None
        self.cloud_storage      = CloudStorageService()
        self._cost_stats: dict  = {}

    # =========================================================================
    # EMBEDDING CON RETRY
    # =========================================================================

    def _embed_with_retry(self, batch: List[str], max_retries: int = 5) -> List[List[float]]:
        """
        Embeds a single batch with exponential backoff on 429.

        Args:
            batch       : Lista de textos a embeber.
            max_retries : Número máximo de reintentos (default 5).

        Returns:
            Lista de vectores de embeddings.

        Raises:
            google_exceptions.ResourceExhausted: Si se agotan los reintentos.
        """
        delay = 30  # espera inicial en segundos
        for attempt in range(max_retries):
            try:
                return self.embeddings_model.embed_documents(batch)
            except google_exceptions.ResourceExhausted:
                if attempt == max_retries - 1:
                    raise
                print(
                    f"[WARN] 429 Rate limit hit. Esperando {delay}s antes de reintentar "
                    f"(intento {attempt + 1}/{max_retries})..."
                )
                time.sleep(delay)
                delay *= 2  # 30 → 60 → 120 → 240 → 480
        return []  # inalcanzable, satisface el type checker

    # =========================================================================
    # CHECKPOINT / RESUME
    # =========================================================================

    def _checkpoint_path(self, persist_path: str) -> Path:
        """Ruta del archivo de checkpoint para un índice dado."""
        return Path(persist_path) / "_embedding_checkpoint.json"

    def _load_checkpoint(self, persist_path: str) -> dict:
        """
        Carga el checkpoint de un run anterior si existe.

        Returns:
            Dict con 'completed_batches' (int) y 'embeddings' (List[List[float]]),
            o valores vacíos si no hay checkpoint.
        """
        cp_path = self._checkpoint_path(persist_path)
        if cp_path.exists():
            try:
                with open(cp_path, "r") as f:
                    data = json.load(f)
                completed = data.get("completed_batches", 0)
                embeddings = data.get("embeddings", [])
                print(
                    f"[Checkpoint] Retomando desde batch {completed + 1} "
                    f"({len(embeddings)} embeddings ya generados)."
                )
                return {"completed_batches": completed, "embeddings": embeddings}
            except Exception as e:
                print(f"[Checkpoint] No se pudo leer checkpoint, empezando desde cero: {e}")
        return {"completed_batches": 0, "embeddings": []}

    def _save_checkpoint(self, persist_path: str, completed_batches: int, embeddings: List) -> None:
        """Persiste el progreso actual a disco."""
        cp_path = self._checkpoint_path(persist_path)
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cp_path, "w") as f:
            json.dump({"completed_batches": completed_batches, "embeddings": embeddings}, f)

    def _clear_checkpoint(self, persist_path: str) -> None:
        """Elimina el checkpoint una vez que el índice se construyó exitosamente."""
        cp_path = self._checkpoint_path(persist_path)
        if cp_path.exists():
            cp_path.unlink()
            print("[Checkpoint] Checkpoint eliminado (índice completo).")

    # =========================================================================
    # GENERACIÓN DE EMBEDDINGS
    # =========================================================================

    def create_embeddings(
        self,
        texts: List[str],
        persist_path: str = None,
    ) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos con batch processing y checkpoint.

        Si persist_path se proporciona, el progreso se guarda a disco después de
        cada batch. Si el proceso falla y se relanza, retoma desde el último batch
        completado en lugar de empezar desde cero.

        Args:
            texts        : Lista de textos para generar embeddings.
            persist_path : Ruta del índice (para checkpoint). None desactiva checkpoint.

        Returns:
            Lista de embeddings (un vector por texto).

        Raises:
            ValueError: Si la lista está vacía o contiene textos vacíos.
        """
        if not texts:
            raise ValueError("La lista de textos está vacía.")

        for i, text in enumerate(texts):
            if not text or text.strip() == "":
                raise ValueError(f"Texto en posición {i} está vacío o es inválido.")

        # Batch pequeño — procesar directamente con retry
        if len(texts) <= self.max_batch_size:
            print(f"[INFO] Generando embeddings para {len(texts)} textos (batch único)...")
            return self._embed_with_retry(texts)

        # Cargar checkpoint si existe
        checkpoint     = self._load_checkpoint(persist_path) if persist_path else {"completed_batches": 0, "embeddings": []}
        start_batch    = checkpoint["completed_batches"]
        all_embeddings = checkpoint["embeddings"]

        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        remaining     = total_batches - start_batch

        print(f"\n{'='*60}")
        print(f"GENERACIÓN DE EMBEDDINGS")
        print(f"Total textos:    {len(texts)}")
        print(f"Batch size:      {self.batch_size}")
        print(f"Total batches:   {total_batches}")
        if start_batch > 0:
            print(f"Retomando desde: batch {start_batch + 1} ({remaining} restantes)")
        print(f"Delay entre batches: {self.request_delay}s")
        print(f"{'='*60}\n")

        for batch_idx in range(start_batch, total_batches):
            batch_num  = batch_idx + 1
            start_idx  = batch_idx * self.batch_size
            batch      = texts[start_idx: start_idx + self.batch_size]

            print(f"[INFO] Procesando batch {batch_num}/{total_batches} ({len(batch)} textos)...")
            batch_start = time.time()

            batch_embeddings = self._embed_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

            batch_time = time.time() - batch_start
            print(f"[INFO] ✓ Batch {batch_num} completado en {batch_time:.2f}s")

            # Guardar checkpoint tras cada batch completado
            if persist_path:
                self._save_checkpoint(persist_path, batch_num, all_embeddings)

            # Delay entre batches (excepto en el último)
            if batch_num < total_batches:
                print(f"[INFO] Esperando {self.request_delay}s...")
                time.sleep(self.request_delay)

        print(f"\n{'='*60}")
        print(f"✓ EMBEDDINGS COMPLETADOS: {len(all_embeddings)} vectores generados")
        print(f"{'='*60}\n")

        return all_embeddings

    # =========================================================================
    # CONSTRUCCIÓN DEL ÍNDICE FAISS
    # =========================================================================

    def build_vectorstore(
        self,
        chunks: List[Document],
        persist_path: str = "./faiss_index",
        collection_name: str = None,
    ) -> FAISS:
        """
        Construye un índice vectorial FAISS a partir de chunks.

        Usa checkpoint/resume: si el proceso falla durante los embeddings,
        relanzar build_vectorstore retoma desde el último batch completado.
        El checkpoint se elimina automáticamente al finalizar con éxito.

        Args:
            chunks          : Lista de Documents con contenido y metadatos.
            persist_path    : Ruta donde se guardará el índice FAISS.
            collection_name : Nombre de colección para Cloud Storage (opcional).

        Returns:
            Objeto FAISS vectorstore listo para búsquedas.

        Raises:
            ValueError: Si la lista de chunks está vacía.
        """
        if not chunks:
            raise ValueError("La lista de chunks está vacía.")

        print(f"\n{'='*60}")
        print(f"CONSTRUCCIÓN DE VECTORSTORE FAISS")
        print(f"Total chunks: {len(chunks)}")
        print(f"Modelo: {self.model}")
        print(f"{'='*60}\n")

        texts     = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Generar embeddings — pasa persist_path para habilitar checkpoint
        embeddings = self.create_embeddings(texts, persist_path=persist_path)

        print(f"[INFO] Construyendo índice FAISS desde embeddings pre-generados...")

        text_embedding_pairs = list(zip(texts, embeddings))
        vectorstore = FAISS.from_embeddings(
            text_embeddings = text_embedding_pairs,
            embedding       = self.embeddings_model,
            metadatas       = metadatas,
        )

        print(f"[INFO] ✓ Índice FAISS construido")

        # Persistir en disco
        print(f"[INFO] Guardando índice en {persist_path}...")
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(persist_path)
        print(f"[INFO] ✓ Índice guardado en disco")

        # Subir a Cloud Storage si está configurado
        if collection_name and self.cloud_storage.client:
            print(f"[INFO] Subiendo índice a Cloud Storage...")
            self.cloud_storage.save_index_to_cloud(persist_path, collection_name)
            print(f"[INFO] ✓ Índice subido a Cloud Storage")

        # Siempre asignar vectorstore (independiente de cloud storage)
        self.vectorstore = vectorstore

        # Eliminar checkpoint — build completado con éxito
        self._clear_checkpoint(persist_path)

        # Calcular estadísticas de costo
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        est_tokens  = total_chars // CHARS_PER_TOKEN
        cost_usd    = (est_tokens / 1_000_000) * EMBEDDING_PRICE_PER_M_TOKENS

        self._cost_stats = {
            "total_chunks":       len(chunks),
            "total_chars":        total_chars,
            "estimated_tokens":   est_tokens,
            "embedding_cost_usd": round(cost_usd, 6),
            "model":              self.model,
            "price_per_m_tokens": EMBEDDING_PRICE_PER_M_TOKENS,
        }

        print(f"[INFO] Embedding cost stats: ~{est_tokens:,} tokens | ${cost_usd:.6f} USD")

        return vectorstore

    # =========================================================================
    # CARGA DEL ÍNDICE
    # =========================================================================

    def load_vectorstore(
        self,
        persist_path: str = "./faiss_index",
        collection_name: str = None,
    ) -> FAISS:
        """
        Carga un índice FAISS previamente guardado.

        Intenta cargar desde Cloud Storage primero si está configurado,
        con fallback a almacenamiento local.

        Args:
            persist_path    : Ruta donde está guardado el índice FAISS.
            collection_name : Nombre de colección para cargar desde Cloud Storage.

        Returns:
            Objeto FAISS vectorstore cargado.

        Raises:
            FileNotFoundError: Si el índice no existe en ninguna ubicación.
        """
        if collection_name and self.cloud_storage.client:
            loaded = self.cloud_storage.load_index_from_cloud(collection_name, persist_path)
            if loaded:
                self.vectorstore = FAISS.load_local(
                    persist_path,
                    self.embeddings_model,
                    allow_dangerous_deserialization=True,
                )
                return self.vectorstore

        index_path = Path(persist_path)
        if not index_path.exists():
            raise FileNotFoundError(f"El índice no existe en la ruta: {persist_path}")

        self.vectorstore = FAISS.load_local(
            persist_path,
            self.embeddings_model,
            allow_dangerous_deserialization=True,
        )
        return self.vectorstore

    # =========================================================================
    # GETTERS
    # =========================================================================

    def get_cost_stats(self) -> dict:
        """
        Retorna las estadísticas de costo de la última operación build_vectorstore().

        Returns:
            Dict con total_chunks, total_chars, estimated_tokens,
            embedding_cost_usd, model, price_per_m_tokens.
            Retorna dict vacío si build_vectorstore no se ha ejecutado aún.
        """
        return self._cost_stats

    def get_vectorstore(self) -> Optional[FAISS]:
        """
        Retorna el vectorstore FAISS actual.

        Returns:
            Vectorstore FAISS, o None si no se ha construido/cargado todavía.
        """
        return self.vectorstore
