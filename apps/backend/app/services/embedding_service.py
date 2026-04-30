from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class EmbeddingService:
    """
    Servicio de embeddings del backend.

    En backend productivo NO construye índices.
    Solo carga el índice FAISS generado por job-indexer y descargado desde GCS.
    """

    def __init__(self, model_name: str = "models/gemini-embedding-001") -> None:
        self.model_name = model_name
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model=model_name)
        self.vectorstore: Optional[FAISS] = None

    def load_vectorstore(self, persist_path: str) -> FAISS:
        index_path = Path(persist_path)

        if not index_path.exists():
            raise FileNotFoundError(f"El índice no existe en la ruta: {persist_path}")

        self.vectorstore = FAISS.load_local(
            str(index_path),
            self.embeddings_model,
            allow_dangerous_deserialization=True,
        )

        return self.vectorstore

    def get_vectorstore(self) -> Optional[FAISS]:
        return self.vectorstore

    def get_embeddings_model(self) -> GoogleGenerativeAIEmbeddings:
        return self.embeddings_model
