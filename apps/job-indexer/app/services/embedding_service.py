from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings


class EmbeddingService:
    """
    Servicio de embeddings para construir el índice.

    A diferencia del backend, aquí sí se generan embeddings para todos los
    Documents y luego se construye FAISS.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model_name)

    def get_embeddings_model(self) -> GoogleGenerativeAIEmbeddings:
        return self.embeddings
