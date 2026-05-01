from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request para consultas generales de mercado inmobiliario."""

    question: str = Field(..., min_length=1)
    collection: str = Field(..., min_length=1)
    use_reranking: bool = False
    use_query_rewriting: bool = False


class FuenteContexto(BaseModel):
    """Fragmento usado como contexto para responder una pregunta RAG."""

    file_name: Optional[str] = None
    page_number: Optional[int] = None
    chunk_type: Optional[str] = None
    priority: Optional[str] = None
    snippet: Optional[str] = None
    content: Optional[str] = None
    rerank_score: Optional[float] = None
    semantic_score: Optional[float] = None


class AskResponse(BaseModel):
    """Response del endpoint /api/v1/ask."""

    question: str
    final_query: str
    answer: str
    collection: str
    files_consulted: list[str] = Field(default_factory=list)
    context_docs: list[FuenteContexto] = Field(default_factory=list)
    reranker_used: bool = False
    query_rewriting_used: bool = False
    response_time_sec: float