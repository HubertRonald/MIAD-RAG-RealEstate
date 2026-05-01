from __future__ import annotations

from typing import Any

from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langdetect import DetectorFactory, detect

from app.services.bigquery_listing_service import BigQueryListingService
from miad_rag_common.schemas.listing import ListingInfo

DetectorFactory.seed = 0

_SPANISH_MARKERS = {
    "qué",
    "que",
    "cómo",
    "como",
    "cuál",
    "cual",
    "cuáles",
    "cuales",
    "por",
    "para",
    "una",
    "uno",
    "los",
    "las",
    "del",
    "con",
    "son",
    "es",
    "en",
    "de",
    "la",
    "el",
    "se",
    "no",
    "si",
    "más",
    "mas",
}


class GenerationService:
    """
    Genera respuestas para /ask y narrativa de recomendaciones para /recommend.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
    ) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
        )

        self.market_prompt = ChatPromptTemplate.from_template(
            """
Eres un experto en el mercado inmobiliario de Montevideo, Uruguay.

Responde ÚNICAMENTE usando el contexto proporcionado.
No inventes datos. Si el contexto no alcanza, dilo claramente.

Contexto:
{context}

Pregunta:
{question}

Responde en {language}, con tono profesional, claro y natural.
Respuesta:
"""
        )

        self.recommendation_prompt = ChatPromptTemplate.from_template(
            """
Eres un asesor inmobiliario experto en Montevideo, Uruguay.

Tu tarea es recomendar propiedades usando ÚNICAMENTE los listings proporcionados.
No inventes disponibilidad, precios negociables ni condiciones no presentes.
No ofrezcas coordinar visitas ni contactar vendedores.

LISTINGS:
{context}

SOLICITUD:
{question}

INSTRUCCIONES:
- Selecciona las mejores propiedades según el match con la solicitud.
- Menciona barrio, precio, dormitorios, superficie y amenities cuando estén disponibles.
- Explica por qué cada propiedad coincide.
- Si hay limitaciones o faltan datos, indícalo de forma breve.
- Usa tono cercano y profesional.

Responde en {language}.
Respuesta:
"""
        )

    def _detect_language(self, text: str) -> str:
        words = set((text or "").lower().split())

        if words & _SPANISH_MARKERS:
            return "español"

        try:
            lang = detect(text)
            return "español" if lang == "es" else "inglés"
        except Exception:
            return "español"

    def _format_context(self, documents: list[Document]) -> str:
        if not documents:
            return "No se encontró contexto relevante."

        parts = []

        for i, doc in enumerate(documents, start=1):
            metadata = doc.metadata or {}
            source = metadata.get("source_file") or metadata.get("source") or "unknown"
            parts.append(
                f"[Documento {i} | source={source}]\n{doc.page_content}"
            )

        return "\n\n".join(parts)

    def _extract_sources(self, documents: list[Document]) -> list[str]:
        sources: list[str] = []

        for doc in documents:
            metadata = doc.metadata or {}
            source = metadata.get("source_file") or metadata.get("source") or "unknown"
            sources.append(str(source))

        return sources

    def generate_response(
        self,
        question: str,
        retrieved_docs: list[Document],
    ) -> dict[str, Any]:
        language = self._detect_language(question)
        context = self._format_context(retrieved_docs)

        chain = self.market_prompt | self.llm
        response = chain.invoke(
            {
                "context": context,
                "question": question,
                "language": language,
            }
        )

        return {
            "answer": response.content,
            "sources": self._extract_sources(retrieved_docs),
        }

    def generate_recommendations(
        self,
        question: str,
        retrieved_docs: list[Document],
        max_recommendations: int = 5,
        semantic_scores: list[float] | None = None,
        listing_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        selected_docs = retrieved_docs[:max_recommendations]
        selected_scores = semantic_scores[:max_recommendations] if semantic_scores else []

        language = self._detect_language(question)
        context = self._format_context(selected_docs)

        chain = self.recommendation_prompt | self.llm
        response = chain.invoke(
            {
                "context": context,
                "question": question,
                "language": language,
            }
        )

        bq_helper = BigQueryListingService()
        listings: list[ListingInfo] = []

        for idx, doc in enumerate(selected_docs):
            metadata = doc.metadata or {}
            listing_id = (
                metadata.get("id")
                or metadata.get("listing_id")
                or metadata.get("property_id")
            )
            override = {}

            if listing_id is not None and listing_overrides:
                override = listing_overrides.get(str(listing_id), {})

            score = selected_scores[idx] if idx < len(selected_scores) else None

            listings.append(
                bq_helper.document_to_listing(
                    doc=doc,
                    semantic_score=score,
                    override=override,
                )
            )

        return {
            "answer": response.content,
            "sources": self._extract_sources(selected_docs),
            "listings_used": listings,
        }
