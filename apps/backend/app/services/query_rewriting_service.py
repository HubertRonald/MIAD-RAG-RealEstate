from __future__ import annotations

import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

ZERO_SHOT_PROMPT = """
Reformula la siguiente consulta inmobiliaria de {num_queries} maneras distintas
para mejorar la recuperación semántica en un índice de propiedades de Montevideo.

REGLAS:
- Mantén el idioma original.
- No inventes filtros no mencionados.
- Usa sinónimos inmobiliarios útiles.
- Devuelve solo una consulta por línea, sin numeración.

Consulta original:
{query}

Reformulaciones:
"""

FEW_SHOT_PROMPT = """
Eres un especialista en búsqueda inmobiliaria para Montevideo.

Reformula la consulta de {num_queries} maneras distintas para mejorar retrieval.
Incluye equivalencias naturales del dominio: rambla/playa/costa, familiar/niños,
luminoso/buena luz, moderno/reciente, terraza/balcón amplio.

No inventes presupuesto, barrio ni tipo de operación si el usuario no lo dijo.

Ejemplos:
Consulta: "algo tranquilo cerca del mar"
Reformulaciones:
propiedad en zona tranquila cerca de rambla o playa
apartamento luminoso próximo a la costa
vivienda residencial con cercanía al mar

Consulta: "para familia con niños"
Reformulaciones:
propiedad familiar con dormitorios y entorno seguro
apartamento apto para familia con niños cerca de plazas o servicios
vivienda cómoda con amenities familiares

Consulta actual:
{query}

Reformulaciones:
"""

DECOMPOSE_PROMPT = """
Descompón la siguiente consulta inmobiliaria en subconsultas simples.
Cada subconsulta debe ser autocontenida. Si ya es simple, devuélvela igual.

Consulta:
{query}

Subconsultas:
"""

STEP_BACK_PROMPT = """
Genera una consulta inmobiliaria más general que capture la intención principal
de esta consulta específica, sin inventar datos.

Consulta específica:
{query}

Consulta general:
"""

HYDE_PROMPT = """
Escribe un párrafo breve que describa la propiedad ideal que respondería
a la consulta del usuario. No inventes barrio, precio ni operación si no aparecen.

Consulta:
{query}

Párrafo:
"""


class QueryRewritingService:
    """
    Reescritura de consultas adaptada al dominio inmobiliario.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        rewriting_temperature: float = 0.3,
        precise_temperature: float = 0.2,
    ) -> None:
        self.llm_diverse = ChatGoogleGenerativeAI(
            model=model,
            temperature=rewriting_temperature,
        )
        self.llm_precise = ChatGoogleGenerativeAI(
            model=model,
            temperature=precise_temperature,
        )

        self.zero_shot_prompt = ChatPromptTemplate.from_template(ZERO_SHOT_PROMPT)
        self.few_shot_prompt = ChatPromptTemplate.from_template(FEW_SHOT_PROMPT)
        self.decompose_prompt = ChatPromptTemplate.from_template(DECOMPOSE_PROMPT)
        self.step_back_prompt = ChatPromptTemplate.from_template(STEP_BACK_PROMPT)
        self.hyde_prompt = ChatPromptTemplate.from_template(HYDE_PROMPT)

    def _clean_output(self, raw: str, limit: int | None = None) -> list[str]:
        prefix_pattern = re.compile(r"^\s*(?:\d+[\.\)\-]\s+|[-•*·]\s+)")

        cleaned: list[str] = []

        for line in raw.strip().split("\n"):
            line = line.strip()

            if not line or line.endswith(":"):
                continue

            line = prefix_pattern.sub("", line).strip()
            line = line.strip('"').strip("'").strip()

            if line:
                cleaned.append(line)

        return cleaned[:limit] if limit else cleaned

    def zero_shot_rewrite(self, query: str, num_queries: int = 3) -> list[str]:
        if not query or not query.strip():
            raise ValueError("La pregunta no puede estar vacía.")

        try:
            chain = self.zero_shot_prompt | self.llm_diverse
            response = chain.invoke({"query": query, "num_queries": num_queries})
            return self._clean_output(response.content, limit=num_queries)
        except Exception as exc:
            logger.warning("[QueryRewriting] zero_shot failed: %s", exc)
            return []

    def few_shot_rewrite(self, query: str, num_queries: int = 3) -> list[str]:
        if not query or not query.strip():
            raise ValueError("La pregunta no puede estar vacía.")

        try:
            chain = self.few_shot_prompt | self.llm_diverse
            response = chain.invoke({"query": query, "num_queries": num_queries})
            return self._clean_output(response.content, limit=num_queries)
        except Exception as exc:
            logger.warning("[QueryRewriting] few_shot failed: %s", exc)
            return []

    def decompose_query(self, query: str) -> list[str]:
        if not query or not query.strip():
            raise ValueError("La pregunta no puede estar vacía.")

        try:
            chain = self.decompose_prompt | self.llm_diverse
            response = chain.invoke({"query": query})
            subqueries = self._clean_output(response.content)
            return subqueries or [query]
        except Exception as exc:
            logger.warning("[QueryRewriting] decompose failed: %s", exc)
            return [query]

    def step_back_rewrite(self, query: str) -> str:
        if not query or not query.strip():
            raise ValueError("La pregunta no puede estar vacía.")

        try:
            chain = self.step_back_prompt | self.llm_precise
            response = chain.invoke({"query": query})
            output = response.content.strip()
            return output or query
        except Exception as exc:
            logger.warning("[QueryRewriting] step_back failed: %s", exc)
            return query

    def hyde_rewrite(self, query: str) -> str:
        if not query or not query.strip():
            raise ValueError("La pregunta no puede estar vacía.")

        try:
            chain = self.hyde_prompt | self.llm_precise
            response = chain.invoke({"query": query})
            output = response.content.strip()
            return output or query
        except Exception as exc:
            logger.warning("[QueryRewriting] hyde failed: %s", exc)
            return query
