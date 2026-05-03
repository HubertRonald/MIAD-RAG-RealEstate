from __future__ import annotations

import re
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.runtime import get_settings
from miad_rag_common.logging.structured_logging import get_logger

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator


settings = get_settings()
logger = get_logger(__name__)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DEFAULT_REWRITING_MODEL = settings.GEMINI_GENERATION_MODEL
DEFAULT_REWRITING_TEMPERATURE = 0.3
DEFAULT_PRECISE_TEMPERATURE = 0.2
DEFAULT_HYDE_MAX_TOKENS = 300


# =============================================================================
# PROMPTS INMOBILIARIOS
# =============================================================================

TONE_AND_LANGUAGE_RULES = """
REGLAS DE TONO Y LENGUAJE:
- Si el usuario escribe en español, usa español rioplatense/uruguayo natural.
- Usa términos inmobiliarios locales cuando apliquen: rambla, costa, parrillero,
  cochera, garaje, gastos comunes, barrio, zona tranquila, apartamento y casa.
- Si el usuario usa groserías, enojo, ironía o sarcasmo, NO repitas insultos.
  Extrae la intención real y reformúlala en lenguaje neutral y buscable.
- Si el usuario exagera de forma coloquial, interpreta la necesidad subyacente
  sin inventar datos nuevos.
- Si el usuario dice algo como "carísimo", "un disparate", "me matan con el precio",
  tradúcelo como preferencia por precio razonable, presupuesto acotado o buena relación costo-beneficio.
- Si el usuario dice algo como "ni loco cerca del ruido", tradúcelo como zona tranquila
  o baja exposición a ruido, sin inventar barrio.
"""


ZERO_SHOT_PROMPT = TONE_AND_LANGUAGE_RULES + """
Reformula la siguiente consulta inmobiliaria de {num_queries} maneras distintas
para mejorar la recuperación semántica en un índice de propiedades de Montevideo.

REGLAS:
- Mantén el idioma original de la consulta.
- No inventes filtros no mencionados.
- No inventes barrio, precio, tipo de operación ni tipo de propiedad.
- Usa sinónimos inmobiliarios útiles.
- Puedes usar equivalencias naturales del dominio:
  rambla/playa/costa, familiar/niños, luminoso/buena luz,
  moderno/reciente, terraza/balcón amplio, cochera/garaje.
- Devuelve solo una consulta por línea.
- No uses numeración, viñetas ni markdown.

Consulta original:
{query}

Reformulaciones:
"""


FEW_SHOT_PROMPT = TONE_AND_LANGUAGE_RULES + """
Eres un especialista en búsqueda inmobiliaria para Montevideo, Uruguay.

Reformula la consulta de {num_queries} maneras distintas para mejorar el retrieval
semántico en un sistema RAG de propiedades inmobiliarias.

REGLAS:
- Mantén el idioma predominante del usuario.
- No inventes presupuesto, barrio, operación ni tipo de propiedad si no aparecen.
- Conserva la intención original.
- Usa vocabulario inmobiliario equivalente.
- Devuelve solo una consulta por línea.
- No uses numeración, viñetas ni markdown.

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

Consulta: "no quiero algo carísimo, que no me maten con el precio"
Reformulaciones:
propiedad con precio razonable y buen equilibrio costo beneficio
vivienda dentro de presupuesto acotado
apartamento accesible comparado con opciones premium

Consulta: "ni loco quiero vivir en una zona ruidosa"
Reformulaciones:
propiedad en zona tranquila y residencial
vivienda con bajo nivel de ruido en el entorno
apartamento en barrio tranquilo para vivir cómodo

Consulta actual:
{query}

Reformulaciones:
"""


DECOMPOSE_PROMPT = TONE_AND_LANGUAGE_RULES + """
Descompón la siguiente consulta inmobiliaria en subconsultas simples.

REGLAS:
- Cada subconsulta debe ser autocontenida.
- Mantén el idioma original.
- No inventes datos nuevos.
- Si la consulta ya es simple, devuélvela igual.
- Devuelve solo una subconsulta por línea.
- No uses numeración, viñetas ni markdown.

Consulta:
{query}

Subconsultas:
"""


STEP_BACK_PROMPT = TONE_AND_LANGUAGE_RULES + """
Genera una consulta inmobiliaria más general que capture la intención principal
de esta consulta específica.

REGLAS:
- No inventes barrio, precio, operación ni tipo de propiedad.
- Mantén el idioma original.
- Devuelve solo la consulta general, sin explicación.

Ejemplos:

Consulta específica:
"apartamento con terraza en Pocitos para familia con niños"

Consulta general:
"propiedad familiar con espacio exterior y buen entorno urbano"

Consulta específica:
{query}

Consulta general:
"""


HYDE_PROMPT = TONE_AND_LANGUAGE_RULES + """
Escribe un párrafo breve que describa la propiedad ideal que respondería
a la consulta del usuario.

Esta técnica se usa para mejorar retrieval semántico mediante HyDE:
el párrafo hipotético debe compartir vocabulario con los documentos indexados.

REGLAS:
- No inventes barrio, precio, operación ni tipo de propiedad si no aparecen.
- No prometas disponibilidad.
- No menciones datos externos.
- Mantén el idioma original.
- Escribe un único párrafo.
- Máximo 120 palabras.
- No uses markdown.

Consulta:
{query}

Párrafo hipotético:
"""

class QueryRewritingService:
    """
    Servicio de reescritura de consultas para el backend RAG inmobiliario.

    Responsabilidades:
      - Generar variantes semánticas de una consulta.
      - Reducir brecha entre lenguaje coloquial del usuario y texto indexado.
      - Apoyar /ask cuando use_query_rewriting=True.
      - Mantener compatibilidad con RAGGraphService mediante métodos explícitos.

    No hace retrieval.
    No consulta BigQuery.
    No construye filtros.
    No se usa en job-indexer.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        rewriting_temperature: float = DEFAULT_REWRITING_TEMPERATURE,
        precise_temperature: float = DEFAULT_PRECISE_TEMPERATURE,
        hyde_max_tokens: int = DEFAULT_HYDE_MAX_TOKENS,
    ) -> None:
        self.model = model or DEFAULT_REWRITING_MODEL
        self.rewriting_temperature = rewriting_temperature
        self.precise_temperature = precise_temperature
        self.hyde_max_tokens = hyde_max_tokens

        self.llm_diverse = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=self.rewriting_temperature,
        )

        self.llm_precise = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=self.precise_temperature,
            max_output_tokens=self.hyde_max_tokens,
        )

        self.zero_shot_prompt = ChatPromptTemplate.from_template(
            ZERO_SHOT_PROMPT
        )
        self.few_shot_prompt = ChatPromptTemplate.from_template(
            FEW_SHOT_PROMPT
        )
        self.decompose_prompt = ChatPromptTemplate.from_template(
            DECOMPOSE_PROMPT
        )
        self.step_back_prompt = ChatPromptTemplate.from_template(
            STEP_BACK_PROMPT
        )
        self.hyde_prompt = ChatPromptTemplate.from_template(
            HYDE_PROMPT
        )
        
        logger.info(
            "query_rewriting_service_initialized",
            extra={
                "model": self.model,
                "rewriting_temperature": self.rewriting_temperature,
                "precise_temperature": self.precise_temperature,
                "hyde_max_tokens": self.hyde_max_tokens,
            },
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _validate_query(query: str) -> str:
        normalized_query = (query or "").strip()

        if not normalized_query:
            raise ValueError("La pregunta no puede estar vacía.")

        return normalized_query

    @staticmethod
    def _clean_output(raw: str, limit: Optional[int] = None) -> list[str]:
        """
        Limpia el texto crudo del LLM y lo convierte en lista de queries.

        Elimina:
          - numeración;
          - viñetas;
          - comillas envolventes;
          - encabezados;
          - líneas vacías.
        """
        prefix_pattern = re.compile(
            r"^\s*(?:\d+[\.\)\-]\s+|[-•*·]\s+)"
        )

        cleaned: list[str] = []

        for line in (raw or "").strip().split("\n"):
            line = line.strip()

            if not line:
                continue

            if line.endswith(":"):
                continue

            line = prefix_pattern.sub("", line).strip()
            line = line.strip('"').strip("'").strip()

            if line:
                cleaned.append(line)

        return cleaned[:limit] if limit else cleaned

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []

        for value in values:
            normalized = value.strip()

            if not normalized:
                continue

            key = normalized.lower()

            if key not in seen:
                seen.add(key)
                result.append(normalized)

        return result

    # =========================================================================
    # Técnicas individuales
    # =========================================================================

    @traceable(name="zero_shot_rewrite", run_type="llm")
    def zero_shot_rewrite(
        self,
        query: str,
        num_queries: int = 3,
    ) -> list[str]:
        """
        Genera reformulaciones sin ejemplos guía.
        """
        normalized_query = self._validate_query(query)

        logger.info(
            "zero_shot_rewrite_started",
            extra={
                "query_length": len(normalized_query),
                "num_queries": num_queries,
            },
        )

        try:
            chain = self.zero_shot_prompt | self.llm_diverse

            response = chain.invoke(
                {
                    "query": normalized_query,
                    "num_queries": num_queries,
                }
            )

            queries = self._clean_output(
                response.content,
                limit=num_queries,
            )

            queries = self._dedupe_preserve_order(queries)

            logger.info(
                "zero_shot_rewrite_completed",
                extra={
                    "generated_count": len(queries),
                    "queries": queries,
                },
            )

            return queries

        except Exception as exc:
            logger.warning(
                "zero_shot_rewrite_failed",
                extra={"error": str(exc)},
            )
            return []

    @traceable(name="few_shot_rewrite", run_type="llm")
    def few_shot_rewrite(
        self,
        query: str,
        num_queries: int = 3,
    ) -> list[str]:
        """
        Genera reformulaciones guiadas por ejemplos inmobiliarios.

        Esta es la técnica recomendada por defecto para /ask.
        """
        normalized_query = self._validate_query(query)

        logger.info(
            "few_shot_rewrite_started",
            extra={
                "query_length": len(normalized_query),
                "num_queries": num_queries,
            },
        )

        try:
            chain = self.few_shot_prompt | self.llm_diverse

            response = chain.invoke(
                {
                    "query": normalized_query,
                    "num_queries": num_queries,
                }
            )

            queries = self._clean_output(
                response.content,
                limit=num_queries,
            )

            queries = self._dedupe_preserve_order(queries)

            logger.info(
                "few_shot_rewrite_completed",
                extra={
                    "generated_count": len(queries),
                    "queries": queries,
                },
            )

            return queries

        except Exception as exc:
            logger.warning(
                "few_shot_rewrite_failed",
                extra={"error": str(exc)},
            )
            return []

    @traceable(name="decompose_query", run_type="llm")
    def decompose_query(self, query: str) -> list[str]:
        """
        Descompone una consulta compuesta en subconsultas simples.
        """
        normalized_query = self._validate_query(query)

        logger.info(
            "decompose_query_started",
            extra={"query_length": len(normalized_query)},
        )

        try:
            chain = self.decompose_prompt | self.llm_diverse
            response = chain.invoke({"query": normalized_query})

            subqueries = self._clean_output(response.content)
            subqueries = self._dedupe_preserve_order(subqueries)

            if not subqueries:
                subqueries = [normalized_query]

            logger.info(
                "decompose_query_completed",
                extra={
                    "generated_count": len(subqueries),
                    "queries": subqueries,
                },
            )

            return subqueries

        except Exception as exc:
            logger.warning(
                "decompose_query_failed",
                extra={"error": str(exc)},
            )
            return [normalized_query]

    @traceable(name="step_back_rewrite", run_type="llm")
    def step_back_rewrite(self, query: str) -> str:
        """
        Genera una consulta más general que capture la intención principal.
        """
        normalized_query = self._validate_query(query)

        logger.info(
            "step_back_rewrite_started",
            extra={"query_length": len(normalized_query)},
        )

        try:
            chain = self.step_back_prompt | self.llm_precise
            response = chain.invoke({"query": normalized_query})

            output = (response.content or "").strip()

            if not output:
                output = normalized_query

            logger.info(
                "step_back_rewrite_completed",
                extra={
                    "output_length": len(output),
                    "output": output,
                },
            )

            return output

        except Exception as exc:
            logger.warning(
                "step_back_rewrite_failed",
                extra={"error": str(exc)},
            )
            return normalized_query

    @traceable(name="hyde_rewrite", run_type="llm")
    def hyde_rewrite(self, query: str) -> str:
        """
        Genera un documento hipotético breve estilo HyDE.

        Útil cuando la consulta del usuario es muy coloquial y se busca aproximar
        el vocabulario del índice FAISS.
        """
        normalized_query = self._validate_query(query)

        logger.info(
            "hyde_rewrite_started",
            extra={"query_length": len(normalized_query)},
        )

        try:
            chain = self.hyde_prompt | self.llm_precise
            response = chain.invoke({"query": normalized_query})

            output = (response.content or "").strip()

            if not output:
                output = normalized_query

            logger.info(
                "hyde_rewrite_completed",
                extra={
                    "output_length": len(output),
                    "word_count": len(output.split()),
                },
            )

            return output

        except Exception as exc:
            logger.warning(
                "hyde_rewrite_failed",
                extra={"error": str(exc)},
            )
            return normalized_query

    # =========================================================================
    # API conveniente para RAGGraphService
    # =========================================================================

    def rewrite(
        self,
        query: str,
        strategy: str = "few_shot_rewrite",
        num_queries: int = 3,
        include_original: bool = True,
    ) -> list[str]:
        """
        Ejecuta una estrategia de reescritura y retorna lista de queries.

        Args:
            query:
                Consulta original.
            strategy:
                Nombre de la estrategia:
                  - zero_shot_rewrite
                  - few_shot_rewrite
                  - decompose_query
                  - step_back_rewrite
                  - hyde_rewrite
            num_queries:
                Número de variantes para estrategias multi-query.
            include_original:
                Si True, incluye la consulta original como primera query.

        Returns:
            Lista deduplicada de consultas.
        """
        normalized_query = self._validate_query(query)

        logger.info(
            "query_rewrite_started",
            extra={
                "strategy": strategy,
                "query_length": len(normalized_query),
                "num_queries": num_queries,
                "include_original": include_original,
            },
        )

        generated: list[str]

        if strategy == "zero_shot_rewrite":
            generated = self.zero_shot_rewrite(
                normalized_query,
                num_queries=num_queries,
            )

        elif strategy == "few_shot_rewrite":
            generated = self.few_shot_rewrite(
                normalized_query,
                num_queries=num_queries,
            )

        elif strategy == "decompose_query":
            generated = self.decompose_query(normalized_query)

        elif strategy == "step_back_rewrite":
            generated = [self.step_back_rewrite(normalized_query)]

        elif strategy == "hyde_rewrite":
            generated = [self.hyde_rewrite(normalized_query)]

        else:
            logger.warning(
                "query_rewrite_unknown_strategy",
                extra={"strategy": strategy},
            )
            generated = []

        queries = (
            [normalized_query, *generated]
            if include_original
            else generated
        )

        queries = self._dedupe_preserve_order(queries)

        logger.info(
            "query_rewrite_completed",
            extra={
                "strategy": strategy,
                "queries_count": len(queries),
                "queries": queries,
            },
        )

        return queries
