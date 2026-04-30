"""
Servicio Query Rewriting
==========================================

Este módulo implementa la funcionalidad de Resscritura de 
Consultas realizadas por el usuario. Implementa 5 técnicas de reescritura de consultas para mejorar el retrieval
del sistema RAG de Tutor-IA, todas trazables con LangSmith via @traceable.

Técnicas disponibles:
    1. zero_shot_rewrite   
    2. few_shot_rewrite    1a Opción
    3. decompose_query     3a Opción
    4. step_back_rewrite   
    5. hyde_rewrite        2a Opción

"""

import logging
import re
from typing import List

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

logger = logging.getLogger(__name__)


# ===========================================================================
# CONFIGURACIÓN DEL SERVICIO
# ===========================================================================

# Modelo de lenguaje 
REWRITING_MODEL = "gemini-2.5-flash"

# Temperatura para técnicas que necesitan diversidad (zero_shot, few_shot, decompose)
REWRITING_TEMPERATURE = 0.3  # Más alta que generation_service (0.2) para variedad entre variantes

# Temperatura para técnicas que necesitan coherencia técnica (step_back, hyde)
PRECISE_TEMPERATURE = 0.3

# Límite de tokens para hyde_rewrite (conciso = menos alucinaciones)
HYDE_MAX_TOKENS = 300


# ===========================================================================
# PROMPTS DEL SISTEMA
# ===========================================================================

ZERO_SHOT_PROMPT = """\
Reformula la siguiente pregunta de {num_queries} maneras distintas para mejorar
la búsqueda en una base de conocimientos técnica sobre Python, JavaScript, React, GO, 
Git, Machine Learning y LLMs.

REGLAS:
- Mantén el idioma de la pregunta original (español o inglés).
- Varía el nivel técnico: al menos una variante formal/técnica y una coloquial.
- Usa sinónimos y términos alternativos del dominio.
- NO añadas información nueva; solo reformula la intención original.
- Devuelve ÚNICAMENTE las {num_queries} preguntas, una por línea, sin numeración ni viñetas.

Pregunta original: {query}
Reformulaciones:"""


# Few-shot examples as structured dicts for FewShotChatMessagePromptTemplate.
# Each "input" is the original query; each "output" is the expected reformulations,
# one per line with no numbering — exactly what the LLM should produce.
FEW_SHOT_EXAMPLES = [
    {
        "input": "¿Cómo funciona React?",
        "output": (
            "¿Cuál es la arquitectura de componentes y el ciclo de renderizado de React?\n"
            "How does React's virtual DOM reconciliation work?\n"
            "Explícame React de forma sencilla, como si empezara desde cero"
        ),
    },
    {
        "input": "What is overfitting in machine learning?",
        "output": (
            "How does a model overfit training data and how is it prevented?\n"
            "¿Qué es el sobreajuste en ML y cómo se detecta con curvas de validación?\n"
            "Why does my model perform well on training data but poorly on new data?"
        ),
    },
    {
        "input": "¿Qué es un closure en JavaScript?",
        "output": (
            "¿Cómo funciona el alcance léxico y los closures en JS?\n"
            "What is a JavaScript closure and when should you use one?\n"
            "¿Por qué una función puede acceder a variables de su función contenedora en JS?"
        ),
    },
    {
        "input": "How do I use Git branches?",
        "output": (
            "What is the Git branching model and how are branches created and merged?\n"
            "¿Cómo se crean, fusionan y eliminan ramas en Git?\n"
            "Explain Git branches for someone who only knows linear commit history"
        ),
    },
]

# System message for few_shot_rewrite (injected once, before the examples)
FEW_SHOT_SYSTEM_MESSAGE =  """\
Reformula la siguiente pregunta de {num_queries} maneras distintas para mejorar
la búsqueda en una base de conocimientos técnica sobre Python, JavaScript, React,
Git, Machine Learning y LLMs.

REGLAS:
- Mantén el idioma predominante de la pregunta original (español o inglés).
- Varía el nivel técnico: al menos una variante técnica/formal y una coloquial/conceptual.
- Usa sinónimos y términos alternativos del dominio (ej: "lista" → "array" → "colección").
- Si la pregunta mezcla idiomas, genera variantes en ambos.
- NO añadas información nueva; solo reformula la intención original.
- Devuelve ÚNICAMENTE las {num_queries} preguntas, una por línea, sin numeración ni viñetas.
"""


DECOMPOSE_PROMPT = """\
La siguiente pregunta puede contener múltiples conceptos o sub-preguntas.
Descompónla en preguntas atómicas más simples que puedan responderse de forma independiente.

REGLAS:
- Cada sub-pregunta debe ser autocontenida (comprensible sin leer las demás).
- Mantén el idioma de la pregunta original (español o inglés).
- Si la pregunta ya es simple y atómica, devuélvela tal cual en una sola línea.
- Devuelve ÚNICAMENTE las sub-preguntas, una por línea, sin numeración ni viñetas.

EJEMPLOS:

Pregunta: ¿Qué es el overfitting, cómo se detecta y qué técnicas existen para evitarlo?
Sub-preguntas:
¿Qué es el overfitting en machine learning?
¿Cómo se detecta el overfitting durante el entrenamiento de un modelo?
¿Qué técnicas se usan para prevenir el overfitting?

Pregunta: How do React hooks work and what are the rules for using useState and useEffect?
Sub-preguntas:
What are React hooks and why were they introduced?
How does the useState hook work in React?
How does the useEffect hook work and when does it run?
What are the rules of hooks in React?

---
Pregunta: {query}
Sub-preguntas:"""


STEP_BACK_PROMPT = """\
Aplica la técnica "step-back": dado una pregunta específica, genera UNA SOLA pregunta
más general que capture el principio o concepto subyacente. Esto permite recuperar
contexto conceptual de mayor nivel antes de responder la pregunta concreta.

REGLAS:
- La pregunta generalizada debe cubrir el concepto fundamental de la original.
- Mantén el idioma de la pregunta original (español o inglés).
- Devuelve ÚNICAMENTE la pregunta generalizada, sin explicaciones ni prefijos.

EJEMPLOS:

Pregunta específica: ¿Cómo uso useEffect para hacer una llamada a una API en React?
Pregunta general: ¿Cómo funciona el ciclo de vida de los componentes y los efectos secundarios en React?

Pregunta específica: Why does my Python list comprehension run slower than a for loop?
Pregunta general: What are the performance characteristics of different iteration methods in Python?

Pregunta específica: ¿Cómo hago un git rebase interactivo para limpiar mis commits?
Pregunta general: ¿Cuáles son las estrategias de gestión del historial de commits en Git?

---
Pregunta específica: {query}
Pregunta general:"""


HYDE_PROMPT = """\
Eres un autor técnico experto redactando un fragmento de un libro educativo.

Dado la siguiente pregunta de un estudiante, escribe UN ÚNICO PÁRRAFO CONCISO
que respondería esa pregunta tal como aparecería en un libro técnico de referencia
sobre Python, JavaScript, React, Git, GO, Machine Learning o LLMs.

REGLAS:
- Usa terminología técnica precisa del área correspondiente.
- Responde en el mismo idioma de la pregunta (español o inglés).
- Si la pregunta es sobre código, incluye un fragmento corto de ejemplo en el párrafo.
- Escribe solo prosa técnica continua: sin encabezados, listas ni markdown.
- Sé conciso (máximo 150 palabras). El objetivo es compartir vocabulario con los
  fragmentos reales del libro, no generar una respuesta completa.

EJEMPLOS:

Pregunta: ¿Qué es un closure en JavaScript?
Párrafo hipotético:
Un closure en JavaScript es la combinación de una función y el entorno léxico en el que
fue declarada, lo que permite que la función interior acceda a variables de su ámbito
exterior incluso después de que la función exterior haya terminado de ejecutarse. Por
ejemplo, `function contador() {{ let n = 0; return () => ++n; }}` crea un closure donde
`n` persiste entre llamadas sucesivas. Los closures son fundamentales para encapsulación
de estado privado y patrones como el módulo en JavaScript moderno.

Pregunta: How does gradient descent work?
Párrafo hipotético:
Gradient descent is an iterative optimization algorithm that minimizes a loss function
by updating model parameters in the direction of the negative gradient. At each step,
weights are adjusted as `w = w - lr * ∂L/∂w`, where `lr` is the learning rate. Choosing
an appropriate learning rate is critical: too large causes divergence, too small leads to
slow convergence. Variants such as stochastic gradient descent and Adam improve
convergence by using mini-batches and adaptive learning rates respectively.

---
Pregunta: {query}
Párrafo hipotético:"""


# ===========================================================================
# CLASE PRINCIPAL
# ===========================================================================

class QueryRewritingService:
    """
    Servicio de reescritura de consultas para el sistema RAG Tutor-IA.

    Mejora la calidad del retrieval FAISS atacando los problemas identificados
    en la evaluación RAGAS:
        - Vocabulario coloquial vs. técnico: few_shot_rewrite, zero_shot_rewrite
        - Brecha semántica pregunta vs. libro: hyde_rewrite
        - Preguntas compuestas que diluyen: decompose_query
        - Falta de contexto conceptual: step_back_rewrite

    Técnicas recomendadas para Tutor-IA:
        ★ Primaria:   few_shot_rewrite  (controla cambio ES/EN y nivel técnico)
        ★ Secundaria: hyde_rewrite      (reduce brecha embedding pregunta vs. libro)
    """

    def __init__(
        self,
        model: str = REWRITING_MODEL,
        rewriting_temperature: float = REWRITING_TEMPERATURE,
        precise_temperature: float = PRECISE_TEMPERATURE,
        hyde_max_tokens: int = HYDE_MAX_TOKENS,
    ):
        """
        Inicializa el servicio con dos instancias LLM especializadas por temperatura.

        Se usan dos instancias de ChatGoogleGenerativeAI porque las técnicas tienen
        necesidades opuestas: diversidad (zero_shot, few_shot, decompose) vs.
        coherencia técnica (step_back, hyde).

        Args:
            model:                  Modelo Gemini a utilizar.
            rewriting_temperature:  Temperatura para técnicas de reformulación.
                                    Más alta = más diversidad entre variantes.
            precise_temperature:    Temperatura para técnicas de abstracción/hipótesis.
                                    Más baja = respuestas técnicamente coherentes.
            hyde_max_tokens:        Límite de tokens para hyde_rewrite.
        """

        # LLM para diversidad: zero_shot, few_shot, decompose
        self.llm_diverse = ChatGoogleGenerativeAI(
            model=model,
            temperature=rewriting_temperature,
        )

        # LLM para coherencia técnica: step_back, hyde
        self.llm_precise = ChatGoogleGenerativeAI(
            model=model,
            temperature=precise_temperature,
            max_output_tokens=hyde_max_tokens,
        )

        # --- Prompts simples (zero_shot, decompose, step_back, hyde) ---
        self.zero_shot_prompt = ChatPromptTemplate.from_template(ZERO_SHOT_PROMPT)
        self.decompose_prompt = ChatPromptTemplate.from_template(DECOMPOSE_PROMPT)
        self.step_back_prompt = ChatPromptTemplate.from_template(STEP_BACK_PROMPT)
        self.hyde_prompt      = ChatPromptTemplate.from_template(HYDE_PROMPT)

        # --- Prompt few-shot con FewShotChatMessagePromptTemplate ---
        # Plantilla para formatear cada ejemplo como par (human, ai)
        example_prompt = ChatPromptTemplate.from_messages([
            ("human",  "{input}"),
            ("ai",     "{output}"),
        ])

        # Bloque de ejemplos estructurados (human/ai alternados)
        few_shot_examples = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=FEW_SHOT_EXAMPLES,
        )

        # Prompt final: system → ejemplos → query actual del usuario
        self.few_shot_prompt = ChatPromptTemplate.from_messages([
            ("system", FEW_SHOT_SYSTEM_MESSAGE),
            few_shot_examples,
            ("human",  "{query}"),
        ])

        logger.info(
            f"[QueryRewritingService] Inicializado | model={model} | "
            f"temp_diverse={rewriting_temperature} | temp_precise={precise_temperature}"
        )

    # -----------------------------------------------------------------------
    # HELPER INTERNO
    # -----------------------------------------------------------------------

    def _clean_output(self, raw: str, limit: int = None) -> List[str]:
        """
        Convierte el texto crudo del LLM en una lista de strings limpios.

        Gemini puede devolver su salida con distintos artefactos de formato
        dependiendo de la temperatura y el prompt. Este método los elimina todos:

            Numeración:   "1. pregunta"  "2) pregunta"  "1- pregunta"
            Viñetas:      "- pregunta"   "• pregunta"   "* pregunta"   "· pregunta"
            Comillas:     '"pregunta"'   "'pregunta'"
            Encabezados:  líneas que terminan en ":" (ej: "Reformulaciones:")
            Espacios:     strip() en cada línea y en el resultado final

        Args:
            raw:   Texto crudo devuelto por el LLM (response.content).
            limit: Número máximo de líneas a retornar (None = sin límite).

        Returns:
            Lista de strings limpios, listos para usar como queries.
        """
        # Patrón que captura prefijos de numeración o viñetas al inicio de línea:
        # "1. " | "1) " | "1- " | "- " | "• " | "* " | "· "
        prefix_pattern = re.compile(r"^\s*(?:\d+[\.\)\-]\s+|[-•*·]\s+)")

        cleaned = []
        for line in raw.strip().split("\n"):
            line = line.strip()

            # Descartar líneas vacías
            if not line:
                continue

            # Descartar líneas que son encabezados del modelo (terminan en ":")
            if line.endswith(":"):
                continue

            # Eliminar prefijo de numeración o viñeta
            line = prefix_pattern.sub("", line).strip()

            # Eliminar comillas envolventes que el modelo a veces añade
            if (line.startswith('"') and line.endswith('"')) or \
               (line.startswith("'") and line.endswith("'")):
                line = line[1:-1].strip()

            if line:
                cleaned.append(line)

        return cleaned[:limit] if limit else cleaned

    # -----------------------------------------------------------------------
    # TÉCNICA 1: ZERO-SHOT REWRITE
    # -----------------------------------------------------------------------

    @traceable(name="zero_shot_rewrite", run_type="llm")
    def zero_shot_rewrite(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Genera N reformulaciones de la pregunta sin ejemplos guía (zero-shot).

        Útil como fallback rápido y de bajo costo. Sin ejemplos, el modelo tiene
        más libertad pero menos control sobre el idioma o nivel técnico resultante.
        Para el corpus bilingüe de Tutor-IA, preferir few_shot_rewrite.

        Args:
            query:       Pregunta original del usuario.
            num_queries: Número de reformulaciones a generar (default: 3).

        Returns:
            Lista de hasta `num_queries` reformulaciones.
            Retorna lista vacía si ocurre un error (el pipeline usa la original).

        Raises:
            ValueError: Si query está vacío.
        """
        if not query or not query.strip():
            raise ValueError("La pregunta no puede estar vacía.")

        logger.info(f"[zero_shot_rewrite] Generando {num_queries} variantes para: '{query}'")

        try:
            chain    = self.zero_shot_prompt | self.llm_diverse
            response = chain.invoke({"query": query, "num_queries": num_queries})
            queries  = self._clean_output(response.content, limit=num_queries)

            logger.info(f"[zero_shot_rewrite] {len(queries)} variantes generadas.")
            return queries

        except Exception as e:
            logger.error(f"[zero_shot_rewrite] Error: {e}")
            return []

    # -----------------------------------------------------------------------
    # TÉCNICA 2: FEW-SHOT REWRITE  ★ RECOMENDADA PRIMARIA
    # -----------------------------------------------------------------------

    @traceable(name="few_shot_rewrite", run_type="llm")
    def few_shot_rewrite(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Genera N reformulaciones guiadas por ejemplos few-shot en ES/EN.

        Técnica primaria recomendada para Tutor-IA. Los ejemplos en el prompt
        modelan explícitamente el cambio de idioma y nivel técnico, dando al LLM
        patrones concretos para manejar el vocabulario bilingüe del corpus.

        Args:
            query:       Pregunta original del usuario.
            num_queries: Número de reformulaciones a generar (default: 3).

        Returns:
            Lista de hasta `num_queries` reformulaciones.
            Retorna lista vacía si ocurre un error (el pipeline usa la original).

        Raises:
            ValueError: Si query está vacío.
        """
        if not query or not query.strip():
            raise ValueError("La pregunta no puede estar vacía.")

        logger.info(f"[few_shot_rewrite] Generando {num_queries} variantes para: '{query}'")

        try:
            chain    = self.few_shot_prompt | self.llm_diverse
            response = chain.invoke({"query": query, "num_queries": num_queries})
            queries  = self._clean_output(response.content, limit=num_queries)

            logger.info(f"[few_shot_rewrite] {len(queries)} variantes generadas.")
            return queries

        except Exception as e:
            logger.error(f"[few_shot_rewrite] Error: {e}")
            return []

    # -----------------------------------------------------------------------
    # TÉCNICA 3: DECOMPOSE QUERY
    # -----------------------------------------------------------------------

    @traceable(name="decompose_query", run_type="llm")
    def decompose_query(self, query: str) -> List[str]:
        """
        Descompone una pregunta compuesta en sub-preguntas atómicas independientes.

        Cuando un estudiante pregunta sobre varios conceptos a la vez, un solo
        retrieval no cubre todos los aspectos. Esta técnica genera sub-preguntas
        autocontenidas para recuperar contexto relevante para cada una.

        Si la pregunta ya es atómica, el modelo la retorna tal cual.

        Args:
            query: Pregunta original del usuario (posiblemente compuesta).

        Returns:
            Lista de sub-preguntas atómicas.
            Retorna [query] si ocurre un error (el pipeline usa la original).

        Raises:
            ValueError: Si query está vacío.
        """
        if not query or not query.strip():
            raise ValueError("La pregunta no puede estar vacía.")

        logger.info(f"[decompose_query] Descomponiendo: '{query}'")

        try:
            chain       = self.decompose_prompt | self.llm_diverse
            response    = chain.invoke({"query": query})
            sub_queries = self._clean_output(response.content)

            # Si el modelo devolvió algo vacío, conservar la original
            if not sub_queries:
                return [query]

            logger.info(f"[decompose_query] {len(sub_queries)} sub-preguntas generadas.")
            return sub_queries

        except Exception as e:
            logger.error(f"[decompose_query] Error: {e}")
            return [query]

    # -----------------------------------------------------------------------
    # TÉCNICA 4: STEP-BACK REWRITE
    # -----------------------------------------------------------------------

    @traceable(name="step_back_rewrite", run_type="llm")
    def step_back_rewrite(self, query: str) -> str:
        """
        Genera una pregunta más general que captura el principio subyacente.

        Útil para preguntas muy específicas o procedurales donde el corpus puede
        no tener la respuesta exacta pero sí los conceptos fundamentales.
        Ej: "¿Cómo hago git rebase -i?" → "¿Cómo se gestiona el historial en Git?"

        La pregunta generalizada se combina con la original en el retrieval para
        recuperar contexto tanto conceptual como específico.

        Args:
            query: Pregunta original del usuario.

        Returns:
            Pregunta generalizada como string.
            Retorna query original si ocurre un error.

        Raises:
            ValueError: Si query está vacío.
        """
        if not query or not query.strip():
            raise ValueError("La pregunta no puede estar vacía.")

        logger.info(f"[step_back_rewrite] Generando pregunta general para: '{query}'")

        try:
            chain     = self.step_back_prompt | self.llm_precise
            response  = chain.invoke({"query": query})
            step_back = response.content.strip()

            if not step_back:
                return query

            logger.info(f"[step_back_rewrite] Pregunta general generada: '{step_back}'")
            return step_back

        except Exception as e:
            logger.error(f"[step_back_rewrite] Error: {e}")
            return query

    # -----------------------------------------------------------------------
    # TÉCNICA 5: HYDE REWRITE  ★ RECOMENDADA SECUNDARIA
    # -----------------------------------------------------------------------

    @traceable(name="hyde_rewrite", run_type="llm")
    def hyde_rewrite(self, query: str) -> str:
        """
        Genera una respuesta hipotética estilo libro técnico (HyDE).

        Técnica secundaria recomendada para Tutor-IA. En lugar de buscar con
        la pregunta del usuario (vocabulario coloquial), HyDE genera un párrafo
        con vocabulario técnico de libro. El embedding de ese párrafo es más
        cercano a los fragmentos reales del corpus que el embedding de la pregunta
        original, reduciendo la brecha semántica identificada en la evaluación RAGAS.

        Args:
            query: Pregunta original del usuario.

        Returns:
            Texto del documento hipotético como string.
            Retorna query original si ocurre un error.

        Raises:
            ValueError: Si query está vacío.
        """
        if not query or not query.strip():
            raise ValueError("La pregunta no puede estar vacía.")

        logger.info(f"[hyde_rewrite] Generando documento hipotético para: '{query}'")

        try:
            chain    = self.hyde_prompt | self.llm_precise
            response = chain.invoke({"query": query})
            hyde_doc = response.content.strip()

            if not hyde_doc:
                return query

            logger.info(
                f"[hyde_rewrite] Documento hipotético generado "
                f"({len(hyde_doc.split())} palabras)."
            )
            return hyde_doc

        except Exception as e:
            logger.error(f"[hyde_rewrite] Error: {e}")
            return query
