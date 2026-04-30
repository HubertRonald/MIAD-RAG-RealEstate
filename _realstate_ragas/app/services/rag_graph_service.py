"""
Servicio de Orquestación RAG con LangGraph
==========================================

Este módulo implementa el flujo RAG completo usando LangGraph.

Flujo del grafo (todos los nodos activos):
    START → rewrite_query → retrieve_documents → rerank_documents → generate_answer → END

    - rewrite_query:      Reformula la pregunta con QueryRewritingService para
                          mejorar el recall del retrieval.
    - retrieve_documents: Recupera documentos para cada query reescrita y los
                          deduplica por contenido antes del reranking.
    - rerank_documents:   Reordena los documentos por relevancia con Cross-Encoder,
                          filtrando al top-k más relevante antes de la generación.
    - generate_answer:    Genera la respuesta final con el contexto rerankeado.

Cada nodo es opcional vía flags de configuración en __init__:
    use_query_rewriting=False → omite rewrite_query
    use_reranking=False       → omite rerank_documents

"""
import logging
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langchain.schema import Document

from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService
from app.services.query_rewriting_service import QueryRewritingService
from app.services.reranking_service import RerankingService

logger = logging.getLogger(__name__)


# ===========================================================================
# ESTADO DEL GRAFO
# ===========================================================================

class RAGState(TypedDict):
    """
    Representa el estado compartido que fluye entre los nodos del grafo RAG.
    
     Atributos:
        question:          Pregunta original del usuario (inmutable durante el flujo).
        rewritten_queries: Lista de queries generadas por el nodo rewrite_query.
                           Incluye la pregunta original + variantes reescritas.
                           Vacía hasta que el nodo de reescritura la popula.
        documents:         Fragmentos de texto recuperados del vectorstore FAISS.
                           Se convierte en List[str] después del nodo generate_answer.
        sources:           Nombres de archivo fuente de los documentos recuperados.
        answer:            Respuesta final generada por el LLM.
    """
    question:          str
    rewritten_queries: List[str]
    documents: List[str]
    sources: List[str]
    answer: str


# ===========================================================================
# SERVICIO PRINCIPAL
# ===========================================================================

class RAGGraphService:
    """
    Servicio que orquesta el flujo RAG completo usando LangGraph.
    
    Este servicio integra QueryRewritingService, RetrievalService y GenerationService en un
    grafo de estados explícito y trazable con LangSmith.
    """
    
    def __init__(
        self,
        retrieval_service: RetrievalService,
        generation_service: GenerationService,
        query_rewriting_service: Optional[QueryRewritingService] = None,
        reranking_service: Optional[RerankingService] = None,
        rewriting_strategy: Optional[str] = "few_shot_rewrite",
        use_query_rewriting: bool = True,
        use_reranking: bool = True,
        text_reranking_service: Optional[RerankingService] = None,
        bypass_text_reranking: bool = True,
    ):
        """
        Inicializa el servicio de orquestación RAG.

        Args:
            retrieval_service: Servicio de recuperación de documentos
            generation_service: Servicio de generación de respuestas
            query_rewriting_service: Servicio de reescritura de consultas.
                                     Si es None, el nodo rewrite_query se omite
                                     y el grafo corre el flujo original.
            rewriting_strategy:      Técnica de reescritura a usar. 
            use_query_rewriting:     Si True y query_rewriting_service no es None,
                                     activa el nodo rewrite_query en el grafo.
            use_reranking:           Si True y reranking_service no es None,
                                     activa el nodo rerank_documents en el grafo.

        """

        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
        self.query_rewriting_service = query_rewriting_service
        self.reranking_service       = reranking_service
        self.rewriting_strategy      = rewriting_strategy
        # Reranker de texto: si no se provee, el routing omitirá el reranking
        # para chunks de texto puro (comportamiento óptimo según los resultados)
        self.text_reranking_service  = text_reranking_service
        # Si True, omite el reranking cuando todos los chunks recuperados son texto puro.
        # Permite comparar el impacto del reranking en chunks de texto vs visuales.
        self.bypass_text_reranking   = bypass_text_reranking

        # Flags efectivos: requieren tanto el servicio inyectado como el flag activo
        self.rewriting_enabled = (
            use_query_rewriting
            and query_rewriting_service is not None
            and rewriting_strategy is not None
        )
        self.reranking_enabled = (
            use_reranking
            and reranking_service is not None
        )
        
        self.rag_app = self._build_graph()

        logger.info(
            f"[RAGGraphService] Inicializado | "
            f"rewriting={'ON' + rewriting_strategy if self.rewriting_enabled else 'OFF'} | "
            f"reranking={'ON' if self.reranking_enabled else 'OFF'}"
        )

    # -----------------------------------------------------------------------
    # NODO 1: REWRITE QUERY  (nuevo)
    # -----------------------------------------------------------------------

    def _rewrite_query_node(self, state: RAGState) -> RAGState:
        """
        Nodo de reescritura de consultas del flujo RAG.

        Llama a la técnica configurada en `self.rewriting_strategy` del
        QueryRewritingService y almacena todas las queries resultantes
        (incluida la original) en `state["rewritten_queries"]`.

        Si la reescritura falla, el nodo degrada gracefully: popula
        `rewritten_queries` solo con la pregunta original para que el
        nodo de retrieval siempre tenga algo con qué trabajar.

        Args:
            state: Estado actual con la pregunta original del usuario.

        Returns:
            Estado con `rewritten_queries` diligenciado.
        """
        question = state["question"]
        logger.info(f"[rewrite_query_node] Reescribiendo: '{question}' | estrategia='{self.rewriting_strategy}'")

        try:
            technique = getattr(self.query_rewriting_service, self.rewriting_strategy)
            result    = technique(question)

            # Normalizar la salida a List[str] (hyde y step_back retornan str)
            if isinstance(result, str):
                rewritten_queries = [question, result] if result != question else [question]
            else:
                # List[str]: incluir la original al inicio si no está ya
                rewritten_queries = [question] + [q for q in result if q != question]

            logger.info(
                f"[rewrite_query_node] {len(rewritten_queries)} queries generadas: "
                + str(rewritten_queries)
            )

        except Exception as e:
            logger.error(f"[rewrite_query_node] Error en reescritura, usando original. Error: {e}")
            rewritten_queries = [question]

        state["rewritten_queries"] = rewritten_queries
        return state

    # -----------------------------------------------------------------------
    # NODO 2: RETRIEVE DOCUMENTS  (actualizado)
    # -----------------------------------------------------------------------

    def _retrieve_documents_node(self, state: RAGState) -> RAGState:
        """
        Nodo de recuperación del flujo RAG.
        
                Itera sobre todas las queries en `state["rewritten_queries"]`,
        llama al retriever una vez por query y deduplica los resultados
        por `page_content` antes de almacenarlos en el estado.

        La deduplicación preserva el orden de aparición (first-seen wins),
        lo que favorece los documentos recuperados por la query original
        o las primeras reformulaciones (generalmente las más precisas).

        Args:
            state: Estado actual con `rewritten_queries` diligenciado.

        Returns:
            Estado con `documents` (List[Document]) diligenciado.
        """
        queries = state.get("rewritten_queries") or [state["question"]]

        logger.info(f"[retrieve_documents_node] Recuperando documentos para {len(queries)} queries.")

        # Recuperar documentos para cada query y deduplicar por contenido
        seen_contents: set  = set()
        all_documents: List[Document] = []

        for query in queries:
            docs = self.retrieval_service.retrieve_documents(query)
            for doc in docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    all_documents.append(doc)

        logger.info(
            f"[retrieve_documents_node] {len(all_documents)} documentos únicos recuperados "
            f"(de {len(queries)} queries)."
        )

        state["documents"] = all_documents
        return state
    

    # -----------------------------------------------------------------------
    # NODO 3: RERANK DOCUMENTS  (nuevo)
    # -----------------------------------------------------------------------

    def _has_visual_chunks(self, documents: list) -> bool:
        """
        Determina si alguno de los chunks recuperados contiene contenido visual.

        Usa chunk_type del metadata (poblado en ChunkingService) como señal
        primaria, con [FIGURA] en el contenido como fallback para chunks
        sin metadata completa.

        Args:
            documents: Lista de Documents recuperados por FAISS.

        Returns:
            True si al menos un chunk es de tipo figure o table.
        """
        for doc in documents:
            chunk_type = doc.metadata.get("chunk_type", "unknown")
            if chunk_type in {"figure", "table"}:
                return True
            # Fallback: VLM descriptions are always tagged with [FIGURA]
            if "[FIGURA]" in doc.page_content:
                return True
        return False

    def _rerank_documents_node(self, state: RAGState) -> RAGState:
        """
        Nodo de reranking del flujo RAG con routing por tipo de contenido.

        Examina los chunks recuperados y decide qué reranker aplicar:

            - Chunks visuales (figure / table / [FIGURA]):
              Usa self.reranking_service (BGE multilingüe), que evalúa
              correctamente descripciones VLM en español.

            - Chunks de texto puro:
              Usa self.text_reranking_service si está configurado, o
              devuelve los documentos sin reordenar si no lo está.
              (Los resultados muestran que el reranking perjudica el
              context_recall en preguntas de texto puro.)

        Usa la pregunta ORIGINAL del usuario (no las reescritas) porque el
        reranking mide relevancia para la intención real, no para las variantes
        generadas para ampliar el recall del retrieval.

        Args:
            state: Estado con `documents` (List[Document]) del nodo anterior.

        Returns:
            Estado con `documents` reordenados y filtrados al top-k.
        """
        question  = state["question"]
        documents = state["documents"]

        is_visual = self._has_visual_chunks(documents)

        if is_visual or not self.bypass_text_reranking:
            # Rerank if visual content detected OR bypass is explicitly disabled
            logger.info(
                f"[rerank_documents_node] {'Contenido visual' if is_visual else 'bypass_text_reranking=False'} → BGE reranker | "
                f"{len(documents)} documentos para query: '{question[:60]}'"
            )
            reranked = self.reranking_service.rerank_documents(
                query=question,
                documents=documents,
            )
        elif self.text_reranking_service is not None:
            logger.info(
                f"[rerank_documents_node] Contenido texto → text reranker | "
                f"{len(documents)} documentos para query: '{question[:60]}'"
            )
            reranked = self.text_reranking_service.rerank_documents(
                query=question,
                documents=documents,
            )
        else:
            logger.info(
                f"[rerank_documents_node] Bypassing reranking — text-only content detected | "
                f"{len(documents)} documentos pasados directamente a generación."
            )
            reranked = documents

        logger.info(f"[rerank_documents_node] {len(reranked)} documentos tras routing.")
        state["documents"] = reranked
        return state
    
    # -----------------------------------------------------------------------
    # NODO 4: GENERATE ANSWER  
    # -----------------------------------------------------------------------

    def _generate_answer_node(self, state: RAGState) -> RAGState:
        """
        Nodo de generación del flujo RAG.
        
        Este nodo utiliza el contexto recuperado para generar
        la respuesta final mediante el LLM.

        Args:
            state: Estado actual con `documents` (List[Document]) populado.
            
        Returns:
            Estado con `answer`, `sources` y `documents` (List[str]) populados.
        """
        
        # Obtener la pregunta y los documentos del estado
        question = state["question"]
        documents = state["documents"]  # List[Document]
        
        # Generar respuesta usando generation_service
        result = self.generation_service.generate_response(question, documents)
        
        # Actualizar estado con los resultados de la generación
        state["answer"] = result["answer"]
        state["sources"] = result["sources"]      # List[str] de generation_service
        state["documents"] = result["context"]    # List[str] de generation_service
        
        return state

    
    # -----------------------------------------------------------------------
    # CONSTRUCCIÓN DEL GRAFO 
    # -----------------------------------------------------------------------
     
    def _build_graph(self):
        """
        Construye y compila el grafo de estados RAG.
        
        Los nodos se añaden según los flags activos, generando hasta 4 flujos:

            Todos activos:
                START → rewrite_query → retrieve_documents → rerank_documents → generate_answer → END

            Sin reranking:
                START → rewrite_query → retrieve_documents → generate_answer → END

            Sin rewriting:
                START → retrieve_documents → rerank_documents → generate_answer → END

            Sin ninguno (flujo base):
                START → retrieve_documents → generate_answer → END

        Returns:
            Aplicación RAG compilada lista para ejecutar
        """
        # Crear el grafo de estados
        graph = StateGraph(RAGState)
        
        # Registrar nodos según flags
        if self.rewriting_enabled:
            graph.add_node("rewrite_query", self._rewrite_query_node)

        graph.add_node("retrieve_documents", self._retrieve_documents_node)

        if self.reranking_enabled:
            graph.add_node("rerank_documents", self._rerank_documents_node)

        graph.add_node("generate_answer", self._generate_answer_node)
        
        # Definir el flujo del grafo
        if self.rewriting_enabled:
            graph.set_entry_point("rewrite_query")
            graph.add_edge("rewrite_query",      "retrieve_documents")
        else:
            graph.set_entry_point("retrieve_documents")

        if self.reranking_enabled:
            graph.add_edge("retrieve_documents", "rerank_documents")
            graph.add_edge("rerank_documents",   "generate_answer")
        else:
            graph.add_edge("retrieve_documents", "generate_answer")

        graph.add_edge("generate_answer", END)
        
        # Compilar el grafo en una aplicación ejecutable
        return graph.compile()
    
    # -----------------------------------------------------------------------
    # PUNTO DE ENTRADA PÚBLICO
    # -----------------------------------------------------------------------
  
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Procesa una pregunta usando el flujo RAG completo.
        
        Este método ejecuta el grafo LangGraph y retorna el resultado
        en un formato compatible con el API REST.
        
        Args:
            question: Pregunta del usuario
            
        Returns:
            Dict con:
                answer:            Respuesta generada por el LLM.
                sources:           Lista de archivos fuente usados.
                context:           Lista de fragmentos de texto recuperados.
                question:          Pregunta original.
                rewritten_queries: Queries usadas en el retrieval (vacío si
                                   rewriting estaba desactivado).
        """
        initial_state: RAGState = {
            "question":          question,
            "rewritten_queries": [],
            "documents":         [],
            "sources":           [],
            "answer":            "",
        }

        result = self.rag_app.invoke(initial_state)

        return {
            "answer":            result["answer"],
            "sources":           result["sources"],
            "context":           result["documents"],
            "question":          result["question"],
            "rewritten_queries": result.get("rewritten_queries", []),
        }
    