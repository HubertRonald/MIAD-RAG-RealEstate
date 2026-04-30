"""
Servicio de Chunking para el Sistema RAG
========================================

Este módulo implementa la funcionalidad de dividir documentos en fragmentos (chunks)
más pequeños y manejables para su procesamiento en el sistema RAG.

Soporte multimodal (opcional):
    Cuando multimodal=True, la carga de PDFs se delega a MultimodalDocumentService,
    que combina texto extraído con descripciones de imágenes (VLM) y tablas
    estructuradas (pdfplumber) antes de que el texto llegue al splitter.
    Los archivos Markdown nunca usan procesamiento multimodal.

"""

from enum import Enum
from typing import List, Dict, Any
from pathlib import Path

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pypdf import PdfReader
import os

from app.utils.text_utils import normalize_text
from app.services.csv_document_service import CSVDocumentService


# Importación lazy para evitar errores si las dependencias multimodales
# (fitz, pdfplumber) no están instaladas en entornos que no las necesitan
def _get_multimodal_service():
    from app.services.multimodal_document_service import MultimodalDocumentService
    return MultimodalDocumentService()


class _LazyEmbeddings:
    """
    Lazy loading wrapper for embeddings model.
    
    Only initializes the embeddings model when actually called,
    not at import time. This prevents errors when GOOGLE_API_KEY
    is not available unless SEMANTIC strategy is actually used.
    """
    _instance = None
    
    def __call__(self):
        """Get or create the embeddings instance"""
        if self._instance is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not found in environment. "
                    "Add it to your .env file or set as environment variable."
                )
            
            self._instance = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key
            )
        
        return self._instance


# Configurar el modelo de embeddings para chunking semántico
# Se inicializa solo cuando se llama (lazy initialization)
SEMANTIC_EMBEDDINGS_MODEL = _LazyEmbeddings()

class ChunkingStrategy(str, Enum):
    """Estrategias de chunking disponibles"""
    RECURSIVE_CHARACTER = "recursive_character"
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    DOCUMENT_STRUCTURE = "document_structure"


class ChunkingService:
    """
    Servicio para segmentar documentos en chunks usando diferentes estrategias.

    """
    
    def __init__(
        self, 
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER
    ):
        """
        Inicializa el servicio de chunking con la estrategia seleccionada.
        
        Args:
            strategy: Estrategia de chunking a utilizar
        
        Raises:
            ValueError: Si la estrategia no está soportada
        """
        self.strategy = strategy
        

        if strategy == ChunkingStrategy.RECURSIVE_CHARACTER:
            self.splitter = self._create_recursive_character_splitter()
        elif strategy == ChunkingStrategy.FIXED_SIZE:
            self.splitter = self._create_fixed_size_splitter()
        elif strategy == ChunkingStrategy.SEMANTIC:
            self.splitter = self._create_semantic_splitter()
        elif strategy == ChunkingStrategy.DOCUMENT_STRUCTURE:
            self.splitter = self._create_document_structure_splitter()
        else:
            raise ValueError(f"Estrategia de chunking no soportada: {strategy}")        


    def _create_fixed_size_splitter(self):
        """
        Crea un splitter de tamaño fijo (CharacterTextSplitter).
        
        Esta estrategia divide documentos según un límite de tamaño definido,
        sin tener en cuenta el contenido semántico. Es simple y genera fragmentos uniformes.
        
        TODO: Implementar CharacterTextSplitter con los siguientes parámetros:
        
        - chunk_size
        - chunk_overlap
        - separator
        
        """
        return CharacterTextSplitter(
                chunk_size = 1500,      # Aprox. 300 palabras
                chunk_overlap = 150,    # 10% solapamiento
                separator = " "         # Se separa el texto por espacios en lugar de \n\n (ABC)
        )

    def _create_recursive_character_splitter(self):
        """
        Crea un splitter basado en estructura del texto (RecursiveCharacterTextSplitter).
        
        Esta estrategia respeta los límites naturales del contenido (párrafos, oraciones, etc.),
        manteniendo el flujo narrativo y preservando la coherencia semántica.
        
        Se implementa RecursiveCharacterTextSplitter con los siguientes parámetros:
        
        - chunk_size
        - chunk_overlap
        - separators
        
        Returns:
            RecursiveCharacterTextSplitter configurado

        """
        return RecursiveCharacterTextSplitter(
                chunk_size=1500,      # Aprox. 300 palabras
                chunk_overlap=300,    # 20% overlap
                separators = [           # Separadores en orden de prioridad
                        "\n\n",    # Párrafos
                        "\n",      # Líneas
                        ". ",      # Oraciones
                        ", ",      # Cláusulas
                        " ",       # Palabras
                        ""         # Caracteres individuales
                ]
        )

    def _create_multimodal_splitter(self):
        """
        Crea un splitter optimizado para contenido multimodal (PDFs con imágenes y tablas).

        Usa un chunk_size mayor que el splitter estándar para acomodar las descripciones
        de imágenes generadas por VLM y las tablas en formato Markdown, evitando que
        se corten a mitad de una descripción visual o una fila de tabla.

        Incremento justificado:
        - Las descripciones VLM de [FIGURA] tienen ~300-600 chars adicionales por imagen
        - Las tablas Markdown añaden ~200-400 chars adicionales por tabla
        - Una página con texto + 1 figura + 1 tabla puede superar fácilmente 1,500 chars
        - chunk_size=2000 da margen suficiente para páginas visualmente densas

        Returns:
            RecursiveCharacterTextSplitter configurado para contenido multimodal
        """
        return RecursiveCharacterTextSplitter(
                chunk_size=2000,      # +33% sobre el splitter estándar (1500)
                chunk_overlap=400,    # 20% overlap, proporcional al nuevo chunk_size
                separators=[
                        "\n\n",    # Párrafos
                        "\n",      # Líneas
                        ". ",       # Oraciones
                        ", ",       # Cláusulas
                        " ",        # Palabras
                        ""          # Caracteres individuales
                ]
        )

    def _create_document_structure_splitter(self):
        """
        Crea un splitter basado en estructura de documento (MarkdownHeaderTextSplitter).
        
        Esta estrategia aprovecha la organización jerárquica de documentos Markdown,
        dividiendo por encabezados para generar fragmentos semánticamente coherentes.
        Cada chunk representa una unidad conceptual completa.
        
        Se implementa MarkdownHeaderTextSplitter con los siguientes parámetros:
        
        - headers_to_split_on
        - strip_headers
        
        Returns:
            MarkdownHeaderTextSplitter configurado

        """
        return MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
                strip_headers=False  # Se conservan los encabezados dentro de los fragmentos
        )
    
    
    def _create_semantic_splitter(self):
        """
        Crea un splitter semántico (SemanticChunker).
        
        Esta estrategia considera directamente el contenido y significado del texto,
        utilizando embeddings para detectar transiciones temáticas significativas.
        Divide el texto donde el significado cambia de forma sustancial.
        
        Se implementa SemanticChunker con los siguientes pasos:
        
        1. Inicializar modelo de embeddings (usar SEMANTIC_EMBEDDINGS_MODEL)
        2. Crear SemanticChunker con el modelo de embeddings

        """
        # Llamar a SEMANTIC_EMBEDDINGS_MODEL() para obtener la instancia
        embeddings = SEMANTIC_EMBEDDINGS_MODEL()

        return SemanticChunker(
                                embeddings,
                                breakpoint_threshold_type = "percentile",
                                breakpoint_threshold_amount = 85.0 # Sensible a temas técnicos
                        )


    def load_documents_from_collection(self, collection_name: str, multimodal: bool = False) -> List[Document]:
        """
        Carga todos los documentos de una colección.
        
        Busca archivos PDF y Markdown en ./docs/{collection_name}, extrae el texto
        y crea objetos Document con metadatos enriquecidos.
        
        Args:
            collection_name: Nombre de la colección
            multimodal: Si True, los PDFs se procesan con MultimodalDocumentService
                        (texto + descripciones de imágenes + tablas estructuradas).
                        Los archivos Markdown ignoran este flag.
            
        Returns:
            Lista de objetos Document con contenido y metadatos
        """
        collection_path = Path("docs") / collection_name
        
        if not collection_path.exists():
            raise FileNotFoundError(f"La colección '{collection_name}' no existe en ./docs/")
        
        # Buscar archivos PDF y Markdown en la colección
        pdf_files = list(collection_path.glob("*.pdf"))
        md_files = list(collection_path.glob("*.md"))
        
        all_files = pdf_files + md_files
        
        if not all_files:
            raise FileNotFoundError(
                f"No se encontraron archivos PDF o Markdown en '{collection_path}'"
            )
        
        documents = []

        # Procesar PDFs
        for pdf_path in pdf_files:
            try:
                document = self._load_pdf(pdf_path, multimodal=multimodal)
                documents.append(document)
            except Exception as e:
                print(f"Error cargando PDF {pdf_path.name}: {str(e)}")
                continue
        
        # Procesar Markdown (nunca multimodal — no hay contenido visual)
        for md_path in md_files:
            try:
                document = self._load_markdown(md_path)
                documents.append(document)
            except Exception as e:
                print(f"Error cargando Markdown {md_path.name}: {str(e)}")
                continue
        
        return documents


    def _load_pdf(self, pdf_path: Path, multimodal: bool = False) -> Document:
        """
        Carga un archivo PDF y extrae su contenido.

        Modo texto (multimodal=False, default):
            Extrae texto plano con pypdf. Las imágenes y tablas se ignoran.

        Modo multimodal (multimodal=True):
            Delega a MultimodalDocumentService, que combina:
            - Texto extraído con PyMuPDF
            - Descripciones de imágenes generadas por VLM (Gemini)
            - Tablas estructuradas extraídas con pdfplumber
            El contenido resultante es más rico pero tiene mayor costo de
            procesamiento por las llamadas al VLM.

        Args:
            pdf_path: Ruta al archivo PDF
            multimodal: Si True, activa el procesamiento multimodal
            
        Returns:
            Document con el contenido del PDF y metadatos.
            En modo multimodal, los metadatos incluyen has_images, has_tables
            y vlm_calls para trazabilidad de costos.
        """
        if multimodal:
            mm_service = _get_multimodal_service()
            result = mm_service.process_pdf(pdf_path)
            return result
            

        # Modo texto: PyMuPDF para mejor resolución de fuentes y encoding
        # Rastrea page_boundaries (offsets de caracteres) para estimar
        # el número de página de cada chunk después del splitting.
        import fitz
        doc = fitz.open(str(pdf_path))
        full_text = ""
        page_boundaries = []   # offsets acumulados al final de cada página

        for page in doc:
            page_text = page.get_text("text") + "\n\n"
            full_text += page_text
            page_boundaries.append(len(full_text))

        total_pages = len(doc)
        doc.close()

        full_text = normalize_text(full_text)

        return Document(
            page_content=full_text,
            metadata={
                "source": str(pdf_path),
                "source_file": pdf_path.name,
                "source_path": str(pdf_path),
                "file_type": "pdf",
                "file_size": pdf_path.stat().st_size,
                "total_pages": total_pages,
                "page_boundaries": page_boundaries,
                "preprocessed": False,
                "multimodal": False
            }
        )


    def _load_markdown(self, md_path: Path) -> Document:
        """
        Carga un archivo Markdown.
        
        Args:
            md_path: Ruta al archivo Markdown
            
        Returns:
            Document con el contenido del Markdown y metadatos
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return Document(
            page_content=text,
            metadata={
                "source": str(md_path),
                "source_file": md_path.name,
                "source_path": str(md_path),
                "file_type": "markdown",
                "file_size": md_path.stat().st_size,
                "preprocessed": False
            }
        )



    def process_collection(self, collection_name: str, multimodal: bool = False) -> List[Document]:
        """
        Procesa una colección completa de documentos aplicando chunking.
        
        Carga los documentos de la colección, aplica la estrategia de chunking
        seleccionada y enriquece cada chunk con metadatos adicionales.
        
        Args:
            collection_name: Nombre de la colección a procesar
            multimodal: Si True, los PDFs se procesan con MultimodalDocumentService.
                        El flag se propaga a load_documents_from_collection → _load_pdf.
                        Los archivos Markdown no se ven afectados.
            
        Returns:
            Lista de chunks (objetos Document) con metadatos enriquecidos
        """
        # Cargar documentos de la colección (multimodal se propaga aquí)
        documents = self.load_documents_from_collection(collection_name, multimodal=multimodal)
        
        all_chunks = []
        
        for doc in documents:
            # Aplicar splitting según la estrategia y tipo de archivo
            chunks = self._split_document(doc)
            
            # Enriquecer cada chunk con metadatos adicionales
            for i, chunk in enumerate(chunks):
                # Asegurar que chunk es un Document
                if isinstance(chunk, str):
                    chunk = Document(page_content=chunk, metadata={})
                
                # Copiar metadatos originales del documento
                chunk.metadata.update(doc.metadata)
                
                # Estimar número de página desde page_boundaries del documento
                page_boundaries = doc.metadata.get("page_boundaries", [])
                total_pages     = doc.metadata.get("total_pages", 1)
                if page_boundaries and len(chunks) > 0:
                    # Aproximar posición del chunk en el documento original
                    # usando su índice relativo al total de chunks
                    approx_char_pos = int((i / max(len(chunks) - 1, 1)) * page_boundaries[-1]) if len(chunks) > 1 else 0
                    page_num = next(
                        (p + 1 for p, boundary in enumerate(page_boundaries) if approx_char_pos <= boundary),
                        total_pages
                    )
                else:
                    # Fallback: estimación proporcional sin boundaries
                    page_num = max(1, round((i / max(len(chunks) - 1, 1)) * total_pages)) if len(chunks) > 1 else 1

                # Clasificar tipo de chunk según el contenido
                content = chunk.page_content
                if "[FIGURA]" in content:
                    chunk_type = "figure"
                elif "|---" in content or ("| " in content and " |" in content):
                    chunk_type = "table"
                elif any(marker in content for marker in ["def ", "class ", "import ", "```", ">>>"]):
                    chunk_type = "code"
                else:
                    chunk_type = "text"

                # Agregar metadatos específicos del chunk
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks_in_doc": len(chunks),
                    "chunking_strategy": self.strategy.value,
                    "chunk_size": len(chunk.page_content),
                    "page_number": page_num,
                    "chunk_type": chunk_type,
                    "rerank_score": None,
                })
            
            all_chunks.extend(chunks)
        
        return all_chunks
    

    def process_csv_collection(
        self,
        collection_name: str,
        operation_type: str = None,
        property_type: str = None,
        barrio: str = None,
    ) -> List[Document]:
        """
        Procesa una colección de listings inmobiliarios desde un CSV.
 
        A diferencia de process_collection() (que divide PDFs/MD en chunks),
        este método NO aplica splitting: cada listing es ya un Document completo.
        Sí enriquece los metadatos con los mismos campos que process_collection()
        produce, para que el resto del pipeline (EmbeddingService, RetrievalService)
        funcione sin cambios.
 
        Espera encontrar un CSV en: ./docs/{collection_name}/*.csv
 
        Args:
            collection_name : Nombre de la colección (carpeta en ./docs/).
            operation_type  : Filtro opcional: "venta" | "alquiler"
            property_type   : Filtro opcional: "apartamentos" | "casas"
            barrio          : Filtro opcional: nombre exacto del barrio
 
        Returns:
            Lista de Document con page_content en lenguaje natural y
            metadata estructurada lista para FAISS.
 
        Raises:
            FileNotFoundError: Si la carpeta o el CSV no existen.
        """
        collection_path = Path("docs") / collection_name
 
        if not collection_path.exists():
            raise FileNotFoundError(
                f"La colección '{collection_name}' no existe en ./docs/"
            )
 
        csv_files = list(collection_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No se encontró ningún CSV en '{collection_path}'. "
                f"Coloca el archivo de listings en ./docs/{collection_name}/"
            )
        if len(csv_files) > 1:
            print(
                f"[WARNING] Se encontraron {len(csv_files)} CSVs en '{collection_path}'. "
                f"Se usará el primero: {csv_files[0].name}"
            )
 
        csv_path = csv_files[0]
        print(f"\n[ChunkingService] Procesando CSV: {csv_path}")
 
        # Delegar carga, preprocesado y conversión a CSVDocumentService
        csv_service = CSVDocumentService()
        documents = csv_service.load_documents(
            str(csv_path),
            operation_type=operation_type,
            property_type=property_type,
            barrio=barrio,
        )
 
        # Enriquecer metadata para compatibilidad con el pipeline existente
        # (los mismos campos que process_collection() agrega en su loop)
        for i, doc in enumerate(documents):
            content = doc.page_content
 
            # Clasificar tipo de chunk (los listings son siempre texto)
            chunk_type = "listing"
 
            doc.metadata.update({
                "chunk_index":         i,
                "total_chunks_in_doc": 1,       # cada listing es su propio documento
                "chunking_strategy":   "csv",   # identificador para trazabilidad
                "chunk_size":          len(content),
                "page_number":         1,
                "chunk_type":          chunk_type,
                "rerank_score":        None,
                "file_type":           "csv",
            })
 
        print(
            f"[ChunkingService] {len(documents)} documents listos "
            f"(colección: {collection_name})"
        )
        return documents
    

    def _split_document(self, doc: Document) -> List[Document]:
        """
        Aplica la estrategia de chunking apropiada según el tipo de documento.

        Para documentos multimodales (PDFs procesados con VLM), usa un splitter
        con chunk_size=2000 para evitar que las descripciones de imágenes [FIGURA]
        y las tablas Markdown se corten a mitad de contenido.

        Para todos los demás documentos, usa el splitter configurado en __init__.
        
        Args:
            doc: Document a dividir
            
        Returns:
            Lista de chunks (Documents)
        """
        file_type = doc.metadata.get("file_type", "pdf")
        is_multimodal = doc.metadata.get("multimodal", False)

        # DOCUMENT_STRUCTURE solo para Markdown
        if self.strategy == ChunkingStrategy.DOCUMENT_STRUCTURE:
            if file_type == "markdown":
                # split_text retorna strings, convertir a Documents
                chunks = self.splitter.split_text(doc.page_content)
                return [
                    Document(page_content=c if isinstance(c, str) else c.page_content)
                    for c in chunks
                ]
            else:
                # Para PDFs, usar RECURSIVE como fallback
                print(f"[WARNING] DOCUMENT_STRUCTURE no aplicable a PDF, usando RECURSIVE")
                fallback_splitter = self._create_recursive_character_splitter()
                return fallback_splitter.split_documents([doc])

        # Contenido multimodal: usar splitter con chunk_size mayor
        # para preservar descripciones VLM y tablas Markdown completas
        if is_multimodal:
            multimodal_splitter = self._create_multimodal_splitter()
            return multimodal_splitter.split_documents([doc])

        # Otras estrategias (RECURSIVE, FIXED, SEMANTIC) — comportamiento original
        return self.splitter.split_documents([doc])


    
    def get_chunking_statistics(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Calcula estadísticas sobre los chunks generados.
        
        Proporciona métricas útiles sobre el proceso de chunking para
        análisis y validación de la estrategia utilizada.
        Incluye métricas multimodales cuando los chunks provienen de
        procesamiento con VLM (para comparación de costos text vs multimodal).
        
        Args:
            chunks: Lista de chunks procesados
            
        Returns:
            Diccionario con estadísticas de chunking
        """
        if not chunks:
            return {
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "total_chunks": 0,
                "total_characters_processed": 0,
                "chunks_by_file_type": {},
                "multimodal_stats": {
                    "multimodal_chunks": 0,
                    "chunks_with_images": 0,
                    "chunks_with_tables": 0,
                    "total_vlm_calls": 0
                }
            }
        
        # Calcular tamaños de chunks
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]

        # Agrupar por tipo de archivo
        chunks_by_type = {}
        for chunk in chunks:
            file_type = chunk.metadata.get("file_type", "unknown")
            chunks_by_type[file_type] = chunks_by_type.get(file_type, 0) + 1

        # Estadísticas multimodales (para la tabla de comparación de costos)
        multimodal_chunks  = sum(1 for c in chunks if c.metadata.get("multimodal", False))
        chunks_with_images = sum(1 for c in chunks if c.metadata.get("has_images", False))
        chunks_with_tables = sum(1 for c in chunks if c.metadata.get("has_tables", False))
        total_vlm_calls    = sum(c.metadata.get("vlm_calls", 0) for c in chunks)
        
        return {
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_chunks": len(chunks),
            "total_characters_processed": sum(chunk_sizes),
            "chunks_by_file_type": chunks_by_type,
            "multimodal_stats": {
                "multimodal_chunks": multimodal_chunks,
                "chunks_with_images": chunks_with_images,
                "chunks_with_tables": chunks_with_tables,
                "total_vlm_calls": total_vlm_calls
            }
        }