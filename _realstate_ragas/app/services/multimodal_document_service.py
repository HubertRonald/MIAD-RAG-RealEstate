"""
Servicio de Procesamiento Multimodal de Documentos
===================================================

Este módulo extiende la carga de PDFs del ChunkingService para incorporar
contenido visual al pipeline RAG. Combina tres fuentes de información por página:

    1. Texto (PyMuPDF)         → extracción span-by-span, preserva orden visual
    2. Tablas (pdfplumber)     → detecta tablas con bordes o alineación de texto,
                                 las serializa como Markdown sin llamar al VLM
    3. Imágenes (Gemini VLM)  → genera descripciones textuales de figuras,
                                 diagramas y gráficas mediante gemini-2.5-flash

Estrategia de costos:
    Las tablas estructuradas se extraen con pdfplumber (costo cero de API).
    El VLM solo se invoca para imágenes reales (≥ MIN_IMAGE_SIZE píxeles).
    Las imágenes decorativas (iconos, separadores) se filtran por tamaño.

Integración:
    ChunkingService._load_pdf() delega aquí cuando multimodal=True.
    El método process_pdf() devuelve un Document con la misma interfaz
    que el _load_pdf() original, por lo que el resto del pipeline no cambia.

Metadatos de trazabilidad (para tabla de comparación de costos):
    multimodal   : True
    has_images   : bool — indica si la página tenía imágenes procesables
    has_tables   : bool — indica si se extrajeron tablas estructuradas
    vlm_calls    : int  — total de llamadas al VLM (para estimación de costos)

"""

import os
import time
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

import fitz                          # PyMuPDF

from app.utils.text_utils import normalize_text
import pdfplumber
from google import genai
from google.genai import types
from langchain.schema import Document

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Tamaño mínimo (ancho o alto) en píxeles para procesar una imagen con el VLM.
# Imágenes por debajo de este umbral se consideran decorativas (iconos, viñetas,
# separadores) y se omiten para ahorrar llamadas de API.
MIN_IMAGE_SIZE = 150

# Modelo VLM para descripción de imágenes.
# Debe coincidir con el usado en los labs del curso.
VLM_MODEL = "gemini-2.5-flash"

# Segundos de pausa entre llamadas consecutivas al VLM.
# Respeta el límite de 100 RPM del free tier de Google AI.
VLM_REQUEST_DELAY = 2

# =============================================================================
# PRECIOS GEMINI (verificar en https://ai.google.dev/pricing antes de reportar)
# Gemini 2.5 Flash — precios por millón de tokens (USD)
# =============================================================================
VLM_INPUT_PRICE_PER_M_TOKENS  = 0.075   # $0.075 / 1M input tokens
VLM_OUTPUT_PRICE_PER_M_TOKENS = 0.30    # $0.30  / 1M output tokens

# Prompt para describir imágenes técnicas de libros educativos.
# Adaptado del lab 20 pero orientado al corpus de Tutor-IA
# (Python, ML, LLMs, JavaScript, React, Git, Go).
IMAGE_PROMPT = """
Eres un asistente especializado en describir contenido visual de libros técnicos
sobre programación, machine learning e inteligencia artificial.

Describe la siguiente imagen con precisión técnica.

Según el tipo de contenido:
- Diagrama de arquitectura: describe componentes, flujo de datos y conexiones.
- Gráfica o plot: describe ejes, variables, tendencias y valores relevantes.
- Figura conceptual: describe el concepto ilustrado y sus elementos clave.
- Tabla visual: describe columnas, filas y los valores más importantes.
- Captura de pantalla / código: describe qué muestra y su propósito.

Sé preciso con los términos técnicos. Describe la imagen en el mismo lenguaje del texto en el documento.
Si hay texto visible en la imagen (títulos, etiquetas, leyendas), inclúyelo.
""".strip()

# Prompt específico para extraer tablas complejas vía VLM
# (fallback cuando pdfplumber no detecta estructura).
TABLE_IMAGE_PROMPT = """
La siguiente imagen contiene una tabla de un libro técnico.
Extrae su contenido de forma estructurada y fiel.
Identifica encabezados de columnas y etiquetas de filas exactamente como aparecen.
Devuelve la tabla como Markdown. No resumas ni infieras valores faltantes.
""".strip()


# =============================================================================
# SERVICIO PRINCIPAL
# =============================================================================

class MultimodalDocumentService:
    """
    Procesa PDFs combinando texto, tablas e imágenes en un único bloque
    de texto enriquecido apto para chunking y retrieval semántico.
    """

    def __init__(self):
        """
        Inicializa el cliente de Google Generative AI de forma lazy.
        El cliente se crea en el primer uso para evitar errores de importación
        cuando GOOGLE_API_KEY no está disponible en el entorno.
        """
        self._client: Optional[Any] = None

    # -------------------------------------------------------------------------
    # Propiedad lazy para el cliente VLM
    # -------------------------------------------------------------------------

    @property
    def client(self):
        """Devuelve el cliente Gemini, creándolo la primera vez que se accede."""
        if self._client is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY no encontrada en el entorno. "
                    "Agrégala al archivo .env antes de usar procesamiento multimodal."
                )
            self._client = genai.Client()
            logger.info("[MultimodalDocumentService] Cliente Gemini inicializado.")
        return self._client

    # =========================================================================
    # PUNTO DE ENTRADA PÚBLICO
    # =========================================================================

    def process_pdf(self, pdf_path: Path) -> Document:
        """
        Procesa un PDF completo y devuelve un Document enriquecido con
        texto, descripciones de imágenes y contenido de tablas.

        Este método es la interfaz pública que llama ChunkingService._load_pdf()
        cuando multimodal=True. El Document devuelto tiene la misma estructura
        que el producido por la ruta texto-only, por lo que el resto del
        pipeline (splitter → embeddings → FAISS) no requiere cambios.

        Flujo interno:
            1. Extraer tablas del PDF completo con pdfplumber (sin costo de API).
            2. Abrir el PDF con PyMuPDF e iterar página por página.
            3. Por cada página: combinar texto + tablas + descripciones de imágenes.
            4. Unir todo el contenido y construir el Document con metadatos.

        Args:
            pdf_path: Ruta al archivo PDF a procesar.

        Returns:
            Document con page_content enriquecido y metadatos de trazabilidad.
        """
        pdf_path = Path(pdf_path)
        logger.info(f"[MultimodalDocumentService] Procesando: {pdf_path.name}")

        # --- Paso 1: extraer tablas una sola vez para todo el documento ---
        # pdfplumber abre el archivo por separado; más eficiente que por página.
        tables_by_page = self._extract_all_tables(pdf_path)
        logger.info(
            f"[MultimodalDocumentService] Tablas detectadas en páginas: "
            f"{sorted(tables_by_page.keys())}"
        )

        # --- Paso 2: iterar páginas con PyMuPDF ---
        document = fitz.open(str(pdf_path))
        pages_text: List[str] = []

        # Contadores para metadatos y trazabilidad de costos
        total_vlm_calls       = 0
        total_vlm_input_tok   = 0
        total_vlm_output_tok  = 0
        pages_with_images     = 0
        pages_with_tables     = 0

        for page_num, page in enumerate(document):
            page_content, vlm_calls, input_tok, output_tok = self._process_page(
                page=page,
                page_num=page_num,
                tables_by_page=tables_by_page
            )
            pages_text.append(page_content)
            total_vlm_calls      += vlm_calls
            total_vlm_input_tok  += input_tok
            total_vlm_output_tok += output_tok

            if vlm_calls > 0:
                pages_with_images += 1
            if page_num in tables_by_page:
                pages_with_tables += 1

        document.close()

        full_text = "\n\n".join(pages_text)

        # Calcular costo VLM en USD
        vlm_cost_usd = (
            (total_vlm_input_tok  / 1_000_000) * VLM_INPUT_PRICE_PER_M_TOKENS +
            (total_vlm_output_tok / 1_000_000) * VLM_OUTPUT_PRICE_PER_M_TOKENS
        )

        logger.info(
            f"[MultimodalDocumentService] {pdf_path.name} procesado | "
            f"páginas={len(pages_text)} | vlm_calls={total_vlm_calls} | "
            f"input_tokens={total_vlm_input_tok} | output_tokens={total_vlm_output_tok} | "
            f"vlm_cost_usd=${vlm_cost_usd:.6f}"
        )

        return Document(
            page_content=full_text,
            metadata={
                "source":       str(pdf_path),
                "source_file":  pdf_path.name,
                "source_path":  str(pdf_path),
                "file_type":    "pdf",
                "file_size":    pdf_path.stat().st_size,
                "total_pages":  len(pages_text),
                "preprocessed": False,
                # --- metadatos multimodales para trazabilidad de costos ---
                "multimodal":          True,
                "has_images":          pages_with_images > 0,
                "has_tables":          pages_with_tables > 0,
                "vlm_calls":           total_vlm_calls,
                "vlm_input_tokens":    total_vlm_input_tok,
                "vlm_output_tokens":   total_vlm_output_tok,
                "vlm_cost_usd":        round(vlm_cost_usd, 6),
            }
        )

    # =========================================================================
    # PROCESAMIENTO POR PÁGINA
    # =========================================================================

    def _process_page(
        self,
        page: fitz.Page,
        page_num: int,
        tables_by_page: Dict[int, List[str]]
    ):
        """
        Combina texto, tablas e imágenes de una sola página en un string.

        Orden de composición por página:
            [texto de bloques tipo 0]
            [tablas en Markdown, si las hay]
            [descripciones de imágenes, si las hay]

        Args:
            page:            Página PyMuPDF.
            page_num:        Índice de página (base 0), usado para buscar tablas.
            tables_by_page:  Dict {page_num: [markdown_table, ...]}.

        Returns:
            Tuple(str, int, int, int):
                (texto_de_la_página, vlm_calls, input_tokens, output_tokens)
        """
        page_text    = ""
        vlm_calls    = 0
        input_tokens  = 0
        output_tokens = 0

        page_dict  = page.get_text("dict")
        blocks     = page_dict.get("blocks", [])

        # --- Bloques de la página ---
        for block in blocks:
            block_type = block.get("type")

            if block_type == 0:   # texto
                page_text += self._process_text_block(block)

            elif block_type == 1: # imagen
                description, called, in_tok, out_tok = self._process_image_block(block)
                if description:
                    page_text += description
                vlm_calls    += called
                input_tokens  += in_tok
                output_tokens += out_tok

        # --- Tablas extraídas por pdfplumber para esta página ---
        # Se agregan después del texto para no interrumpir el flujo narrativo.
        if page_num in tables_by_page:
            for table_md in tables_by_page[page_num]:
                page_text += f"\n\n{table_md}\n"

        return page_text.strip(), vlm_calls, input_tokens, output_tokens

    # =========================================================================
    # PROCESAMIENTO DE BLOQUES (texto e imagen) — basado en lab 20
    # =========================================================================

    def _process_text_block(self, block: dict) -> str:
        """
        Reconstruye el texto de un bloque tipo 0 de PyMuPDF.

        Itera líneas y spans manteniendo el orden visual de la página.
        Cada línea se separa con \\n para conservar estructura de párrafos.

        Args:
            block: Bloque dict de page.get_text("dict") con type=0.

        Returns:
            Texto reconstruido con saltos de línea entre líneas del bloque.
        """
        lines = block.get("lines", [])
        text  = ""
        for line in lines:
            for span in line.get("spans", []):
                text += span.get("text", "")
            text += "\n"
        return normalize_text(text)

    def _process_image_block(self, block: dict):
        """
        Genera una descripción textual de un bloque de imagen tipo 1.

        Filtro de tamaño: imágenes menores a MIN_IMAGE_SIZE píxeles en ancho
        o alto se omiten (son iconos o elementos decorativos sin valor semántico).

        Flujo:
            1. Extraer bytes de imagen del bloque.
            2. Verificar dimensiones (filtro de tamaño).
            3. Guardar en archivo temporal.
            4. Subir al Gemini Files API.
            5. Invocar VLM con IMAGE_PROMPT.
            6. Retornar descripción como texto marcado con etiqueta [FIGURA].

        Args:
            block: Bloque dict de page.get_text("dict") con type=1.

        Returns:
            Tuple(str, int, int, int):
                (descripción_o_vacío, vlm_calls, input_tokens, output_tokens)
        """
        # Filtro por tamaño — omitir imágenes decorativas
        width  = block.get("width",  0)
        height = block.get("height", 0)
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            logger.debug(
                f"[_process_image_block] Imagen omitida ({width}x{height}px < {MIN_IMAGE_SIZE}px)"
            )
            return "", 0, 0, 0

        image_bytes = block.get("image")
        if not image_bytes:
            return "", 0, 0, 0
        try:
            # Guardar imagen en archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_bytes)
                image_path = tmp.name

            # Subir al Files API de Google para obtener URI
            uploaded_file = self.client.files.upload(file=image_path)

            # Llamar al VLM — retorna texto y conteos de tokens
            description, in_tok, out_tok = self._generate_description(
                file_uri  = uploaded_file.uri,
                prompt    = IMAGE_PROMPT,
                mime_type = "image/jpeg"
            )

            # Limpiar archivo temporal
            os.unlink(image_path)

            # Respetar rate limits entre llamadas consecutivas
            time.sleep(VLM_REQUEST_DELAY)

            # Etiquetar la descripción para que el LLM pueda referenciarla
            tagged = f"\n[FIGURA] {description.strip()}\n"

            logger.debug(
                f"[_process_image_block] Imagen {width}x{height}px | "
                f"input_tokens={in_tok} | output_tokens={out_tok}"
            )
            return tagged, 1, in_tok, out_tok

        except Exception as e:
            logger.warning(f"[_process_image_block] Error procesando imagen: {e}")
            return "", 0, 0, 0

    # =========================================================================
    # EXTRACCIÓN DE TABLAS — basado en lab 19
    # =========================================================================

    def _extract_all_tables(self, pdf_path: Path) -> Dict[int, List[str]]:
        """
        Extrae todas las tablas del documento con pdfplumber y las serializa
        como strings Markdown listos para insertarse en el texto de la página.

        Estrategia de detección en dos pasos (del lab 19):
            1. "lines"  → detecta tablas con bordes explícitos (la mayoría de
                          libros técnicos O'Reilly usan este estilo).
            2. "text"   → fallback para tablas sin bordes, basadas en alineación
                          de columnas de texto.
        Si ninguna estrategia detecta tablas en una página, esa página no aparece
        en el dict devuelto (no se agrega nada al texto).

        Args:
            pdf_path: Ruta al PDF.

        Returns:
            Dict {page_num (base 0): [markdown_table_str, ...]}.
            Páginas sin tablas no están en el dict.
        """
        tables_by_page: Dict[int, List[str]] = {}

        # Solo detectar tablas con bordes explícitos ("lines").
        # La estrategia "text" produce demasiados falsos positivos en libros
        # con texto justificado o layouts multi-columna, interpretando la
        # alineación de columnas de texto como estructura de tabla.
        # Los libros técnicos O'Reilly y similares usan bordes en sus tablas reales.
        strategies = [
            {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
        ]

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):

                    page_tables = []

                    for settings in strategies:
                        raw_tables = page.extract_tables(settings)

                        if raw_tables:
                            for raw_table in raw_tables:
                                # Omitir tablas vacías
                                if not any(row for row in raw_table if any(cell for cell in row if cell)):
                                    continue
                                md = self._table_to_markdown(raw_table)
                                if md:
                                    page_tables.append(md)
                            # Si "lines" encontró tablas, no intentar "text"
                            break

                    if page_tables:
                        tables_by_page[page_num] = page_tables

        except Exception as e:
            logger.warning(
                f"[_extract_all_tables] Error extrayendo tablas de {pdf_path.name}: {e}"
            )

        return tables_by_page

    def _table_to_markdown(self, raw_table: List[List]) -> str:
        """
        Convierte una tabla cruda (lista de listas) a formato Markdown.

        La primera fila se trata como encabezados de columna.
        Las celdas None se reemplazan por cadena vacía.
        Si la tabla tiene menos de 2 filas (sin datos), retorna cadena vacía.

        Args:
            raw_table: Lista de listas con el contenido de la tabla.

        Returns:
            String Markdown de la tabla, o "" si la tabla está vacía.
        """
        if not raw_table or len(raw_table) < 2:
            return ""

        def clean(cell) -> str:
            """Limpia una celda: None → '', reemplaza saltos de línea y normaliza encoding."""
            if cell is None:
                return ""
            text = str(cell).replace("\n", " ").strip()
            return normalize_text(text)

        header = raw_table[0]
        rows   = raw_table[1:]

        # Fila de encabezados
        header_row    = "| " + " | ".join(clean(c) for c in header) + " |"
        separator_row = "| " + " | ".join("---" for _ in header) + " |"

        # Filas de datos
        data_rows = []
        for row in rows:
            # Asegurar que la fila tenga el mismo número de columnas que el header
            padded = row + [None] * (len(header) - len(row))
            data_rows.append("| " + " | ".join(clean(c) for c in padded) + " |")

        return "\n".join([header_row, separator_row] + data_rows)

    # =========================================================================
    # LLAMADA AL VLM — basado en lab 20
    # =========================================================================

    def _generate_description(
        self,
        file_uri:  str,
        prompt:    str,
        mime_type: str = "image/jpeg"
    ) -> str:
        """
        Invoca el modelo VLM de Gemini para generar una descripción textual
        de una imagen a partir de su URI en el Files API.

        Implementación directa del patrón del lab 20:
            - Construye contenido multimodal (texto + imagen por URI).
            - Configura el modelo para retornar solo texto.
            - Retorna el texto generado.

        Args:
            file_uri:  URI del archivo subido al Files API de Google.
            prompt:    Instrucción que guía la descripción.
            mime_type: MIME type de la imagen (default "image/jpeg").

        Returns:
            Tuple(str, int, int): (texto_generado, input_tokens, output_tokens).
            Los conteos de tokens provienen de response.usage_metadata de Gemini.
            Si usage_metadata no está disponible, retorna 0 para ambos conteos.

        Raises:
            Exception: Si la llamada al modelo falla. El caller en
                       _process_image_block captura y loguea el error.
        """
        contents = [
            types.UserContent(parts=[
                types.Part.from_text(text=prompt)
            ]),
            types.UserContent(parts=[
                types.Part.from_uri(
                    file_uri  = file_uri,
                    mime_type = mime_type
                )
            ])
        ]

        response = self.client.models.generate_content(
            model    = VLM_MODEL,
            contents = contents,
            config   = types.GenerateContentConfig(
                response_modalities = ["TEXT"]
            )
        )

        # Extraer conteos de tokens desde usage_metadata
        usage        = response.usage_metadata
        input_tokens  = getattr(usage, "prompt_token_count",     0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        return response.text, input_tokens, output_tokens
