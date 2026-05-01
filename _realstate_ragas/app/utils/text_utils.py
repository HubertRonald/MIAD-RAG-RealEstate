"""
Utilidades de normalización de texto para el pipeline RAG
==========================================================

Centraliza la corrección de artefactos de encoding comunes en PDFs técnicos
en español, para que tanto el pipeline de texto (chunking_service) como el
multimodal (multimodal_document_service) apliquen exactamente la misma lógica.

Problemas que resuelve:
    - Ligaduras tipográficas: ﬁ (U+FB01) → 'fi', ﬂ (U+FB02) → 'fl', etc.
      Ocurren cuando PyMuPDF extrae glifos de fuentes con ligaduras OpenType.
    - Spacing accents: 'm´as' → 'más', 'dise˜nado' → 'diseñado'
      Ocurren en PDFs con encoding Latin-1/Windows-1252 mal interpretado como UTF-8.
    - Artefactos con pilcrow: 'protecio¶n' → 'protección'
      Variante del problema anterior con el carácter U+00B6 como separador.
    - Guiones de corte de línea: 'signi-\\nfica' → 'significa'
      El chunker RecursiveCharacterTextSplitter puede separar estas palabras
      en chunks distintos, degradando la calidad de los embeddings.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Mapa de reemplazos para spacing accents y artefactos Latin-1/pilcrow.
# IMPORTANTE: aplicar ANTES de unicodedata.normalize('NFKC') porque NFKC
# convierte U+00B4 (ACUTE ACCENT ´) en combining acute (U+0301), lo que
# rompe los patrones de búsqueda de este mapa.
# ---------------------------------------------------------------------------
_ACCENT_MAP = {
    # Vocales minúsculas con acento agudo
    '´a': 'á', '´e': 'é', '´ı': 'í', '´o': 'ó', '´u': 'ú',
    # Vocales mayúsculas con acento agudo
    '´A': 'Á', '´E': 'É', '´I': 'Í', '´O': 'Ó', '´U': 'Ú',
    # Diéresis
    '¨u': 'ü', '¨U': 'Ü',
    # Eñe
    '˜n': 'ñ', '˜N': 'Ñ',
    # Variantes con pilcrow (artefacto Latin-1: U+00B6 como separador)
    'a¶': 'á', 'e¶': 'é', 'o¶': 'ó', 'u¶': 'ú', 'i¶': 'í',
    # Pilcrow residual sin contexto claro
    '¶': '',
}


def normalize_text(text: str) -> str:
    """
    Normaliza artefactos de encoding comunes en PDFs técnicos en español.

    Aplica tres transformaciones en orden estricto:
        1. Reunificación de guiones de corte de línea
           'signi-\\nfica' → 'significa'
        2. Corrección de spacing accents y artefactos Latin-1/pilcrow
           'm´as' → 'más', 'protecio¶n' → 'protección'
        3. Normalización NFKC para descomponer ligaduras tipográficas
           ﬁ (U+FB01) → 'fi', ﬂ (U+FB02) → 'fl'

    El orden importa: el paso 2 debe preceder al 3 porque NFKC transforma
    U+00B4 (´) en U+0301 (combining acute), invalidando los patrones del mapa.

    Args:
        text: Texto extraído de PDF (por pypdf o PyMuPDF) antes del chunking.

    Returns:
        Texto limpio listo para embeddings.
    """
    # 1. Reunificar palabras cortadas por guión al final de línea
    text = re.sub(r'-\n(\S)', r'\1', text)

    # 2. Corregir spacing accents y artefactos Latin-1/pilcrow
    for bad, good in _ACCENT_MAP.items():
        text = text.replace(bad, good)

    # 3. NFKC: descompone ligaduras tipográficas y normaliza Unicode
    text = unicodedata.normalize('NFKC', text)

    return text
