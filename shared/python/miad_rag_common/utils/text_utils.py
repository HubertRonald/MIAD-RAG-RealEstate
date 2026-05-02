"""
Utilidades de normalización de texto para el pipeline RAG
==========================================================

Centraliza la corrección de artefactos de encoding comunes en PDFs técnicos
en español, para que tanto el pipeline de texto como el pipeline de indexación
apliquen una lógica consistente.

Funciones principales:
    normalize_text:
        Mantiene el comportamiento original. Corrige artefactos de encoding,
        ligaduras y guiones de corte, pero NO colapsa todos los espacios.

    clean_for_embedding:
        Aplica normalize_text y además colapsa espacios. Útil para documentos
        tipo listing donde se quiere un texto compacto para embeddings.

    normalize_for_match:
        Normaliza texto para comparación robusta: mayúsculas, sin tildes y
        espacios colapsados.

    safe_*:
        Helpers para convertir valores provenientes de BigQuery, Pandas,
        metadata de FAISS o JSON.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Mapa de reemplazos para spacing accents y artefactos Latin-1/pilcrow.
#
# IMPORTANTE:
# Aplicar ANTES de unicodedata.normalize("NFKC"), porque NFKC convierte
# U+00B4 (ACUTE ACCENT ´) en combining acute (U+0301), lo que rompe los
# patrones de búsqueda de este mapa.
# ---------------------------------------------------------------------------
_ACCENT_MAP = {
    # Vocales minúsculas con acento agudo
    "´a": "á",
    "´e": "é",
    "´ı": "í",
    "´i": "í",
    "´o": "ó",
    "´u": "ú",

    # Vocales mayúsculas con acento agudo
    "´A": "Á",
    "´E": "É",
    "´I": "Í",
    "´O": "Ó",
    "´U": "Ú",

    # Diéresis
    "¨u": "ü",
    "¨U": "Ü",

    # Eñe
    "˜n": "ñ",
    "˜N": "Ñ",

    # Variantes con pilcrow
    "a¶": "á",
    "e¶": "é",
    "i¶": "í",
    "o¶": "ó",
    "u¶": "ú",

    # Pilcrow residual sin contexto claro
    "¶": "",
}


def normalize_text(text: str) -> str:
    """
    Normaliza artefactos de encoding comunes en PDFs técnicos en español.

    Mantiene la intención de la función original.

    Aplica tres transformaciones en orden estricto:

        1. Reunificación de guiones de corte de línea:
           "signi-\\nfica" -> "significa"

        2. Corrección de spacing accents y artefactos Latin-1/pilcrow:
           "m´as" -> "más"
           "protecio¶n" -> "protección"

        3. Normalización NFKC:
           ﬁ -> "fi"
           ﬂ -> "fl"

    Nota:
        Esta función NO colapsa todos los espacios ni saltos de línea.
        Eso se hace en clean_for_embedding(), para no romper flujos que
        dependan de estructura textual antes del chunking.

    Args:
        text: Texto extraído de PDF, HTML, CSV o BigQuery.

    Returns:
        Texto normalizado.
    """
    if text is None:
        return ""

    text = str(text)

    # 1. Reunificar palabras cortadas por guion al final de línea.
    text = re.sub(r"-\n(\S)", r"\1", text)

    # 2. Corregir spacing accents y artefactos Latin-1/pilcrow.
    for bad, good in _ACCENT_MAP.items():
        text = text.replace(bad, good)

    # 3. NFKC: descompone ligaduras tipográficas y normaliza Unicode.
    text = unicodedata.normalize("NFKC", text)

    return text


def collapse_spaces(text: str) -> str:
    """
    Colapsa espacios, tabs y saltos de línea múltiples en un solo espacio.

    Útil para:
        - texto final antes de embeddings,
        - matching simple,
        - limpieza de campos de listings.

    No usar como reemplazo directo de normalize_text() si se quiere preservar
    estructura de párrafos antes del chunking.
    """
    return re.sub(r"\s+", " ", text or "").strip()


def clean_for_embedding(text: str) -> str:
    """
    Limpieza recomendada antes de generar embeddings.

    Aplica:
        - normalize_text()
        - collapse_spaces()

    Esta función sí transforma el texto en una línea compacta.
    Es adecuada para listings inmobiliarios o textos ya estructurados.
    """
    return collapse_spaces(normalize_text(text))


def remove_accents(text: str) -> str:
    """
    Quita tildes para comparación robusta.

    Ejemplo:
        "Cordón" -> "Cordon"
        "Peñarol" -> "Penarol"
    """
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_for_match(text: str) -> str:
    """
    Normaliza texto para matching simple.

    Aplica:
        - conversión a string,
        - remoción de tildes,
        - mayúsculas,
        - colapso de espacios.
    """
    return collapse_spaces(remove_accents(text).upper())


def is_nullish(value: Any) -> bool:
    """
    Detecta valores nulos frecuentes sin depender de pandas.

    Cubre:
        - None
        - NaN
        - float("inf") / float("-inf")
        - strings vacíos
        - strings "nan", "none", "null"
    """
    if value is None:
        return True

    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)

    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null", "na", "n/a"}

    # Compatibilidad ligera con pandas.NA / numpy.nan sin importar pandas.
    try:
        if value != value:
            return True
    except Exception:
        pass

    return False


def safe_str(value: Any, default: str = "") -> str:
    """
    Convierte un valor a string de forma segura.
    """
    if is_nullish(value):
        return default

    return str(value)


def safe_float(value: Any) -> Optional[float]:
    """
    Convierte un valor a float de forma segura.

    Retorna None si el valor es nulo, NaN, infinito o no convertible.
    """
    if is_nullish(value):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(result) or math.isinf(result):
        return None

    return result


def safe_int(value: Any) -> Optional[int]:
    """
    Convierte un valor a int de forma segura.

    Soporta valores tipo:
        "2"
        "2.0"
        2.0

    Retorna None si no es convertible.
    """
    number = safe_float(value)

    if number is None:
        return None

    return int(number)


def safe_bool(value: Any) -> bool:
    """
    Convierte valores comunes a booleano.

    True:
        True, 1, 1.0, "1", "true", "yes", "y", "si", "sí", "s"

    False:
        False, 0, 0.0, "0", "false", "no", "n", None, NaN, otros.
    """
    if is_nullish(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value == 1

    if isinstance(value, float):
        return value == 1.0

    if isinstance(value, str):
        return value.strip().lower() in {
            "1",
            "true",
            "t",
            "yes",
            "y",
            "si",
            "sí",
            "s",
        }

    return False
