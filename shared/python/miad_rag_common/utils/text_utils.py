from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional


_ACCENT_MAP = {
    "´a": "á",
    "´e": "é",
    "´ı": "í",
    "´i": "í",
    "´o": "ó",
    "´u": "ú",
    "´A": "Á",
    "´E": "É",
    "´I": "Í",
    "´O": "Ó",
    "´U": "Ú",
    "¨u": "ü",
    "¨U": "Ü",
    "˜n": "ñ",
    "˜N": "Ñ",
    "a¶": "á",
    "e¶": "é",
    "i¶": "í",
    "o¶": "ó",
    "u¶": "ú",
    "¶": "",
}


def normalize_text(text: str) -> str:
    """
    Normaliza artefactos comunes de extracción de texto.

    Corrige:
    - guiones de corte de línea,
    - spacing accents,
    - ligaduras tipográficas,
    - espacios repetidos.
    """
    if text is None:
        return ""

    text = str(text)

    # Reunifica palabras cortadas por guion al final de línea.
    text = re.sub(r"-\n(\S)", r"\1", text)

    # Corrige artefactos de acentos antes de NFKC.
    for bad, good in _ACCENT_MAP.items():
        text = text.replace(bad, good)

    # Normaliza unicode y ligaduras.
    text = unicodedata.normalize("NFKC", text)

    return text


def collapse_spaces(text: str) -> str:
    """Colapsa espacios, tabs y saltos de línea múltiples."""
    return re.sub(r"\s+", " ", text or "").strip()


def clean_for_embedding(text: str) -> str:
    """
    Limpieza recomendada antes de generar embeddings.

    Mantiene contenido semántico, pero reduce ruido.
    """
    text = normalize_text(text)
    text = collapse_spaces(text)
    return text


def remove_accents(text: str) -> str:
    """Quita tildes para comparación robusta."""
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_for_match(text: str) -> str:
    """Normaliza texto para matching simple."""
    text = remove_accents(text)
    text = text.upper()
    text = collapse_spaces(text)
    return text


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value == 1

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "si", "sí"}

    return False