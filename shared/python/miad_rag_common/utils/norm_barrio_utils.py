from __future__ import annotations

import unicodedata
from typing import Optional

try:
    from rapidfuzz import fuzz, process

    _HAS_RAPIDFUZZ = True
except ImportError:
    import difflib

    _HAS_RAPIDFUZZ = False


KNOWN_BARRIOS = {
    "AGUADA",
    "AIRES PUROS",
    "ATAHUALPA",
    "BARRIO SUR",
    "BAÑADOS DE CARRASCO",
    "BELVEDERE",
    "BRAZO ORIENTAL",
    "BUCEO",
    "CAPURRO BELLA VISTA",
    "CARRASCO",
    "CARRASCO NORTE",
    "CASABO PAJAS BLANCAS",
    "CASAVALLE",
    "CASTRO CASTELLANOS",
    "CENTRO",
    "CERRITO",
    "CERRO",
    "CIUDAD VIEJA",
    "COLON CENTRO Y NOROESTE",
    "COLON SURESTE ABAYUBA",
    "CONCILIACION",
    "CORDON",
    "FIGURITA",
    "FLOR DE MAROÑAS",
    "ITUZAINGO",
    "JACINTO VERA",
    "JARDINES DEL HIPODROMO",
    "LA BLANQUEADA",
    "LA COMERCIAL",
    "LA TEJA",
    "LARRAÑAGA",
    "LAS ACACIAS",
    "LAS CANTERAS",
    "LEZICA MELILLA",
    "MALVIN",
    "MALVIN NORTE",
    "MANGA",
    "MANGA TOLEDO CHICO",
    "MAROÑAS PARQUE GUARANI",
    "MERCADO MODELO Y BOLIVAR",
    "NUEVO PARIS",
    "PALERMO",
    "PARQUE RODO",
    "PASO DE LA ARENA",
    "PASO DE LAS DURANAS",
    "PEÑAROL LAVALLEJA",
    "PIEDRAS BLANCAS",
    "POCITOS",
    "PQUE BATLLE VILLA DOLORES",
    "PRADO NUEVA SAVONA",
    "PUNTA CARRETAS",
    "PUNTA GORDA",
    "PUNTA RIELES BELLA ITALIA",
    "REDUCTO",
    "SAYAGO",
    "TRES CRUCES",
    "TRES OMBUES PBLO VICTORIA",
    "UNION",
    "VILLA ESPAÑOLA",
    "VILLA GARCIA MANGA RURAL",
    "VILLA MUÑOZ RETIRO",
}


BARRIO_ALIASES = {
    "PARQUE RODÓ": "PARQUE RODO",
    "CORDÓN": "CORDON",
    "PEÑAROL": "PEÑAROL LAVALLEJA",
    "PQUE BATLLE": "PQUE BATLLE VILLA DOLORES",
    "PARQUE BATLLE": "PQUE BATLLE VILLA DOLORES",
    "VILLA DOLORES": "PQUE BATLLE VILLA DOLORES",
    "CAPURRO": "CAPURRO BELLA VISTA",
    "BELLA VISTA": "CAPURRO BELLA VISTA",
    "PRADO": "PRADO NUEVA SAVONA",
    "NUEVA SAVONA": "PRADO NUEVA SAVONA",
    "VILLA MUÑOZ": "VILLA MUÑOZ RETIRO",
    "RETIRO": "VILLA MUÑOZ RETIRO",
    "COLON": "COLON CENTRO Y NOROESTE",
    "COLÓN": "COLON CENTRO Y NOROESTE",
}


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_key(value: str) -> str:
    return " ".join(strip_accents(value).upper().strip().split())


def normalize_barrio(raw: Optional[str], cutoff: int = 80) -> Optional[str]:
    """
    Normaliza un barrio de Montevideo hacia su forma canónica.

    Ejemplos:
      "pocitos" -> "POCITOS"
      "Cordón" -> "CORDON"
      "Parque Rodó" -> "PARQUE RODO"
    """
    if not raw:
        return None

    candidate_original = " ".join(raw.upper().strip().split())
    candidate = normalize_key(candidate_original)

    if candidate_original in BARRIO_ALIASES:
        return BARRIO_ALIASES[candidate_original]

    if candidate in BARRIO_ALIASES:
        return BARRIO_ALIASES[candidate]

    normalized_known = {normalize_key(barrio): barrio for barrio in KNOWN_BARRIOS}

    if candidate in normalized_known:
        return normalized_known[candidate]

    if _HAS_RAPIDFUZZ:
        result = process.extractOne(
            candidate,
            list(normalized_known.keys()),
            scorer=fuzz.WRatio,
            score_cutoff=cutoff,
        )
        if result:
            match_key, _, _ = result
            return normalized_known[match_key]

    else:
        matches = difflib.get_close_matches(
            candidate,
            list(normalized_known.keys()),
            n=1,
            cutoff=cutoff / 100,
        )
        if matches:
            return normalized_known[matches[0]]

    return None