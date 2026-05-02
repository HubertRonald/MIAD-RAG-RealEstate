from __future__ import annotations

import logging
import unicodedata
from typing import Optional

from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)


# Known barrios — mantener la lista canónica original
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


# Aliases explícitos, pero sin reemplazar la lista canónica.
# Estos ayudan cuando el usuario escribe nombres abreviados o con tildes.
BARRIO_ALIASES = {
    "PARQUE RODÓ": "PARQUE RODO",
    "PARQUE RODO": "PARQUE RODO",
    "CORDÓN": "CORDON",
    "CORDON": "CORDON",
    "MALVÍN": "MALVIN",
    "MALVIN": "MALVIN",
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
    "COLÓN": "COLON CENTRO Y NOROESTE",
    "COLON": "COLON CENTRO Y NOROESTE",
}


def strip_accents(value: str) -> str:
    """
    Quita tildes para comparación robusta.
    """
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_key(value: str) -> str:
    """
    Normaliza texto para comparación:
    - convierte a str
    - mayúsculas
    - quita tildes
    - colapsa espacios
    """
    return " ".join(strip_accents(str(value)).upper().strip().split())


def normalize_barrio(raw: Optional[str], cutoff: int = 80) -> Optional[str]:
    """
    Normaliza un nombre de barrio a su forma canónica en mayúsculas.

    Mantiene la intención de la función original:

      - "pocitos" -> "POCITOS"
      - "Posiitos" -> "POCITOS" si supera cutoff
      - sin match suficiente -> None

    Mejoras frente a la original:

      - soporta tildes/no tildes: "Cordón" -> "CORDON"
      - aliases explícitos: "Prado" -> "PRADO NUEVA SAVONA"
      - matching determinístico usando lista ordenada
      - usa logging en vez de print
    """
    if not raw:
        return None

    raw_text = str(raw)
    candidate_original = " ".join(raw_text.upper().strip().split())
    candidate_key = normalize_key(candidate_original)

    # 1. Exact match original
    if candidate_original in KNOWN_BARRIOS:
        return candidate_original

    # 2. Alias con forma original
    if candidate_original in BARRIO_ALIASES:
        match = BARRIO_ALIASES[candidate_original]
        logger.debug("[normalize_barrio] alias '%s' -> '%s'", raw, match)
        return match

    # 3. Alias normalizado sin tildes
    normalized_aliases = {
        normalize_key(alias): canonical
        for alias, canonical in BARRIO_ALIASES.items()
    }

    if candidate_key in normalized_aliases:
        match = normalized_aliases[candidate_key]
        logger.debug("[normalize_barrio] normalized alias '%s' -> '%s'", raw, match)
        return match

    # 4. Exact match normalizado contra barrios conocidos
    normalized_known = {
        normalize_key(barrio): barrio
        for barrio in KNOWN_BARRIOS
    }

    if candidate_key in normalized_known:
        return normalized_known[candidate_key]

    # 5. Fuzzy match determinístico
    choices = sorted(normalized_known.keys())

    result = process.extractOne(
        candidate_key,
        choices,
        scorer=fuzz.WRatio,
        score_cutoff=cutoff,
    )

    if result:
        match_key, score, _ = result
        match = normalized_known[match_key]

        logger.debug(
            "[normalize_barrio] fuzzy '%s' -> '%s' score=%s",
            raw,
            match,
            score,
        )

        return match

    logger.debug(
        "[normalize_barrio] '%s' no match found cutoff=%s",
        raw,
        cutoff,
    )

    return None
