from rapidfuzz import process, fuzz
from typing import Optional


# Known barrios 

KNOWN_BARRIOS = {'AGUADA', 'AIRES PUROS', 'ATAHUALPA', 'BARRIO SUR', 'BAÑADOS DE CARRASCO', 'BELVEDERE',
                 'BRAZO ORIENTAL', 'BUCEO', 'CAPURRO BELLA VISTA', 'CARRASCO', 'CARRASCO NORTE',
                 'CASABO PAJAS BLANCAS', 'CASAVALLE', 'CASTRO CASTELLANOS', 'CENTRO', 'CERRITO',
                 'CERRO', 'CIUDAD VIEJA', 'COLON CENTRO Y NOROESTE', 'COLON SURESTE ABAYUBA',
                 'CONCILIACION', 'CORDON', 'FIGURITA', 'FLOR DE MAROÑAS', 'ITUZAINGO', 'JACINTO VERA',
                 'JARDINES DEL HIPODROMO', 'LA BLANQUEADA', 'LA COMERCIAL', 'LA TEJA', 'LARRAÑAGA',
                 'LAS ACACIAS', 'LAS CANTERAS', 'LEZICA MELILLA', 'MALVIN', 'MALVIN NORTE', 'MANGA',
                 'MANGA TOLEDO CHICO', 'MAROÑAS PARQUE GUARANI', 'MERCADO MODELO Y BOLIVAR',
                 'NUEVO PARIS', 'PALERMO', 'PARQUE RODO', 'PASO DE LA ARENA', 'PASO DE LAS DURANAS',
                 'PEÑAROL LAVALLEJA', 'PIEDRAS BLANCAS', 'POCITOS', 'PQUE BATLLE VILLA DOLORES',
                 'PRADO NUEVA SAVONA', 'PUNTA CARRETAS', 'PUNTA GORDA', 'PUNTA RIELES BELLA ITALIA',
                 'REDUCTO', 'SAYAGO', 'TRES CRUCES', 'TRES OMBUES PBLO VICTORIA', 'UNION',
                 'VILLA ESPAÑOLA', 'VILLA GARCIA MANGA RURAL', 'VILLA MUÑOZ RETIRO'}

def normalize_barrio(raw: str, cutoff: int = 80) -> Optional[str]:
    """
    Normaliza un nombre de barrio a su forma canónica en mayúsculas.

    Maneja:
      - Diferencias de capitalización: "pocitos" → "POCITOS"
      - Typos leves: "Posiitos" → "POCITOS" (score ~85)
      - Sin match: retorna None si el score es menor al cutoff

    Args:
        raw    : Nombre ingresado por el usuario o extraído por el LLM.
        cutoff : Score mínimo de similitud (0-100). Default 80.

    Returns:
        Nombre canónico del barrio, o None si no hay match suficiente.
    """
    if not raw:
        return None

    candidate = raw.upper().strip()

    # Exact match after normalization — no fuzzy needed
    if candidate in KNOWN_BARRIOS:
        return candidate

    # Fuzzy match
    result = process.extractOne(
        candidate,
        KNOWN_BARRIOS,
        scorer=fuzz.WRatio,  # handles partial matches and transpositions
        score_cutoff=cutoff,
    )

    if result:
        match, score, _ = result
        print(f"[normalize_barrio] '{raw}' → '{match}' (score={score})")
        return match

    print(f"[normalize_barrio] '{raw}' no match found (cutoff={cutoff})")
    return None