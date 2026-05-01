"""
Servicio de Documentos CSV para el Sistema RAG Inmobiliario
===========================================================

Convierte las filas del CSV de listings inmobiliarios en objetos Document
de LangChain, listos para ser embebidos y almacenados en FAISS.

Diseño general:
- page_content : texto en lenguaje natural que describe la propiedad,
                 construido a partir de los campos estructurados + título/descripción
                 original. Este texto es el que se embebe y contra el que se busca.
- metadata     : todos los campos estructurados como valores nativos (int, float, str),
                 usados para filtrado pre-semántico (barrio, tipo de operación, etc.)

NO se aplica chunking: cada listing es un documento completo por sí mismo.
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from langchain.schema import Document


# =============================================================================
# COLUMNAS A ELIMINAR
# =============================================================================

COLUMNS_TO_DROP = [
    # Artifacts de scraping
    "url", "scraped_at", "status", "image_urls", "thumbnail_url",
    # Seller info
    "seller_name", "seller_type", "seller_id",
    # Geometría redundante (lat/lon ya están como numéricos)
    "geometry",
    # barrio is replaced with barrio_fixed + Códigos administrativos internos (nombres en minúsculas desde el CSV)
    "barrio", "nrobarrio", "codba", "barrio_check",
    "status", "zona_legal", "departamen", "seccion_pol",
    # Precios sin limpiar (sustituidos por price_fixed / currency_fixed)
    "price", "currency",
    # Amenities como texto crudo (se decodifican en flags binarios)
    "amenities",
    # Columnas geográficas con correlación débil o nula con price/m²
    # (n_industrial, n_destinos, n_gubernamental se conservan — ver análisis)
    "n_tecnica_800m", "dist_tecnica",
    "n_formacion_docente_800m", "dist_formacion_docente",
    # Ordenanzas de tránsito — irrelevante para búsqueda inmobiliaria
    "n_ord_transito_800m", "dist_ord_transito", "area_ord_transito_800m",
]

# Nota: 'condition' se conserva deliberadamente.
# 'age' tiene 501 nulls (~15%); 'condition' (sin nulls) actúa como fallback
# en page_content y complementa la información de antigüedad.


# =============================================================================
# AMENITIES: MAPEO PIPE-SEPARATED → FLAGS BINARIOS
# =============================================================================

# Fuente: columna "amenities" (pipe-separated, ej: "Ascensor | Parrillero | Sauna")
# IMPORTANTE: has_parrillero figura aquí Y en AMENITY_DESC_FLAGS.
# En preprocess_dataframe() se aplica lógica OR para no perder señal de ninguna fuente.
AMENITY_PIPE_FLAGS = {
    "has_elevator":          r"ascensor",
    "has_gym":               r"gimnasio",
    "has_rooftop":           r"azotea",
    "has_party_room":        r"salón de fiestas",
    "has_multipurpose_room": r"salón de usos múltiples",
    "has_laundry":           r"área de lavandería",
    "has_green_area":        r"con área verde",
    "has_cowork":            r"cowork",
    "has_internet":          r"acceso a internet",
    "has_wheelchair":        r"rampa para silla de ruedas",
    "has_fireplace":         r"chimenea",
    "has_fridge":            r"heladera",
    # Amenities incorporadas tras análisis de frecuencias
    "has_parrillero":        r"parrillero",         # 232 en amenities, 1172 en desc — OR en step 3
    "has_reception":         r"recepción|recepcion", # 271 ocurrencias
    "has_playground":        r"área de juegos infantiles|parque infantil",  # 182 + 7
    "has_visitor_parking":   r"estacionamiento para visitas",  # 129 ocurrencias
    "has_sauna":             r"sauna",               # 53 ocurrencias
}

# Fuente: columna title + description (regex en texto libre)
# Estos amenities raramente aparecen en la columna amenities estructurada,
# o se complementan con señal adicional desde la descripción (has_parrillero).
AMENITY_DESC_FLAGS = {
    "has_pool":       r"piscina|pileta",
    "has_parrillero": r"parrillero|parrilla",  # OR con pipe: 48 listings solo en amenities
    "has_terrace":    r"terraza",
    "has_storage":    r"deposito|depósito|baulera",
    "has_security":   r"seguridad|vigilancia|portero",
}

# Labels en español para incluir en page_content
AMENITY_LABELS = {
    "has_elevator":          "ascensor",
    "has_gym":               "gimnasio",
    "has_rooftop":           "azotea",
    "has_party_room":        "salón de fiestas",
    "has_multipurpose_room": "salón de usos múltiples",
    "has_laundry":           "área de lavandería",
    "has_green_area":        "área verde",
    "has_cowork":            "cowork",
    "has_internet":          "acceso a internet",
    "has_wheelchair":        "acceso para silla de ruedas",
    "has_fireplace":         "chimenea",
    "has_fridge":            "heladera incluida",
    "has_parrillero":        "parrillero",
    "has_reception":         "recepción",
    "has_playground":        "área de juegos infantiles",
    "has_visitor_parking":   "estacionamiento para visitas",
    "has_sauna":             "sauna",
    "has_pool":              "piscina",
    "has_terrace":           "terraza",
    "has_storage":           "depósito/baulera",
    "has_security":          "seguridad/vigilancia",
}

ALL_AMENITY_FLAGS = list(AMENITY_PIPE_FLAGS.keys()) + [
    f for f in AMENITY_DESC_FLAGS.keys() if f not in AMENITY_PIPE_FLAGS
]


# =============================================================================
# CAMPOS DE METADATA
# =============================================================================

METADATA_FIELDS = [
    # Identificadores
    "id",
    # Segmentación principal
    "operation_type", "property_type", "barrio_fixed",
    # Calidad del barrio — resultado del pipeline de limpieza geoespacial
    # Valores: 'consistent' | 'no_barrio_in_text' | 'genuine_ambiguity' | 'marketing_inflation'
    "barrio_confidence",
    # Operación dual — True si el listing está disponible para venta Y alquiler
    "is_dual_intent",
    # Coordenadas
    "lat", "lon",
    # Precio
    "price_fixed", "currency_fixed", "price_m2", "price_m2_basis",
    # Características físicas
    "surface_covered", "surface_total", "bedrooms", "bathrooms",
    "floor", "age", "condition", "garages", "expenses",
    # Amenity flags (todos)
    *ALL_AMENITY_FLAGS,
    # Geografía urbana — distancias (metros)
    "dist_nearest_public_space", "dist_espacio_libre", "dist_plaza",
    "dist_plazoleta", "dist_isla", "dist_playa",
    "dist_nearest_escuela", "dist_primaria", "dist_secundaria",
    "dist_comercial", "dist_gubernamental", "dist_industrial",
    "dist_nearest_destino",
    # Geografía urbana — conteos dentro de 800m
    "n_public_spaces_800m", "n_espacio_libre_800m", "n_plaza_800m",
    "n_plazoleta_800m", "n_isla_800m", "n_playa_800m",
    "n_escuelas_800m", "n_primaria_800m", "n_secundaria_800m",
    "n_comercial_800m", "n_gubernamental_800m", "n_industrial_800m",
    "n_destinos_800m",
    # Geografía urbana — áreas (m²)
    "public_space_area_800m", "area_espacio_libre_800m",
    "area_plaza_800m", "area_plazoleta_800m",
    "area_isla_800m", "area_playa_800m",
]

OPERATION_LABEL = {"venta": "en venta", "alquiler": "en alquiler"}
PROPERTY_LABEL  = {"apartamentos": "apartamento", "casas": "casa"}


# =============================================================================
# HELPERS
# =============================================================================

def _safe(value, default=None):
    """
    Retorna default si el valor es NaN / None / pd.NA / pd.NaT.

    Usa pd.isna() para cubrir todos los tipos nulos de pandas, incluyendo
    pd.NA (columnas Int64 nullable) que isinstance(v, float) no detecta.
    El try/except protege contra tipos no hashables (listas, dicts).
    """
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _format_price(price, currency) -> Optional[str]:
    p = _safe(price)
    c = _safe(currency, "")
    if p is None:
        return None
    return f"{c} {int(p):,}".replace(",", ".").strip()


# =============================================================================
# PREPROCESADO DEL DATAFRAME
# =============================================================================

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y enriquece el DataFrame antes de la conversión a Documents.

    Pasos:
      1. Extrae flags binarios desde la columna 'amenities' (pipe-separated).
      2. Elimina columnas irrelevantes para el RAG.
      3. Extrae flags binarios adicionales desde title + description (regex).
         Para flags presentes en ambas fuentes (has_parrillero), aplica OR
         para no sobreescribir los resultados del paso 1.

    Args:
        df: DataFrame tal como sale del notebook de limpieza.

    Returns:
        DataFrame listo para ser convertido a Documents por CSVDocumentService.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Flags desde columna 'amenities' (pipe-separated)
    #    Debe ejecutarse ANTES del drop ya que 'amenities' está en COLUMNS_TO_DROP.
    # ------------------------------------------------------------------
    amenities_col = (
        df["amenities"].fillna("").str.lower()
        if "amenities" in df.columns
        else pd.Series("", index=df.index)
    )

    for flag, pattern in AMENITY_PIPE_FLAGS.items():
        df[flag] = amenities_col.str.contains(pattern, regex=True, na=False).astype(int)

    pipe_counts = {k: int(df[k].sum()) for k in AMENITY_PIPE_FLAGS if df[k].sum() > 0}
    print(f"[preprocess] Flags desde amenities:   {pipe_counts}")

    # ------------------------------------------------------------------
    # 2. Eliminar columnas irrelevantes
    # ------------------------------------------------------------------
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_present)
    print(f"[preprocess] Columnas eliminadas: {len(cols_present)}")
    print(f"[preprocess] Columnas restantes:  {df.shape[1]}")

    # ------------------------------------------------------------------
    # 3. Flags desde title + description (texto libre)
    #    Para flags que ya existen del paso 1 (ej: has_parrillero),
    #    se aplica OR para combinar señal de ambas fuentes sin sobreescribir.
    # ------------------------------------------------------------------
    title_series = (
        df["title"].fillna("") if "title" in df.columns
        else pd.Series("", index=df.index)
    )
    desc_series = (
        df["description"].fillna("") if "description" in df.columns
        else pd.Series("", index=df.index)
    )
    desc_col = (title_series + " " + desc_series).str.lower()

    for flag, pattern in AMENITY_DESC_FLAGS.items():
        from_desc = desc_col.str.contains(pattern, regex=True, na=False).astype(int)
        if flag in df.columns:
            # OR: preserva el 1 si ya fue seteado por el paso 1
            df[flag] = (df[flag] | from_desc).astype(int)
        else:
            df[flag] = from_desc

    desc_counts = {k: int(df[k].sum()) for k in AMENITY_DESC_FLAGS if df[k].sum() > 0}
    print(f"[preprocess] Flags desde description: {desc_counts}")

    return df


# =============================================================================
# CONSTRUCCIÓN DE page_content
# =============================================================================

def _build_property_summary(row: pd.Series) -> str:
    """Encabezado + características físicas + precio."""
    parts = []

    prop   = PROPERTY_LABEL.get(_safe(row.get("property_type"), ""), "propiedad")
    op     = OPERATION_LABEL.get(_safe(row.get("operation_type"), ""), "")
    barrio = _safe(row.get("barrio_fixed"), "")   # corrected ground truth barrio

    header = f"{prop.capitalize()} {op}".strip()
    if barrio:
        header += f" en {barrio}"
    parts.append(header + ".")

    attrs = []
    bedrooms = _safe(row.get("bedrooms"))
    if bedrooms is not None:
        n = int(bedrooms)
        attrs.append("monoambiente" if n == 0 else f"{n} dormitorio{'s' if n > 1 else ''}")

    bathrooms = _safe(row.get("bathrooms"))
    if bathrooms is not None:
        n = int(bathrooms)
        attrs.append(f"{n} baño{'s' if n > 1 else ''}")

    garages = _safe(row.get("garages"))
    if garages is not None and int(garages) > 0:
        n = int(garages)
        attrs.append(f"{n} cochera{'s' if n > 1 else ''}")

    floor = _safe(row.get("floor"))
    if floor is not None:
        floor_int = int(floor)
        attrs.append("planta baja" if floor_int == 0 else f"piso {floor_int}")

    # Antigüedad: age tiene precedencia; condition actúa como fallback cuando age es null.
    age       = _safe(row.get("age"))
    condition = _safe(row.get("condition"), "")
    if age is not None:
        attrs.append("a estrenar" if int(age) == 0 else f"{int(age)} años de antigüedad")
    elif condition == "new":
        attrs.append("a estrenar")
    # Si age es null y condition es 'used', se omite (dato insuficiente para precisar)

    if attrs:
        parts.append(", ".join(attrs) + ".")

    sc = _safe(row.get("surface_covered"))
    st = _safe(row.get("surface_total"))
    if sc is not None and st is not None and sc != st:
        parts.append(f"Superficie cubierta {int(sc)} m², total {int(st)} m².")
    elif sc is not None:
        parts.append(f"Superficie cubierta {int(sc)} m².")
    elif st is not None:
        parts.append(f"Superficie total {int(st)} m².")

    price_str = _format_price(row.get("price_fixed"), row.get("currency_fixed"))
    if price_str:
        pm2 = _safe(row.get("price_m2"))
        suffix = f" ({int(pm2):,} USD/m²)".replace(",", ".") if pm2 else ""
        parts.append(f"Precio: {price_str}{suffix}.")

    expenses = _safe(row.get("expenses"))
    if expenses is not None and expenses > 0:
        parts.append(f"Expensas: UYU {int(expenses):,}.".replace(",", "."))

    return "\n".join(parts)


def _build_amenities_text(row: pd.Series) -> Optional[str]:
    """Frase con los amenities presentes."""
    present = [
        AMENITY_LABELS[flag]
        for flag in ALL_AMENITY_FLAGS
        if _safe(row.get(flag), 0) == 1
    ]
    if not present:
        return None
    return f"Amenities: {', '.join(present)}."


def _build_geo_context(row: pd.Series) -> Optional[str]:
    """
    Párrafo de contexto urbano en lenguaje natural.

    Convierte distancias y conteos en frases útiles para búsquedas como
    "cerca de una plaza", "zona con colegios", "a pasos de la playa",
    "zona industrial", etc. Solo incluye frases cuando los datos están
    disponibles y son relevantes.
    """
    phrases = []

    # Playa
    dist_playa = _safe(row.get("dist_playa"))
    if dist_playa is not None and dist_playa <= 800:
        phrases.append(f"a {int(dist_playa)} m de la playa")

    # Plaza
    dist_plaza = _safe(row.get("dist_plaza"))
    n_plaza    = _safe(row.get("n_plaza_800m"), 0)
    if dist_plaza is not None and dist_plaza <= 800:
        phrases.append(f"plaza a {int(dist_plaza)} m")
        if n_plaza and int(n_plaza) > 1:
            phrases.append(f"{int(n_plaza)} plazas en radio de 800 m")

    # Espacios verdes (solo si no hay plaza para no repetir)
    if dist_plaza is None or dist_plaza > 800:
        n_verde = _safe(row.get("n_espacio_libre_800m"), 0)
        if n_verde and int(n_verde) > 0:
            phrases.append(
                f"{int(n_verde)} espacio{'s' if int(n_verde) > 1 else ''} "
                f"verde{'s' if int(n_verde) > 1 else ''} en radio de 800 m"
            )

    # Escuelas
    n_escuelas   = _safe(row.get("n_escuelas_800m"), 0)
    dist_escuela = _safe(row.get("dist_nearest_escuela"))
    if n_escuelas and int(n_escuelas) > 0:
        suffix = f" (la más cercana a {int(dist_escuela)} m)" if dist_escuela else ""
        phrases.append(
            f"{int(n_escuelas)} escuela{'s' if int(n_escuelas) > 1 else ''} "
            f"en radio de 800 m{suffix}"
        )

    school_parts = []
    n_primaria   = _safe(row.get("n_primaria_800m"), 0)
    n_secundaria = _safe(row.get("n_secundaria_800m"), 0)
    if n_primaria and int(n_primaria) > 0:
        school_parts.append(f"{int(n_primaria)} primaria{'s' if int(n_primaria) > 1 else ''}")
    if n_secundaria and int(n_secundaria) > 0:
        school_parts.append(f"{int(n_secundaria)} liceo{'s' if int(n_secundaria) > 1 else ''}")
    if school_parts:
        phrases.append(", ".join(school_parts) + " en zona")

    # Comercios
    n_comercial = _safe(row.get("n_comercial_800m"), 0)
    if n_comercial and int(n_comercial) > 0:
        if int(n_comercial) >= 10:
            phrases.append("zona comercial con múltiples servicios")
        else:
            phrases.append(
                f"{int(n_comercial)} comercio{'s' if int(n_comercial) > 1 else ''} "
                f"en radio de 800 m"
            )

    # Zona industrial — correlación negativa fuerte con price/m² (-0.28 a -0.37)
    # Se menciona explícitamente para que la búsqueda semántica pueda discriminar
    # consultas como "zona residencial tranquila" vs. listados en áreas industriales.
    n_industrial = _safe(row.get("n_industrial_800m"), 0)
    if n_industrial and int(n_industrial) > 0:
        phrases.append(
            f"zona con {int(n_industrial)} establecimiento{'s' if int(n_industrial) > 1 else ''} "
            f"industrial{'es' if int(n_industrial) > 1 else ''} en radio de 800 m"
        )

    if not phrases:
        return None
    return "Entorno: " + ", ".join(phrases) + "."


def _build_page_content(row: pd.Series) -> str:
    """
    Construye el texto completo que representa un listing para el embedding.

    Estructura:
      1. Resumen de la propiedad (tipo, operación, barrio, características, precio)
      2. Ubicación l3 (calle, esquina o zona específica)
      3. Amenities presentes
      4. Contexto urbano/geográfico
      5. Título original
      6. Descripción original (truncada a 5000 chars)
    """
    sections = []

    summary = _build_property_summary(row)
    if summary:
        sections.append(summary)

    l3 = _safe(row.get("l3"), "")
    if l3:
        # Strip spurious numeric ID prefix (e.g. "7639281 | Calle X...")
        l3 = re.sub(r"^\d+ \| ", "", str(l3)).strip()
        if l3:
            sections.append(f"Ubicación: {l3}")

    amenities = _build_amenities_text(row)
    if amenities:
        sections.append(amenities)

    geo = _build_geo_context(row)
    if geo:
        sections.append(geo)

    # Dual intent note — appears before title so it's prominent in the embedding
    if _safe(row.get("is_dual_intent"), False):
        sections.append("Disponible para venta y alquiler.")

    # Title and description — use cleaned versions for marketing_inflation listings
    # (false premium barrio name replaced with barrio_fixed in the cleaning pipeline).
    # For all other listings, use original title and description.
    is_marketing_inflation = (
        _safe(row.get("barrio_confidence"), "") == "marketing_inflation"
    )

    title_field = "title_clean" if is_marketing_inflation else "title"
    desc_field  = "desc_clean"  if is_marketing_inflation else "description"

    title = _safe(row.get(title_field), "") or _safe(row.get("title"), "")
    if title:
        sections.append(f"Título: {str(title).strip()}")

    description = _safe(row.get(desc_field), "") or _safe(row.get("description"), "")

    # Maximum description length before truncation.
    # Gemini embedding-001 limit: 2048 tokens.
    # At 3.2 chars/token (measured on 200-doc sample), non-description content
    # peaks at ~286 tokens (full corpus), leaving ~1,762 tokens for description.
    # 5,000 chars ≈ 1,562 tokens — well within budget with margin to spare.
    # In practice, 0/3,377 listings exceed this; cap is purely defensive. 
    # In place in case we decide to try other embedding models


    MAX_DESC_CHARS = 5_000
    if description:
        desc_text = str(description).strip()
        if len(desc_text) > MAX_DESC_CHARS:
            desc_text = desc_text[:MAX_DESC_CHARS] + "..."
        sections.append(f"Descripción: {desc_text}")

    return "\n".join(sections)


# =============================================================================
# CONSTRUCCIÓN DE METADATA
# =============================================================================

def _build_metadata(row: pd.Series) -> dict:
    """
    Extrae campos estructurados como Python nativos para metadata de FAISS.
    NaN / None / pd.NA → None; numpy types → Python types.
    """
    metadata = {}
    for field in METADATA_FIELDS:
        raw = row.get(field)
        val = _safe(raw)
        if val is None:
            metadata[field] = None
        elif isinstance(val, (np.integer,)):
            metadata[field] = int(val)
        elif isinstance(val, (np.floating,)):
            metadata[field] = float(val)
        elif isinstance(val, (np.bool_,)):
            metadata[field] = bool(val)
        else:
            metadata[field] = val
    return metadata


# =============================================================================
# CLASE PRINCIPAL
# =============================================================================

class CSVDocumentService:
    """
    Servicio para convertir un CSV de listings inmobiliarios en Documents de LangChain.

    Flujo típico:

        service = CSVDocumentService()

        # Opción A: desde CSV crudo (sale directo del notebook de limpieza)
        df = pd.read_csv("./data/listings.csv")
        df = service.preprocess(df)
        documents = service.dataframe_to_documents(df)

        # Opción B: shortcut en un solo paso
        documents = service.load_documents("./data/listings.csv")
    """

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesa el DataFrame: elimina columnas, extrae flags de amenities."""
        return preprocess_dataframe(df)

    def dataframe_to_documents(
        self,
        df: pd.DataFrame,
        operation_type: Optional[str] = None,
        property_type: Optional[str] = None,
        barrio: Optional[str] = None,
    ) -> List[Document]:
        """
        Convierte un DataFrame ya preprocesado en una lista de Documents.

        Args:
            df             : DataFrame limpio y preprocesado.
            operation_type : Filtro opcional: "venta" | "alquiler"
            property_type  : Filtro opcional: "apartamentos" | "casas"
            barrio         : Filtro opcional: nombre exacto del barrio (en mayúsculas)

        Returns:
            Lista de Document, uno por listing.
        """
        df = df.copy()

        if operation_type:
            df = df[df["operation_type"] == operation_type].copy()
            print(f"[filter] operation_type='{operation_type}': {len(df)} filas")
        if property_type:
            df = df[df["property_type"] == property_type].copy()
            print(f"[filter] property_type='{property_type}': {len(df)} filas")
        if barrio:
            df = df[df["barrio_fixed"] == barrio].copy()
            print(f"[filter] barrio_fixed='{barrio}': {len(df)} filas")

        if len(df) == 0:
            print("[WARNING] El filtro dejó 0 filas.")
            return []

        documents = []
        skipped   = 0

        for _, row in df.iterrows():
            try:
                page_content = _build_page_content(row)
                metadata     = _build_metadata(row)
                listing_id   = metadata.get("id", "unknown")
                metadata["source"]      = f"listing_{listing_id}"
                metadata["source_file"] = "listings.csv"
                documents.append(Document(page_content=page_content, metadata=metadata))
            except Exception as e:
                skipped += 1
                print(f"[WARNING] Fila omitida: {e}")

        print(
            f"[documents] Creados: {len(documents)}"
            + (f" | Omitidos: {skipped}" if skipped else "")
        )
        return documents

    def load_documents(
        self,
        csv_path: str,
        operation_type: Optional[str] = None,
        property_type: Optional[str] = None,
        barrio: Optional[str] = None,
    ) -> List[Document]:
        """Carga, preprocesa y convierte el CSV en Documents en un solo paso."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV no encontrado: {csv_path}")

        print(f"\n{'='*60}")
        print(f"CARGA DESDE CSV: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"Filas cargadas: {len(df)}")
        df   = self.preprocess(df)
        docs = self.dataframe_to_documents(df, operation_type, property_type, barrio)
        print(f"{'='*60}\n")
        return docs

    def get_available_segments(self, csv_path: str) -> dict:
        """Valores únicos de los campos de segmentación disponibles en el CSV."""
        df = pd.read_csv(csv_path, low_memory=False)
        return {
            "operation_types": sorted(df["operation_type"].dropna().unique().tolist()),
            "property_types":  sorted(df["property_type"].dropna().unique().tolist()),
            "barrios":         sorted(df["barrio_fixed"].dropna().unique().tolist()),
            "total_listings":  len(df),
        }

    def preview_document(self, csv_path: str, n: int = 3) -> None:
        """Muestra una preview de los primeros n Documents. Útil para validar antes de embeber."""
        docs = self.load_documents(csv_path)
        for i, doc in enumerate(docs[:n]):
            print(f"\n{'─'*60}")
            print(f"DOCUMENTO {i+1}")
            print(f"{'─'*60}")
            print("PAGE CONTENT:")
            print(doc.page_content)
            print("\nMETADATA (no-null):")
            for k, v in doc.metadata.items():
                if v is not None:
                    print(f"  {k}: {v}")
