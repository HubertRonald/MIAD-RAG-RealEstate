from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from langchain.schema import Document

from miad_rag_common.logging.structured_logging import get_logger
from miad_rag_common.utils.norm_barrio_utils import normalize_barrio
from miad_rag_common.utils.text_utils import clean_for_embedding

logger = get_logger(__name__)


# =============================================================================
# COLUMNAS A ELIMINAR
# =============================================================================

COLUMNS_TO_DROP = [
    # Artifacts de scraping
    "url",
    "scraped_at",
    "status",
    "image_urls",
    "thumbnail_url",

    # Seller info
    "seller_name",
    "seller_type",
    "seller_id",

    # Geometría redundante: BigQuery ya trae lat/lon como numéricos
    "geometry",

    # barrio original se reemplaza por barrio_fixed para FAISS
    "barrio",
    "nrobarrio",
    "codba",
    "barrio_check",
    "zona_legal",
    "departamen",
    "seccion_pol",

    # Precios sin limpiar, sustituidos por price_fixed / currency_fixed
    "price",
    "currency",

    # Amenities como texto crudo: se decodifican en flags binarios
    "amenities",

    # Columnas geográficas con correlación débil o irrelevantes para el RAG
    "n_tecnica_800m",
    "dist_tecnica",
    "n_formacion_docente_800m",
    "dist_formacion_docente",

    # Ordenanzas de tránsito — irrelevante para búsqueda inmobiliaria
    "n_ord_transito_800m",
    "dist_ord_transito",
    "area_ord_transito_800m",
]


# =============================================================================
# AMENITIES: MAPEO PIPE-SEPARATED → FLAGS BINARIOS
# =============================================================================

AMENITY_PIPE_FLAGS = {
    "has_elevator": r"ascensor",
    "has_gym": r"gimnasio",
    "has_rooftop": r"azotea",
    "has_party_room": r"salón de fiestas",
    "has_multipurpose_room": r"salón de usos múltiples",
    "has_laundry": r"área de lavandería",
    "has_green_area": r"con área verde",
    "has_cowork": r"cowork",
    "has_internet": r"acceso a internet",
    "has_wheelchair": r"rampa para silla de ruedas",
    "has_fireplace": r"chimenea",
    "has_fridge": r"heladera",

    # Amenities incorporadas tras análisis de frecuencias
    "has_parrillero": r"parrillero",
    "has_reception": r"recepción|recepcion",
    "has_playground": r"área de juegos infantiles|parque infantil",
    "has_visitor_parking": r"estacionamiento para visitas",
    "has_sauna": r"sauna",
}


AMENITY_DESC_FLAGS = {
    "has_pool": r"piscina|pileta",
    "has_parrillero": r"parrillero|parrilla",
    "has_terrace": r"terraza",
    "has_storage": r"deposito|depósito|baulera",
    "has_security": r"seguridad|vigilancia|portero",
}


AMENITY_LABELS = {
    "has_elevator": "ascensor",
    "has_gym": "gimnasio",
    "has_rooftop": "azotea",
    "has_party_room": "salón de fiestas",
    "has_multipurpose_room": "salón de usos múltiples",
    "has_laundry": "área de lavandería",
    "has_green_area": "área verde",
    "has_cowork": "cowork",
    "has_internet": "acceso a internet",
    "has_wheelchair": "acceso para silla de ruedas",
    "has_fireplace": "chimenea",
    "has_fridge": "heladera incluida",
    "has_parrillero": "parrillero",
    "has_reception": "recepción",
    "has_playground": "área de juegos infantiles",
    "has_visitor_parking": "estacionamiento para visitas",
    "has_sauna": "sauna",
    "has_pool": "piscina",
    "has_terrace": "terraza",
    "has_storage": "depósito/baulera",
    "has_security": "seguridad/vigilancia",
}


ALL_AMENITY_FLAGS = list(AMENITY_PIPE_FLAGS.keys()) + [
    flag for flag in AMENITY_DESC_FLAGS.keys()
    if flag not in AMENITY_PIPE_FLAGS
]


# =============================================================================
# CAMPOS DE METADATA PARA FAISS
# =============================================================================

METADATA_FIELDS = [
    # Identificador
    "id",

    # Segmentación principal
    "operation_type",
    "property_type",
    "barrio_fixed",

    # Calidad del barrio
    # Valores esperados:
    # 'consistent' | 'no_barrio_in_text' | 'genuine_ambiguity' | 'marketing_inflation'
    "barrio_confidence",

    # Operación dual: True si el listing está disponible para venta y alquiler
    "is_dual_intent",

    # Coordenadas reales de BigQuery
    # Importante: en BigQuery NO existen latitude/longitude, solo lat/lon.
    "lat",
    "lon",

    # Precio
    "price_fixed",
    "currency_fixed",
    "price_m2",
    "price_m2_basis",

    # Características físicas
    "surface_covered",
    "surface_total",
    "bedrooms",
    "bathrooms",
    "floor",
    "age",
    "condition",
    "garages",
    "expenses",

    # Amenity flags
    *ALL_AMENITY_FLAGS,

    # Geografía urbana — distancias
    "dist_nearest_public_space",
    "dist_espacio_libre",
    "dist_plaza",
    "dist_plazoleta",
    "dist_isla",
    "dist_playa",
    "dist_nearest_escuela",
    "dist_primaria",
    "dist_secundaria",
    "dist_comercial",
    "dist_gubernamental",
    "dist_industrial",
    "dist_nearest_destino",

    # Geografía urbana — conteos dentro de 800m
    "n_public_spaces_800m",
    "n_espacio_libre_800m",
    "n_plaza_800m",
    "n_plazoleta_800m",
    "n_isla_800m",
    "n_playa_800m",
    "n_escuelas_800m",
    "n_primaria_800m",
    "n_secundaria_800m",
    "n_comercial_800m",
    "n_gubernamental_800m",
    "n_industrial_800m",
    "n_destinos_800m",

    # Geografía urbana — áreas
    "public_space_area_800m",
    "area_espacio_libre_800m",
    "area_plaza_800m",
    "area_plazoleta_800m",
    "area_isla_800m",
    "area_playa_800m",
]


OPERATION_LABEL = {
    "venta": "en venta",
    "alquiler": "en alquiler",
}


PROPERTY_LABEL = {
    "apartamentos": "apartamento",
    "casas": "casa",
}


# =============================================================================
# HELPERS
# =============================================================================

def _safe(value, default=None):
    """
    Retorna default si el valor es NaN / None / pd.NA / pd.NaT.

    Usa pd.isna() para cubrir tipos nulos de pandas, incluyendo pd.NA.
    El try/except protege contra tipos no escalares.
    """
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    return value


def _normalize_text_key(value: str) -> str:
    """
    Normalización ligera para preservar valores cuando normalize_barrio()
    no encuentra match.
    """
    return " ".join(str(value).upper().strip().split())


def _normalize_barrio_or_original(value) -> Optional[str]:
    """
    Normaliza barrio usando catálogo conocido.

    Si no hay match suficiente, preserva el valor original normalizado.
    Esto evita perder barrio_fixed por un fallo de fuzzy matching.
    """
    raw = _safe(value)

    if raw is None:
        return None

    raw_text = _normalize_text_key(str(raw))

    if not raw_text:
        return None

    return normalize_barrio(raw_text) or raw_text


def _format_price(price, currency) -> Optional[str]:
    p = _safe(price)
    c = _safe(currency, "")

    if p is None:
        return None

    try:
        return f"{c} {int(float(p)):,}".replace(",", ".").strip()
    except (TypeError, ValueError):
        return None


def _to_python_native(value):
    """
    Convierte valores numpy/pandas a tipos nativos Python para metadata FAISS.
    """
    value = _safe(value)

    if value is None:
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def _as_int(value) -> Optional[int]:
    value = _safe(value)

    if value is None:
        return None

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value) -> Optional[float]:
    value = _safe(value)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# =============================================================================
# SERVICIO PRINCIPAL
# =============================================================================

class ListingDocumentService:
    """
    Convierte filas de BigQuery en Documents de LangChain.

    Es la evolución del antiguo CSVDocumentService:

    - ya no lee CSV;
    - recibe un DataFrame desde BigQueryReader;
    - conserva la lógica de un listing = un Document;
    - conserva metadata estructurada para filtros pre-semánticos;
    - usa barrio_fixed como barrio canónico;
    - conserva lat/lon sin crear latitude/longitude.

    No incluye:
    - load_documents();
    - preview_document();
    - get_available_segments().

    Esas utilidades eran propias del flujo local con CSV.
    """

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y enriquece el DataFrame antes de convertirlo a Documents.

        Pasos:
          1. Normaliza barrio_fixed sin perder el valor original si no hay match.
          2. Extrae flags binarios desde amenities.
          3. Extrae flags adicionales desde title/description limpios y crudos.
          4. Aplica OR en flags presentes en ambas fuentes.
          5. Elimina columnas irrelevantes para FAISS.
        """
        df = df.copy()

        # --------------------------------------------------------------
        # 1. Normalizar barrio_fixed sin reemplazarlo por barrio.
        # --------------------------------------------------------------
        if "barrio_fixed" in df.columns:
            df["barrio_fixed"] = df["barrio_fixed"].apply(
                _normalize_barrio_or_original
            )

        # --------------------------------------------------------------
        # 2. Flags desde amenities.
        # Debe correr antes de eliminar amenities.
        # --------------------------------------------------------------
        amenities_col = (
            df["amenities"].fillna("").astype(str).str.lower()
            if "amenities" in df.columns
            else pd.Series("", index=df.index)
        )

        for flag, pattern in AMENITY_PIPE_FLAGS.items():
            df[flag] = amenities_col.str.contains(
                pattern,
                regex=True,
                na=False,
            ).astype(int)

        pipe_counts = {
            key: int(df[key].sum())
            for key in AMENITY_PIPE_FLAGS
            if key in df.columns and int(df[key].sum()) > 0
        }

        logger.info(
            "amenity_flags_from_pipe_created",
            extra={"counts": pipe_counts},
        )

        # --------------------------------------------------------------
        # 3. Flags desde title_clean / description_clean.
        # Para embeddings y detección de amenities usamos solo texto limpio.
        # Los campos crudos quedan en BigQuery para enriquecimiento posterior
        # del frontend, pero no alimentan el índice FAISS.
        # --------------------------------------------------------------
        title_clean_series = (
            df["title_clean"].fillna("").astype(str)
            if "title_clean" in df.columns
            else pd.Series("", index=df.index)
        )

        description_clean_series = (
            df["description_clean"].fillna("").astype(str)
            if "description_clean" in df.columns
            else pd.Series("", index=df.index)
        )

        text_col = (
            title_clean_series
            + " "
            + description_clean_series
        ).str.lower()

        for flag, pattern in AMENITY_DESC_FLAGS.items():
            from_description = text_col.str.contains(
                pattern,
                regex=True,
                na=False,
            ).astype(int)

            if flag in df.columns:
                df[flag] = (df[flag].astype(int) | from_description).astype(int)
            else:
                df[flag] = from_description

        desc_counts = {
            key: int(df[key].sum())
            for key in AMENITY_DESC_FLAGS
            if key in df.columns and int(df[key].sum()) > 0
        }

        logger.info(
            "amenity_flags_from_description_created",
            extra={"counts": desc_counts},
        )

        # --------------------------------------------------------------
        # 4. Eliminar columnas irrelevantes para FAISS.
        # --------------------------------------------------------------
        cols_to_drop = [
            column for column in COLUMNS_TO_DROP
            if column in df.columns
        ]

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        logger.info(
            "listing_dataframe_preprocessed",
            extra={
                "columns_dropped": len(cols_to_drop),
                "columns_remaining": df.shape[1],
                "rows": len(df),
            },
        )

        return df

    def dataframe_to_documents(
        self,
        df: pd.DataFrame,
        operation_type: Optional[str] = None,
        property_type: Optional[str] = None,
        barrio: Optional[str] = None,
    ) -> list[Document]:
        """
        Convierte un DataFrame ya preprocesado en Documents.

        Args:
            df:
                DataFrame leído desde BigQuery y preprocesado.
            operation_type:
                Filtro opcional: "venta" | "alquiler".
            property_type:
                Filtro opcional: "apartamentos" | "casas".
            barrio:
                Filtro opcional contra barrio_fixed.

        Returns:
            Lista de Document, uno por listing.
        """
        df = df.copy()

        if operation_type and "operation_type" in df.columns:
            df = df[df["operation_type"] == operation_type].copy()
            logger.info(
                "document_filter_applied",
                extra={
                    "field": "operation_type",
                    "value": operation_type,
                    "rows": len(df),
                },
            )

        if property_type and "property_type" in df.columns:
            df = df[df["property_type"] == property_type].copy()
            logger.info(
                "document_filter_applied",
                extra={
                    "field": "property_type",
                    "value": property_type,
                    "rows": len(df),
                },
            )

        if barrio and "barrio_fixed" in df.columns:
            normalized_barrio = _normalize_barrio_or_original(barrio)
            df = df[df["barrio_fixed"] == normalized_barrio].copy()
            logger.info(
                "document_filter_applied",
                extra={
                    "field": "barrio_fixed",
                    "value": normalized_barrio,
                    "rows": len(df),
                },
            )

        if df.empty:
            logger.warning(
                "document_filter_result_empty",
                extra={
                    "operation_type": operation_type,
                    "property_type": property_type,
                    "barrio": barrio,
                },
            )
            return []

        documents: list[Document] = []
        skipped = 0

        for _, row in df.iterrows():
            try:
                page_content = self._build_page_content(row)
                metadata = self._build_metadata(row)

                listing_id = metadata.get("id", "unknown")
                metadata["source"] = f"listing_{listing_id}"
                metadata["source_file"] = "bigquery.real_estate_listings"

                documents.append(
                    Document(
                        page_content=page_content,
                        metadata=metadata,
                    )
                )

            except Exception as exc:
                skipped += 1
                logger.warning(
                    "listing_document_skipped",
                    extra={"error": str(exc)},
                )

        logger.info(
            "listing_documents_created",
            extra={
                "documents_count": len(documents),
                "skipped": skipped,
            },
        )

        return documents

    def _build_page_content(self, row: pd.Series) -> str:
        """
        Construye el texto que representa el listing para embeddings.

        Estructura:
          1. Resumen de la propiedad.
          2. Ubicación l3.
          3. Amenities.
          4. Contexto urbano.
          5. Nota de operación dual.
          6. Título limpio/raw.
          7. Descripción limpia/raw.

        BigQuery tiene title_clean y description_clean.
        No existe desc_clean.
        """
        sections: list[str] = []

        summary = self._build_property_summary(row)
        if summary:
            sections.append(summary)

        l3 = _safe(row.get("l3"), "")
        if l3:
            l3 = re.sub(r"^\d+ \| ", "", str(l3)).strip()
            if l3:
                sections.append(f"Ubicación: {l3}")

        amenities = self._build_amenities_text(row)
        if amenities:
            sections.append(amenities)

        geo = self._build_geo_context(row)
        if geo:
            sections.append(geo)

        if _safe(row.get("is_dual_intent"), False):
            sections.append("Disponible para venta y alquiler.")

        # Title and description — usar SIEMPRE campos limpios para embeddings.
        #
        # En BigQuery existen:
        #   - title_clean
        #   - description_clean
        #
        # No existe desc_clean.
        #
        # Decisión de diseño:
        #   Para el índice FAISS usamos texto limpio, no texto crudo.
        #   El texto crudo queda en BigQuery y puede ser recuperado por el backend
        #   para cards/frontend si se necesita, pero no se usa para embeddings.
        title = _safe(row.get("title_clean"), "")

        if title:
            sections.append(f"Título: {str(title).strip()}")

        description = _safe(row.get("description_clean"), "")

        # Maximum description length before truncation.
        # Gemini embedding-001 limit: 2048 tokens.
        # At 3.2 chars/token, 5,000 chars ≈ 1,562 tokens.
        # El cap es defensivo para mantener margen junto con el resumen,
        # amenities y contexto urbano.
        max_desc_chars = 5_000

        if description:
            description_text = str(description).strip()
            if len(description_text) > max_desc_chars:
                description_text = description_text[:max_desc_chars] + "..."
            sections.append(f"Descripción: {description_text}")

        return clean_for_embedding("\n".join(sections))

    def _build_property_summary(self, row: pd.Series) -> str:
        """
        Encabezado + características físicas + precio.
        """
        parts: list[str] = []

        property_type = PROPERTY_LABEL.get(
            _safe(row.get("property_type"), ""),
            "propiedad",
        )

        operation_type = OPERATION_LABEL.get(
            _safe(row.get("operation_type"), ""),
            "",
        )

        barrio = _safe(row.get("barrio_fixed"), "")

        header = f"{property_type.capitalize()} {operation_type}".strip()
        if barrio:
            header += f" en {barrio}"

        parts.append(header + ".")

        attrs: list[str] = []

        bedrooms = _as_int(row.get("bedrooms"))
        if bedrooms is not None:
            attrs.append(
                "monoambiente"
                if bedrooms == 0
                else f"{bedrooms} dormitorio{'s' if bedrooms > 1 else ''}"
            )

        bathrooms = _as_int(row.get("bathrooms"))
        if bathrooms is not None:
            attrs.append(f"{bathrooms} baño{'s' if bathrooms > 1 else ''}")

        garages = _as_int(row.get("garages"))
        if garages is not None and garages > 0:
            attrs.append(f"{garages} cochera{'s' if garages > 1 else ''}")

        floor = _as_int(row.get("floor"))
        if floor is not None:
            attrs.append("planta baja" if floor == 0 else f"piso {floor}")

        age = _as_int(row.get("age"))
        condition = _safe(row.get("condition"), "")

        if age is not None:
            attrs.append("a estrenar" if age == 0 else f"{age} años de antigüedad")
        elif condition == "new":
            attrs.append("a estrenar")

        if attrs:
            parts.append(", ".join(attrs) + ".")

        surface_covered = _as_float(row.get("surface_covered"))
        surface_total = _as_float(row.get("surface_total"))

        if surface_covered is not None and surface_total is not None:
            if surface_covered != surface_total:
                parts.append(
                    f"Superficie cubierta {int(surface_covered)} m², "
                    f"total {int(surface_total)} m²."
                )
            else:
                parts.append(f"Superficie cubierta {int(surface_covered)} m².")
        elif surface_covered is not None:
            parts.append(f"Superficie cubierta {int(surface_covered)} m².")
        elif surface_total is not None:
            parts.append(f"Superficie total {int(surface_total)} m².")

        price = _format_price(
            row.get("price_fixed"),
            row.get("currency_fixed"),
        )

        if price:
            price_m2 = _as_float(row.get("price_m2"))
            suffix = ""
            if price_m2:
                suffix = f" ({int(price_m2):,} USD/m²)".replace(",", ".")
            parts.append(f"Precio: {price}{suffix}.")

        expenses = _as_float(row.get("expenses"))
        if expenses is not None and expenses > 0:
            parts.append(f"Expensas: UYU {int(expenses):,}.".replace(",", "."))

        return "\n".join(parts)

    def _build_amenities_text(self, row: pd.Series) -> Optional[str]:
        """
        Frase con los amenities presentes.
        """
        present: list[str] = []

        for flag in ALL_AMENITY_FLAGS:
            if _as_int(row.get(flag)) == 1:
                label = AMENITY_LABELS.get(flag)
                if label:
                    present.append(label)

        if not present:
            return None

        return f"Amenities: {', '.join(present)}."

    def _build_geo_context(self, row: pd.Series) -> Optional[str]:
        """
        Párrafo de contexto urbano en lenguaje natural.
        """
        phrases: list[str] = []

        # Playa
        dist_playa = _as_float(row.get("dist_playa"))
        if dist_playa is not None and dist_playa <= 800:
            phrases.append(f"a {int(dist_playa)} m de la playa")

        # Plaza
        dist_plaza = _as_float(row.get("dist_plaza"))
        n_plaza = _as_int(row.get("n_plaza_800m"))

        if dist_plaza is not None and dist_plaza <= 800:
            phrases.append(f"plaza a {int(dist_plaza)} m")

            if n_plaza and n_plaza > 1:
                phrases.append(f"{n_plaza} plazas en radio de 800 m")

        # Espacios verdes, solo si no hay plaza cercana para no repetir.
        if dist_plaza is None or dist_plaza > 800:
            n_verde = _as_int(row.get("n_espacio_libre_800m"))
            if n_verde and n_verde > 0:
                phrases.append(
                    f"{n_verde} espacio{'s' if n_verde > 1 else ''} "
                    f"verde{'s' if n_verde > 1 else ''} en radio de 800 m"
                )

        # Escuelas
        n_escuelas = _as_int(row.get("n_escuelas_800m"))
        dist_escuela = _as_float(row.get("dist_nearest_escuela"))

        if n_escuelas and n_escuelas > 0:
            suffix = (
                f" (la más cercana a {int(dist_escuela)} m)"
                if dist_escuela
                else ""
            )
            phrases.append(
                f"{n_escuelas} escuela{'s' if n_escuelas > 1 else ''} "
                f"en radio de 800 m{suffix}"
            )

        school_parts: list[str] = []

        n_primaria = _as_int(row.get("n_primaria_800m"))
        n_secundaria = _as_int(row.get("n_secundaria_800m"))

        if n_primaria and n_primaria > 0:
            school_parts.append(
                f"{n_primaria} primaria{'s' if n_primaria > 1 else ''}"
            )

        if n_secundaria and n_secundaria > 0:
            school_parts.append(
                f"{n_secundaria} liceo{'s' if n_secundaria > 1 else ''}"
            )

        if school_parts:
            phrases.append(", ".join(school_parts) + " en zona")

        # Comercios
        n_comercial = _as_int(row.get("n_comercial_800m"))
        if n_comercial and n_comercial > 0:
            if n_comercial >= 10:
                phrases.append("zona comercial con múltiples servicios")
            else:
                phrases.append(
                    f"{n_comercial} comercio{'s' if n_comercial > 1 else ''} "
                    f"en radio de 800 m"
                )

        # Zona industrial
        n_industrial = _as_int(row.get("n_industrial_800m"))
        if n_industrial and n_industrial > 0:
            phrases.append(
                f"zona con {n_industrial} establecimiento"
                f"{'s' if n_industrial > 1 else ''} industrial"
                f"{'es' if n_industrial > 1 else ''} "
                f"en radio de 800 m"
            )

        if not phrases:
            return None

        return "Entorno: " + ", ".join(phrases) + "."

    def _build_metadata(self, row: pd.Series) -> dict:
        """
        Extrae campos estructurados como Python nativos para metadata de FAISS.

        Importante:
          - conserva lat/lon;
          - no crea latitude/longitude porque BigQuery no tiene esos campos;
          - incluye todas las claves de METADATA_FIELDS para estructura estable.
        """
        metadata: dict = {}

        for field in METADATA_FIELDS:
            metadata[field] = _to_python_native(row.get(field))

        return metadata
