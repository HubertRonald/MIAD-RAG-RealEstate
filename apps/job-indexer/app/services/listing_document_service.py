from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from langchain.schema import Document

from miad_rag_common.utils.norm_barrio_utils import normalize_barrio
from miad_rag_common.utils.text_utils import clean_for_embedding


COLUMNS_TO_DROP = [
    "scraped_at",
    "status",
    "image_urls",
    "thumbnail_url",
    "seller_name",
    "seller_type",
    "seller_id",
    "geometry",
    "nrobarrio",
    "codba",
    "zona_legal",
    "departamen",
    "seccion_pol",
    "price",
    "currency",
    "n_tecnica_800m",
    "dist_tecnica",
    "n_formacion_docente_800m",
    "dist_formacion_docente",
    "n_ord_transito_800m",
    "dist_ord_transito",
    "area_ord_transito_800m",
]


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
    "has_parrillero": r"parrillero",
    "has_reception": r"recepción|recepcion",
    "has_playground": r"área de juegos infantiles|parque infantil",
    "has_visitor_parking": r"estacionamiento para visitas",
    "has_sauna": r"sauna",
}


AMENITY_DESC_FLAGS = {
    "has_pool": r"piscina|pileta",
    "has_parrillero": r"parrillero|parrilla",
    "has_terrace": r"terraza|balcón|balcon",
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
    "has_terrace": "terraza o balcón",
    "has_storage": "depósito o baulera",
    "has_security": "seguridad o vigilancia",
}


ALL_AMENITY_FLAGS = list(AMENITY_PIPE_FLAGS.keys()) + [
    flag for flag in AMENITY_DESC_FLAGS if flag not in AMENITY_PIPE_FLAGS
]


METADATA_FIELDS = [
    "id",
    "operation_type",
    "property_type",
    "barrio",
    "lat",
    "lon",
    "latitude",
    "longitude",
    "url",
    "title",
    "price_fixed",
    "currency_fixed",
    "price_m2",
    "price_m2_basis",
    "surface_covered",
    "surface_total",
    "bedrooms",
    "bathrooms",
    "floor",
    "age",
    "condition",
    "garages",
    "expenses",
    *ALL_AMENITY_FLAGS,
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


def _safe(value, default=None):
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

    try:
        return f"{c} {int(float(p)):,}".replace(",", ".").strip()
    except (TypeError, ValueError):
        return None


def _to_python_native(value):
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


class ListingDocumentService:
    """
    Convierte filas de BigQuery en Documents de LangChain.

    Es la evolución del antiguo CSVDocumentService:
    - ya no lee CSV,
    - recibe DataFrame desde BigQueryReader,
    - conserva la lógica de un listing = un Document,
    - conserva metadata para filtros estructurados.
    """

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "barrio" in df.columns:
            df["barrio"] = df["barrio"].apply(
                lambda value: normalize_barrio(str(value)) if _safe(value) else None
            )

        if "amenities" in df.columns:
            amenities_col = df["amenities"].fillna("").astype(str).str.lower()
        else:
            amenities_col = pd.Series("", index=df.index)

        for flag, pattern in AMENITY_PIPE_FLAGS.items():
            df[flag] = amenities_col.str.contains(
                pattern,
                regex=True,
                na=False,
            ).astype(int)

        title_series = (
            df["title"].fillna("").astype(str)
            if "title" in df.columns
            else pd.Series("", index=df.index)
        )

        description_series = (
            df["description"].fillna("").astype(str)
            if "description" in df.columns
            else pd.Series("", index=df.index)
        )

        text_col = (title_series + " " + description_series).str.lower()

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

        cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df

    def dataframe_to_documents(
        self,
        df: pd.DataFrame,
        operation_type: Optional[str] = None,
        property_type: Optional[str] = None,
        barrio: Optional[str] = None,
    ) -> list[Document]:
        df = df.copy()

        if operation_type and "operation_type" in df.columns:
            df = df[df["operation_type"] == operation_type].copy()

        if property_type and "property_type" in df.columns:
            df = df[df["property_type"] == property_type].copy()

        if barrio and "barrio" in df.columns:
            normalized_barrio = normalize_barrio(barrio)
            df = df[df["barrio"] == normalized_barrio].copy()

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
            except Exception:
                skipped += 1

        if not documents:
            raise ValueError(
                f"No se generaron Documents desde BigQuery. Filas omitidas: {skipped}"
            )

        return documents

    def _build_page_content(self, row: pd.Series) -> str:
        sections: list[str] = []

        summary = self._build_property_summary(row)
        if summary:
            sections.append(summary)

        location = _safe(row.get("l3"), "")
        if location:
            location = re.sub(r"^\d+ \| ", "", str(location)).strip()
            if location:
                sections.append(f"Ubicación: {location}")

        amenities = self._build_amenities_text(row)
        if amenities:
            sections.append(amenities)

        geo = self._build_geo_context(row)
        if geo:
            sections.append(geo)

        title = _safe(row.get("title"), "")
        if title:
            sections.append(f"Título: {str(title).strip()}")

        description = _safe(row.get("description"), "")
        if description:
            description_text = str(description).strip()
            if len(description_text) > 1200:
                description_text = description_text[:1200] + "..."
            sections.append(f"Descripción: {description_text}")

        return clean_for_embedding("\n".join(sections))

    def _build_property_summary(self, row: pd.Series) -> str:
        parts: list[str] = []

        property_type = PROPERTY_LABEL.get(
            _safe(row.get("property_type"), ""),
            "propiedad",
        )
        operation_type = OPERATION_LABEL.get(
            _safe(row.get("operation_type"), ""),
            "",
        )
        barrio = _safe(row.get("barrio"), "")

        header = f"{property_type.capitalize()} {operation_type}".strip()
        if barrio:
            header += f" en {barrio}"
        parts.append(header + ".")

        attrs: list[str] = []

        bedrooms = _safe(row.get("bedrooms"))
        if bedrooms is not None:
            bedrooms_int = int(float(bedrooms))
            attrs.append(
                "monoambiente"
                if bedrooms_int == 0
                else f"{bedrooms_int} dormitorio{'s' if bedrooms_int > 1 else ''}"
            )

        bathrooms = _safe(row.get("bathrooms"))
        if bathrooms is not None:
            bathrooms_int = int(float(bathrooms))
            attrs.append(f"{bathrooms_int} baño{'s' if bathrooms_int > 1 else ''}")

        garages = _safe(row.get("garages"))
        if garages is not None and int(float(garages)) > 0:
            garages_int = int(float(garages))
            attrs.append(f"{garages_int} cochera{'s' if garages_int > 1 else ''}")

        floor = _safe(row.get("floor"))
        if floor is not None:
            floor_int = int(float(floor))
            attrs.append("planta baja" if floor_int == 0 else f"piso {floor_int}")

        age = _safe(row.get("age"))
        condition = _safe(row.get("condition"), "")
        if age is not None:
            age_int = int(float(age))
            attrs.append(
                "a estrenar" if age_int == 0 else f"{age_int} años de antigüedad"
            )
        elif condition == "new":
            attrs.append("a estrenar")

        if attrs:
            parts.append(", ".join(attrs) + ".")

        surface_covered = _safe(row.get("surface_covered"))
        surface_total = _safe(row.get("surface_total"))

        if surface_covered is not None and surface_total is not None:
            if float(surface_covered) != float(surface_total):
                parts.append(
                    f"Superficie cubierta {int(float(surface_covered))} m², "
                    f"total {int(float(surface_total))} m²."
                )
            else:
                parts.append(
                    f"Superficie cubierta {int(float(surface_covered))} m²."
                )
        elif surface_covered is not None:
            parts.append(f"Superficie cubierta {int(float(surface_covered))} m².")
        elif surface_total is not None:
            parts.append(f"Superficie total {int(float(surface_total))} m².")

        price = _format_price(row.get("price_fixed"), row.get("currency_fixed"))
        if price:
            price_m2 = _safe(row.get("price_m2"))
            suffix = ""
            if price_m2:
                suffix = f" ({int(float(price_m2)):,} USD/m²)".replace(",", ".")
            parts.append(f"Precio: {price}{suffix}.")

        expenses = _safe(row.get("expenses"))
        if expenses is not None and float(expenses) > 0:
            parts.append(f"Expensas: UYU {int(float(expenses)):,}.".replace(",", "."))

        return "\n".join(parts)

    def _build_amenities_text(self, row: pd.Series) -> Optional[str]:
        present = []

        for flag in ALL_AMENITY_FLAGS:
            if int(_safe(row.get(flag), 0) or 0) == 1:
                label = AMENITY_LABELS.get(flag)
                if label:
                    present.append(label)

        if not present:
            return None

        return f"Amenities: {', '.join(present)}."

    def _build_geo_context(self, row: pd.Series) -> Optional[str]:
        phrases: list[str] = []

        dist_playa = _safe(row.get("dist_playa"))
        if dist_playa is not None and float(dist_playa) <= 800:
            phrases.append(f"a {int(float(dist_playa))} m de la playa")

        dist_plaza = _safe(row.get("dist_plaza"))
        n_plaza = _safe(row.get("n_plaza_800m"), 0)

        if dist_plaza is not None and float(dist_plaza) <= 800:
            phrases.append(f"plaza a {int(float(dist_plaza))} m")

        if n_plaza and int(float(n_plaza)) > 1:
            phrases.append(f"{int(float(n_plaza))} plazas en radio de 800 m")

        n_escuelas = _safe(row.get("n_escuelas_800m"), 0)
        dist_escuela = _safe(row.get("dist_nearest_escuela"))

        if n_escuelas and int(float(n_escuelas)) > 0:
            suffix = (
                f" (la más cercana a {int(float(dist_escuela))} m)"
                if dist_escuela
                else ""
            )
            phrases.append(
                f"{int(float(n_escuelas))} escuela"
                f"{'s' if int(float(n_escuelas)) > 1 else ''} "
                f"en radio de 800 m{suffix}"
            )

        n_comercial = _safe(row.get("n_comercial_800m"), 0)
        if n_comercial and int(float(n_comercial)) > 0:
            if int(float(n_comercial)) >= 10:
                phrases.append("zona comercial con múltiples servicios")
            else:
                phrases.append(
                    f"{int(float(n_comercial))} comercio"
                    f"{'s' if int(float(n_comercial)) > 1 else ''} "
                    f"en radio de 800 m"
                )

        n_industrial = _safe(row.get("n_industrial_800m"), 0)
        if n_industrial and int(float(n_industrial)) > 0:
            phrases.append(
                f"zona con {int(float(n_industrial))} establecimiento"
                f"{'s' if int(float(n_industrial)) > 1 else ''} industrial"
                f"{'es' if int(float(n_industrial)) > 1 else ''} "
                f"en radio de 800 m"
            )

        if not phrases:
            return None

        return "Entorno: " + ", ".join(phrases) + "."

    def _build_metadata(self, row: pd.Series) -> dict:
        metadata = {}

        for field in METADATA_FIELDS:
            if field in row.index:
                metadata[field] = _to_python_native(row.get(field))

        if metadata.get("latitude") is None and metadata.get("lat") is not None:
            metadata["latitude"] = metadata.get("lat")

        if metadata.get("longitude") is None and metadata.get("lon") is not None:
            metadata["longitude"] = metadata.get("lon")

        return metadata
