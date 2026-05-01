from __future__ import annotations

from typing import Any

from langchain.schema import Document

from app.config.runtime import get_settings
from miad_rag_common.gcp.bigquery_client import BigQueryClient
from miad_rag_common.schemas.listing import ListingInfo, MapPoint
from miad_rag_common.utils.text_utils import safe_float, safe_int

settings = get_settings()


class BigQueryListingService:
    """
    Enriquece resultados recuperados por FAISS con la fuente canónica en BigQuery.
    """

    def __init__(self) -> None:
        self.bq = BigQueryClient(
            project_id=settings.BQ_PROJECT_ID,
            location=settings.BQ_LOCATION,
        )

    def extract_listing_ids(self, docs: list[Document]) -> list[str]:
        ids: list[str] = []

        for doc in docs:
            metadata = doc.metadata or {}
            value = (
                metadata.get("id")
                or metadata.get("listing_id")
                or metadata.get("property_id")
            )

            if value is not None:
                ids.append(str(value))

        seen = set()
        ordered_unique = []

        for value in ids:
            if value not in seen:
                seen.add(value)
                ordered_unique.append(value)

        return ordered_unique

    def fetch_by_ids(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}

        df = self.bq.fetch_rows_by_ids(
            project_id=settings.BQ_PROJECT_ID,
            dataset_id=settings.BQ_DATASET_ID,
            table_id=settings.BQ_LISTINGS_TABLE,
            ids=ids,
            id_column=settings.BQ_LISTING_ID_COLUMN,
        )

        if df.empty:
            return {}

        records = df.to_dict(orient="records")

        return {
            str(record.get(settings.BQ_LISTING_ID_COLUMN)): record
            for record in records
            if record.get(settings.BQ_LISTING_ID_COLUMN) is not None
        }

    def document_to_listing(
        self,
        doc: Document,
        semantic_score: float | None = None,
        override: dict[str, Any] | None = None,
    ) -> ListingInfo:
        """
        Une metadata FAISS + fila BigQuery.

        BigQuery tiene precedencia porque es la fuente canónica para render.
        """
        metadata = dict(doc.metadata or {})
        data = {**metadata, **(override or {})}

        listing_id = (
            data.get(settings.BQ_LISTING_ID_COLUMN)
            or data.get("id")
            or data.get("listing_id")
        )

        latitude = data.get("latitude") or data.get("lat")
        longitude = data.get("longitude") or data.get("lon") or data.get("lng")

        return ListingInfo(
            id=str(listing_id) if listing_id is not None else None,
            title=data.get("title"),
            description=data.get("description") or doc.page_content[:500],
            source=data.get("source") or data.get("source_file"),
            url=data.get("url"),
            barrio=data.get("barrio"),
            operation_type=data.get("operation_type"),
            property_type=data.get("property_type"),
            price_fixed=safe_float(data.get("price_fixed")),
            currency_fixed=data.get("currency_fixed"),
            price_m2=safe_float(data.get("price_m2")),
            bedrooms=safe_float(data.get("bedrooms")),
            bathrooms=safe_float(data.get("bathrooms")),
            surface_covered=safe_float(data.get("surface_covered")),
            surface_total=safe_float(data.get("surface_total")),
            floor=safe_float(data.get("floor")),
            age=safe_float(data.get("age")),
            garages=safe_float(data.get("garages")),
            latitude=safe_float(latitude),
            longitude=safe_float(longitude),
            dist_plaza=safe_float(data.get("dist_plaza")),
            dist_playa=safe_float(data.get("dist_playa")),
            n_escuelas_800m=safe_int(data.get("n_escuelas_800m")),
            has_pool=self._safe_bool_or_none(data.get("has_pool")),
            has_gym=self._safe_bool_or_none(data.get("has_gym")),
            has_elevator=self._safe_bool_or_none(data.get("has_elevator")),
            has_parrillero=self._safe_bool_or_none(data.get("has_parrillero")),
            has_terrace=self._safe_bool_or_none(data.get("has_terrace")),
            has_rooftop=self._safe_bool_or_none(data.get("has_rooftop")),
            has_security=self._safe_bool_or_none(data.get("has_security")),
            has_storage=self._safe_bool_or_none(data.get("has_storage")),
            has_parking=self._safe_bool_or_none(data.get("has_parking")),
            has_party_room=self._safe_bool_or_none(data.get("has_party_room")),
            has_green_area=self._safe_bool_or_none(data.get("has_green_area")),
            has_playground=self._safe_bool_or_none(data.get("has_playground")),
            has_visitor_parking=self._safe_bool_or_none(data.get("has_visitor_parking")),
            semantic_score=semantic_score,
        )

    def build_map_points(self, listings: list[ListingInfo]) -> list[MapPoint]:
        points: list[MapPoint] = []

        for listing in listings:
            if not listing.id or listing.latitude is None or listing.longitude is None:
                continue

            points.append(
                MapPoint(
                    id=listing.id,
                    lat=listing.latitude,
                    lon=listing.longitude,
                    label=listing.title or listing.barrio or listing.id,
                    barrio=listing.barrio,
                    price_fixed=listing.price_fixed,
                    currency_fixed=listing.currency_fixed,
                    match_score=listing.match_score,
                    rank=listing.rank,
                )
            )

        return points

    @staticmethod
    def _safe_bool_or_none(value: Any) -> bool | None:
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, int):
            return value == 1

        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "si", "sí", "y"}

        return None
