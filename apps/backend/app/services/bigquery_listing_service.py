from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from langchain.schema import Document

from app.config.runtime import get_settings
from miad_rag_common.gcp.bigquery_client import BigQueryClient
from miad_rag_common.logging.structured_logging import get_logger
from miad_rag_common.schemas.listing import ListingInfo
from miad_rag_common.utils.text_utils import safe_float, safe_int

settings = get_settings()
logger = get_logger(__name__)


class BigQueryListingService:
    """
    Servicio de enriquecimiento desde BigQuery.

    Responsabilidad:
      - Recibir Documents recuperados por FAISS.
      - Extraer IDs de propiedades desde metadata.
      - Consultar BigQuery con esos IDs.
      - Devolver:
          1. registros completos para frontend;
          2. objetos ListingInfo compactos compatibles con el contrato original;
          3. puntos de mapa basados en lat/lon.

    Importante:
      - El job-indexer puede dropear columnas para construir FAISS.
      - Este servicio NO dropea columnas: BigQuery es la fuente canónica.
      - BigQuery usa lat/lon, no latitude/longitude.
    """

    def __init__(self) -> None:
        self.bq = BigQueryClient(
            project_id=settings.BQ_PROJECT_ID,
            location=settings.BQ_LOCATION,
        )

    # -------------------------------------------------------------------------
    # Extracción de IDs desde FAISS Documents
    # -------------------------------------------------------------------------

    def extract_listing_ids(self, docs: list[Document]) -> list[str]:
        """
        Extrae IDs únicos desde metadata de Documents FAISS.

        Conserva el orden de recuperación.
        """
        ids: list[str] = []

        for doc in docs:
            metadata = doc.metadata or {}

            value = (
                metadata.get(settings.BQ_LISTING_ID_COLUMN)
                or metadata.get("id")
                or metadata.get("listing_id")
                or metadata.get("property_id")
            )

            if value is not None:
                ids.append(str(value))

        seen: set[str] = set()
        ordered_unique: list[str] = []

        for value in ids:
            if value not in seen:
                seen.add(value)
                ordered_unique.append(value)

        return ordered_unique

    # -------------------------------------------------------------------------
    # Consulta BigQuery
    # -------------------------------------------------------------------------

    def fetch_by_ids(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """
        Consulta BigQuery por IDs y retorna registros completos.

        Retorna:
            {
              "<id>": {
                 ... todas las columnas de real_estate_listings ...
              }
            }

        BigQueryClient.fetch_rows_by_ids() usa SELECT * si no se pasan
        selected_columns, por lo tanto aquí se trae la fila completa.
        """
        if not ids:
            return {}

        logger.info(
            "bigquery_fetch_listings_started",
            extra={
                "ids_count": len(ids),
                "table": (
                    f"{settings.BQ_PROJECT_ID}."
                    f"{settings.BQ_DATASET_ID}."
                    f"{settings.BQ_LISTINGS_TABLE}"
                ),
            },
        )

        df = self.bq.fetch_rows_by_ids(
            project_id=settings.BQ_PROJECT_ID,
            dataset_id=settings.BQ_DATASET_ID,
            table_id=settings.BQ_LISTINGS_TABLE,
            ids=ids,
            id_column=settings.BQ_LISTING_ID_COLUMN,
            selected_columns=None,  # SELECT * <- more cost
        )

        if df.empty:
            logger.warning(
                "bigquery_fetch_listings_empty",
                extra={"ids_count": len(ids)},
            )
            return {}

        records = [
            self._sanitize_record(record)
            for record in df.to_dict(orient="records")
        ]

        result = {
            str(record.get(settings.BQ_LISTING_ID_COLUMN)): record
            for record in records
            if record.get(settings.BQ_LISTING_ID_COLUMN) is not None
        }

        logger.info(
            "bigquery_fetch_listings_completed",
            extra={
                "requested_ids": len(ids),
                "records_found": len(result),
            },
        )

        return result

    def fetch_records_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """
        Variante conveniente para frontend: retorna lista en el mismo orden
        de los IDs recuperados por FAISS.
        """
        by_id = self.fetch_by_ids(ids)

        return [
            by_id[str(listing_id)]
            for listing_id in ids
            if str(listing_id) in by_id
        ]

    # -------------------------------------------------------------------------
    # Conversión compacta a ListingInfo
    # -------------------------------------------------------------------------
    def document_to_listing(
        self,
        doc: Document,
        semantic_score: float | None = None,
        override: dict[str, Any] | None = None,
        rank: int | None = None,
        match_score: int | None = None,
    ) -> ListingInfo:
        """
        Convierte un Document FAISS + fila BigQuery en ListingInfo compacto.

        Este método mantiene compatibilidad con el contrato original:
        - no mete todo el SELECT *;
        - no usa latitude/longitude;
        - usa barrio_fixed como barrio principal;
        - usa barrio como fallback;
        - conserva scores internos y user-facing.

        Para frontend completo usar document_to_frontend_record().
        """
        data = self._merge_document_and_bigquery_row(
            doc=doc,
            override=override,
        )

        listing_id = self._extract_id_from_data(data)

        return ListingInfo(
            id=str(listing_id) if listing_id is not None else None,

            # Barrio principal para contrato compacto.
            # barrio_fixed viene del proceso de limpieza.
            # barrio viene del join espacial oficial.
            barrio=data.get("barrio_fixed") or data.get("barrio"),
            barrio_confidence=data.get("barrio_confidence"),

            operation_type=data.get("operation_type"),
            is_dual_intent=self._safe_bool_or_none(data.get("is_dual_intent")),
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

            dist_plaza=safe_float(data.get("dist_plaza")),
            dist_playa=safe_float(data.get("dist_playa")),
            n_escuelas_800m=safe_int(data.get("n_escuelas_800m")),

            source=data.get("source") or data.get("source_file"),

            semantic_score=semantic_score,
            rerank_score=safe_float(data.get("rerank_score")),
            match_score=match_score,
            rank=rank,
        )

    def documents_to_listings(
        self,
        docs: list[Document],
        semantic_scores: list[float] | None = None,
        listing_overrides: dict[str, dict[str, Any]] | None = None,
        match_score_fn: Optional[Callable[[float], int]] = None,
    ) -> list[ListingInfo]:
        """
        Convierte varios Documents en ListingInfo compactos.

        Útil para alimentar RecommendResponse.listings_used.
        """
        listings: list[ListingInfo] = []

        for index, doc in enumerate(docs):
            metadata = doc.metadata or {}
            listing_id = (
                metadata.get(settings.BQ_LISTING_ID_COLUMN)
                or metadata.get("id")
                or metadata.get("listing_id")
                or metadata.get("property_id")
            )

            override = {}
            if listing_id is not None and listing_overrides:
                override = listing_overrides.get(str(listing_id), {})

            semantic_score = (
                semantic_scores[index]
                if semantic_scores and index < len(semantic_scores)
                else None
            )

            match_score = (
                match_score_fn(semantic_score)
                if semantic_score is not None and match_score_fn is not None
                else None
            )

            listings.append(
                self.document_to_listing(
                    doc=doc,
                    semantic_score=semantic_score,
                    override=override,
                    rank=index + 1,
                    match_score=match_score,
                )
            )

        return listings

    # -------------------------------------------------------------------------
    # Payload completo para frontend
    # -------------------------------------------------------------------------

    def document_to_frontend_record(
        self,
        doc: Document,
        semantic_score: float | None = None,
        override: dict[str, Any] | None = None,
        rank: int | None = None,
        match_score: int | None = None,
        snippet_chars: int = 700,
    ) -> dict[str, Any]:
        """
        Construye un registro completo para frontend.

        Contiene:
        - todas las columnas traídas de BigQuery con SELECT *;
        - metadata útil de FAISS;
        - semantic_score;
        - rerank_score;
        - match_score;
        - rank;
        - retrieval_snippet;
        - map_point usando lat/lon.

        Este método NO depende de ListingInfo.
        """
        data = self._merge_document_and_bigquery_row(
            doc=doc,
            override=override,
        )

        listing_id = self._extract_id_from_data(data)
        lat = safe_float(data.get("lat"))
        lon = safe_float(data.get("lon"))

        title_display = (
            data.get("title_clean")
            or data.get("title")
        )

        description_display = (
            data.get("description_clean")
            or data.get("description")
        )

        barrio_display = (
            data.get("barrio_fixed")
            or data.get("barrio")
        )

        data["id"] = str(listing_id) if listing_id is not None else None
        data["title_display"] = title_display
        data["description_display"] = description_display
        data["barrio_display"] = barrio_display

        data["source"] = data.get("source") or data.get("source_file")
        data["semantic_score"] = semantic_score
        data["rerank_score"] = safe_float(data.get("rerank_score"))
        data["match_score"] = match_score
        data["rank"] = rank

        data["retrieval_snippet"] = (
            doc.page_content[:snippet_chars]
            if doc.page_content
            else None
        )

        if lat is not None and lon is not None and listing_id is not None:
            data["map_point"] = {
                "id": str(listing_id),
                "lat": lat,
                "lon": lon,
                "label": title_display or barrio_display or str(listing_id),
                "barrio": barrio_display,
                "price_fixed": safe_float(data.get("price_fixed")),
                "currency_fixed": data.get("currency_fixed"),
                "match_score": match_score,
                "rank": rank,
            }
        else:
            data["map_point"] = None

        return self._sanitize_record(data)


    def documents_to_frontend_records(
        self,
        docs: list[Document],
        semantic_scores: list[float] | None = None,
        listing_overrides: dict[str, dict[str, Any]] | None = None,
        match_score_fn: Optional[Callable[[float], int]] = None,
    ) -> list[dict[str, Any]]:
        """
        Convierte Documents FAISS + filas BigQuery en payload completo para frontend.

        Esto es lo que debería usar el endpoint nuevo/enriquecido cuando el
        frontend necesite cards, tabla y mapa.
        """
        records: list[dict[str, Any]] = []

        for index, doc in enumerate(docs):
            metadata = doc.metadata or {}

            listing_id = (
                metadata.get(settings.BQ_LISTING_ID_COLUMN)
                or metadata.get("id")
                or metadata.get("listing_id")
                or metadata.get("property_id")
            )

            override = {}
            if listing_id is not None and listing_overrides:
                override = listing_overrides.get(str(listing_id), {})

            semantic_score = (
                semantic_scores[index]
                if semantic_scores and index < len(semantic_scores)
                else None
            )

            match_score = (
                match_score_fn(semantic_score)
                if semantic_score is not None and match_score_fn is not None
                else None
            )

            records.append(
                self.document_to_frontend_record(
                    doc=doc,
                    semantic_score=semantic_score,
                    override=override,
                    rank=index + 1,
                    match_score=match_score,
                )
            )

        return records


    def build_map_points_from_records(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Construye puntos de mapa desde registros completos del frontend.

        Usa lat/lon, no latitude/longitude.
        """
        points: list[dict[str, Any]] = []

        for record in records:
            map_point = record.get("map_point")

            if map_point:
                points.append(map_point)
                continue

            listing_id = record.get("id")
            lat = safe_float(record.get("lat"))
            lon = safe_float(record.get("lon"))

            if not listing_id or lat is None or lon is None:
                continue

            points.append(
                {
                    "id": str(listing_id),
                    "lat": lat,
                    "lon": lon,
                    "label": (
                        record.get("title_clean")
                        or record.get("title")
                        or record.get("barrio_fixed")
                        or str(listing_id)
                    ),
                    "barrio": record.get("barrio_fixed") or record.get("barrio"),
                    "price_fixed": safe_float(record.get("price_fixed")),
                    "currency_fixed": record.get("currency_fixed"),
                    "match_score": record.get("match_score"),
                    "rank": record.get("rank"),
                }
            )

        return points

    # -------------------------------------------------------------------------
    # Helpers internos
    # -------------------------------------------------------------------------

    def _merge_document_and_bigquery_row(
        self,
        doc: Document,
        override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Une metadata FAISS + fila BigQuery.

        Regla:
          - metadata FAISS aporta flags derivados y source;
          - BigQuery override aporta la verdad canónica del inmueble;
          - si BigQuery no tiene has_* flags, se conservan los de FAISS.
        """
        metadata = self._sanitize_record(dict(doc.metadata or {}))
        row = self._sanitize_record(override or {})

        merged = {**metadata, **row}

        # Evitar perder source si BigQuery no lo tiene.
        if not merged.get("source"):
            merged["source"] = metadata.get("source")

        if not merged.get("source_file"):
            merged["source_file"] = metadata.get("source_file")

        return merged

    def _extract_id_from_data(self, data: dict[str, Any]) -> Any:
        return (
            data.get(settings.BQ_LISTING_ID_COLUMN)
            or data.get("id")
            or data.get("listing_id")
            or data.get("property_id")
        )

    def _sanitize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Convierte un record a tipos JSON-friendly.

        Esto evita problemas con:
          - numpy int/float/bool;
          - pandas Timestamp;
          - Decimal;
          - NaN / NaT / pd.NA.
        """
        return {
            str(key): self._to_jsonable(value)
            for key, value in record.items()
        }

    def _to_jsonable(self, value: Any) -> Any:
        if self._is_nullish(value):
            return None

        if isinstance(value, dict):
            return {
                str(key): self._to_jsonable(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [self._to_jsonable(item) for item in value]

        if isinstance(value, tuple):
            return [self._to_jsonable(item) for item in value]

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.bool_):
            return bool(value)

        if isinstance(value, Decimal):
            return float(value)

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, date):
            return value.isoformat()

        return value

    @staticmethod
    def _is_nullish(value: Any) -> bool:
        if value is None:
            return True

        try:
            result = pd.isna(value)
            if isinstance(result, bool):
                return result
        except (TypeError, ValueError):
            pass

        return False

    @staticmethod
    def _safe_bool_or_none(value: Any) -> bool | None:
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, np.bool_):
            return bool(value)

        if isinstance(value, int):
            return value == 1

        if isinstance(value, float):
            return value == 1.0

        if isinstance(value, str):
            return value.strip().lower() in {
                "1",
                "true",
                "yes",
                "si",
                "sí",
                "y",
                "s",
            }

        return None
