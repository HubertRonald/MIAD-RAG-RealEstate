from __future__ import annotations

import pandas as pd

from app.config.runtime import get_settings
from miad_rag_common.gcp.bigquery_client import BigQueryClient, build_table_fqn
from miad_rag_common.logging.structured_logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class BigQueryReader:
    """
    Lee listings inmobiliarios desde BigQuery.

    Fuente esperada:
      miad-paad-rs-dev.ds_miad_rag_rs.real_estate_listings
    """

    def __init__(self) -> None:
        self.client = BigQueryClient(
            project_id=settings.BQ_PROJECT_ID,
            location=settings.BQ_LOCATION,
        )

    def read_listings(self) -> pd.DataFrame:
        table_fqn = build_table_fqn(
            project_id=settings.BQ_PROJECT_ID,
            dataset_id=settings.BQ_DATASET_ID,
            table_id=settings.BQ_LISTINGS_TABLE,
        )

        query = f"""
        SELECT *
        FROM {table_fqn}
        """

        if settings.BQ_WHERE_CLAUSE:
            query += f"\nWHERE {settings.BQ_WHERE_CLAUSE}"

        if settings.BQ_LIMIT:
            query += f"\nLIMIT {int(settings.BQ_LIMIT)}"

        logger.info(
            "reading_bigquery_listings",
            extra={
                "source_table": settings.source_table_fqn,
                "limit": settings.BQ_LIMIT,
                "where_clause": settings.BQ_WHERE_CLAUSE,
            },
        )

        df = self.client.query_to_dataframe(query)

        logger.info(
            "bigquery_listings_loaded",
            extra={
                "rows": len(df),
                "columns": len(df.columns),
                "source_table": settings.source_table_fqn,
            },
        )

        if df.empty:
            raise ValueError(
                f"BigQuery no retornó filas desde {settings.source_table_fqn}"
            )

        return df
