from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd
from google.cloud import bigquery


def build_table_fqn(project_id: str, dataset_id: str, table_id: str) -> str:
    """Construye un FQN de BigQuery con backticks."""
    if not project_id or not dataset_id or not table_id:
        raise ValueError("project_id, dataset_id y table_id son requeridos.")
    return f"`{project_id}.{dataset_id}.{table_id}`"


class BigQueryClient:
    """
    Cliente liviano para BigQuery.

    Sirve tanto para:
    - job-indexer: leer tabla completa o subset para construir FAISS.
    - backend: enriquecer listings por IDs después de recuperar candidatos.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        client: Optional[bigquery.Client] = None,
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.client = client or bigquery.Client(project=project_id, location=location)

    def query_to_dataframe(
        self,
        query: str,
        parameters: Optional[Sequence[bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter]] = None,
    ) -> pd.DataFrame:
        job_config = bigquery.QueryJobConfig(query_parameters=list(parameters or []))
        query_job = self.client.query(query, job_config=job_config)
        return query_job.result().to_dataframe()

    def read_table(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
        selected_columns: Optional[list[str]] = None,
        where_clause: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        table_fqn = build_table_fqn(project_id, dataset_id, table_id)

        columns = ", ".join(selected_columns) if selected_columns else "*"
        query = f"SELECT {columns} FROM {table_fqn}"

        if where_clause:
            query += f"\nWHERE {where_clause}"

        if limit:
            query += f"\nLIMIT {int(limit)}"

        return self.query_to_dataframe(query)

    def fetch_rows_by_ids(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
        ids: list[str],
        id_column: str = "id",
        selected_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Recupera filas de una tabla por lista de IDs.

        Nota:
        - El orden original de ids se restaura en Pandas.
        - id_column debe ser un nombre controlado por código, no por usuario final.
        """
        if not ids:
            return pd.DataFrame()

        table_fqn = build_table_fqn(project_id, dataset_id, table_id)
        columns = ", ".join(selected_columns) if selected_columns else "*"

        query = f"""
        SELECT {columns}
        FROM {table_fqn}
        WHERE CAST({id_column} AS STRING) IN UNNEST(@ids)
        """

        df = self.query_to_dataframe(
            query,
            parameters=[
                bigquery.ArrayQueryParameter("ids", "STRING", [str(x) for x in ids])
            ],
        )

        if df.empty or id_column not in df.columns:
            return df

        order = {str(value): idx for idx, value in enumerate(ids)}
        df["_input_order"] = df[id_column].astype(str).map(order)
        df = df.sort_values("_input_order").drop(columns=["_input_order"])

        return df

    def dry_run(self, query: str) -> int:
        """
        Ejecuta dry run y retorna bytes estimados a procesar.
        """
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query_job = self.client.query(query, job_config=job_config)
        return int(query_job.total_bytes_processed or 0)