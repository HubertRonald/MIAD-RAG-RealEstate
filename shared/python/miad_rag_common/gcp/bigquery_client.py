from __future__ import annotations

import re
from typing import Optional, Sequence

import pandas as pd
from google.cloud import bigquery


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def build_table_fqn(project_id: str, dataset_id: str, table_id: str) -> str:
    """
    Construye un FQN de BigQuery con backticks.

    Ejemplo:
      `miad-paad-rs-dev.ds_miad_rag_rs.real_estate_listings`
    """
    if not project_id or not dataset_id or not table_id:
        raise ValueError("project_id, dataset_id y table_id son requeridos.")

    return f"`{project_id}.{dataset_id}.{table_id}`"


def _validate_identifier(value: str, field_name: str = "identifier") -> str:
    """
    Valida nombres de columnas controlados por código.

    No usar con valores provenientes del usuario final.
    """
    if not value or not _IDENTIFIER_RE.match(value):
        raise ValueError(f"{field_name} inválido: {value!r}")

    return value


def _format_columns(selected_columns: Optional[list[str]]) -> str:
    if not selected_columns:
        return "*"

    safe_columns = [
        _validate_identifier(column, "selected_column")
        for column in selected_columns
    ]

    return ", ".join(f"`{column}`" for column in safe_columns)


class BigQueryClient:
    """
    Cliente liviano para BigQuery.

    Sirve para:
      - job-indexer: leer tabla completa o subset para construir FAISS.
      - backend: enriquecer listings por IDs después de recuperar candidatos.

    Este cliente no mantiene estado de negocio.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        client: Optional[bigquery.Client] = None,
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.client = client or bigquery.Client(
            project=project_id,
            location=location,
        )

    def query_to_dataframe(
        self,
        query: str,
        parameters: Optional[
            Sequence[
                bigquery.ScalarQueryParameter
                | bigquery.ArrayQueryParameter
            ]
        ] = None,
    ) -> pd.DataFrame:
        job_config = bigquery.QueryJobConfig(
            query_parameters=list(parameters or []),
        )

        query_job = self.client.query(
            query,
            job_config=job_config,
        )

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
        table_fqn = build_table_fqn(
            project_id=project_id,
            dataset_id=dataset_id,
            table_id=table_id,
        )

        columns = _format_columns(selected_columns)

        query = f"SELECT {columns} FROM {table_fqn}"

        if where_clause:
            # where_clause debe ser controlado por configuración interna,
            # no por input directo del usuario final.
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

        Notas:
          - Usa parámetro ARRAY para evitar interpolar IDs en SQL.
          - Restaura el orden original de ids en Pandas.
          - id_column debe ser controlado por código/configuración.
        """
        if not ids:
            return pd.DataFrame()

        safe_id_column = _validate_identifier(id_column, "id_column")

        table_fqn = build_table_fqn(
            project_id=project_id,
            dataset_id=dataset_id,
            table_id=table_id,
        )

        columns = _format_columns(selected_columns)

        query = f"""
        SELECT {columns}
        FROM {table_fqn}
        WHERE CAST(`{safe_id_column}` AS STRING) IN UNNEST(@ids)
        """

        df = self.query_to_dataframe(
            query=query,
            parameters=[
                bigquery.ArrayQueryParameter(
                    "ids",
                    "STRING",
                    [str(value) for value in ids],
                )
            ],
        )

        if df.empty or safe_id_column not in df.columns:
            return df

        order = {
            str(value): index
            for index, value in enumerate(ids)
        }

        df["_input_order"] = df[safe_id_column].astype(str).map(order)
        df = df.sort_values("_input_order").drop(columns=["_input_order"])

        return df

    def dry_run(self, query: str) -> int:
        """
        Ejecuta dry run y retorna bytes estimados a procesar.
        """
        job_config = bigquery.QueryJobConfig(
            dry_run=True,
            use_query_cache=False,
        )

        query_job = self.client.query(
            query,
            job_config=job_config,
        )

        return int(query_job.total_bytes_processed or 0)
