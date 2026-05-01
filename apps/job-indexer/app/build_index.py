from __future__ import annotations

import sys
from datetime import datetime, timezone

from app.config.runtime import get_settings
from app.services.bigquery_reader import BigQueryReader
from app.services.embedding_service import EmbeddingService
from app.services.faiss_builder import FAISSBuilder, utc_now_iso
from app.services.gcs_service import GCSIndexPublisher
from app.services.listing_document_service import ListingDocumentService
from app.services.mlflow_service import MLflowService
from miad_rag_common.logging.structured_logging import configure_logging

settings = get_settings()
logger = configure_logging(level=settings.LOG_LEVEL, json_logs=settings.JSON_LOGS)


def build_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    version = build_version()
    started_at = utc_now_iso()

    logger.info(
        "indexer_job_started",
        extra={
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "env": settings.ENV,
            "collection": settings.COLLECTION,
            "source_table": settings.source_table_fqn,
            "index_bucket": settings.INDEX_BUCKET,
            "index_prefix": settings.INDEX_PREFIX,
            "dry_run": settings.DRY_RUN,
        },
    )

    try:
        # 1. Leer BigQuery
        reader = BigQueryReader()
        df = reader.read_listings()

        # 2. Convertir filas en Documents
        document_service = ListingDocumentService()
        df_preprocessed = document_service.preprocess(df)
        documents = document_service.dataframe_to_documents(df_preprocessed)

        logger.info(
            "documents_created",
            extra={
                "documents_count": len(documents),
                "rows_input": len(df),
                "rows_preprocessed": len(df_preprocessed),
            },
        )

        if settings.DRY_RUN:
            logger.info(
                "dry_run_completed_without_building_index",
                extra={
                    "documents_count": len(documents),
                    "collection": settings.COLLECTION,
                },
            )
            return

        # 3. Construir FAISS
        embedding_service = EmbeddingService(
            model_name=settings.GEMINI_EMBEDDING_MODEL,
        )
        faiss_builder = FAISSBuilder(embedding_service=embedding_service)

        vectorstore = faiss_builder.build_vectorstore(documents)
        local_index_dir = faiss_builder.save_vectorstore(
            vectorstore=vectorstore,
            collection=settings.COLLECTION,
            version=version,
        )

        finished_at = utc_now_iso()

        # 4. Archivos auxiliares: manifest + ids
        manifest = faiss_builder.write_auxiliary_files(
            output_dir=local_index_dir,
            documents=documents,
            version=version,
            started_at=started_at,
            finished_at=finished_at,
        )

        # 5. Publicar en GCS
        publisher = GCSIndexPublisher()
        uploaded = publisher.publish_index(
            local_index_dir=local_index_dir,
            version=version,
        )

        # 6. MLflow opcional
        mlflow_service = MLflowService()
        mlflow_service.log_index_build(
            manifest=manifest,
            local_index_dir=local_index_dir,
            metrics={
                "documents_count": len(documents),
                "input_rows": len(df),
                "preprocessed_rows": len(df_preprocessed),
                "uploaded_latest_files": len(uploaded.get("latest", [])),
                "uploaded_versioned_files": len(uploaded.get("versioned", [])),
            },
            params={
                "collection": settings.COLLECTION,
                "source_table": settings.source_table_fqn,
                "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
                "index_bucket": settings.INDEX_BUCKET,
                "index_prefix": settings.INDEX_PREFIX,
                "environment": settings.ENV,
            },
        )

        logger.info(
            "indexer_job_completed",
            extra={
                "collection": settings.COLLECTION,
                "version": version,
                "documents_count": len(documents),
                "local_index_dir": str(local_index_dir),
                "gcs_latest_prefix": settings.gcs_latest_prefix,
                "uploaded": uploaded,
            },
        )

    except Exception as exc:
        logger.exception(
            "indexer_job_failed",
            extra={
                "collection": settings.COLLECTION,
                "source_table": settings.source_table_fqn,
                "error": str(exc),
            },
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
