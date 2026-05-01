"""
Endpoint de indexación de listings inmobiliarios desde CSV
==========================================================

Procesa el CSV de listings limpio y construye el índice FAISS
para búsqueda semántica y recomendaciones.

Flujo: CSV en ./docs/{collection_name}/ → Chunking → Embeddings → FAISS
"""

from fastapi import APIRouter, HTTPException, status
from typing import Optional
import asyncio
import datetime
import json
import os
import time
import uuid

from app.services.load_documents_service import load_from_csv
from app.models.load_documents import LoadFromCsvRequest

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


# =============================================================================
# BACKGROUND TASK
# =============================================================================

async def _process_csv(
    payload: LoadFromCsvRequest,
    processing_id: str,
    timestamp: str,
):
    """
    Tarea en background: indexa el CSV y persiste el resultado en ./logs/.
    """
    print(f"[load_from_csv] Iniciando procesamiento en background: {processing_id}")

    try:
        result = await load_from_csv(
            collection_name=payload.collection_name,
            operation_type=payload.operation_type,
            property_type=payload.property_type,
            barrio=payload.barrio,
        )

        if result.get("data") is not None:
            result["data"]["processing_id"] = processing_id
            result["data"]["timestamp"]     = timestamp
        else:
            result["processing_id"] = processing_id
            result["timestamp"]     = timestamp

    except Exception as e:
        result = {
            "success":       False,
            "message":       "Error al indexar CSV",
            "data":          None,
            "processing_id": processing_id,
            "timestamp":     timestamp,
            "error":         str(e),
        }

    # Persistir resultado en ./logs/ para consultar con /validate-load
    os.makedirs("./logs", exist_ok=True)
    with open(f"./logs/{processing_id}.json", "w") as f:
        json.dump(result, f, indent=4)

    status_word = "completado" if result.get("success") else "fallido"
    print(f"[load_from_csv] Procesamiento {status_word}: {processing_id}")


# =============================================================================
# ENDPOINT
# =============================================================================

@router.post("/load-from-csv")
async def load_from_csv_endpoint(payload: LoadFromCsvRequest):
    """
    Indexa una colección de listings inmobiliarios desde un CSV local.

    Procesa el archivo en background y retorna inmediatamente un
    `processing_id` que puedes usar para consultar el estado en
    GET /api/v1/documents/validate-load/{processing_id}.

    **Prerequisito**: CSV limpio en ./docs/{collection_name}/

    **Flujo:**
    1. Validación del request
    2. Inicio de tarea asíncrona en background
    3. Retorno inmediato con processing_id

    **Filtros opcionales** — útiles para indexar solo un segmento del mercado
    y reducir el tamaño del índice FAISS:
    - `operation_type`: "venta" | "alquiler"
    - `property_type`: "apartamentos" | "casas"
    - `barrio`: nombre exacto del barrio (ej: "POCITOS")

    Args:
        payload: Configuración de la indexación

    Returns:
        dict con processing_id para tracking del proceso en background
    """
    # Verificar que la carpeta de la colección existe antes de encolar
    collection_path = f"./docs/{payload.collection_name}"
    if not os.path.exists(collection_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "success": False,
                "error": {
                    "code":    "COLLECTION_NOT_FOUND",
                    "message": f"No existe la carpeta '{collection_path}'",
                    "details": {
                        "field":            "collection_name",
                        "provided_value":   payload.collection_name,
                        "expected_path":    collection_path,
                        "reason":           f"Crea la carpeta y coloca el CSV dentro",
                    },
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            }
        )

    processing_id = f"csv_{uuid.uuid4().hex[:12]}"
    timestamp     = datetime.datetime.now().isoformat()

    asyncio.create_task(
        _process_csv(payload, processing_id, timestamp)
    )

    filters = {
        k: v for k, v in {
            "operation_type": payload.operation_type,
            "property_type":  payload.property_type,
            "barrio":         payload.barrio,
        }.items() if v is not None
    }

    return {
        "success":       True,
        "message":       "Indexación iniciada en background",
        "processing_id": processing_id,
        "timestamp":     timestamp,
        "collection":    payload.collection_name,
        "filters":       filters or None,
    }
