"""
Endpoint de Validación de Carga de Documentos
=============================================

Este módulo proporciona el endpoint para consultar el estado y resultados
del procesamiento asíncrono de documentos iniciado por load_from_url.

"""

from fastapi import APIRouter, HTTPException, status
from pathlib import Path
import os, json, aiofiles
from pydantic import ValidationError
from app.services.load_documents_service import validate_processing_with_rag


# Router para endpoints de documentos
router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

# Directorios configurados por variables de entorno
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "./docs")
LOG_DIR = os.getenv("LOG_DIR", "./logs")

# ==================== FUNCIÓN AUXILIAR ====================

async def read_processing_file(processing_id: str) -> dict:
    """
    Lee y parsea el archivo de log de procesamiento de manera asíncrona.
    
    Esta función maneja la lectura de archivos JSON generados por el
    procesamiento en background, con manejo robusto de errores.
    
    Args:
        processing_id: Identificador único del procesamiento
    
    Returns:
        dict: Contenido parseado del archivo JSON
    
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el JSON está corrupto
    """
    file_path = Path(LOG_DIR) / f"{processing_id}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"No existe {file_path}")

    # Lectura asíncrona del archivo
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        content = await f.read()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Archivo JSON corrupto: {e}")


# ==================== ENDPOINT PRINCIPAL ====================

@router.get("/validate-load/{processing_id}")
async def validate_load(processing_id: str):
    """
    Endpoint para consultar el estado y resultados de un procesamiento.
    
    Permite verificar el progreso y obtener los resultados completos
    de un procesamiento de documentos iniciado con load_from_csv.
    
    - Lee logs generados por load_from_csv
    - Valida con RAG service cuando está disponible
    - Retorna información detallada del procesamiento
    
    Flujo del endpoint:
    1. Validación del processing_id
    2. Lectura del archivo de log correspondiente
    3. Validación adicional con RAG service
    4. Retorno de datos completos
    
    Args:
        processing_id: Identificador único del procesamiento a consultar
    
    Returns:
        dict: Datos completos del procesamiento incluyendo:
            - Estado de éxito/fallo
            - Estadísticas de procesamiento  
            - Documentos procesados y fallidos
            - Información de colección y chunks
            - Metadatos y timestamps
    
    Raises:
        HTTPException:
            - 400: Si processing_id está vacío
            - 404: Si el processing_id no existe
            - 422: Si el archivo JSON está corrupto
            - 500: Si hay errores internos
    """
    
    # Validación de entrada
    if not processing_id or not processing_id.strip():
        raise HTTPException(
            status_code=400, 
            detail="El processing_id es requerido"
        )

    try:
        # Usar la función de validación del load_documents_service
        data = await validate_processing_with_rag(processing_id)
        return data

    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail="El processing_id no existe"
        )

    except ValueError as ve:
        # JSON corrupto o inválido
        raise HTTPException(
            status_code=422, 
            detail=str(ve)
        )

    except ValidationError as ve:
        # Datos no cumplen con el schema esperado
        raise HTTPException(
            status_code=500, 
            detail={"schema_errors": ve.errors()}
        )

    except Exception as e:
        # Errores inesperados
        raise HTTPException(
            status_code=500, 
            detail=f"Error al validar el processing_id: {str(e)}"
        )