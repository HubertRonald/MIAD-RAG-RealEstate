"""
Modelos para el sistema de carga de documentos

Define las estructuras de datos para la carga, procesamiento y validación
de documentos desde URLs externas, incluyendo configuraciones y estadísticas.
"""

from pydantic import BaseModel, Field, HttpUrl, validator, ConfigDict
from typing import Optional, List, Dict, Any, Literal


class ProcessingOptions(BaseModel):
    """
    Opciones de procesamiento para documentos descargados.
    
    Controla qué tipos de archivos procesar y cómo manejarlos.
    OPCIONAL: Si no se proporciona, usa valores por defecto.

    """
    file_extensions: List[str] = Field(
        default=["pdf", "txt", "docx", "md", "csv", "xlsx", "png", "jpg", "jpeg"],
        description="Extensiones de archivo permitidas"
    )
    max_file_size_mb: int = Field(
        default=100, 
        ge=1, 
        le=1000,
        description="Tamaño máximo de archivo en MB"
    )
    extract_metadata: bool = Field(
        default=True,
        description="Extraer metadatos de documentos"
    )
    preserve_formatting: bool = Field(
        default=False,
        description="Preservar formato original"
    )
    timeout_per_file_seconds: int = Field(
        default=300, 
        ge=30, 
        le=3600,
        description="Timeout por archivo en segundos"
    )


class LoadFromUrlRequest(BaseModel):
    """
    Modelo principal para solicitudes de carga de documentos desde URL.
    
    Encapsula toda la configuración necesaria para descargar y procesar
    documentos de manera asíncrona con pipeline completo: Descarga → Chunking → Embeddings.
    
    IMPORTANTE: La configuración de embeddings (modelo, batch_size) se debe definir
    internamente en app/services/embedding_service.py
    
    Atributos:
        source_url: URL donde se encuentran los documentos (Google Drive)
        collection_name: Nombre identificador de la colección
        chunking_strategy: Estrategia de chunking a utilizar (OBLIGATORIO)
        processing_options: Opciones de procesamiento (opcional, usa defaults)
        
    """
    source_url: HttpUrl = Field(
        ..., 
        description="URL de Google Drive (carpeta o archivo individual)"
    )
    collection_name: str = Field(
        ..., 
        min_length=1, 
        description="Nombre de la colección"
    )
    chunking_strategy: Literal[
        "recursive_character",
        "fixed_size", 
        "semantic",
        "document_structure"
    ] = Field(
        ...,
        description="Estrategia de chunking a utilizar"
    )
    processing_options: Optional[ProcessingOptions] = Field(
        default=None,
        description="Opciones de procesamiento (opcional, usa defaults si no se proporciona)"
    )
    multimodal: bool = Field(
        default=False,
        description=(
            "Activa el procesamiento multimodal de PDFs. "
            "Cuando es True, combina texto extraído con descripciones de imágenes "
            "(Gemini VLM) y tablas estructuradas (pdfplumber). "
            "Incrementa el costo y tiempo de indexación. "
            "Solo tiene efecto sobre archivos PDF; los Markdown no se ven afectados."
        )
    )

    @validator('collection_name')
    def validate_collection_name(cls, v):
        import re
        v = v.strip()
        if not v:
            raise ValueError('collection_name no puede estar vacío')
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('collection_name solo puede contener letras, números, guiones y guiones bajos')
        return v


class LoadFromCsvRequest(BaseModel):
    """
    Configuración para indexar una colección de listings desde CSV.
 
    Atributos:
        collection_name : Nombre de la carpeta en ./docs/ que contiene el CSV.
        operation_type  : Filtro opcional — indexar solo "venta" o "alquiler".
        property_type   : Filtro opcional — indexar solo "apartamentos" o "casas".
        barrio          : Filtro opcional — indexar solo un barrio específico.
    """
    collection_name: str = Field(
        ...,
        min_length=1,
        description="Nombre de la colección (carpeta en ./docs/)"
    )
    operation_type: Optional[str] = Field(
        default=None,
        description="Filtro opcional: 'venta' | 'alquiler'"
    )
    property_type: Optional[str] = Field(
        default=None,
        description="Filtro opcional: 'apartamentos' | 'casas'"
    )
    barrio: Optional[str] = Field(
        default=None,
        description="Filtro opcional: nombre exacto del barrio"
    )
 
    @validator("collection_name")
    def validate_collection_name(cls, v):
        import re
        v = v.strip()
        if not v:
            raise ValueError("collection_name no puede estar vacío")
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "collection_name solo puede contener letras, números, "
                "guiones y guiones bajos"
            )
        return v
 
    @validator("operation_type")
    def validate_operation_type(cls, v):
        if v is not None and v not in ("venta", "alquiler"):
            raise ValueError("operation_type debe ser 'venta' o 'alquiler'")
        return v
 
    @validator("property_type")
    def validate_property_type(cls, v):
        if v is not None and v not in ("apartamentos", "casas"):
            raise ValueError("property_type debe ser 'apartamentos' o 'casas'")
        return v
    
    
class ProcessingSummary(BaseModel):
    """Resumen estadístico del procesamiento de documentos."""
    documents_found: int
    documents_loaded: int
    documents_failed: int
    total_chunks_created: int
    total_processing_time_seconds: float


class CollectionInfo(BaseModel):
    """Información sobre la colección de documentos."""
    name: str
    documents_count_before: int
    documents_count_after: int
    total_chunks_before: int
    total_chunks_after: int
    storage_size_mb: float


class DocumentMetadata(BaseModel):
    """Metadatos extraídos de documentos procesados."""
    pages: Optional[int] = None
    author: Optional[str] = None
    creation_date: Optional[str] = None
    file_type: str
    lines: Optional[int] = None
    encoding: Optional[str] = None


class ProcessedDocument(BaseModel):
    """Información detallada de un documento procesado exitosamente."""
    filename: str
    file_size_bytes: int
    download_url: str
    processing_status: str
    chunks_created: int
    processing_time_seconds: float
    metadata: DocumentMetadata


class FailedDocument(BaseModel):
    """Información sobre documentos que fallaron en el procesamiento."""
    filename: str
    download_url: str
    error_code: str
    error_message: str
    processing_time_seconds: float


class ChunkingStatistics(BaseModel):
    """Estadísticas sobre la fragmentación de documentos."""
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    chunks_with_overlap: int
    total_characters_processed: int


class EmbeddingStatistics(BaseModel):
    """Estadísticas sobre la generación de embeddings."""
    model_config = {"protected_namespaces": ()}

    model_used: str
    total_embeddings_generated: int
    batch_processing_time_seconds: float
    failed_embeddings: int
    # --- campos multimodales (presentes cuando multimodal=True) ---
    multimodal_processing: bool = False
    multimodal_chunks: int = 0
    chunks_with_images: int = 0
    chunks_with_tables: int = 0
    total_vlm_calls: int = 0


class Warning(BaseModel):
    """Advertencias generadas durante el procesamiento."""
    code: str
    message: str
    affected_documents: List[str]
    recommendation: str


class ResponseData(BaseModel):
    """Datos completos de respuesta del procesamiento."""
    processing_summary: ProcessingSummary
    collection_info: CollectionInfo
    documents_processed: List[ProcessedDocument]
    failed_documents: List[FailedDocument]
    chunking_statistics: ChunkingStatistics
    embedding_statistics: EmbeddingStatistics


class LoadFromUrlResponse(BaseModel):
    """
    Respuesta completa del endpoint de carga de documentos.
    
    Incluye toda la información sobre el procesamiento realizado,
    estadísticas y posibles advertencias.
    """
    success: bool
    message: str
    data: ResponseData
    warnings: List[Warning]
    processing_id: str
    timestamp: str
