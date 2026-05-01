"""
Servicio de descarga y procesamiento de documentos

Este módulo orquesta la descarga de documentos desde diferentes proveedores
de almacenamiento (Google Drive, etc.) y los organiza en colecciones locales.
Además, procesa los documentos para crear la base de datos vectorial RAG.
"""

import os
import time
import logging
from typing import Tuple, List, Dict, Any
from pathlib import Path
from app.services.google_drive import GoogleDriveProvider

logger = logging.getLogger(__name__)

# Directorio base para descargas configurado por variable de entorno
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "./docs")

# Ruta a credenciales de Google Drive (puede venir de Secret Manager en Cloud Run)
GOOGLE_DRIVE_CREDENTIALS = os.getenv("GOOGLE_DRIVE_CREDENTIALS", "apikey.json")


async def download_documents(
        url: str, 
        collection_name: str, 
        timeout_per_file: int = 300,
        credentials_path: str = None,
        payload: Any = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Descarga documentos desde una URL a una colección local.
        
        **Tipos de URL soportadas:**
        - Carpeta de Google Drive: descarga todos los documentos de la carpeta
        - Archivo individual de Google Drive: descarga ese único archivo
        
        **Flujo de descarga:**
        1. Creación del directorio de colección
        2. Inicialización del proveedor de almacenamiento
        3. Listado de documentos disponibles (carpeta completa o archivo único)
        4. Descarga de todos los documentos encontrados
        5. Manejo individual de errores por documento
        6. Retorno de listas de éxitos y fallos
        
        Args:
            url: URL fuente (formatos soportados):
                - Carpeta: https://drive.google.com/drive/folders/FOLDER_ID
                - Archivo: https://drive.google.com/file/d/FILE_ID/view
            collection_name: Nombre de la colección destino
            timeout_per_file: Timeout por archivo en segundos (default: 300)
            credentials_path: Ruta al archivo de credenciales (default: variable GOOGLE_DRIVE_CREDENTIALS)
            payload: Configuración de procesamiento para validación
        
        Returns:
            Tuple[List, List]: (documentos_exitosos, documentos_fallidos)
                - documentos_exitosos: Lista con metadatos de descargas exitosas
                - documentos_fallidos: Lista con información de errores
        """
        # === CONFIGURACIÓN DE CREDENCIALES ===
        if credentials_path is None:
            credentials_path = GOOGLE_DRIVE_CREDENTIALS
        
        # === CONFIGURACIÓN DE DIRECTORIO ===
        collection_dir = os.path.join(DOWNLOAD_DIR, collection_name)
        os.makedirs(collection_dir, exist_ok=True)
        
        # === INICIALIZACIÓN DEL PROVEEDOR ===
        # Para este esqueleto solo se implementa Google Drive
        provider = GoogleDriveProvider(url, credentials_path)
        documents = provider.list_documents()
        
        # === INFORMACIÓN DE DOCUMENTOS ENCONTRADOS ===
        print(f"Documentos encontrados: {len(documents)}")
        
        # === INICIALIZACIÓN DE LISTAS DE RESULTADOS ===
        processed_docs = []
        failed_docs = []
        
        # === PROCESAMIENTO INDIVIDUAL DE DOCUMENTOS ===
        for doc in documents:
            doc_start_time = time.time()
            output_path = os.path.join(collection_dir, doc["name"])
            
            # Determinar URL de descarga apropiada
            # Si la URL original ya apunta a un archivo específico, usar esa
            # Si es una carpeta, agregar el nombre del archivo
            if "/file/d/" in url:
                doc_download_url = url  # URL de archivo individual
            else:
                doc_download_url = f"{url}/{doc['name']}"  # URL de carpeta + archivo
            
            try:
                print(f"Descargando: {doc['name']}")
                
                # Descarga usando el proveedor configurado
                download_result = provider.download_document(
                    doc["id"], 
                    output_path, 
                    timeout_seconds=timeout_per_file,
                    payload=payload
                )
                
                # === MANEJO DE RESULTADO DE DESCARGA ===
                if download_result == "":  # Descarga exitosa
                    file_stats = os.stat(output_path)
                    processed_docs.append({
                        "filename": doc["name"],
                        "file_path": output_path,
                        "file_size_bytes": file_stats.st_size,
                        "download_url": doc_download_url,
                        "processing_time_seconds": round(time.time() - doc_start_time, 2),
                        "doc_metadata": doc
                    })
                    print(f"Descargado: {doc['name']} ({file_stats.st_size} bytes)")
                else:
                    # Descarga fallida con información de error
                    failed_docs.append({
                        "filename": doc["name"],
                        "download_url": doc_download_url,
                        "error_message": str(download_result),
                        "processing_time_seconds": round(time.time() - doc_start_time, 2)
                    })
                    print(f"Error: {doc['name']} - {download_result}")
                    
            except Exception as e:
                # === MANEJO DE EXCEPCIONES ===
                failed_docs.append({
                    "filename": doc["name"],
                    "download_url": doc_download_url,
                    "error_message": str(e),
                    "processing_time_seconds": round(time.time() - doc_start_time, 2)
                })
                print(f"Excepción: {doc['name']} - {e}")
        
        # === RETORNO DE RESULTADOS ===
        return processed_docs, failed_docs


async def download_and_process_documents(
        url: str, 
        collection_name: str, 
        timeout_per_file: int = 300,
        credentials_path: str = None,
        payload: Any = None
    ) -> Dict[str, Any]:
        """
        Descarga documentos y procesa completamente para RAG.
        
        Esta función orquesta todo el proceso:
        1. Descarga documentos
        2. Procesamiento de chunks con ChunkingService
        3. Generación de embeddings con EmbeddingService
        4. Creación de base de datos vectorial con RetrievalService
        
        Args:
            url: URL fuente de los documentos
            collection_name: Nombre de la colección destino
            timeout_per_file: Timeout por archivo en segundos
            credentials_path: Ruta al archivo de credenciales
            payload: Configuración de procesamiento (chunk_size, overlap, etc.)
        
        Returns:
            Dict con resultados completos del procesamiento
        """
        # === CONFIGURACIÓN DE CREDENCIALES ===
        if credentials_path is None:
            credentials_path = GOOGLE_DRIVE_CREDENTIALS
        
        start_time = time.time()
        
        logger.info(f"Iniciando descarga y procesamiento RAG para colección: {collection_name}")
        
        # === STEP 1: DESCARGA DE DOCUMENTOS ===
        try:
            processed_docs, failed_docs = await download_documents(
                url=url,
                collection_name=collection_name,
                timeout_per_file=timeout_per_file,
                credentials_path=credentials_path,
                payload=payload
            )
            
            logger.info(f"Descarga completada. Exitosos: {len(processed_docs)}, Fallidos: {len(failed_docs)}")
            
        except Exception as e:
            logger.error(f"Error en descarga de documentos: {str(e)}")
            return {
                "success": False,
                "message": f"Error en descarga: {str(e)}",
                "data": None,
                "processing_time_sec": round(time.time() - start_time, 3)
            }
        
        # === STEP 2: PROCESAMIENTO RAG ===
        rag_success = False
        rag_error = None
        chunking_stats = {}
        embedding_stats = {}
        total_chunks = 0
        
        if processed_docs:  # Solo si hay documentos exitosamente descargados
            try:
                logger.info("Iniciando procesamiento RAG...")
                
                # Intentar importar servicios RAG
                # Si no están implementados, continuar sin ellos
                try:
                    from app.services.chunking_service import ChunkingService, ChunkingStrategy
                    from app.services.embedding_service import EmbeddingService
                    from app.services.retrieval_service import RetrievalService
                except ImportError as import_error:
                    logger.warning(f"Servicios RAG no disponibles: {import_error}")
                    rag_error = "Servicios RAG no implementados aún"
                    raise Exception(rag_error)
                
                # Configurar estrategia de chunking desde payload
                # Mapear el string de estrategia al enum
                strategy_map = {
                    "recursive_character": ChunkingStrategy.RECURSIVE_CHARACTER,
                    "fixed_size": ChunkingStrategy.FIXED_SIZE,
                    "semantic": ChunkingStrategy.SEMANTIC,
                    "document_structure": ChunkingStrategy.DOCUMENT_STRUCTURE
                }
                
                # Obtener estrategia desde payload
                chunking_strategy = ChunkingStrategy.RECURSIVE_CHARACTER  # Default fallback
                if payload and hasattr(payload, 'chunking_strategy'):
                    chunking_strategy = strategy_map.get(
                        payload.chunking_strategy, 
                        ChunkingStrategy.RECURSIVE_CHARACTER
                    )
                
                logger.info(f"Estrategia de chunking: {chunking_strategy.value}")

                # Leer flag multimodal desde payload (default False para no romper
                # clientes existentes que no envíen el campo)
                multimodal = getattr(payload, "multimodal", False)
                logger.info(f"Procesamiento multimodal: {'ACTIVADO' if multimodal else 'DESACTIVADO'}")
                
                # Inicializar servicios
                # ChunkingService: cada estrategia define sus propios parámetros internos
                chunking_service = ChunkingService(strategy=chunking_strategy)
                
                # EmbeddingService: usa la configuración interna del servicio
                embedding_service = EmbeddingService()
                
                logger.info(f"Procesando documentos con estrategia={chunking_strategy.value}")
                
                # === CHUNKING ===
                # multimodal=True activa PyMuPDF + pdfplumber + VLM en _load_pdf()
                chunks = chunking_service.process_collection(collection_name, multimodal=multimodal)
                
                # Verificar que process_collection retornó algo válido
                if chunks is None:
                    raise Exception("ChunkingService.process_collection() no está implementado (retorna None)")
                
                total_chunks = len(chunks)
                
                logger.info(f"Chunking completado: {total_chunks} chunks generados")
                
                if chunks and total_chunks > 0:
                    # Crear estadísticas de chunking
                    chunking_stats = chunking_service.get_chunking_statistics(chunks)
                    
                    # === EMBEDDINGS + FAISS INDEX CREATION ===
                    # EmbeddingService es responsable de generar embeddings y construir índice FAISS
                    logger.info("Generando embeddings y construyendo índice FAISS...")
                    
                    # Construir índice FAISS con path específico para la colección
                    faiss_index_path = f"./faiss_index/{collection_name}"
                    
                    try:
                        vectorstore = embedding_service.build_vectorstore(
                            chunks=chunks,
                            persist_path=faiss_index_path,
                            collection_name=collection_name
                        )
                        
                        rag_success = True

                        # Recopilar estadísticas multimodales desde metadatos de chunks
                        # (poblados por MultimodalDocumentService cuando multimodal=True)
                        mm_stats       = chunking_stats.get("multimodal_stats", {})

                        # Costo de embeddings desde EmbeddingService
                        emb_cost_stats = embedding_service.get_cost_stats()

                        # Costo VLM acumulado desde metadatos de chunks
                        vlm_input_tok  = sum(
                            c.metadata.get("vlm_input_tokens",  0) for c in chunks
                        )
                        vlm_output_tok = sum(
                            c.metadata.get("vlm_output_tokens", 0) for c in chunks
                        )
                        vlm_cost_usd   = sum(
                            c.metadata.get("vlm_cost_usd", 0.0) for c in chunks
                        )

                        emb_cost_usd   = emb_cost_stats.get("embedding_cost_usd", 0.0)
                        total_cost_usd = round(vlm_cost_usd + emb_cost_usd, 6)

                        embedding_stats = {
                            "model_used": embedding_service.model,
                            "total_embeddings_generated": total_chunks,
                            "batch_processing_time_seconds": 0.0,
                            "failed_embeddings": 0,
                            "vector_store_created": True,
                            "vector_store_path": faiss_index_path,
                            "vector_store_type": "FAISS",
                            # --- métricas multimodales ---
                            "multimodal_processing":  multimodal,
                            "multimodal_chunks":      mm_stats.get("multimodal_chunks", 0),
                            "chunks_with_images":     mm_stats.get("chunks_with_images", 0),
                            "chunks_with_tables":     mm_stats.get("chunks_with_tables", 0),
                            "total_vlm_calls":        mm_stats.get("total_vlm_calls", 0),
                            # --- costo de indexación (para tabla de comparación) ---
                            "cost_breakdown": {
                                "embedding_estimated_tokens": emb_cost_stats.get("estimated_tokens", 0),
                                "embedding_cost_usd":         emb_cost_usd,
                                "vlm_input_tokens":           vlm_input_tok,
                                "vlm_output_tokens":          vlm_output_tok,
                                "vlm_cost_usd":               round(vlm_cost_usd, 6),
                                "total_indexing_cost_usd":    total_cost_usd,
                            },
                        }
                        
                        logger.info(f"Índice FAISS creado exitosamente: {total_chunks} embeddings en {faiss_index_path}")
                        
                    except Exception as e:
                        rag_error = f"Error creando índice FAISS: {str(e)}"
                        logger.error(rag_error)
                        raise
                        
                else:
                    rag_error = "No se pudieron procesar documentos para chunking"
                    logger.warning(rag_error)
                    
            except Exception as e:
                rag_error = f"Error en procesamiento RAG: {str(e)}"
                logger.error(rag_error)
        
        # === STEP 3: CALCULAR ESTADÍSTICAS ADICIONALES ===
        collection_path = f"./docs/{collection_name}"
        storage_size = 0
        if os.path.exists(collection_path):
            for root, dirs, files in os.walk(collection_path):
                for file in files:
                    storage_size += os.path.getsize(os.path.join(root, file))
        storage_size_mb = round(storage_size / (1024 * 1024), 2)
        
        # === STEP 4: COMPILAR RESULTADOS ===
        warnings = []
        if not rag_success and rag_error:
            warnings.append(f"RAG Warning: {rag_error}")
        
        processing_time = round(time.time() - start_time, 3)
        
        # Mensaje apropiado según si RAG está disponible
        if rag_success:
            message = "Documentos descargados y procesados exitosamente con RAG"
        else:
            message = "Documentos descargados exitosamente (RAG pendiente de implementación)"
        
        result = {
            "success": True,
            "message": message,
            "data": {
                "processing_summary": {
                    "rag_processing": rag_success,
                    "vector_store_created": rag_success,
                    "total_processing_time_sec": processing_time
                },
                "collection_info": {
                    "name": collection_name,
                    "documents_found": len(processed_docs) + len(failed_docs),
                    "documents_processed_successfully": len(processed_docs),
                    "documents_failed": len(failed_docs),
                    "documents_count_before": 0,
                    "documents_count_after": len(processed_docs),
                    "total_chunks_before": 0,
                    "total_chunks_after": total_chunks,
                    "storage_size_mb": storage_size_mb
                },
                "documents_processed": processed_docs,
                "failed_documents": failed_docs,
                "chunking_statistics": chunking_stats,
                "embedding_statistics": embedding_stats,
                "warnings": warnings
            }
        }
        
        logger.info(f"Procesamiento completo finalizado en {processing_time}s. RAG: {rag_success}")
        
        return result


async def load_from_csv(
        collection_name: str,
        operation_type: str = None,
        property_type: str = None,
        barrio: str = None,
    ) -> Dict[str, Any]:
        """
        Indexa una colección de listings inmobiliarios desde un CSV local.
 
        Reemplaza el flujo Google Drive → chunk → embed para datos CSV.
        Espera encontrar el archivo en: ./docs/{collection_name}/*.csv
 
        Flujo:
          1. ChunkingService.process_csv_collection() → List[Document]
          2. EmbeddingService.build_vectorstore()     → FAISS index
 
        Args:
            collection_name : Nombre de la colección (carpeta en ./docs/).
            operation_type  : Filtro opcional: "venta" | "alquiler"
            property_type   : Filtro opcional: "apartamentos" | "casas"
            barrio          : Filtro opcional: nombre exacto del barrio
 
        Returns:
            Dict con resultados del procesamiento en el mismo formato
            que download_and_process_documents().
        """
        start_time = time.time()
        logger.info(f"Iniciando indexación CSV para colección: {collection_name}")
 
        try:
            from app.services.chunking_service import ChunkingService
            from app.services.embedding_service import EmbeddingService
        except ImportError as e:
            return {
                "success": False,
                "message": f"Servicios RAG no disponibles: {e}",
                "data": None,
            }
 
        # === STEP 1: CHUNK ===
        try:
            chunking_service = ChunkingService()
            chunks = chunking_service.process_csv_collection(
                collection_name,
                operation_type=operation_type,
                property_type=property_type,
                barrio=barrio,
            )
        except FileNotFoundError as e:
            return {"success": False, "message": str(e), "data": None}
        except Exception as e:
            logger.error(f"Error en process_csv_collection: {e}")
            return {"success": False, "message": f"Error en chunking: {e}", "data": None}
 
        if not chunks:
            return {
                "success": False,
                "message": "No se generaron documentos. Verifica los filtros.",
                "data": None,
            }
 
        total_chunks = len(chunks)
        chunking_stats = chunking_service.get_chunking_statistics(chunks)
        logger.info(f"Chunking completado: {total_chunks} documentos")
 
        # === STEP 2: EMBED + FAISS ===
        try:
            embedding_service  = EmbeddingService()
            faiss_index_path   = f"./faiss_index/{collection_name}"
 
            vectorstore = embedding_service.build_vectorstore(
                chunks=chunks,
                persist_path=faiss_index_path,
                collection_name=collection_name,
            )
 
            emb_cost_stats = embedding_service.get_cost_stats()
 
            embedding_stats = {
                "model_used":                    embedding_service.model,
                "total_embeddings_generated":    total_chunks,
                "batch_processing_time_seconds": round(time.time() - start_time, 2),
                "failed_embeddings":             0,
                "vector_store_created":          True,
                "vector_store_path":             faiss_index_path,
                "vector_store_type":             "FAISS",
                "multimodal_processing":         False,
                "cost_breakdown": {
                    "embedding_estimated_tokens": emb_cost_stats.get("estimated_tokens", 0),
                    "embedding_cost_usd":         emb_cost_stats.get("embedding_cost_usd", 0.0),
                    "total_indexing_cost_usd":    emb_cost_stats.get("embedding_cost_usd", 0.0),
                },
            }
 
        except Exception as e:
            logger.error(f"Error creando índice FAISS: {e}")
            return {"success": False, "message": f"Error en embeddings: {e}", "data": None}
 
        processing_time = round(time.time() - start_time, 3)
        logger.info(f"Indexación CSV completada en {processing_time}s — {total_chunks} embeddings")
 
        return {
            "success": True,
            "message": f"Colección '{collection_name}' indexada exitosamente desde CSV",
            "data": {
                "processing_summary": {
                    "rag_processing":             True,
                    "vector_store_created":       True,
                    "total_documents":            total_chunks,
                    "total_processing_time_sec":  processing_time,
                    "filters_applied": {
                        "operation_type": operation_type,
                        "property_type":  property_type,
                        "barrio":         barrio,
                    },
                },
                "collection_info": {
                    "name":                         collection_name,
                    "total_chunks_after":           total_chunks,
                    "storage_path":                 faiss_index_path,
                },
                "chunking_statistics":  chunking_stats,
                "embedding_statistics": embedding_stats,
            },
        }


async def validate_processing_with_rag(processing_id: str) -> Dict[str, Any]:
    """
    Valida el procesamiento y agrega información de estado RAG.
    
    Esta función lee el archivo de log del procesamiento y agrega
    información actualizada sobre el estado de la base de datos vectorial.
    
    Args:
        processing_id: Identificador único del procesamiento
    
    Returns:
        Datos completos del procesamiento con estado RAG actualizado
    
    Raises:
        FileNotFoundError: Si el processing_id no existe
        ValueError: Si el archivo JSON está corrupto
    """
    import json
    from pathlib import Path
    
    LOG_DIR = os.getenv("LOG_DIR", "./logs")
    log_dir = Path(LOG_DIR)
    
    # Leer archivo de log
    file_path = log_dir / f"{processing_id}.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"No existe {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        data = json.loads(content)
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Archivo JSON corrupto: {e}")
    
    # Agregar información de estado RAG si el procesamiento fue exitoso
    if data.get("success") and data.get("data"):
        collection_name = data["data"].get("collection_info", {}).get("name")
        
        if collection_name:
            logger.info(f"Agregando estado RAG actualizado para colección: {collection_name}")
            
            # Obtener estado RAG actualizado
            rag_status = get_rag_status(collection_name)
            
            # Agregar información RAG a la respuesta
            if "rag_status" not in data["data"]:
                data["data"]["rag_status"] = {}
            
            data["data"]["rag_status"].update(rag_status)
            
            # Agregar timestamp de validación
            data["data"]["rag_status"]["validated_at"] = time.time()
    
    return data


def get_rag_status(collection_name: str) -> Dict[str, Any]:
    """
    Obtiene el estado actual del índice vectorial FAISS.
    
    Args:
        collection_name: Nombre de la colección a verificar
    
    Returns:
        Diccionario con el estado RAG completo
    """
    try:
        # Verificar documentos en la carpeta docs
        docs_path = f"./docs/{collection_name}"
        csv_count = 0
        csv_files = []
        
        if os.path.exists(docs_path):
            csv_files = [f for f in os.listdir(docs_path) if f.endswith('.csv')]
            csv_count = len(csv_files)
        
        # Verificar si existe el índice FAISS
        faiss_index_path = f"./faiss_index/{collection_name}"
        vector_store_exists = os.path.exists(faiss_index_path)
        
        # Verificar archivos específicos de FAISS
        index_faiss_file = os.path.join(faiss_index_path, "index.faiss")
        index_pkl_file = os.path.join(faiss_index_path, "index.pkl")
        
        vector_store_ready = (
            vector_store_exists and 
            os.path.exists(index_faiss_file) and 
            os.path.exists(index_pkl_file)
        )
        
        # Intentar obtener número de documentos del índice
        document_count = 0
        if vector_store_ready:
            try:
                from app.services.embedding_service import EmbeddingService
                embedder = EmbeddingService()
                embedder.load_vectorstore(faiss_index_path, collection_name=collection_name)
                # Obtener número de vectores directamente del índice FAISS
                if embedder.vectorstore:
                    document_count = embedder.vectorstore.index.ntotal
            except Exception as e:
                logger.warning(f"No se pudo cargar estadísticas del índice FAISS: {e}")
        
        rag_status = {
            "vector_store_exists": vector_store_exists,
            "vector_store_ready": vector_store_ready,
            "vector_store_type": "FAISS",
            "collection_name": collection_name,
            "vector_store_path": faiss_index_path,
            "rag_ready": vector_store_ready,
            "document_count": document_count,
            "documents_in_collection": csv_count,
            "csv_files": csv_files  
        }
        
        logger.info(f"Estado RAG para '{collection_name}': ready={rag_status['rag_ready']}, "
                   f"docs={csv_count}, embeddings={rag_status['document_count']}")
        
        return rag_status
        
    except Exception as e:
        logger.error(f"Error verificando estado RAG: {str(e)}")
        return {
            "error": f"Error verificando estado RAG: {str(e)}",
            "vector_store_exists": False,
            "rag_ready": False,
            "collection_name": collection_name
        }
    