"""
Servicio de integración con Google Drive

Este módulo proporciona funcionalidades para interactuar con Google Drive
como proveedor de almacenamiento de documentos. Incluye validación de archivos,
descarga con timeout y manejo robusto de errores.
"""

import re
from typing import List
from fastapi import APIRouter, HTTPException, status
from google.oauth2 import service_account
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload
from datetime import datetime
import signal
import time
from abc import ABC, abstractmethod

# Scopes necesarios para acceso de solo lectura a Google Drive
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# ==================== CLASE ABSTRACTA ====================

class StorageProvider(ABC):
    """
    Clase abstracta que define la interfaz para proveedores de almacenamiento.
    
    Establece el contrato que deben implementar todos los proveedores
    de almacenamiento (Google Drive, Dropbox, etc.).
    """
    @abstractmethod
    def list_documents(self) -> list[dict]:
        """Lista todos los documentos disponibles en el proveedor."""
        pass

    @abstractmethod
    def download_document(self, document_id: str, output_path: str) -> None:
        """Descarga un documento específico al sistema local."""
        pass


# ==================== FUNCIONES AUXILIARES ====================

def validate_file(filename, filesize, payload):
    """
    Valida si un archivo cumple con los criterios de extensión y tamaño.
    
    Verifica que el archivo tenga una extensión permitida y no exceda
    el tamaño máximo configurado en las opciones de procesamiento.
    
    Args:
        filename: Nombre del archivo a validar
        filesize: Tamaño del archivo en bytes
        payload: Configuración de procesamiento con opciones de validación
    
    Returns:
        tuple: (is_valid: bool, error_message: str)

    Solo modificar esta función en caso de que se quiera agregar más extensiones de archivos permitidas.
    También se puede modificar el tamaño máximo de archivo permitido.
    """
    # Usar valores por defecto si processing_options no está presente
    if payload and hasattr(payload, 'processing_options') and payload.processing_options:
        allowed_extensions = [ext.lower() for ext in payload.processing_options.file_extensions]
        max_size_mb = payload.processing_options.max_file_size_mb
    else:
        # Valores por defecto
        allowed_extensions = ["pdf", "txt", "docx", "md", "csv", "xlsx", "png", "jpg", "jpeg"]
        max_size_mb = 100
    
    # Validación de extensión
    file_extension = filename.split(".")[-1].lower() if "." in filename else ""
    
    if file_extension not in allowed_extensions:
        return False, f"INVALID_EXTENSION"
    
    # Validación de tamaño
    max_size_bytes = max_size_mb * 1024 * 1024
    if int(filesize) > max_size_bytes:
        return False, f"FILE_TOO_LARGE"
    
    return True, "VALID"
    

def extract_folder_id(folder_url: str) -> str:
    """
    Extrae el ID de carpeta de una URL de Google Drive.
    
    Parsea la URL para obtener el identificador único de la carpeta
    que se utilizará en las consultas a la API de Google Drive.
    
    Args:
        folder_url: URL completa de la carpeta de Google Drive
    
    Returns:
        str: ID de la carpeta extraído de la URL
    
    Raises:
        HTTPException: Si la URL no tiene el formato esperado
    """
    time_stamp = datetime.now().isoformat()
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", folder_url)
    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "success": False,
                "error": {
                    "code": "SOURCE_URL_NOT_FOUND",
                    "message": "La URL proporcionada no es accesible",
                    "details": {
                    "url": folder_url,
                    "http_status": 404,
                    "retry_attempts": 3
                    },
                    "timestamp": time_stamp
                }
            }
        )
    return match.group(1)



# ==================== IMPLEMENTACIÓN GOOGLE DRIVE ====================

class GoogleDriveProvider(StorageProvider):
    """
    Implementación concreta del proveedor de almacenamiento para Google Drive.
    
    Maneja la autenticación, listado y descarga de documentos desde
    una carpeta específica de Google Drive.
    """
    
    def __init__(self, folder_url: str, credentials_path: str):
        """
        Inicializa el proveedor de Google Drive.
        
        Args:
            folder_url: URL de la carpeta de Google Drive
            credentials_path: Ruta al archivo de credenciales JSON
        """
        self.folder_id = extract_folder_id(folder_url)
        self.service = self._build_service(credentials_path)

    def _build_service(self, credentials_path: str):
        """
        Construye el cliente de la API de Google Drive.
        
        Args:
            credentials_path: Ruta al archivo de credenciales
        
        Returns:
            Resource: Cliente autenticado de Google Drive API
        """
        creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES)
        return build("drive", "v3", credentials=creds)

    def list_documents(self) -> List[dict]:
        """
        Lista todos los documentos en la carpeta configurada.
        
        Consulta la API de Google Drive para obtener la lista de archivos
        en la carpeta especificada, excluyendo archivos en papelera.
        
        Returns:
            List[dict]: Lista de documentos con metadatos (id, name, mimeType, size)
        """
        query = f"'{self.folder_id}' in parents and trashed = false"
        results = self.service.files().list(
            q=query,
            fields="files(id, name, mimeType, size)",
            pageSize=100
        ).execute()
        return results.get("files", [])

    
    def download_document(self, document_id: str, output_path: str, payload, timeout_seconds: int = 300) -> dict:
        """
        Descarga un documento con timeout y validación de errores.
        
        Implementa descarga robusta con múltiples mecanismos de timeout,
        validación previa de archivos y limpieza automática en caso de error.
        
        **Flujo de descarga:**
        1. Configuración de timeout (signal + manual)
        2. Obtención de metadatos del archivo
        3. Validación de extensión y tamaño
        4. Descarga por chunks con verificación de timeout
        5. Limpieza automática en caso de error
        
        Args:
            document_id: ID del documento en Google Drive
            output_path: Ruta donde guardar el archivo
            payload: Configuración de procesamiento para validación
            timeout_seconds: Timeout en segundos (default: 300 = 5 minutos)
            
        Returns:
            dict: Resultado de la descarga:
                - "" (string vacío) si es exitosa
                - dict con error_code, error_message, etc. si falla
        """
        
        start_time = time.time()
        filename = None
        
        # === CONFIGURACIÓN DE TIMEOUT ===
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Download timeout after {timeout_seconds} seconds")

        # Configurar timeout con signal (solo en sistemas Unix)
        use_alarm_timeout = hasattr(signal, "SIGALRM")
        if use_alarm_timeout:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        
        try:
            # === OBTENCIÓN DE METADATOS ===
            file_info = self.service.files().get(fileId=document_id, fields="name,size").execute()
            filename = file_info.get("name", "unknown_file")
            filesize = file_info.get("size", 0)
            
            # === VALIDACIÓN PREVIA ===
            validation_result = validate_file(filename, filesize, payload)
            if not validation_result[0]:
                processing_time = time.time() - start_time
                return {
                    "filename": filename or "unknown_file",
                    "download_url": f"https://drive.google.com/file/d/{document_id}",
                    "error_code": "VALIDATION_ERROR", 
                    "error_message": validation_result[1],
                    "processing_time_seconds": round(processing_time, 2),
                    "success": False
                }
            
            # === INICIO DE DESCARGA ===
            request = self.service.files().get_media(fileId=document_id)
            fh = io.FileIO(output_path, "wb")
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            
            # === DESCARGA POR CHUNKS ===
            while not done:
                # Verificación manual de timeout
                if time.time() - start_time > timeout_seconds:
                    fh.close()
                    raise TimeoutError(f"Download timeout after {timeout_seconds} seconds")
                
                _, done = downloader.next_chunk()
            
            # === DESCARGA EXITOSA ===
            processing_time = time.time() - start_time
            return ""
                
        except TimeoutError:
            # === MANEJO DE TIMEOUT ===
            # Cerrar archivo si existe
            try:
                fh.close()
            except:
                pass
            # Limpiar archivo parcial
            try:
                import os
                if os.path.exists(output_path):
                    os.remove(output_path)
            except:
                pass
            
            processing_time = time.time() - start_time
            return {
                "filename": filename or "unknown_file",
                "download_url": f"https://drive.google.com/file/d/{document_id}",
                "error_code": "DOWNLOAD_TIMEOUT",
                "error_message": f"La descarga del documento excedió el tiempo límite de {timeout_seconds} segundos",
                "processing_time_seconds": round(processing_time, 2),
                "success": False
            }
            
        except Exception as e:
            # === MANEJO DE ERRORES GENERALES ===
            # Cerrar archivo si existe
            try:
                fh.close()
            except:
                pass
            # Limpiar archivo parcial
            try:
                import os
                if os.path.exists(output_path):
                    os.remove(output_path)
            except:
                pass
            
            processing_time = time.time() - start_time
            return {
                "filename": filename or "unknown_file",
                "download_url": f"https://drive.google.com/file/d/{document_id}",
                "error_code": "CORRUPTED_FILE",
                "error_message": f"Error al descargar el documento: {str(e)}",
                "processing_time_seconds": round(processing_time, 2),
                "success": False
            }
            
        finally:
            # === LIMPIEZA FINAL ===
            # Cancelar el timeout si se configuró
            if use_alarm_timeout:
                signal.alarm(0)
            # Asegurar que el archivo se cierre
            try:
                fh.close()
            except:
                pass
            

