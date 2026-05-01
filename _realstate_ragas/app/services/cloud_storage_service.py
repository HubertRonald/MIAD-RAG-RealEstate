"""
Servicio de integración con Google Cloud Storage

Este módulo proporciona funcionalidades para guardar y cargar
índices FAISS en Google Cloud Storage de manera persistente.
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional
from google.cloud import storage

logger = logging.getLogger(__name__)

class CloudStorageService:
    """
    Servicio para gestionar índices FAISS en Google Cloud Storage.
    
    Permite guardar y cargar índices FAISS de manera persistente
    en buckets de Google Cloud Storage.
    """
    
    def __init__(self, bucket_name: Optional[str] = None):
        """
        Inicializa el servicio de Cloud Storage.
        
        Args:
            bucket_name: Nombre del bucket de GCS. Si no se proporciona,
                        se obtiene de la variable de entorno CLOUD_STORAGE_BUCKET
        """
        self.bucket_name = bucket_name or os.getenv("CLOUD_STORAGE_BUCKET")
        
        if not self.bucket_name:
            logger.warning(
                "CLOUD_STORAGE_BUCKET no configurada. "
                "Los índices se guardarán localmente."
            )
            self.client = None
            self.bucket = None
        else:
            try:
                self.client = storage.Client()
                self.bucket = self.client.bucket(self.bucket_name)
                logger.info(f"Cloud Storage inicializado con bucket: {self.bucket_name}")
            except Exception as e:
                logger.error(f"Error inicializando Cloud Storage: {str(e)}")
                self.client = None
                self.bucket = None
    
    def save_index_to_cloud(self, local_path: str, collection_name: str) -> bool:
        """
        Guarda un índice FAISS local a Google Cloud Storage.
        
        Comprime el índice y lo sube al bucket.
        
        Args:
            local_path: Ruta local del índice FAISS
            collection_name: Nombre de la colección (para organizar en el bucket)
        
        Returns:
            bool: True si se guardó exitosamente, False si ocurrió un error
        """
        if not self.client or not self.bucket:
            logger.warning("Cloud Storage no configurado. Índice guardado solo localmente.")
            return True
        
        try:
            local_path = Path(local_path)
            if not local_path.exists():
                logger.error(f"Índice local no encontrado: {local_path}")
                return False
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_archive = Path(temp_dir) / f"{collection_name}_index"
                shutil.make_archive(
                    str(temp_archive),
                    'zip',
                    str(local_path.parent),
                    local_path.name
                )
                
                cloud_path = f"indexes/{collection_name}/{collection_name}_index.zip"
                blob = self.bucket.blob(cloud_path)
                blob.upload_from_filename(f"{temp_archive}.zip")
                
                logger.info(f"Índice guardado en Cloud Storage: {cloud_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error al guardar índice en Cloud Storage: {str(e)}")
            return False
    
    def load_index_from_cloud(self, collection_name: str, local_path: str) -> bool:
        """
        Carga un índice FAISS desde Google Cloud Storage al almacenamiento local.
        
        Args:
            collection_name: Nombre de la colección
            local_path: Ruta local donde guardar el índice
        
        Returns:
            bool: True si se cargó exitosamente, False si ocurrió un error
        """
        if not self.client or not self.bucket:
            logger.warning("Cloud Storage no configurado.")
            return False
        
        try:
            cloud_path = f"indexes/{collection_name}/{collection_name}_index.zip"
            blob = self.bucket.blob(cloud_path)
            
            if not blob.exists():
                logger.warning(f"Índice no encontrado en Cloud Storage: {cloud_path}")
                return False
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_archive = Path(temp_dir) / f"{collection_name}_index.zip"
                blob.download_to_filename(str(temp_archive))
                
                local_parent = Path(local_path).parent
                local_parent.mkdir(parents=True, exist_ok=True)
                
                shutil.unpack_archive(str(temp_archive), str(local_parent), 'zip')
                
                logger.info(f"Índice cargado desde Cloud Storage: {cloud_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error al cargar índice desde Cloud Storage: {str(e)}")
            return False
    
    def index_exists_in_cloud(self, collection_name: str) -> bool:
        """
        Verifica si un índice existe en Cloud Storage.
        
        Args:
            collection_name: Nombre de la colección
        
        Returns:
            bool: True si existe, False si no
        """
        if not self.client or not self.bucket:
            return False
        
        try:
            cloud_path = f"indexes/{collection_name}/{collection_name}_index.zip"
            blob = self.bucket.blob(cloud_path)
            return blob.exists()
        except Exception as e:
            logger.error(f"Error al verificar índice en Cloud Storage: {str(e)}")
            return False
