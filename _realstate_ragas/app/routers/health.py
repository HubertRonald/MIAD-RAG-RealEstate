"""
Módulo de verificación de salud del sistema

Este módulo proporciona un endpoint simple para verificar que la API
esté funcionando correctamente. Es útil para monitoring y balanceadores de carga.
"""

from datetime import datetime
from fastapi import APIRouter

# Router sin prefijo para acceso directo al health check
router = APIRouter()


@router.get("/api/v1/health")
def health_check():
    """
    Endpoint de verificación de salud del sistema.
    
    Este endpoint permite verificar que la API esté funcionando correctamente.
    Es una función síncrona (no async) porque no requiere operaciones complejas.
    
    Returns:
        dict: Estado del sistema con:
            - status: Siempre "healthy" si la API responde
            - success: Siempre True
            - timestamp: Momento exacto de la verificación
            - service: Identificador del servicio
    """
    time_stamp = datetime.now().isoformat()
    return {
        "status": "healthy",
        "success": True,
        "timestamp": time_stamp,
        "service": "API",
    }

