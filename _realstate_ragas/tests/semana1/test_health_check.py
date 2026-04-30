"""
Tests de health check del API

Este módulo contiene tests para verificar que el endpoint de salud
del sistema funciona correctamente y retorna la información esperada.

Tests incluidos:
- Verificación de status code 200
- Validación de formato de respuesta JSON
- Verificación de campos requeridos
- Validación de formato de timestamp ISO
- Pruebas de estabilidad con múltiples llamadas
"""

import pytest
import re
from datetime import datetime
from fastapi.testclient import TestClient


def test_health_check_responds_200(client: TestClient):
    """
    Test básico que verifica que el health check responde con status 200.
    
    Este es el test más fundamental que confirma que el endpoint
    está disponible y responde correctamente.
    
    Args:
        client: Cliente de prueba de FastAPI
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_check_response_format(client: TestClient):
    """
    Test que verifica el formato completo de la respuesta del health check.
    
    Valida que la respuesta contenga todos los campos requeridos
    con los valores esperados y que esté en formato JSON válido.
    
    Args:
        client: Cliente de prueba de FastAPI
    """
    response = client.get("/api/v1/health")
    
    # === VERIFICACIÓN DE STATUS CODE ===
    assert response.status_code == 200
    
    # === VERIFICACIÓN DE CONTENT TYPE ===
    assert response.headers["content-type"].startswith("application/json")
    
    # === OBTENCIÓN Y VALIDACIÓN DE DATOS ===
    data = response.json()
    
    # Verificar campos requeridos
    required_fields = ["status", "success", "timestamp", "service"]
    for field in required_fields:
        assert field in data, f"Campo '{field}' no encontrado en la respuesta"
    
    # === VALIDACIÓN DE VALORES ESPECÍFICOS ===
    assert data["success"] is True, "El campo 'success' debe ser True"
    assert data["status"] == "healthy", "El campo 'status' debe ser 'healthy'"
    assert data["service"] == "API", "El campo 'service' debe ser 'API'"


def test_health_check_timestamp_format(client: TestClient):
    """
    Test que verifica que el timestamp esté en formato ISO 8601 válido.
    
    Valida tanto el formato de la cadena como la capacidad de parsearla
    como un objeto datetime válido.
    
    Args:
        client: Cliente de prueba de FastAPI
    """
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    
    # === VERIFICACIÓN DE PRESENCIA ===
    assert "timestamp" in data
    timestamp_str = data["timestamp"]
    
    # === VALIDACIÓN DE FORMATO ISO 8601 ===
    # Patrón para validar formato ISO: YYYY-MM-DDTHH:MM:SS.ssssss
    iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?$'
    assert re.match(iso_pattern, timestamp_str), f"Timestamp '{timestamp_str}' no está en formato ISO válido"
    
    # === VALIDACIÓN DE PARSING ===
    try:
        parsed_date = datetime.fromisoformat(timestamp_str)
        assert isinstance(parsed_date, datetime)
    except ValueError as e:
        pytest.fail(f"No se pudo parsear el timestamp '{timestamp_str}' como datetime: {e}")


def test_health_check_multiple_calls(client: TestClient):
    """
    Test que verifica la estabilidad del endpoint con múltiples llamadas.
    
    Confirma que el endpoint mantiene su funcionalidad correcta
    después de varias peticiones consecutivas.
    
    Args:
        client: Cliente de prueba de FastAPI
    """
    for i in range(3):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "healthy"
