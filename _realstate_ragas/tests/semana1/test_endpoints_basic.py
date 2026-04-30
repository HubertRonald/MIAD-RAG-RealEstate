"""
Tests de endpoints básicos del API

Este módulo contiene tests para verificar que todos los endpoints
principales están disponibles y responden correctamente.

Tests incluidos:
- Verificación de existencia de endpoints (no 404)
- Validación de formato JSON en answers
- Pruebas de estabilidad del servidor
- Tests individuales por endpoint
- Test integral de todos los endpoints
"""

import pytest
from fastapi.testclient import TestClient


class TestEndpointsBasic:
    """
    Clase que agrupa los tests de endpoints básicos.
    
    Contiene tests para verificar que todos los endpoints principales
    están disponibles y responden correctamente sin errores 404.
    """
    
    def test_load_from_url_endpoint_exists(self, client: TestClient):
        """
        Test que verifica que el endpoint POST /api/v1/documents/load-from-url existe.
        
        Valida que el endpoint de carga de documentos esté disponible
        y responda con formato JSON válido.
        
        Args:
            client: Cliente de prueba de FastAPI
        """
        # === CONFIGURACIÓN DE PAYLOAD ===
        payload = {
            "source_url": "https://example.com",
            "collection_name": "test_collection",
            "chunking_strategy": "recursive_character"
        }
        
        # === REALIZACIÓN DE PETICIÓN ===
        response = client.post("/api/v1/documents/load-from-url", json=payload)
        
        # === VERIFICACIÓN DE EXISTENCIA ===
        assert response.status_code != 404, "El endpoint POST /api/v1/documents/load-from-url no debería retornar 404"
        
        # === VERIFICACIÓN DE FORMATO JSON ===
        assert "application/json" in response.headers.get("content-type", ""), "La answer debe ser JSON"
        
        # === VERIFICACIÓN DE ESTABILIDAD ===
        try:
            data = response.json()
            assert isinstance(data, dict), "La answer debe ser un objeto JSON"
        except Exception as e:
            pytest.fail(f"Error al parsear la answer JSON: {e}")
    
    def test_ask_endpoint_exists(self, client: TestClient):
        """
        Test que verifica que el endpoint POST /api/v1/ask existe.
        
        Valida que el endpoint de consultas esté disponible
        y responda con formato JSON válido.
        
        Args:
            client: Cliente de prueba de FastAPI
        """
        # === CONFIGURACIÓN DE PAYLOAD ===
        payload = {
            "question": "¿Qué es esto?",
            "collection": "test_collection"
        }
        
        # === REALIZACIÓN DE PETICIÓN ===
        response = client.post("/api/v1/ask", json=payload)
        
        # === VERIFICACIÓN DE EXISTENCIA ===
        assert response.status_code != 404, "El endpoint POST /api/v1/ask no debería retornar 404"
        
        # === VERIFICACIÓN DE FORMATO JSON ===
        assert "application/json" in response.headers.get("content-type", ""), "La answer debe ser JSON"
        
        # === VERIFICACIÓN DE ESTABILIDAD ===
        try:
            data = response.json()
            assert isinstance(data, dict), "La answer debe ser un objeto JSON"
        except Exception as e:
            pytest.fail(f"Error al parsear la answer JSON: {e}")
    
    def test_validate_load_endpoint_exists(self, client: TestClient):
        """
        Test que verifica que el endpoint GET /api/v1/documents/load-from-url/{processing_id} existe.
        
        Valida que el endpoint de validación de carga esté disponible
        y responda con formato JSON válido.
        
        Args:
            client: Cliente de prueba de FastAPI
        """
        # === CONFIGURACIÓN DE PARÁMETROS ===
        test_processing_id = "test-id"
        
        # === REALIZACIÓN DE PETICIÓN ===
        response = client.get(f"/api/v1/documents/load-from-url/{test_processing_id}")
        
        # === VERIFICACIÓN DE EXISTENCIA ===
        assert response.status_code != 404, f"El endpoint GET /api/v1/documents/load-from-url/{test_processing_id} no debería retornar 404"
        
        # === VERIFICACIÓN DE FORMATO JSON ===
        assert "application/json" in response.headers.get("content-type", ""), "La answer debe ser JSON"
        
        # === VERIFICACIÓN DE ESTABILIDAD ===
        try:
            data = response.json()
            assert isinstance(data, dict), "La answer debe ser un objeto JSON"
        except Exception as e:
            pytest.fail(f"Error al parsear la answer JSON: {e}")
    
    def test_all_endpoints_respond_json(self, client: TestClient):
        """
        Test integral que verifica que todos los endpoints principales respondan con JSON válido.
        
        Ejecuta una batería de tests contra todos los endpoints principales
        para verificar que están disponibles y responden correctamente.
        
        Args:
            client: Cliente de prueba de FastAPI
        """
        # === CONFIGURACIÓN DE ENDPOINTS ===
        endpoints_to_test = [
            {
                "method": "POST",
                "url": "/api/v1/documents/load-from-url",
                "payload": {
                    "source_url": "https://example.com",
                    "collection_name": "test_collection",
                    "chunking_strategy": "recursive_character"
                }
            },
            {
                "method": "POST", 
                "url": "/api/v1/ask",
                "payload": {
                    "question": "¿Qué es esto?",
                    "collection": "test_collection"
                }
            },
            {
                "method": "GET",
                "url": "/api/v1/documents/load-from-url/test-id",
                "payload": None
            }
        ]
        
        # === EJECUCIÓN DE TESTS ===
        for endpoint in endpoints_to_test:
            method = endpoint["method"]
            url = endpoint["url"]
            payload = endpoint["payload"]
            
            # Realizar petición según método HTTP
            if method == "POST":
                response = client.post(url, json=payload)
            elif method == "GET":
                response = client.get(url)
            else:
                pytest.fail(f"Método HTTP no soportado: {method}")
            
            # === VERIFICACIONES ===
            # Verificar que no sea 404
            assert response.status_code != 404, f"El endpoint {method} {url} no debería retornar 404"
            
            # Verificar que sea JSON
            content_type = response.headers.get("content-type", "")
            assert "application/json" in content_type, f"El endpoint {method} {url} debe retornar JSON"
            
            # Verificar que el JSON sea válido
            try:
                data = response.json()
                assert isinstance(data, dict), f"El endpoint {method} {url} debe retornar un objeto JSON"
            except Exception as e:
                pytest.fail(f"Error al parsear JSON del endpoint {method} {url}: {e}")
    
    def test_server_stability(self, client: TestClient):
        """
        Test que verifica la estabilidad del servidor con múltiples peticiones.
        
        Realiza varias peticiones consecutivas para verificar que el servidor
        mantiene su estabilidad y no presenta degradación de rendimiento.
        
        Args:
            client: Cliente de prueba de FastAPI
        """
        # === BATERÍA DE PETICIONES ===
        for i in range(5):
            # Test health check para verificar estado del servidor
            health_response = client.get("/api/v1/health")
            assert health_response.status_code == 200, f"El servidor debe seguir respondiendo después de {i} peticiones"
            
            # Test endpoint de consultas para verificar funcionalidad
            ask_payload = {"question": f"Pregunta de prueba {i}", "top_k": 3}
            ask_response = client.post("/api/v1/ask", json=ask_payload)
            assert ask_response.status_code != 404, f"El endpoint ask debe seguir existiendo después de {i} peticiones"
