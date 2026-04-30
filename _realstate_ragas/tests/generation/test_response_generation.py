"""
Tests de Lógica de Generación de Respuestas
============================================

Valida el comportamiento del método generate_response() del GenerationService.

"""

import pytest
from langchain.schema import Document
from app.services.generation_service import GenerationService


@pytest.mark.integration
def test_generate_response_structure(mock_documents):
    """
    Verifica que generate_response() retorna la estructura correcta.
    
    Valida:
    - Retorna Dict con "answer", "sources", "context"
    - Sources se extraen de metadata correctamente
    - Context contiene page_content de documentos
    
    ADVERTENCIA: Consume API (~1-2 requests)
    """
    # Crear servicio real
    service = GenerationService()
    
    # Ejecutar generación con documentos reales
    question = "What is machine learning?"
    result = service.generate_response(question, mock_documents)
    
    # Verificar estructura del resultado
    assert isinstance(result, dict), "El resultado debe ser un diccionario"
    assert "answer" in result, "El resultado debe contener 'answer'"
    assert "sources" in result, "El resultado debe contener 'sources'"
    assert "context" in result, "El resultado debe contener 'context'"
    
    # Verificar tipos
    assert isinstance(result["answer"], str), "answer debe ser string"
    assert isinstance(result["sources"], list), "sources debe ser una lista"
    assert isinstance(result["context"], list), "context debe ser una lista"
    
    # Verificar que no están vacíos
    assert len(result["answer"]) > 0, "answer no debe estar vacío"
    assert len(result["sources"]) == len(mock_documents), "sources debe tener un elemento por documento"
    assert len(result["context"]) == len(mock_documents), "context debe tener un elemento por documento"


@pytest.mark.integration
def test_generate_response_preserves_sources(mock_documents):
    """
    Verifica que las fuentes (sources) se extraen correctamente de metadata.
    
    Valida:
    - Extrae 'source_file' de metadata de cada documento
    - Retorna lista de sources en el orden correcto
    
    ADVERTENCIA: Consume API (~1-2 requests)
    """
    service = GenerationService()
    
    # Ejecutar generación
    result = service.generate_response("test question", mock_documents)
    
    # Verificar sources
    assert len(result["sources"]) == len(mock_documents), \
        "Debe haber una source por cada documento"
    
    # Verificar que las sources son correctas
    expected_sources = [
        "ai_guide.pdf",
        "python_handbook.pdf",
        "deep_learning.pdf"
    ]
    assert result["sources"] == expected_sources, \
        f"Sources incorrectas. Esperadas: {expected_sources}, Obtenidas: {result['sources']}"


