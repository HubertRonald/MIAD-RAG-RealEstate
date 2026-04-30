"""
Tests de Casos Extremos y Edge Cases
=====================================

Valida que el GenerationService maneja correctamente situaciones especiales.

"""

import pytest
from langchain.schema import Document
from app.services.generation_service import GenerationService


@pytest.mark.integration
def test_generate_response_with_empty_documents():
    """
    Verifica el comportamiento con lista vacía de documentos.
    
    Valida:
    - Maneja retrieved_docs = [] sin errores
    - Retorna respuesta válida (contexto vacío)
    - sources = []
    - context = []
    
    ADVERTENCIA: Consume API (~1 request)
    """
    service = GenerationService()
    
    # Ejecutar generación con lista vacía de documentos
    question = "What is machine learning?"
    result = service.generate_response(question, [])
    
    # Verificar que no hay errores y la estructura es correcta
    assert isinstance(result, dict), "El resultado debe ser un diccionario"
    assert "answer" in result, "El resultado debe contener 'answer'"
    assert "sources" in result, "El resultado debe contener 'sources'"
    assert "context" in result, "El resultado debe contener 'context'"
    
    # Verificar que sources y context están vacíos
    assert result["sources"] == [], "sources debe ser lista vacía"
    assert result["context"] == [], "context debe ser lista vacía"
    
    # Verificar que se generó una respuesta (aunque sea con contexto vacío)
    assert result["answer"] is not None, "answer no debe ser None"
    assert isinstance(result["answer"], str), "answer debe ser string"


