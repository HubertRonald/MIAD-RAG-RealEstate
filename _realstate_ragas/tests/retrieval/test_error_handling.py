"""
Tests de Manejo de Errores
===========================

Valida que el RetrievalService maneja correctamente situaciones de error.

ESTRATEGIA: Mock (sin API, sin vectorstore real)
TIEMPO: ~1 segundo
COSTO API: $0.00
"""

import pytest
from unittest.mock import Mock
from app.services.retrieval_service import RetrievalService


def test_service_handles_retriever_exceptions(mock_embedding_service, mock_vectorstore):
    """
    Verifica que el servicio propaga excepciones del retriever correctamente.
    
    Si el retriever.invoke() falla, la excepción debe propagarse al caller.
    """
    # Configurar mock retriever que lanza excepción
    mock_retriever = Mock()
    mock_retriever.invoke.side_effect = Exception("Retriever failed")
    mock_vectorstore.as_retriever.return_value = mock_retriever
    
    # Crear servicio
    service = RetrievalService(mock_embedding_service, k=3)
    
    # Debe propagar la excepción
    with pytest.raises(Exception) as exc_info:
        service.retrieve_documents("test query")
    
    assert "Retriever failed" in str(exc_info.value)
