"""
Tests de Configuración del Retriever
=====================================

Valida que el método _build_retriever() de RetrievalService configura
correctamente el retriever con los parámetros esperados.

ESTRATEGIA: Mock (sin API, sin vectorstore real)
TIEMPO: ~1 segundo
COSTO API: $0.00
"""

import pytest
from unittest.mock import Mock
from app.services.retrieval_service import RetrievalService


def test_retriever_configured_with_similarity_search(mock_embedding_service, mock_vectorstore):
    """
    Verifica que _build_retriever() configura el retriever correctamente.
    
    Valida:
    - Inicialización correcta con embedding_service y k
    - as_retriever() se llama con search_type="similarity"
    - search_kwargs contiene {"k": k}
    - El retriever está listo para usar
    """
    # Configurar mock retriever
    mock_retriever = Mock()
    mock_vectorstore.as_retriever.return_value = mock_retriever
    
    # Crear servicio con k=5
    service = RetrievalService(mock_embedding_service, k=5)
    
    # Verificar configuración del retriever
    mock_vectorstore.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Verificar atributos del servicio
    assert service.k == 5
    assert service.vectorstore == mock_vectorstore
    assert service.retriever == mock_retriever
    assert service.embedding_service == mock_embedding_service


def test_retriever_initialization_fails_with_none_vectorstore(mock_embedding_service):
    """
    Verifica que el servicio falla si el vectorstore es None.
    
    Esto previene errores silenciosos cuando el vectorstore no está inicializado.
    """
    # Configurar mock para retornar None
    mock_embedding_service.get_vectorstore.return_value = None
    
    # Debe lanzar ValueError
    with pytest.raises(ValueError) as exc_info:
        RetrievalService(mock_embedding_service, k=3)
    
    # Verificar mensaje de error
    assert "debe tener un vectorstore inicializado" in str(exc_info.value).lower()

