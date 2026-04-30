"""
Tests de Lógica de Recuperación de Documentos
==============================================

Valida el comportamiento del método retrieve_documents() y la interacción
correcta con el retriever usando invoke().

ESTRATEGIA: Mock (sin API, sin vectorstore real)
TIEMPO: ~1 segundo
COSTO API: $0.00
"""

import pytest
from unittest.mock import Mock
from langchain.schema import Document
from app.services.retrieval_service import RetrievalService


def test_retrieve_documents_calls_invoke_and_returns_documents(
    mock_embedding_service, 
    mock_vectorstore, 
    mock_documents
):
    """
    Verifica que retrieve_documents() funciona correctamente.
    
    Valida:
    - Llama a retriever.invoke() con el query
    - Retorna List[Document]
    - Preserva contenido y metadata de documentos
    """
    # Configurar mock retriever
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = mock_documents
    mock_vectorstore.as_retriever.return_value = mock_retriever
    
    # Crear servicio
    service = RetrievalService(mock_embedding_service, k=3)
    
    # Ejecutar retrieval
    query = "What is machine learning?"
    results = service.retrieve_documents(query)
    
    # Verificar que invoke fue llamado correctamente
    mock_retriever.invoke.assert_called_once_with(query)
    
    # Verificar tipo de retorno
    assert isinstance(results, list)
    assert len(results) == len(mock_documents)
    assert all(isinstance(doc, Document) for doc in results)
    
    # Verificar que contenido y metadata se preservan
    for original, retrieved in zip(mock_documents, results):
        assert retrieved.page_content == original.page_content
        assert retrieved.metadata == original.metadata


def test_retrieve_documents_with_edge_cases(
    mock_embedding_service, 
    mock_vectorstore
):
    """
    Verifica que retrieve_documents() maneja casos extremos correctamente.
    
    Incluye:
    - Query vacío
    - Sin resultados
    - Documentos con metadata vacía
    """
    # Documentos con metadata mínima
    docs_minimal = [
        Document(page_content="Doc without full metadata", metadata={})
    ]
    
    # Configurar mock retriever
    mock_retriever = Mock()
    mock_vectorstore.as_retriever.return_value = mock_retriever
    
    # Crear servicio
    service = RetrievalService(mock_embedding_service, k=3)
    
    # Test 1: Query vacío retorna lista
    mock_retriever.invoke.return_value = []
    results = service.retrieve_documents("")
    assert isinstance(results, list)
    assert len(results) == 0
    
    # Test 2: Sin resultados retorna lista vacía
    mock_retriever.invoke.return_value = []
    results = service.retrieve_documents("nonexistent topic")
    assert results == []
    
    # Test 3: Documentos con metadata vacía no causan error
    mock_retriever.invoke.return_value = docs_minimal
    results = service.retrieve_documents("test")
    assert len(results) == 1
    assert hasattr(results[0], 'metadata')

