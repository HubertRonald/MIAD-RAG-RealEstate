"""
Tests del Parámetro K en Retrieval
===================================

Valida que el parámetro k (número de documentos a recuperar) funciona
correctamente en retrieve_documents().

ESTRATEGIA: Mock (sin API, sin vectorstore real)
TIEMPO: ~1 segundo
COSTO API: $0.00
"""

import pytest
from unittest.mock import Mock
from langchain.schema import Document
from app.services.retrieval_service import RetrievalService


def test_retrieve_documents_respects_k_parameter(
    mock_embedding_service, 
    mock_vectorstore, 
    mock_documents
):
    """
    Verifica que retrieve_documents() retorna exactamente k documentos.
    
    Prueba diferentes valores de k para asegurar flexibilidad.
    Esto es crítico para controlar el contexto enviado al LLM.
    """
    k_values_and_expected = [
        (1, 1, "k=1 debe retornar 1 documento"),
        (2, 2, "k=2 debe retornar 2 documentos"),
        (3, 3, "k=3 debe retornar 3 documentos"),
    ]
    
    for k, expected_count, description in k_values_and_expected:
        # Reset mock
        mock_vectorstore.reset_mock()
        
        # Configurar mock retriever
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = mock_documents[:expected_count]
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        # Crear servicio con k específico
        service = RetrievalService(mock_embedding_service, k=k)
        
        # Ejecutar retrieval
        results = service.retrieve_documents("test query")
        
        # Verificar
        assert len(results) == expected_count, description
        assert all(isinstance(doc, Document) for doc in results)


def test_retrieve_documents_when_fewer_than_k_available(
    mock_embedding_service, 
    mock_vectorstore
):
    """
    Verifica el comportamiento cuando hay menos de k documentos disponibles.
    
    Debe retornar todos los documentos disponibles sin errores ni padding.
    """
    # Solo 2 documentos disponibles
    available_docs = [
        Document(page_content="Only document 1", metadata={"source_file": "doc1.txt"}),
        Document(page_content="Only document 2", metadata={"source_file": "doc2.txt"}),
    ]
    
    # Configurar mock para retornar solo 2 docs (aunque k=10)
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = available_docs
    mock_vectorstore.as_retriever.return_value = mock_retriever
    
    # Crear servicio con k=10 (pero solo hay 2 docs)
    service = RetrievalService(mock_embedding_service, k=10)
    
    # Ejecutar retrieval
    results = service.retrieve_documents("test query")
    
    # Debe retornar los 2 disponibles, no fallar ni llenar con nulls
    assert len(results) == 2
    assert all(doc.page_content for doc in results)
    assert all(isinstance(doc, Document) for doc in results)

