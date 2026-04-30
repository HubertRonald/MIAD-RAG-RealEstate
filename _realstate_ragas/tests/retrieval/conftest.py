"""
Fixtures Compartidas para Tests de Retrieval
=============================================

Este módulo define fixtures que se comparten entre todas las pruebas
del servicio de retrieval.

ESTRATEGIA DE TESTING:
----------------------
1. Tests Unitarios: Usan MOCKS (sin API, sin vectorstore real)

2. Tests de Integración: Usan vectorstore REAL (con API)

"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
from langchain.schema import Document

# Agregar el directorio raíz al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================
# FIXTURES PARA TESTS UNITARIOS
# =============================

@pytest.fixture
def mock_documents():
    """
    Documentos controlados para tests con mock.
    
    Incluye 3 documentos sobre diferentes temas:
    - Machine Learning / AI
    - Python / Programming
    - Neural Networks
    
    Cada documento tiene metadata realista.
    """
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence focused on data-driven algorithms.",
            metadata={"source_file": "ai_textbook.pdf", "chunk_id": 0, "page": 1}
        ),
        Document(
            page_content="Python is a popular programming language widely used for data science and machine learning.",
            metadata={"source_file": "python_guide.pdf", "chunk_id": 1, "page": 5}
        ),
        Document(
            page_content="Neural networks are computational models inspired by biological neural networks in the brain.",
            metadata={"source_file": "deep_learning.pdf", "chunk_id": 2, "page": 12}
        ),
    ]


@pytest.fixture
def mock_vectorstore():
    """
    Mock del vectorstore FAISS.
    
    Este mock NO requiere un índice FAISS real.
    Los tests configuran el comportamiento del mock según necesiten.
    
    Métodos mockeados:
    - as_retriever(): Retorna un mock retriever
    """
    mock = Mock()
    mock.as_retriever = Mock()
    return mock


@pytest.fixture
def mock_embedding_service(mock_vectorstore):
    """
    Mock del EmbeddingService.
    
    Retorna un vectorstore mockeado cuando se llama a get_vectorstore().
    NO requiere API key ni hace llamadas reales.
    """
    mock_service = Mock()
    mock_service.get_vectorstore.return_value = mock_vectorstore
    return mock_service


@pytest.fixture
def configured_mock_retriever(mock_documents):
    """
    Mock retriever pre-configurado que retorna documentos controlados.
    
    Este fixture simula el comportamiento de un retriever real:
    - invoke(query) retorna documentos
    - Configurable para diferentes escenarios
    
    Returns:
        Mock: Retriever con invoke() configurado
    """
    mock_retriever = Mock()
    mock_retriever.invoke = Mock(return_value=mock_documents[:2])
    return mock_retriever


# ==================================
# FIXTURE PARA TESTS DE INTEGRACIÓN
# ==================================

@pytest.fixture(scope="session")
def real_vectorstore_cache(tmp_path_factory):
    """
    Vectorstore FAISS REAL para tests de integración.
    
    
    Estructura del Cache:
    ---------------------
    {
        "vectorstore": FAISS vectorstore object,
        "documents": List[Document] (documentos originales),
        "path": str (path al índice persistido),
        "embedding_service": EmbeddingService instance
    }
    
    Returns:
        dict: Cache con vectorstore y metadata
    """
    from app.services.embedding_service import EmbeddingService
    
    # Crear documentos de prueba (3 documentos pequeños)
    test_docs = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
            metadata={"source_file": "test_ai.txt", "chunk_id": 0, "topic": "AI"}
        ),
        Document(
            page_content="Python is a popular programming language for data science, known for its simplicity and powerful libraries like NumPy and Pandas.",
            metadata={"source_file": "test_python.txt", "chunk_id": 1, "topic": "Programming"}
        ),
        Document(
            page_content="Neural networks are computational models inspired by biological neural networks in the brain, used for deep learning applications.",
            metadata={"source_file": "test_nn.txt", "chunk_id": 2, "topic": "Deep Learning"}
        ),
    ]
    
    # Crear servicio de embeddings REAL
    try:
        embedding_service = EmbeddingService()
    except Exception as e:
        pytest.skip(f"No se pudo inicializar EmbeddingService: {e}. "
                   f"Verifica GOOGLE_API_KEY en .env")
    
    # Crear directorio temporal (persiste durante la sesión)
    temp_dir = tmp_path_factory.mktemp("retrieval_test")
    vectorstore_path = str(temp_dir / "test_vectorstore")

    
    # Construir vectorstore (CONSUME API AQUÍ)
    try:
        vectorstore = embedding_service.build_vectorstore(
            chunks=test_docs,
            persist_path=vectorstore_path
        )
    except Exception as e:
        pytest.skip(f"Error al construir vectorstore: {e}. "
                   f"Puede ser un problema de API key o límites de rate.")
    
    return {
        "vectorstore": vectorstore,
        "documents": test_docs,
        "path": vectorstore_path,
        "embedding_service": embedding_service
    }


# ========================
# CONFIGURACIÓN DE PYTEST
# ========================

def pytest_configure(config):
    """
    Configuración personalizada para pytest.
    
    Registra markers personalizados:
    - integration: Tests que usan vectorstore real (consumen API)
    """
    config.addinivalue_line(
        "markers",
        "integration: tests de integración con vectorstore real (consumen API, caché de sesión)"
    )



def pytest_collection_modifyitems(config, items):
    """
    Modifica items de la colección para marcar tests automáticamente.
    
    - Tests que usan real_vectorstore_cache se marcan como 'integration'
    """
    for item in items:
        if "real_vectorstore_cache" in item.fixturenames:
            item.add_marker(pytest.mark.integration)

