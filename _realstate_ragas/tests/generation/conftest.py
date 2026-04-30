"""
Fixtures Compartidas para Tests de Generation
==============================================

Este módulo define fixtures que se comparten entre todas las pruebas
del servicio de generación.

ESTRATEGIA DE TESTING:
----------------------
1. Tests Unitarios: Usan MOCKS (sin API, sin LLM real)

2. Tests de Integración: Usan LLM REAL (con API)
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
    Documentos de prueba con contenido y metadata.
    
    Simula documentos recuperados por el retrieval service.
    """
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
            metadata={"source_file": "ai_guide.pdf", "chunk_id": 0, "page": 1}
        ),
        Document(
            page_content="Python is a popular programming language widely used in data science and machine learning.",
            metadata={"source_file": "python_handbook.pdf", "chunk_id": 1, "page": 5}
        ),
        Document(
            page_content="Neural networks are computational models inspired by the human brain, used in deep learning.",
            metadata={"source_file": "deep_learning.pdf", "chunk_id": 2, "page": 12}
        ),
    ]


@pytest.fixture
def mock_llm_response():
    """
    Mock de respuesta del LLM.
    
    Simula la respuesta que retornaría ChatGoogleGenerativeAI.
    """
    mock_response = Mock()
    mock_response.content = "Machine learning is a subset of AI that enables systems to learn from data without explicit programming."
    return mock_response


@pytest.fixture
def mock_llm(mock_llm_response):
    """
    Mock del LLM (ChatGoogleGenerativeAI).
    
    Retorna una respuesta controlada sin hacer llamadas API reales.
    """
    mock = Mock()
    mock.invoke = Mock(return_value=mock_llm_response)
    return mock


@pytest.fixture
def mock_prompt_template():
    """
    Mock del ChatPromptTemplate.
    
    Simula el comportamiento del prompt template sin procesamiento real.
    """
    mock_template = Mock()
    mock_template.format = Mock(return_value="Formatted prompt")
    return mock_template


@pytest.fixture
def sample_questions():
    """
    Preguntas de ejemplo para tests.
    
    Incluye diferentes tipos de preguntas y casos de uso.
    """
    return {
        "simple": "What is machine learning?",
        "spanish": "¿Qué es el aprendizaje automático?",
        "long": "Can you explain in detail how machine learning works, what algorithms are commonly used, and what are the main applications in the industry?",
        "special_chars": "What is ML? 🤖 How does it work? 🚀",
        "empty": "",
    }


# ==================================
# FIXTURES PARA TESTS DE INTEGRACIÓN
# ==================================

@pytest.fixture(scope="session")
def real_generation_service():
    """
    Servicio de generación REAL para tests de integración.
    
    ADVERTENCIA:
    ------------
    Este fixture consume API. Úsalo solo en tests marcados con @pytest.mark.integration
    
    Returns:
        GenerationService: Servicio configurado con LLM real
    """
    from app.services.generation_service import GenerationService
    
    
    try:
        service = GenerationService(
            model="gemini-2.5-flash",
            temperature=0.1,
            max_tokens=1024
        )
        
        return service
        
    except Exception as e:
        pytest.skip(f"No se pudo inicializar GenerationService: {e}. "
                   f"Verifica GOOGLE_API_KEY en .env")


# ========================
# CONFIGURACIÓN DE PYTEST
# ========================

def pytest_configure(config):
    """
    Configuración personalizada para pytest.
    
    Registra markers personalizados:
    - integration: Tests que usan LLM real (consumen API)
    """
    config.addinivalue_line(
        "markers",
        "integration: tests de integración con LLM real (consumen API)"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modifica items de la colección para marcar tests automáticamente.
    
    - Tests que usan real_generation_service se marcan como 'integration'
    """
    for item in items:
        if "real_generation_service" in item.fixturenames:
            item.add_marker(pytest.mark.integration)

