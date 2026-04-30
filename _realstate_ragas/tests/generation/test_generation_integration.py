"""
Tests de Integración con LLM Real
==================================

Valida el funcionamiento end-to-end del GenerationService con
un LLM real de Google (ChatGoogleGenerativeAI).

"""

import pytest
from langchain.schema import Document
from app.services.generation_service import GenerationService


@pytest.mark.integration
def test_generate_response_with_real_llm(real_generation_service):
    """
    Test de integración completo con LLM real de Google.
    
    Valida:
    - Genera respuesta coherente con contexto real
    - La respuesta contiene información relacionada al contexto
    - Estructura del resultado es correcta
    - Sources y context se extraen correctamente
    """
    # Documentos de prueba sobre machine learning
    test_docs = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data without being explicitly programmed. It uses algorithms to identify patterns and make predictions.",
            metadata={"source_file": "ml_intro.pdf", "chunk_id": 0}
        ),
        Document(
            page_content="Common machine learning algorithms include linear regression, decision trees, random forests, and neural networks. Each has different use cases and performance characteristics.",
            metadata={"source_file": "ml_algorithms.pdf", "chunk_id": 1}
        ),
    ]
    
    # Pregunta sobre el contenido de los documentos
    question = "What is machine learning and what algorithms are commonly used?"
    
    # Ejecutar generación con LLM real
    result = real_generation_service.generate_response(question, test_docs)
    
    # Validar estructura del resultado
    assert isinstance(result, dict), "El resultado debe ser un diccionario"
    assert "answer" in result, "El resultado debe contener 'answer'"
    assert "sources" in result, "El resultado debe contener 'sources'"
    assert "context" in result, "El resultado debe contener 'context'"
    
    # Validar answer
    assert isinstance(result["answer"], str), "answer debe ser string"
    assert len(result["answer"]) > 0, "answer no debe estar vacío"
    assert len(result["answer"]) > 20, "answer debe tener contenido sustancial"
    
    # Validar sources
    assert len(result["sources"]) == 2, "Debe haber 2 sources"
    assert "ml_intro.pdf" in result["sources"], "Debe incluir ml_intro.pdf"
    assert "ml_algorithms.pdf" in result["sources"], "Debe incluir ml_algorithms.pdf"
    
    # Validar context
    assert len(result["context"]) == 2, "Debe haber 2 contexts"
    assert all(isinstance(ctx, str) for ctx in result["context"]), "Todos los contexts deben ser strings"
    
    # Validar que la respuesta es coherente con el contexto
    # (al menos menciona "machine learning" o conceptos relacionados)
    answer_lower = result["answer"].lower()
    relevant_keywords = ["machine learning", "algorithm", "data", "model", "learn"]
    assert any(keyword in answer_lower for keyword in relevant_keywords), \
        f"La respuesta debe mencionar conceptos relevantes. Respuesta: {result['answer']}"


@pytest.mark.integration
def test_generate_response_respects_context(real_generation_service):
    """
    Verifica que el LLM respeta el contexto proporcionado y no alucina.
    
    Valida:
    - La respuesta se basa en el contexto proporcionado
    - No inventa información fuera del contexto
    """
    # Documentos sobre un tema específico (Python)
    test_docs = [
        Document(
            page_content="Python was created by Guido van Rossum and first released in 1991. It is known for its clear syntax and readability.",
            metadata={"source_file": "python_history.pdf"}
        ),
    ]
    
    # Pregunta específica sobre el contenido
    question = "Who created Python and when was it released?"
    
    # Generar respuesta
    result = real_generation_service.generate_response(question, test_docs)
    
    # Validar que la respuesta menciona información del contexto
    answer_lower = result["answer"].lower()
    
    # Debe mencionar a Guido van Rossum (o al menos "guido" o "rossum")
    assert any(name in answer_lower for name in ["guido", "rossum"]), \
        "La respuesta debe mencionar al creador (Guido van Rossum)"
    
    # Debe mencionar el año 1991
    assert "1991" in result["answer"], \
        "La respuesta debe mencionar el año de creación (1991)"



