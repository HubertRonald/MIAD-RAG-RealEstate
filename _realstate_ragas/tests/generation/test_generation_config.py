"""
Tests de Configuración del GenerationService
=============================================

Valida que el servicio se inicializa correctamente con LLM y prompt template.

"""

import pytest
from unittest.mock import Mock, patch
from app.services.generation_service import GenerationService


def test_generation_service_initialization():
    """
    Verifica que el servicio se inicializa correctamente.
    
    Valida:
    - LLM (ChatGoogleGenerativeAI) se crea con parámetros correctos
    - Prompt template se crea correctamente
    - Atributos llm y prompt están disponibles
    """
    # Crear servicio con parámetros personalizados
    service = GenerationService(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_tokens=512
    )
    
    # Verificar que el LLM se inicializó
    assert hasattr(service, 'llm'), "El servicio debe tener atributo 'llm'"
    assert service.llm is not None, "El LLM no debe ser None"
    
    # Verificar que el prompt template se inicializó
    assert hasattr(service, 'prompt'), "El servicio debe tener atributo 'prompt'"
    assert service.prompt is not None, "El prompt no debe ser None"
    
    # Verificar parámetros del LLM
    assert "gemini-2.5-flash" in service.llm.model, "El modelo debe contener gemini-2.5-flash"
    assert service.llm.temperature == 0.2, "La temperatura debe ser 0.2"
    assert service.llm.max_output_tokens == 512, "Los max_tokens deben ser 512"


def test_generation_service_prompt_template_structure():
    """
    Verifica que el prompt template tiene la estructura correcta para RAG.
    
    Valida:
    - El template contiene placeholders {question} y {context}
    - El template tiene instrucciones para responder basándose en contexto
    """
    # Crear servicio
    service = GenerationService()
    
    # Obtener el template string
    template_str = service.prompt.messages[0].prompt.template
    
    # Verificar que contiene los placeholders necesarios
    assert "{question}" in template_str, "El template debe contener placeholder {question}"
    assert "{context}" in template_str, "El template debe contener placeholder {context}"
    
    # Verificar que tiene instrucciones básicas para RAG
    template_lower = template_str.lower()
    assert any(keyword in template_lower for keyword in ["contexto", "context"]), \
        "El template debe mencionar 'contexto' o 'context'"
    assert any(keyword in template_lower for keyword in ["pregunta", "question"]), \
        "El template debe mencionar 'pregunta' o 'question'"

