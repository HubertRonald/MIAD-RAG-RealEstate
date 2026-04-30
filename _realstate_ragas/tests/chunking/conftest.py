"""
Fixtures Compartidas para Tests de Chunking
==============================================

Este módulo define fixtures que se comparten entre todas las pruebas
del servicio de chunking.
"""

import pytest
import sys
from pathlib import Path
from langchain.schema import Document

# Agregar el directorio raíz al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.chunking_service import ChunkingService, ChunkingStrategy, SEMANTIC_EMBEDDINGS_MODEL


def pytest_configure(config):
    """Registrar markers personalizados"""
    config.addinivalue_line(
        "markers", 
        "semantic_api: tests que consumen API de LLM (embeddings) - requieren confirmación"
    )


@pytest.fixture
def sample_text_simple():
    """Texto simple para tests básicos"""
    return """Introduction to Software Architecture

Software architecture represents fundamental structures.
It comprises elements, relations, and properties.
Good architecture enables maintainability and scalability.

Design Patterns

Patterns are reusable solutions to common problems.
They provide templates for solving recurring design issues.
Examples include Singleton, Factory, and Observer patterns.

Conclusion

Understanding architecture is crucial for building robust systems."""


@pytest.fixture
def sample_text_long():
    """Texto largo para tests de múltiples chunks"""
    return "This is a repeated sentence for testing long documents. " * 500


@pytest.fixture
def sample_markdown():
    """Documento Markdown estructurado para DOCUMENT_STRUCTURE"""
    return """# Main Title

## Section 1: Introduction

This is the introduction paragraph with important information.
It contains multiple sentences to test chunking behavior.

## Section 2: Details

### Subsection 2.1

More detailed information here.
This tests nested header structures.

### Subsection 2.2

Final subsection with concluding remarks.

## Section 3: Conclusion

Final thoughts and summary."""


@pytest.fixture
def sample_semantic_text():
    """Texto con cambios semánticos claros para tests de SEMANTIC"""
    return """Artificial Intelligence has revolutionized modern computing.
Machine learning algorithms can now process vast amounts of data.
Neural networks have become increasingly sophisticated over the years.

Climate change poses significant challenges to our planet.
Rising temperatures are affecting ecosystems worldwide.
Scientists are working on solutions to reduce carbon emissions.

The history of ancient Rome spans over a millennium.
Roman emperors ruled vast territories across Europe and Africa.
The fall of Rome marked the end of classical antiquity."""


@pytest.fixture
def create_document():
    """Factory para crear documentos de LangChain"""
    def _create(content: str, source: str = "test.txt"):
        return Document(
            page_content=content,
            metadata={
                "source_file": source,
                "source_path": f"./docs/test/{source}",
            }
        )
    return _create


# Fixtures para servicios de chunking
@pytest.fixture
def recursive_service():
    """Servicio con estrategia RECURSIVE_CHARACTER"""
    try:
        return ChunkingService(strategy=ChunkingStrategy.RECURSIVE_CHARACTER)
    except Exception as e:
        pytest.fail(f"Error al inicializar RECURSIVE_CHARACTER: {str(e)}. "
                   f"Verifica tu implementación en chunking_service.py")


@pytest.fixture
def fixed_service():
    """Servicio con estrategia FIXED_SIZE"""
    try:
        return ChunkingService(strategy=ChunkingStrategy.FIXED_SIZE)
    except Exception as e:
        pytest.fail(f"Error al inicializar FIXED_SIZE: {str(e)}. "
                   f"Verifica tu implementación en chunking_service.py")


@pytest.fixture
def document_service():
    """Servicio con estrategia DOCUMENT_STRUCTURE"""
    try:
        return ChunkingService(strategy=ChunkingStrategy.DOCUMENT_STRUCTURE)
    except Exception as e:
        pytest.fail(f"Error al inicializar DOCUMENT_STRUCTURE: {str(e)}. "
                   f"Verifica tu implementación en chunking_service.py")


@pytest.fixture
def semantic_service():
    """
    Servicio con estrategia SEMANTIC.
    
    IMPORTANTE: Este fixture usa el modelo configurado en chunking_service.py
    Los tests leen la constante SEMANTIC_EMBEDDINGS_MODEL automáticamente.
    """
    try:
        print(f"\n[SEMANTIC CHUNKING] Modelo: {SEMANTIC_EMBEDDINGS_MODEL}")
        
        return ChunkingService(strategy=ChunkingStrategy.SEMANTIC)
    except ValueError as e:
        error_msg = str(e).lower()
        if "api" in error_msg and "key" in error_msg:
            pytest.skip(f"API key no configurada: {e}")
        raise
    except Exception as e:
        pytest.fail(f"Error al inicializar SEMANTIC: {str(e)}. "
                   f"Verifica tu implementación en chunking_service.py")


@pytest.fixture(scope="module")
def semantic_chunks_cached():
    """
    Genera chunks semánticos UNA SOLA VEZ y los cachea para todos los tests.
    
    El scope="module" significa que este fixture se ejecuta una vez por archivo de test,
    y el resultado se reutiliza en todos los tests del archivo.
    
    Returns:
        dict: Diccionario con chunks, texto original, y servicio
    """
    try:
        # Inicializar servicio
        service = ChunkingService(strategy=ChunkingStrategy.SEMANTIC)
        
        # Texto de prueba con 3 temas distintos
        text = """Artificial Intelligence has revolutionized modern computing.
Machine learning algorithms can now process vast amounts of data.
Neural networks have become increasingly sophisticated over the years.

Climate change poses significant challenges to our planet.
Rising temperatures are affecting ecosystems worldwide.
Scientists are working on solutions to reduce carbon emissions.

The history of ancient Rome spans over a millennium.
Roman emperors ruled vast territories across Europe and Africa.
The fall of Rome marked the end of classical antiquity."""
        
        # Crear documento
        from langchain.schema import Document
        doc = Document(
            page_content=text,
            metadata={
                "source_file": "test_semantic.txt",
                "source_path": "./docs/test/test_semantic.txt",
            }
        )
        
        # Generar chunks (ÚNICA llamada API)
        print("\nGenerando chunks semanticos (1 llamada API)...")
        chunks = service.splitter.split_documents([doc])
        print(f"Chunks generados: {len(chunks)} chunks")
        
        return {
            "chunks": chunks,
            "original_text": text,
            "service": service,
            "document": doc
        }
        
    except ValueError as e:
        if "GOOGLE_API_KEY" in str(e):
            pytest.skip("GOOGLE_API_KEY no configurada. Configura tu .env para ejecutar tests de SEMANTIC.")
        raise
    except Exception as e:
        pytest.fail(f"Error al generar chunks semánticos: {str(e)}")
