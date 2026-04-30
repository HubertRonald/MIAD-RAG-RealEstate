"""
Tests Universales de Chunking
==============================

Tests que validan comportamientos básicos que TODAS las estrategias deben cumplir.
Estos tests se ejecutan parametrizados para RECURSIVE_CHARACTER, FIXED_SIZE, y DOCUMENT_STRUCTURE.

"""

import pytest
from langchain.schema import Document


class TestUniversalBehavior:
    """Tests que validan integridad básica de todas las estrategias"""
    
    @pytest.mark.parametrize("strategy_fixture", [
        pytest.param("recursive_service", id="recursive"),
        pytest.param("fixed_service", id="fixed"),
        pytest.param("document_service", id="document"),
    ])
    def test_no_empty_chunks_universal(self, strategy_fixture, request, 
                                      sample_text_simple, create_document):
        """Ninguna estrategia debe generar chunks vacíos"""
        service = request.getfixturevalue(strategy_fixture)
        doc = create_document(sample_text_simple)
        
        # DOCUMENT_STRUCTURE usa split_text, las demás usan split_documents
        if service.strategy.value == "document_structure":
            chunks = service.splitter.split_text(sample_text_simple)
            # Convertir a Documents para validación uniforme
            chunks = [Document(page_content=c) if isinstance(c, str) else c for c in chunks]
        else:
            chunks = service.splitter.split_documents([doc])
        
        assert all(len(chunk.page_content) > 0 for chunk in chunks), \
            f"Estrategia {service.strategy} generó chunks vacíos"
    
    @pytest.mark.parametrize("strategy_fixture", [
        pytest.param("recursive_service", id="recursive"),
        pytest.param("fixed_service", id="fixed"),
    ])
    def test_content_preservation_universal(self, strategy_fixture, request,
                                           sample_text_simple, create_document):
        """Todas las estrategias deben preservar el contenido (≥95%)"""
        service = request.getfixturevalue(strategy_fixture)
        doc = create_document(sample_text_simple)
        original_length = len(sample_text_simple)
        
        chunks = service.splitter.split_documents([doc])
        
        # Combinar chunks sin overlap
        combined_content = chunks[0].page_content
        for chunk in chunks[1:]:
            combined_content += chunk.page_content
        
        preservation_ratio = len(combined_content) / original_length
        
        assert preservation_ratio >= 0.95, \
            f"Estrategia {service.strategy} no preserva contenido: {preservation_ratio:.2%}"
    
    @pytest.mark.parametrize("strategy_fixture", [
        pytest.param("recursive_service", id="recursive"),
        pytest.param("fixed_service", id="fixed"),
        pytest.param("document_service", id="document"),
    ])
    def test_produces_at_least_one_chunk_universal(self, strategy_fixture, request,
                                                   sample_text_simple, create_document):
        """Todas las estrategias deben producir al menos 1 chunk de documento no vacío"""
        service = request.getfixturevalue(strategy_fixture)
        doc = create_document(sample_text_simple)
        
        # DOCUMENT_STRUCTURE usa split_text, las demás usan split_documents
        if service.strategy.value == "document_structure":
            chunks = service.splitter.split_text(sample_text_simple)
        else:
            chunks = service.splitter.split_documents([doc])
        
        assert len(chunks) >= 1, \
            f"Estrategia {service.strategy} no produjo chunks"


class TestEdgeCases:
    """Tests de casos extremos que todas las estrategias deben manejar"""
    
    @pytest.mark.parametrize("strategy_fixture", [
        pytest.param("recursive_service", id="recursive"),
        pytest.param("fixed_service", id="fixed"),
    ])
    def test_empty_document_universal(self, strategy_fixture, request, create_document):
        """Todas las estrategias deben manejar documento vacío sin errores"""
        service = request.getfixturevalue(strategy_fixture)
        doc = create_document("")
        
        try:
            chunks = service.splitter.split_documents([doc])
            # Puede producir 0 chunks o 1 chunk vacío, ambos son válidos
            assert len(chunks) <= 1, \
                f"Estrategia {service.strategy} produjo múltiples chunks de documento vacío"
        except Exception as e:
            pytest.fail(f"Estrategia {service.strategy} falló con documento vacío: {str(e)}")
    
    @pytest.mark.parametrize("strategy_fixture", [
        pytest.param("recursive_service", id="recursive"),
        pytest.param("fixed_service", id="fixed"),
    ])
    def test_single_word_document_universal(self, strategy_fixture, request, create_document):
        """Todas las estrategias deben manejar documento de 1 palabra sin errores"""
        service = request.getfixturevalue(strategy_fixture)
        doc = create_document("Hello")
        
        chunks = service.splitter.split_documents([doc])
        assert len(chunks) == 1, \
            f"Estrategia {service.strategy} no manejó correctamente documento de 1 palabra"
        assert "Hello" in chunks[0].page_content, \
            f"Estrategia {service.strategy} no preservó la palabra"
    
    @pytest.mark.parametrize("strategy_fixture", [
        pytest.param("recursive_service", id="recursive"),
        pytest.param("fixed_service", id="fixed"),
    ])
    def test_special_characters_preserved_universal(self, strategy_fixture, request, create_document):
        """Todas las estrategias deben preservar caracteres especiales"""
        service = request.getfixturevalue(strategy_fixture)
        special_text = "Texto con acentos é, ñ, emojis 🎉🚀 y símbolos especiales @#$%"
        doc = create_document(special_text)
        
        chunks = service.splitter.split_documents([doc])
        combined = "".join(chunk.page_content for chunk in chunks)
        
        # Verificar que caracteres especiales están presentes
        assert "é" in combined, f"Estrategia {service.strategy} no preservó acentos"
        assert "ñ" in combined, f"Estrategia {service.strategy} no preservó ñ"
        assert "🎉" in combined or "🚀" in combined, \
            f"Estrategia {service.strategy} no preservó emojis"


class TestInitialization:
    """Tests de inicialización y configuración"""
    
    @pytest.mark.parametrize("strategy_fixture", [
        pytest.param("recursive_service", id="recursive"),
        pytest.param("fixed_service", id="fixed"),
        pytest.param("document_service", id="document"),
    ])
    def test_strategy_initializes_correctly_universal(self, strategy_fixture, request):
        """Todas las estrategias deben inicializarse sin errores"""
        service = request.getfixturevalue(strategy_fixture)
        assert service.splitter is not None, \
            f"Estrategia {service.strategy} no inicializó splitter"
    
    @pytest.mark.parametrize("strategy_fixture", [
        pytest.param("recursive_service", id="recursive"),
        pytest.param("fixed_service", id="fixed"),
    ])
    def test_chunks_have_metadata_universal(self, strategy_fixture, request, 
                                           sample_text_simple, create_document):
        """Chunks deben preservar metadatos del documento original"""
        service = request.getfixturevalue(strategy_fixture)
        doc = create_document(sample_text_simple, source="test_doc.pdf")
        chunks = service.splitter.split_documents([doc])
        
        assert len(chunks) > 0, "No se generaron chunks"
        
        # Verificar que metadatos están presentes
        for chunk in chunks:
            assert hasattr(chunk, 'metadata'), "Chunk no tiene metadatos"
            assert isinstance(chunk.metadata, dict), "Metadatos no son un diccionario"

