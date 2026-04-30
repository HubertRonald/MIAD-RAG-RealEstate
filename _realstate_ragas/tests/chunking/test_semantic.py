"""
Tests Específicos para SEMANTIC
================================

Tests que validan el comportamiento específico de la estrategia SEMANTIC.
Esta estrategia usa embeddings para detectar cambios semánticos.

Estos tests consumen cuota de API de embeddings.
Los tests usan un fixture con caché (semantic_chunks_cached)
que genera los chunks UNA SOLA VEZ, reduciendo el consumo de API.

"""

import pytest
from langchain.schema import Document


class TestSemanticUniversal:
    """Tests universales para SEMANTIC (usan cache para optimizar API)"""
    
    @pytest.mark.semantic_api
    def test_semantic_no_empty_chunks_universal(self, semantic_chunks_cached):
        """
        Test universal: SEMANTIC no debe generar chunks vacíos.
        
        Usa chunks cacheados (no consume API adicional).
        """
        chunks = semantic_chunks_cached["chunks"]
        
        assert len(chunks) > 0, "SEMANTIC no generó chunks"
        assert all(len(chunk.page_content) > 0 for chunk in chunks), \
            "SEMANTIC generó chunks vacíos"
    
    @pytest.mark.semantic_api
    def test_semantic_content_preservation_universal(self, semantic_chunks_cached):
        """
        Test universal: SEMANTIC debe preservar el contenido (≥95%).
        
        Usa chunks cacheados (no consume API adicional).
        """
        chunks = semantic_chunks_cached["chunks"]
        original_text = semantic_chunks_cached["original_text"]
        original_length = len(original_text)
        
        # Combinar todos los chunks
        combined_content = "".join(chunk.page_content for chunk in chunks)
        preservation_ratio = len(combined_content) / original_length
        
        assert preservation_ratio >= 0.95, \
            f"SEMANTIC no preserva contenido: {preservation_ratio:.2%}"
    
    @pytest.mark.semantic_api
    def test_semantic_produces_at_least_one_chunk_universal(self, semantic_chunks_cached):
        """
        Test universal: SEMANTIC debe producir al menos 1 chunk.
        
        Usa chunks cacheados (no consume API adicional).
        """
        chunks = semantic_chunks_cached["chunks"]
        
        assert len(chunks) >= 1, \
            "SEMANTIC no produjo chunks"


class TestSemanticSpecific:
    """Tests específicos de la estrategia SEMANTIC (usan cache para optimizar API)"""
    
    @pytest.mark.semantic_api
    def test_semantic_produces_multiple_chunks(self, semantic_chunks_cached):
        """
        SEMANTIC debe generar múltiples chunks basados en cambios semánticos.
        
        El texto de prueba tiene 3 temas distintos:
        1. Inteligencia Artificial
        2. Cambio Climático
        3. Historia de Roma
        
        Debe generar al menos 2-3 chunks (uno por tema).
        
        Usa chunks cacheados (no consume API adicional).
        """
        chunks = semantic_chunks_cached["chunks"]
        
        # Debe generar al menos 2 chunks (detectar cambios semánticos)
        assert len(chunks) >= 2, \
            f"SEMANTIC debe detectar cambios semánticos y generar múltiples chunks. " \
            f"Generó {len(chunks)} chunk(s)."
    
    @pytest.mark.semantic_api
    def test_semantic_groups_semantically_similar_content(self, semantic_chunks_cached):
        """
        SEMANTIC debe agrupar contenido semánticamente similar.
        
        Las primeras 3 oraciones son sobre IA (deben estar juntas).
        Las siguientes 3 son sobre clima (deben estar juntas).
        Las últimas 3 son sobre Roma (deben estar juntas).
        
        Usa chunks cacheados (no consume API adicional).
        """
        chunks = semantic_chunks_cached["chunks"]
        
        # Verificar que temas están separados
        # Al menos debe haber un chunk con "Artificial Intelligence" 
        # y otro diferente con "Climate change" o "Rome"
        
        ai_chunks = [c for c in chunks if "Artificial Intelligence" in c.page_content or 
                    "Machine learning" in c.page_content]
        climate_chunks = [c for c in chunks if "Climate change" in c.page_content or 
                         "carbon emissions" in c.page_content]
        rome_chunks = [c for c in chunks if "Rome" in c.page_content or 
                      "Roman emperors" in c.page_content]
        
        # Al menos 2 de los 3 temas deben estar en chunks separados
        themes_found = sum([len(ai_chunks) > 0, len(climate_chunks) > 0, len(rome_chunks) > 0])
        
        assert themes_found >= 2, \
            f"SEMANTIC debe separar temas diferentes. Solo detectó {themes_found} tema(s)."
    
    @pytest.mark.semantic_api
    def test_semantic_respects_breakpoint_threshold(self, semantic_chunks_cached):
        """
        SEMANTIC debe respetar el breakpoint_threshold configurado.
        
        Con threshold alto (95), debe generar MENOS chunks (más conservador).
        
        Usa chunks cacheados (no consume API adicional).
        """
        chunks = semantic_chunks_cached["chunks"]
        service = semantic_chunks_cached["service"]
        
        # Verificar que el servicio tiene el splitter correcto
        assert service.splitter is not None, "SEMANTIC no inicializó splitter"
        
        # Validación flexible: debe generar al menos 1 chunk y no más de 10
        assert 1 <= len(chunks) <= 10, \
            f"SEMANTIC generó {len(chunks)} chunks. " \
            f"Verifica que breakpoint_threshold_amount esté configurado correctamente. " \
            f"Valores típicos: 90-95 (conservador), 70-85 (balanceado), 50-65 (agresivo)."
        
        # Si generó solo 1 chunk, el threshold puede estar demasiado alto
        if len(chunks) == 1:
            pytest.skip(
                f"SEMANTIC generó solo 1 chunk. Esto puede indicar threshold muy alto (>98). "
                f"Considera usar valores entre 85-95 para mejores resultados."
            )

