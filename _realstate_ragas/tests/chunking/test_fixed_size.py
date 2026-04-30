"""
Tests Específicos para FIXED_SIZE
==================================

Tests que validan el comportamiento específico de la estrategia FIXED_SIZE.
Esta estrategia debe generar chunks uniformes de tamaño fijo con overlap mínimo.
"""

import pytest
from langchain.schema import Document


class TestFixedSize:
    """Tests específicos de la estrategia FIXED_SIZE"""
    
    def test_fixed_respects_max_size_strict(self, fixed_service, sample_text_long, create_document):
        """
        FIXED debe respetar chunk_size de manera más estricta que RECURSIVE.
        
        Este test lee el chunk_size configurado por el grupo y verifica
        que los chunks no excedan ese tamaño + 15% de tolerancia.
        """
        doc = create_document(sample_text_long)
        chunks = fixed_service.splitter.split_documents([doc])
        
        assert len(chunks) > 1, "Documento largo debe generar múltiples chunks"
        
        # Leer chunk_size de la configuración del grupo
        configured_chunk_size = fixed_service.splitter._chunk_size
        max_allowed = int(configured_chunk_size * 1.15)  # 15% tolerancia
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        max_size = max(chunk_sizes)
        
        # FIXED debe ser más uniforme, permitir hasta chunk_size + 15%
        assert max_size <= max_allowed, \
            f"FIXED_SIZE excedió límite: {max_size} chars (máximo esperado: {max_allowed} = {configured_chunk_size} + 15%)"
    
    def test_fixed_handles_missing_separator(self, fixed_service, create_document):
        """
        FIXED debe manejar correctamente texto sin el separador configurado.
        
        BUG CONOCIDO: Si CharacterTextSplitter se configura con separator="\\n\\n"
        y el texto NO contiene "\\n\\n", puede generar 1 chunk gigante.
        
        """
        # Texto largo SIN el separador "\n\n" (sin párrafos)
        text_without_separator = "Word " * 500  # 3000 chars aprox, sin \n\n
        doc = create_document(text_without_separator)
        
        chunks = fixed_service.splitter.split_documents([doc])
        
        # Debe generar múltiples chunks, NO un solo chunk gigante
        assert len(chunks) > 1, \
            "FIXED_SIZE debe generar múltiples chunks incluso sin el separador configurado. " \
            "Verifica que tu implementación tenga fallback (ej: separator=' ' o usar RecursiveCharacterTextSplitter)"
        
        # Verificar que ningún chunk es excesivamente grande
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        max_size = max(chunk_sizes)
        
        # Leer chunk_size de la configuración del grupo
        configured_chunk_size = fixed_service.splitter._chunk_size
        max_reasonable = int(configured_chunk_size * 1.4)  # 40% tolerancia para texto sin separador
        
        assert max_size <= max_reasonable, \
            f"FIXED_SIZE generó un chunk de {max_size} chars (máximo razonable: {max_reasonable} = {configured_chunk_size} + 40%). " \
            f"Esto indica que no encontró el separador y no tiene fallback."
    
    def test_fixed_overlap_exists(self, fixed_service, sample_text_long, create_document):
        """
        FIXED debe tener overlap configurado correctamente.
        
        Este test verifica que:
        1. El splitter tiene chunk_overlap configurado
        2. Los chunks tienen tamaño razonable
        3. Hay suficientes chunks generados (indica que el overlap está funcionando)
        """
        doc = create_document(sample_text_long)
        chunks = fixed_service.splitter.split_documents([doc])
        
        if len(chunks) < 2:
            pytest.skip("Se necesitan al menos 2 chunks para validar overlap")
        
        # Verificar que el splitter tiene chunk_overlap configurado
        splitter = fixed_service.splitter
        configured_chunk_size = splitter._chunk_size
        configured_overlap = splitter._chunk_overlap if hasattr(splitter, '_chunk_overlap') else 0
        
        # El overlap debe ser al menos 10% del chunk_size
        min_overlap = int(configured_chunk_size * 0.1)
        assert configured_overlap >= min_overlap, \
            f"chunk_overlap debe ser al menos {min_overlap} (10% de {configured_chunk_size}), pero es {configured_overlap}. " \
            f"Verifica tu implementación en _create_fixed_size_splitter()"
        
        # Verificar que genera múltiples chunks (indica que overlap está funcionando)
        # Calcular chunks esperados: texto de ~27K chars
        text_length = len(sample_text_long)
        expected_min_chunks = int(text_length / (configured_chunk_size + configured_overlap) * 0.5)
        
        assert len(chunks) >= expected_min_chunks, \
            f"Debe generar suficientes chunks con overlap configurado. " \
            f"Generó {len(chunks)}, esperado al menos {expected_min_chunks} " \
            f"(basado en chunk_size={configured_chunk_size}, overlap={configured_overlap}). " \
            f"Verifica que chunk_overlap esté configurado correctamente."
        
        # Verificar que chunks tienen tamaño razonable
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        
        # El tamaño promedio debe ser al menos 50% del chunk_size
        min_avg = int(configured_chunk_size * 0.5)
        assert avg_size >= min_avg, \
            f"Tamaño promedio de chunks es {avg_size:.0f} chars (muy pequeño, mínimo esperado: {min_avg}). " \
            f"Verifica que chunk_size y chunk_overlap estén configurados correctamente."
        
        # El tamaño promedio no debe exceder chunk_size + 20%
        max_avg = int(configured_chunk_size * 1.2)
        assert avg_size <= max_avg, \
            f"Tamaño promedio de chunks es {avg_size:.0f} chars (muy grande, máximo esperado: {max_avg}). " \
            f"Verifica que chunk_size esté configurado correctamente."

