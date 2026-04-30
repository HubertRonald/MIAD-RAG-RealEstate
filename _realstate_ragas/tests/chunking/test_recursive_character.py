"""
Tests Específicos para RECURSIVE_CHARACTER
===========================================

Tests que validan el comportamiento específico de la estrategia RECURSIVE_CHARACTER.
Esta estrategia debe respetar límites naturales del texto (párrafos, oraciones)
mientras mantiene chunks cerca del tamaño objetivo.
"""

import pytest
import inspect
from langchain.schema import Document


class TestRecursiveCharacter:
    """Tests específicos de la estrategia RECURSIVE_CHARACTER"""
    
    def test_recursive_respects_max_size_with_tolerance(self, recursive_service, 
                                                        sample_text_long, create_document):
        """
        RECURSIVE debe respetar chunk_size con tolerancia del 10%.
        
        Este test lee el chunk_size configurado por el grupo y verifica
        que los chunks no excedan ese tamaño + 10% de tolerancia.
        """
        doc = create_document(sample_text_long)
        chunks = recursive_service.splitter.split_documents([doc])
        
        assert len(chunks) > 1, "Documento largo debe generar múltiples chunks"
        
        # Leer chunk_size de la configuración del grupo
        configured_chunk_size = recursive_service.splitter._chunk_size
        max_allowed = int(configured_chunk_size * 1.10)  # 10% tolerancia
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        max_size = max(chunk_sizes)
        
        # Verificar que ningún chunk excede chunk_size + 10%
        assert max_size <= max_allowed, \
            f"RECURSIVE_CHARACTER excedió límite: {max_size} chars (máximo esperado: {max_allowed} = {configured_chunk_size} + 10%)"
    
    def test_recursive_overlap_exists(self, recursive_service, sample_text_long, create_document):
        """
        RECURSIVE debe tener overlap configurado correctamente.
        
        """
        doc = create_document(sample_text_long)
        chunks = recursive_service.splitter.split_documents([doc])
        
        if len(chunks) < 2:
            pytest.skip("Se necesitan al menos 2 chunks para validar overlap")
        
        # Verificar que el splitter tiene chunk_overlap configurado
        splitter = recursive_service.splitter
        configured_chunk_size = splitter._chunk_size
        configured_overlap = splitter._chunk_overlap if hasattr(splitter, '_chunk_overlap') else 0
        
        # El overlap debe ser al menos 20% del chunk_size
        min_overlap = int(configured_chunk_size * 0.2)
        assert configured_overlap >= min_overlap, \
            f"chunk_overlap debe ser al menos {min_overlap} (20% de {configured_chunk_size}), pero es {configured_overlap}. " \
            f"Verifica tu implementación en _create_recursive_character_splitter()"
        
        # Verificar que genera múltiples chunks (indica que overlap está funcionando)
        # Calcular chunks esperados: texto de ~27K chars
        text_length = len(sample_text_long)
        expected_min_chunks = int(text_length / (configured_chunk_size + configured_overlap) * 0.5)
        
        assert len(chunks) >= expected_min_chunks, \
            f"Debe generar suficientes chunks con overlap configurado. " \
            f"Generó {len(chunks)}, esperado al menos {expected_min_chunks} " \
            f"(basado en chunk_size={configured_chunk_size}, overlap={configured_overlap}). " \
            f"Verifica que chunk_overlap esté configurado correctamente."
        
        # Verificar que chunks tienen tamaño razonable (no son excesivamente pequeños)
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        
        # El tamaño promedio debe ser al menos 40% del chunk_size
        min_avg = int(configured_chunk_size * 0.4)
        assert avg_size >= min_avg, \
            f"Tamaño promedio de chunks es {avg_size:.0f} chars (muy pequeño, mínimo esperado: {min_avg}). " \
            f"Verifica que chunk_size y chunk_overlap estén configurados correctamente."
    
    def test_recursive_prefers_user_breaks(self, recursive_service, sample_text_simple, create_document):
        """
        RECURSIVE debe preferir dividir por separadores definidos (párrafos, oraciones).
        
        """
        # Verificar que el grupo configuró separadores
        source_code = inspect.getsource(recursive_service._create_recursive_character_splitter)
        
        assert "separators" in source_code, \
            "La implementación de _create_recursive_character_splitter debe incluir 'separators'"
        
        # Verificar que tiene separadores comunes
        assert "\\n\\n" in source_code or '\\n\\n' in source_code, \
            "Los separadores deben incluir '\\n\\n' para párrafos"
        
        # Probar comportamiento con documento real
        doc = create_document(sample_text_simple)
        chunks = recursive_service.splitter.split_documents([doc])
        
        # Verificar que chunks respetan límites naturales
        for i, chunk in enumerate(chunks):
            content = chunk.page_content.strip()
            
            # No debe terminar en medio de una palabra (excepto último chunk)
            if i < len(chunks) - 1:
                # Verificar que no termina con letra seguida de letra (palabra cortada)
                if len(content) > 1:
                    # Permitir que termine en espacio, punto, coma, o salto de línea
                    last_chars = content[-2:]
                    assert not (last_chars[0].isalpha() and last_chars[1].isalpha() and 
                              i < len(chunks) - 1), \
                        f"Chunk {i} termina en medio de palabra: '...{content[-20:]}'"
    
    def test_recursive_avoids_mid_word_splits(self, recursive_service, create_document):
        """
        RECURSIVE no debe dividir palabras a la mitad de manera obvia.
        
        Esto es resultado de usar separadores jerárquicos que priorizan
        espacios y puntuación sobre caracteres individuales.
        """
        # Texto con párrafos para probar división natural
        text_with_paragraphs = """
        The quick brown fox jumps over the lazy dog. This is a test sentence.
        Another sentence here with more words to test the chunking behavior.
        
        This is a new paragraph with different content. It should be handled properly.
        More content to ensure we have enough text for multiple chunks to be generated.
        
        Third paragraph with additional information. Testing the recursive character splitter.
        Making sure it respects word boundaries and doesn't split in the middle of words.
        """ * 20  # Repetir para generar suficiente texto
        
        doc = create_document(text_with_paragraphs)
        chunks = recursive_service.splitter.split_documents([doc])
        
        assert len(chunks) > 1, "Debe generar múltiples chunks"
        
        # Verificar que no hay splits obvios en medio de palabras
        # (permitir casos donde termina en letra si el siguiente empieza con espacio/puntuación)
        mid_word_splits = 0
        
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i].page_content.strip()
            next_chunk = chunks[i + 1].page_content.strip()
            
            if not current_chunk or not next_chunk:
                continue
            
            last_char = current_chunk[-1]
            first_char = next_chunk[0]
            
            # Split en medio de palabra: termina en letra Y empieza en letra (sin espacio/puntuación)
            if last_char.isalpha() and first_char.isalpha():
                # Verificar si realmente es una palabra cortada
                # (no es cortada si hay espacio en el original entre ellas)
                mid_word_splits += 1
        
        # Permitir algunos splits en medio de palabra (por el overlap y separadores)
        # pero no debe ser la mayoría
        split_percentage = (mid_word_splits / (len(chunks) - 1)) * 100
        
        assert split_percentage <= 30, \
            f"{split_percentage:.1f}% de chunks tienen splits en medio de palabra (máximo 30%). " \
            f"Verifica que los separadores incluyan espacios y puntuación."

