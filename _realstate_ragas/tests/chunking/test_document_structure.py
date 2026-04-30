"""
Tests Específicos para DOCUMENT_STRUCTURE
==========================================

Tests que validan el comportamiento específico de la estrategia DOCUMENT_STRUCTURE.
Esta estrategia divide documentos por headers Markdown (#, ##, ###, ####).

NOTA IMPORTANTE: Esta estrategia funciona con documentos Markdown.
"""

import pytest
from langchain.schema import Document


class TestDocumentStructure:
    """Tests específicos de la estrategia DOCUMENT_STRUCTURE"""
    
    def test_document_splits_by_markdown_headers(self, document_service, sample_markdown):
        """
        DOCUMENT_STRUCTURE debe dividir por headers Markdown.
        
        """
        # sample_markdown tiene: # Main, ## Section 1, ## Section 2, ### 2.1, ### 2.2, ## Section 3
        
        chunks = document_service.splitter.split_text(sample_markdown)
        
        assert len(chunks) >= 3, \
            f"DOCUMENT_STRUCTURE debe generar al menos 3 chunks de documento Markdown. " \
            f"Generó {len(chunks)}. Verifica que headers_to_split_on esté configurado."
        
        contents = [chunk.page_content if isinstance(chunk, Document) else chunk for chunk in chunks]
        combined = " ".join(contents)
        
        assert "Section 1" in combined, "No se encontró 'Section 1' en los chunks"
        assert "Section 2" in combined, "No se encontró 'Section 2' en los chunks"
    
    def test_document_preserves_header_in_content(self, document_service, sample_markdown):
        """
        DOCUMENT_STRUCTURE debe preservar headers en el contenido de chunks.
        
        """
        chunks = document_service.splitter.split_text(sample_markdown)
        
        has_header = False
        for chunk in chunks:
            content = chunk.page_content if isinstance(chunk, Document) else chunk
            if "##" in content or "#" in content:
                has_header = True
                break
        
        assert has_header, \
            "DOCUMENT_STRUCTURE debe preservar headers en chunks. " \
            "Verifica que strip_headers=False en tu implementación."
    
    def test_document_handles_nested_headers(self, document_service, sample_markdown):
        """
        DOCUMENT_STRUCTURE debe manejar headers anidados (jerarquía).
        
        El documento de prueba tiene:
        - # Main Title
        - ## Section 2
        - ### Subsection 2.1
        - ### Subsection 2.2
        
        Debe generar chunks respetando esta jerarquía.
        """
        chunks = document_service.splitter.split_text(sample_markdown)
        
        assert len(chunks) >= 2, \
            "DOCUMENT_STRUCTURE debe generar múltiples chunks de headers anidados"
        
        contents = [chunk.page_content if isinstance(chunk, Document) else chunk for chunk in chunks]
        combined = " ".join(contents)
        
        assert "Subsection" in combined, \
            "DOCUMENT_STRUCTURE debe procesar subsecciones (###). " \
            "Verifica que headers_to_split_on incluya ###"
        
        chunk_sizes = [len(c.page_content if isinstance(c, Document) else c) for c in chunks]
        
        if len(chunk_sizes) > 1:
            assert len(set(chunk_sizes)) > 1, \
                "Los chunks deben tener tamaños variados según la estructura del documento"

