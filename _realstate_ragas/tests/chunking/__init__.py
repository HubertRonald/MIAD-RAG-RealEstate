"""
Tests unitarios para el servicio de chunking.

Este módulo contiene tests para validar las 4 estrategias de chunking:
- RECURSIVE_CHARACTER
- FIXED_SIZE
- SEMANTIC
- DOCUMENT_STRUCTURE

Ejecución:
----------

# Todos los tests (incluye uso de embeddings con API):
pytest tests/chunking/ -v

# Todos los tests excepto Semantic Tests (uso de embeddings con API):
pytest tests/chunking/ -v -m "not semantic_api"

# Ejecutar tests de cada estrategia por separado:
pytest tests/chunking/test_fixed_size.py -v
pytest tests/chunking/test_recursive_character.py -v
pytest tests/chunking/test_document_structure.py -v
pytest tests/chunking/test_semantic.py -v

"""


