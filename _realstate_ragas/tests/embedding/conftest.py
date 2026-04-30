"""
Fixtures Compartidas para Tests de Embeddings
==============================================

Este módulo define fixtures que se comparten entre todas las pruebas
del servicio de embeddings.

"""

import pytest
import os
from pathlib import Path
from typing import List, Dict
from langchain.schema import Document

from app.services.embedding_service import EmbeddingService, EMBEDDINGS_MODEL, BATCH_SIZE


@pytest.fixture(scope="session")
def embedding_service():
    """
    Fixture del servicio de embeddings con configuración estándar.
    
    Scope: session - Se crea UNA SOLA VEZ para toda la sesión de tests.
    Esto minimiza inicializaciones y llamadas a la API.
    
    IMPORTANTE: Este fixture usa el modelo configurado en embedding_service.py
    Los tests leen las constantes EMBEDDINGS_MODEL y BATCH_SIZE automáticamente.
    
    NOTA: Los tests no validan dimensiones específicas, permitiendo
    flexibilidad en la elección del modelo de embeddings.
    """
    try:        
        return EmbeddingService()
    except Exception as e:
        pytest.fail(f"Error al inicializar EmbeddingService: {e}")


@pytest.fixture(scope="session")
def sample_texts():
    """
    Textos de ejemplo para pruebas de similitud y consistencia.
    
    Incluye:
    - Textos similares (mismo tema)
    - Textos disímiles (temas diferentes)
    - Texto idéntico (para reproducibilidad)
    - Edge cases (vacío, largo, especiales)
    """
    return {
        "ai_similar": "Artificial intelligence is transforming modern technology and computing.",
        "food_dissimilar": "Pizza is a delicious Italian dish with cheese and tomato sauce.",
        "identical": "This text will be embedded multiple times to test consistency.",
        "empty": "",
        "very_long": "word " * 200,  # ~1000 caracteres, >512 tokens
        "special_emojis": "Hello 😊 World 🌍 Testing 🚀 Emojis 🎉 Special chars: !@#$%",
    }


@pytest.fixture(scope="session")
def all_embeddings_cache(embedding_service, sample_texts):
    """
    CACHÉ GLOBAL DE EMBEDDINGS - Genera TODOS los embeddings necesarios UNA SOLA VEZ.
    
    Esta es la fixture MÁS IMPORTANTE para optimización de API.
    
    Genera embeddings para:
    1. Similarity tests (2 textos: similar, dissimilar)
    2. Consistency tests (1 texto repetido 3 veces)
    3. Dimension tests (usa los mismos de similarity)
    4. Edge cases (3 textos: empty, very_long, special_emojis)
    5. Performance test (usa los mismos de similarity)
      
    IMPORTANTE: Los embeddings salen de la implementación de cada  en
    embedding_service.py. Si se no implementa caché o batch processing,
    consumirá más API. Estos tests evalúan eso.
    """    
    cache = {}
    
    # 1. Para Similarity Tests (2 textos diferentes)
    similarity_texts = [
        sample_texts["ai_similar"],
        sample_texts["food_dissimilar"]
    ]
    cache["similarity"] = {
        "texts": similarity_texts,
        "embeddings": embedding_service.create_embeddings(similarity_texts)
    }
    
    # 2. Para Consistency Tests (mismo texto 3 veces para determinismo)
    identical_text = sample_texts["identical"]
    cache["consistency"] = {
        "text": identical_text,
        "embedding_1": embedding_service.create_embeddings([identical_text])[0],
        "embedding_2": embedding_service.create_embeddings([identical_text])[0],
        "embedding_3": embedding_service.create_embeddings([identical_text])[0],
    }
    
    # 3. Para Batch vs Single Consistency (2 textos)
    batch_texts = [sample_texts["ai_similar"], sample_texts["food_dissimilar"]]
    cache["batch_consistency"] = {
        "texts": batch_texts,
        "batch_embeddings": embedding_service.create_embeddings(batch_texts),
        "single_embeddings": [
            embedding_service.create_embeddings([batch_texts[0]])[0],
            embedding_service.create_embeddings([batch_texts[1]])[0]
        ]
    }
    
    # 4. Para Edge Cases (3 textos especiales)
    edge_texts = [
        sample_texts["empty"],
        sample_texts["very_long"],
        sample_texts["special_emojis"]
    ]
    
    # Manejar empty string con try/except
    edge_embeddings = []
    for i, text in enumerate(edge_texts):
        try:
            emb = embedding_service.create_embeddings([text])[0]
            edge_embeddings.append({"text": text, "embedding": emb, "error": None})
        except Exception as e:
            edge_embeddings.append({"text": text, "embedding": None, "error": str(e)})
    
    cache["edge_cases"] = edge_embeddings
    
    # 5. Para Performance Test (reutiliza similarity embeddings)
    cache["performance"] = cache["similarity"]  # Reutilizar
    
    return cache


@pytest.fixture(scope="session")
def vectorstore_cache(embedding_service, sample_texts, tmp_path_factory):
    """
    Vectorstore FAISS cacheado para pruebas de índice vectorial.
    
    Genera UN SOLO vectorstore FAISS por sesión y lo reutiliza en todos
    los tests de vectorstore.
    
    IMPORTANTE: Usa tmp_path_factory para crear un directorio temporal
    que persiste durante toda la sesión de tests.
    """
    # Crear documentos de prueba
    from langchain.schema import Document
    
    documents = [
        Document(
            page_content=sample_texts["ai_similar"],
            metadata={"source": "test_ai.txt", "chunk_id": 0}
        ),
        Document(
            page_content=sample_texts["food_dissimilar"],
            metadata={"source": "test_food.txt", "chunk_id": 1}
        ),
        Document(
            page_content=sample_texts["identical"],
            metadata={"source": "test_identical.txt", "chunk_id": 2}
        ),
    ]
    
    # Crear directorio temporal que persiste durante la sesión
    temp_dir = tmp_path_factory.mktemp("faiss_test")
    vectorstore_path = str(temp_dir / "test_vectorstore")

    
    # Construir vectorstore
    vectorstore = embedding_service.build_vectorstore(
        chunks=documents,
        persist_path=vectorstore_path
    )
    
    return {
        "vectorstore": vectorstore,
        "documents": documents,
        "path": vectorstore_path,
        "service": embedding_service
    }



def pytest_collection_modifyitems(config, items):
    """
    Modifica items de la colección para agregar markers automáticamente.
    """
    for item in items:
        # Marcar tests que usan embedding_service
        if "embedding_service" in item.fixturenames:
            item.add_marker(pytest.mark.embedding_api)
        
        # Marcar tests de vectorstore
        if "vectorstore" in item.name or "faiss" in item.name.lower():
            item.add_marker(pytest.mark.embedding_vectorstore)


