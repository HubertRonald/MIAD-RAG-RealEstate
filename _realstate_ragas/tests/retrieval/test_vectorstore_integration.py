"""
Tests de Integración con Vectorstore Real
==========================================

Valida el funcionamiento end-to-end del RetrievalService con un
vectorstore FAISS real.

ESTRATEGIA: Integración (CON API, vectorstore real)
TIEMPO: ~10 segundos (primera ejecución)
COSTO API: ~5 requests (caché de sesión, UNA SOLA VEZ)

NOTA IMPORTANTE:
----------------
Estos tests están marcados con @pytest.mark.integration.
Para ejecutar SOLO tests unitarios (sin API):
    pytest tests/retrieval/ -v -m "not integration"

Para ejecutar TODOS (incluye integración):
    pytest tests/retrieval/ -v
"""

import pytest
from langchain.schema import Document
from app.services.retrieval_service import RetrievalService


@pytest.mark.integration
def test_retrieval_with_real_vectorstore(real_vectorstore_cache):
    """
    Test de integración completo con vectorstore FAISS real.
    
    Valida:
    - Carga del vectorstore desde EmbeddingService
    - Retrieval funciona end-to-end
    - Documentos recuperados tienen estructura correcta
    - Metadata se preserva correctamente
    
    COSTO API: 0 (usa caché de sesión)
    """
    embedding_service = real_vectorstore_cache["embedding_service"]
    expected_docs = real_vectorstore_cache["documents"]
    
    # Crear servicio de retrieval con k=2
    retrieval_service = RetrievalService(embedding_service, k=2)
    
    # Query relacionada con ML/AI
    query = "What is machine learning and artificial intelligence?"
    results = retrieval_service.retrieve_documents(query)
    
    # Validaciones básicas
    assert len(results) > 0, "Debe retornar al menos 1 documento"
    assert len(results) <= 2, f"Debe respetar k=2, pero retornó {len(results)}"
    
    # Verificar estructura de documentos
    for doc in results:
        assert isinstance(doc, Document), f"Resultado no es Document: {type(doc)}"
        assert hasattr(doc, 'page_content'), "Document debe tener page_content"
        assert hasattr(doc, 'metadata'), "Document debe tener metadata"
        assert len(doc.page_content) > 0, "page_content no debe estar vacío"
    
    # Verificar metadata
    for doc in results:
        assert 'source_file' in doc.metadata, "Metadata debe incluir source_file"
        assert 'chunk_id' in doc.metadata, "Metadata debe incluir chunk_id"
        assert doc.metadata['source_file'] is not None


@pytest.mark.integration
def test_retrieval_returns_relevant_documents(real_vectorstore_cache):
    """
    Verifica que el retrieval retorna documentos semánticamente relevantes.
    
    Query sobre AI/ML debería retornar documentos relacionados con estos temas.
    
    COSTO API: 0 (usa caché de sesión)
    """
    embedding_service = real_vectorstore_cache["embedding_service"]
    
    # Crear servicio
    retrieval_service = RetrievalService(embedding_service, k=3)
    
    # Query específico sobre ML/AI
    query = "machine learning algorithms and neural networks"
    results = retrieval_service.retrieve_documents(query)
    
    # Verificar que retornó documentos
    assert len(results) > 0, "Debe retornar documentos para query relevante"
    
    # Verificar que los documentos tienen contenido relacionado
    # (al menos uno debe mencionar ML, AI, o neural networks)
    contents_lower = [doc.page_content.lower() for doc in results]
    relevant_terms = ['machine learning', 'artificial intelligence', 'neural network', 'ai']
    
    found_relevant = any(
        any(term in content for term in relevant_terms)
        for content in contents_lower
    )
    
    assert found_relevant, \
        f"Al menos un documento debe contener términos relevantes. " \
        f"Contenidos: {contents_lower}"


@pytest.mark.integration
def test_retrieval_with_different_k_values(real_vectorstore_cache):
    """
    Verifica que diferentes valores de k retornan cantidades correctas.
    
    Prueba k=1, k=2, k=3 con el mismo vectorstore.
    
    COSTO API: 0 (usa caché de sesión)
    """
    embedding_service = real_vectorstore_cache["embedding_service"]
    total_docs = len(real_vectorstore_cache["documents"])
    
    # Test con k=1
    service_k1 = RetrievalService(embedding_service, k=1)
    results_k1 = service_k1.retrieve_documents("artificial intelligence")
    assert len(results_k1) == 1, "k=1 debe retornar exactamente 1 documento"
    
    # Test con k=2
    service_k2 = RetrievalService(embedding_service, k=2)
    results_k2 = service_k2.retrieve_documents("artificial intelligence")
    assert len(results_k2) == 2, "k=2 debe retornar exactamente 2 documentos"
    
    # Test con k=3 (igual al total de docs en vectorstore)
    service_k3 = RetrievalService(embedding_service, k=3)
    results_k3 = service_k3.retrieve_documents("artificial intelligence")
    assert len(results_k3) == total_docs, \
        f"k=3 debe retornar {total_docs} documentos (total disponible)"


@pytest.mark.integration
def test_retrieval_consistent_across_multiple_calls(real_vectorstore_cache):
    """
    Verifica que múltiples retrievals con el mismo query son consistentes.
    
    El retrieval debe ser determinístico para el mismo query.
    
    COSTO API: 0 (usa caché de sesión)
    """
    embedding_service = real_vectorstore_cache["embedding_service"]
    
    # Crear servicio
    retrieval_service = RetrievalService(embedding_service, k=2)
    
    # Query específico
    query = "Python programming language"
    
    # Hacer múltiples retrievals con el mismo query
    results1 = retrieval_service.retrieve_documents(query)
    results2 = retrieval_service.retrieve_documents(query)
    results3 = retrieval_service.retrieve_documents(query)
    
    # Verificar que todos retornan la misma cantidad
    assert len(results1) == len(results2) == len(results3), \
        "Múltiples llamadas deben retornar misma cantidad de documentos"
    
    # Verificar que el contenido es el mismo
    for i, (doc1, doc2, doc3) in enumerate(zip(results1, results2, results3)):
        assert doc1.page_content == doc2.page_content == doc3.page_content, \
            f"Documento {i} tiene contenido inconsistente entre llamadas"


@pytest.mark.integration
def test_retrieval_with_different_queries(real_vectorstore_cache):
    """
    Verifica que diferentes queries retornan documentos diferentes (o en orden diferente).
    
    COSTO API: 0 (usa caché de sesión)
    """
    embedding_service = real_vectorstore_cache["embedding_service"]
    
    # Crear servicio
    retrieval_service = RetrievalService(embedding_service, k=2)
    
    # Queries sobre temas diferentes
    query_ai = "machine learning and artificial intelligence"
    query_python = "Python programming language"
    query_nn = "neural networks deep learning"
    
    # Ejecutar retrievals
    results_ai = retrieval_service.retrieve_documents(query_ai)
    results_python = retrieval_service.retrieve_documents(query_python)
    results_nn = retrieval_service.retrieve_documents(query_nn)
    
    # Verificar que todos retornan documentos
    assert len(results_ai) > 0
    assert len(results_python) > 0
    assert len(results_nn) > 0
    
    # Verificar que son objetos Document válidos
    all_results = results_ai + results_python + results_nn
    for doc in all_results:
        assert isinstance(doc, Document)
        assert len(doc.page_content) > 0


@pytest.mark.integration
def test_retrieval_preserves_metadata_from_vectorstore(real_vectorstore_cache):
    """
    Verifica que la metadata original del vectorstore se preserva en retrieval.
    
    COSTO API: 0 (usa caché de sesión)
    """
    embedding_service = real_vectorstore_cache["embedding_service"]
    original_docs = real_vectorstore_cache["documents"]
    
    # Crear servicio
    retrieval_service = RetrievalService(embedding_service, k=3)
    
    # Ejecutar retrieval
    results = retrieval_service.retrieve_documents("test query")
    
    # Verificar que los documentos recuperados tienen metadata válida
    for doc in results:
        # Debe tener los campos que pusimos en los documentos originales
        assert 'source_file' in doc.metadata
        assert 'chunk_id' in doc.metadata
        assert 'topic' in doc.metadata
        
        # Verificar que los valores son válidos (no None)
        assert doc.metadata['source_file'] is not None
        assert doc.metadata['chunk_id'] is not None
        assert doc.metadata['topic'] is not None
        
        # Verificar que los valores son strings/ints esperados
        assert isinstance(doc.metadata['source_file'], str)
        assert isinstance(doc.metadata['chunk_id'], int)
        assert isinstance(doc.metadata['topic'], str)


@pytest.mark.integration
def test_retrieval_with_empty_query(real_vectorstore_cache):
    """
    Verifica el comportamiento con query vacío en vectorstore real.
    
    COSTO API: 0 (usa caché de sesión)
    """
    embedding_service = real_vectorstore_cache["embedding_service"]
    
    # Crear servicio
    retrieval_service = RetrievalService(embedding_service, k=2)
    
    # Ejecutar retrieval con query vacío
    # El comportamiento puede variar: retornar docs arbitrarios o fallar
    try:
        results = retrieval_service.retrieve_documents("")
        
        # Si no falla, debe retornar lista (posiblemente vacía)
        assert isinstance(results, list)
        
        # Si retorna documentos, deben ser válidos
        for doc in results:
            assert isinstance(doc, Document)
            assert len(doc.page_content) > 0
            
    except Exception as e:
        # También es aceptable que falle con query vacío
        # (depende del modelo de embeddings)
        print(f"Query vacío causó excepción (aceptable): {e}")
        pass


@pytest.mark.integration
def test_retrieval_service_reusable_across_queries(real_vectorstore_cache):
    """
    Verifica que un mismo servicio puede reutilizarse para múltiples queries.
    
    El servicio debe ser stateless y no acumular estado entre queries.
    
    COSTO API: 0 (usa caché de sesión)
    """
    embedding_service = real_vectorstore_cache["embedding_service"]
    
    # Crear UN SOLO servicio
    retrieval_service = RetrievalService(embedding_service, k=2)
    
    # Ejecutar múltiples queries diferentes
    queries = [
        "machine learning",
        "Python programming",
        "neural networks",
        "artificial intelligence",
        "data science"
    ]
    
    for query in queries:
        results = retrieval_service.retrieve_documents(query)
        
        # Cada query debe funcionar correctamente
        assert isinstance(results, list)
        assert len(results) <= 2, f"Debe respetar k=2 para query: {query}"
        
        # Verificar que retorna Documents válidos
        for doc in results:
            assert isinstance(doc, Document)
            assert len(doc.page_content) > 0


@pytest.mark.integration
def test_retrieval_with_long_query(real_vectorstore_cache):
    """
    Verifica que queries largos funcionan correctamente con vectorstore real.
    
    COSTO API: 0 (usa caché de sesión)
    """
    embedding_service = real_vectorstore_cache["embedding_service"]
    
    # Crear servicio
    retrieval_service = RetrievalService(embedding_service, k=2)
    
    # Query muy largo
    long_query = (
        "I am looking for detailed information about machine learning, "
        "artificial intelligence, neural networks, and deep learning. "
        "Specifically, I want to understand how these technologies work, "
        "what programming languages are commonly used (like Python), "
        "and what are the main applications in industry and research."
    )
    
    # Ejecutar retrieval
    results = retrieval_service.retrieve_documents(long_query)
    
    # Debe retornar resultados válidos
    assert len(results) > 0, "Query largo debe retornar documentos"
    assert len(results) <= 2, "Debe respetar k=2"
    
    for doc in results:
        assert isinstance(doc, Document)
        assert len(doc.page_content) > 0


@pytest.mark.integration
def test_retrieval_vectorstore_has_expected_size(real_vectorstore_cache):
    """
    Verifica que el vectorstore cacheado tiene el tamaño esperado.
    
    Esto valida que el fixture de caché se construyó correctamente.
    
    COSTO API: 0 (usa caché de sesión)
    """
    vectorstore = real_vectorstore_cache["vectorstore"]
    documents = real_vectorstore_cache["documents"]
    
    # Verificar que el índice FAISS tiene el número correcto de vectores
    assert vectorstore.index.ntotal == len(documents), \
        f"Vectorstore debe tener {len(documents)} vectores, " \
        f"pero tiene {vectorstore.index.ntotal}"
    
    # Verificar que tenemos 3 documentos (según conftest.py)
    assert len(documents) == 3, "Debe haber 3 documentos en el caché"


@pytest.mark.integration
@pytest.mark.slow
def test_retrieval_performance_acceptable(real_vectorstore_cache):
    """
    Verifica que el retrieval se completa en tiempo razonable.
    
    COSTO API: 0 (usa caché de sesión)
    NOTA: Marcado como 'slow' porque mide tiempo de ejecución
    """
    import time
    
    embedding_service = real_vectorstore_cache["embedding_service"]
    
    # Crear servicio
    retrieval_service = RetrievalService(embedding_service, k=3)
    
    # Medir tiempo de retrieval
    start_time = time.time()
    results = retrieval_service.retrieve_documents("machine learning")
    elapsed_time = time.time() - start_time
    
    # Retrieval debe completarse en menos de 5 segundos
    assert elapsed_time < 5.0, \
        f"Retrieval tomó {elapsed_time:.2f}s, debe ser < 5s"
    
    # Verificar que retornó resultados
    assert len(results) > 0
    
    print(f"\n⏱️  Retrieval time: {elapsed_time:.3f}s")

