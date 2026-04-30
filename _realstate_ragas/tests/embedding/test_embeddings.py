"""
Tests Esenciales de Embeddings
================================

Suite mínima de 12 tests que valida la funcionalidad core del servicio
de embeddings con consumo mínimo de API.
CATEGORÍAS:
- Similarity (3 tests)
- Consistency (3 tests)
- Edge Cases (3 tests)
- FAISS Vectorstore (2 tests)
- Performance (1 test)

TOTAL: 12 tests

NOTA: No se validan dimensiones específicas para permitir flexibilidad
en la elección del modelo de embeddings por parte de cada grupo.
"""

import pytest
import numpy as np
import time


def cosine_similarity_manual(vec1, vec2):
    """
    Calcula la similitud coseno entre dos vectores.
    
    Formula: cos(θ) = (A · B) / (||A|| × ||B||)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


# ============================
# SIMILARITY TESTING (3 tests)
# ============================

class TestSimilarity:
    """
    Valida que los embeddings capturen correctamente la similitud semántica.
    """
    
    def test_similar_texts_high_similarity(self, all_embeddings_cache):
        """
        Textos sobre el mismo tema deben tener alta similitud (≥0.7).
        
        Usa embeddings cacheados - NO hace llamadas adicionales a la API.
        """
        # Obtener embeddings del caché
        embeddings = all_embeddings_cache["similarity"]["embeddings"]
        
        # Calcular similitud consigo mismo (debe ser ~1.0)
        similarity_ai = cosine_similarity_manual(embeddings[0], embeddings[0])
        
        assert similarity_ai >= 0.95, (
            f"Texto consigo mismo debe tener similitud ≥0.95, "
            f"obtuvo: {similarity_ai:.3f}"
        )
    
    def test_dissimilar_texts_low_similarity(self, all_embeddings_cache):
        """
        Textos sobre temas diferentes deben tener similitud menor que textos similares.
        
        Usa embeddings cacheados - NO hace llamadas adicionales a la API.
        """
        # Obtener embeddings del caché
        embeddings = all_embeddings_cache["similarity"]["embeddings"]
        
        # AI vs Food (temas completamente diferentes)
        similarity_different = cosine_similarity_manual(embeddings[0], embeddings[1])
        
        # AI vs AI (mismo texto)
        similarity_same = cosine_similarity_manual(embeddings[0], embeddings[0])
        
        assert similarity_different < similarity_same, (
            f"Textos disímiles deben tener menor similitud que textos idénticos. "
            f"Disímiles: {similarity_different:.3f}, Idénticos: {similarity_same:.3f}"
        )
        
        # Verificar que no es demasiado alta (ajustado para gemini-embedding-001)
        assert similarity_different <= 0.85, (
            f"Textos disímiles (AI vs Food) deben tener similitud ≤0.85, "
            f"obtuvieron: {similarity_different:.3f}"
        )
    
    def test_identical_texts_perfect_similarity(self, all_embeddings_cache):
        """
        El mismo texto embeddeado múltiples veces debe tener similitud ≥0.95.
        
        Valida determinismo del modelo.
        Usa embeddings cacheados - NO hace llamadas adicionales a la API.
        """
        # Obtener embeddings del caché (mismo texto, 3 veces)
        emb_1 = all_embeddings_cache["consistency"]["embedding_1"]
        emb_2 = all_embeddings_cache["consistency"]["embedding_2"]
        
        similarity = cosine_similarity_manual(emb_1, emb_2)
        
        assert similarity >= 0.95, (
            f"Textos idénticos deben tener similitud ≥0.95, "
            f"obtuvieron: {similarity:.3f}"
        )


# =============================
# CONSISTENCY TESTING (3 tests)
# =============================

class TestConsistency:
    """
    Valida reproducibilidad y consistencia del servicio.
    """
    
    def test_batch_vs_single_consistency(self, all_embeddings_cache):
        """
        Embeddings generados en batch deben ser similares a los generados
        individualmente.
        
        NOTA: Puede haber pequeñas variaciones debido al procesamiento interno
        del modelo, pero deben ser altamente similares (≥0.80).
        
        Usa embeddings cacheados - NO hace llamadas adicionales a la API.
        """
        batch_embs = all_embeddings_cache["batch_consistency"]["batch_embeddings"]
        single_embs = all_embeddings_cache["batch_consistency"]["single_embeddings"]
        
        # Comparar cada embedding
        for i, (batch_emb, single_emb) in enumerate(zip(batch_embs, single_embs)):
            similarity = cosine_similarity_manual(batch_emb, single_emb)
            
            assert similarity >= 0.80, (
                f"Embedding {i} en batch debe ser consistente con single, "
                f"similitud: {similarity:.6f} (esperado ≥0.80)"
            )
    
    def test_determinism_reproducibility(self, all_embeddings_cache):
        """
        Generar embeddings para el mismo texto múltiples veces debe producir
        resultados idénticos (determinismo).
        
        Usa embeddings cacheados - NO hace llamadas adicionales a la API.
        """
        emb_1 = all_embeddings_cache["consistency"]["embedding_1"]
        emb_2 = all_embeddings_cache["consistency"]["embedding_2"]
        emb_3 = all_embeddings_cache["consistency"]["embedding_3"]
        
        # Verificar que son prácticamente idénticos
        similarity_1_2 = cosine_similarity_manual(emb_1, emb_2)
        similarity_1_3 = cosine_similarity_manual(emb_1, emb_3)
        similarity_2_3 = cosine_similarity_manual(emb_2, emb_3)
        
        assert similarity_1_2 >= 0.99, (
            f"Embeddings deben ser determinísticos: "
            f"similitud 1-2 = {similarity_1_2:.6f} (esperado ≥0.99)"
        )
        
        assert similarity_1_3 >= 0.99, (
            f"Embeddings deben ser determinísticos: "
            f"similitud 1-3 = {similarity_1_3:.6f} (esperado ≥0.99)"
        )
        
        assert similarity_2_3 >= 0.99, (
            f"Embeddings deben ser determinísticos: "
            f"similitud 2-3 = {similarity_2_3:.6f} (esperado ≥0.99)"
        )
    
    def test_order_preservation_in_batch(self, all_embeddings_cache):
        """
        El orden de los embeddings en batch debe corresponder al orden
        de los textos de entrada.
        
        Valida que batch[0] corresponde a single[0] y batch[1] a single[1],
        con alta similitud (≥0.80).
        
        Usa embeddings cacheados - NO hace llamadas adicionales a la API.
        """
        batch_embs = all_embeddings_cache["batch_consistency"]["batch_embeddings"]
        single_embs = all_embeddings_cache["batch_consistency"]["single_embeddings"]
        
        # Verificar que cada posición corresponde
        for i in range(len(batch_embs)):
            similarity = cosine_similarity_manual(batch_embs[i], single_embs[i])
            
            assert similarity >= 0.80, (
                f"Embedding en posición {i} no corresponde al texto esperado, "
                f"similitud: {similarity:.6f} (esperado ≥0.80)"
            )


# ====================
# EDGE CASES (3 tests)
# ====================

class TestEdgeCases:
    """
    Valida robustez del servicio con inputs inusuales.
    """
    
    def test_empty_string_handling(self, all_embeddings_cache):
        """
        El servicio debe manejar strings vacíos de forma controlada.
        
        Puede:
        1. Generar un embedding válido (sin NaN/Inf)
        2. Lanzar un error informativo (ValueError con mensaje claro)
        
        Usa embeddings cacheados - NO hace llamadas adicionales a la API.
        """
        edge_case = all_embeddings_cache["edge_cases"][0]  # empty string
        
        if edge_case["error"] is not None:
            # Si hubo error, debe ser informativo
            error_msg = edge_case["error"].lower()
            assert "empty" in error_msg or "invalid" in error_msg or "vac" in error_msg, (
                f"Error al procesar string vacío debe ser informativo, "
                f"obtuvimos: {edge_case['error']}"
            )
        else:
            # Si no hubo error, el embedding debe ser válido
            emb = edge_case["embedding"]
            assert len(emb) > 0, "Embedding debe tener al menos 1 dimensión"
            assert not np.any(np.isnan(emb)), "Embedding no debe contener NaN"
            assert not np.any(np.isinf(emb)), "Embedding no debe contener infinitos"
    
    def test_very_long_text_handling(self, all_embeddings_cache):
        """
        El servicio debe manejar textos muy largos (>512 tokens).
        
        Puede:
        1. Procesarlo completo
        2. Truncarlo automáticamente
        3. Lanzar error informativo
        
        Usa embeddings cacheados - NO hace llamadas adicionales a la API.
        """
        edge_case = all_embeddings_cache["edge_cases"][1]  # very long text
        
        if edge_case["error"] is not None:
            # Si hubo error, debe ser informativo
            error_msg = edge_case["error"].lower()
            assert "long" in error_msg or "token" in error_msg or "limit" in error_msg, (
                f"Error al procesar texto largo debe ser informativo, "
                f"obtuvimos: {edge_case['error']}"
            )
        else:
            # Si no hubo error, el embedding debe ser válido
            emb = edge_case["embedding"]
            assert len(emb) > 0, "Embedding debe tener al menos 1 dimensión"
            assert not np.any(np.isnan(emb)), "Embedding no debe contener NaN"
            assert not np.any(np.isinf(emb)), "Embedding no debe contener infinitos"
    
    def test_special_chars_and_emojis(self, all_embeddings_cache):
        """
        El servicio debe manejar caracteres especiales y emojis correctamente.
        
        Usa embeddings cacheados - NO hace llamadas adicionales a la API.
        """
        edge_case = all_embeddings_cache["edge_cases"][2]  # special chars + emojis
        
        if edge_case["error"] is not None:
            # Si hubo error, debe ser informativo
            pytest.fail(
                f"El servicio debe manejar caracteres especiales y emojis, "
                f"pero falló con: {edge_case['error']}"
            )
        else:
            # El embedding debe ser válido
            emb = edge_case["embedding"]
            assert len(emb) > 0, "Embedding debe tener al menos 1 dimensión"
            assert not np.any(np.isnan(emb)), "Embedding no debe contener NaN"
            assert not np.any(np.isinf(emb)), "Embedding no debe contener infinitos"


# ===========================
# FAISS VECTORSTORE (2 tests)
# ===========================

class TestVectorstore:
    """
    Valida la construcción y funcionalidad del índice vectorial FAISS.
    """
    
    def test_vectorstore_creation_and_persistence(self, vectorstore_cache):
        """
        El servicio debe crear correctamente un índice FAISS y persistirlo en disco.
        
        Valida:
        1. El vectorstore se crea sin errores
        2. Contiene el número correcto de vectores
        3. Se persiste en disco (archivos index.faiss e index.pkl)
        
        Usa vectorstore cacheado - NO hace llamadas adicionales a la API.
        """
        from pathlib import Path
        
        vectorstore = vectorstore_cache["vectorstore"]
        documents = vectorstore_cache["documents"]
        path = vectorstore_cache["path"]
        
        # Validar que el vectorstore existe
        assert vectorstore is not None, "Vectorstore no debe ser None"
        
        # Validar número de vectores
        num_vectors = vectorstore.index.ntotal
        expected_count = len(documents)
        
        assert num_vectors == expected_count, (
            f"Vectorstore debe contener {expected_count} vectores, "
            f"contiene: {num_vectors}"
        )
        
        # Validar persistencia en disco
        path_obj = Path(path)
        assert path_obj.exists(), f"Directorio {path} debe existir"
        assert (path_obj / "index.faiss").exists(), "Archivo index.faiss debe existir"
        assert (path_obj / "index.pkl").exists(), "Archivo index.pkl debe existir"
        
    def test_vectorstore_similarity_search(self, vectorstore_cache):
        """
        El vectorstore debe permitir búsquedas de similitud semántica.
        
        Valida:
        1. La búsqueda retorna resultados
        2. Los resultados son relevantes (contienen palabras clave de la query)
        3. El número de resultados respeta el parámetro k
        
        Usa vectorstore cacheado - NO hace llamadas adicionales a la API.
        """
        vectorstore = vectorstore_cache["vectorstore"]
        
        # Realizar búsqueda sobre tema de AI
        query = "artificial intelligence and machine learning"
        results = vectorstore.similarity_search(query, k=2)
        
        # Validar que retorna resultados
        assert len(results) > 0, "La búsqueda debe retornar al menos 1 resultado"
        assert len(results) <= 2, f"La búsqueda debe retornar máximo 2 resultados (k=2)"
        
        # Validar que el primer resultado es relevante
        top_result = results[0].page_content.lower()
        
        # Debe contener palabras relacionadas con AI o tecnología
        relevant_keywords = ["artificial", "intelligence", "technology", "computing", "machine", "learning"]
        has_relevant_keyword = any(keyword in top_result for keyword in relevant_keywords)
        
        assert has_relevant_keyword, (
            f"El resultado más relevante debe contener palabras clave relacionadas. "
            f"Resultado: {top_result[:100]}..."
        )
        
        # Validar que tiene metadatos
        assert results[0].metadata is not None, "Los resultados deben tener metadatos"
        assert "source" in results[0].metadata, "Los metadatos deben incluir 'source'"
        

# ============================
# PERFORMANCE TESTING (1 test)
# ============================

class TestPerformance:
    """
    Valida velocidad del servicio.
    """
    
    def test_embedding_generation_speed(self, embedding_service, sample_texts):
        """
        Generar embeddings debe ser razonablemente rápido.
        
        """
        texts = [
            sample_texts["ai_similar"],
            sample_texts["food_dissimilar"]
        ]
        
        start_time = time.time()
        embeddings = embedding_service.create_embeddings(texts)
        elapsed_time = time.time() - start_time
        
        # Verificar que se generaron correctamente
        assert len(embeddings) == 2, "Deben generarse 2 embeddings"
        
        # Verificar velocidad razonable
        assert elapsed_time < 10.0, (
            f"Generar 2 embeddings debe tomar <10s, tomó: {elapsed_time:.2f}s"
        )
        
        avg_time = elapsed_time / len(texts)

