"""
Tests Semana 5: Query Rewriting y Reranking

Verifica que los estudiantes implementaron query rewriting y reranking
consultando las trazas en LangSmith.
"""

import os
import pytest
from datetime import datetime, timedelta
from langsmith import Client
from pathlib import Path

# Cargar variables de entorno
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


def get_langsmith_client() -> Client:
    """Obtiene cliente de LangSmith."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        pytest.skip("LANGSMITH_API_KEY not set")
    
    return Client(api_key=api_key)


def test_query_rewriting_implementado():
    """
    TEST 1: Verifica que QUERY REWRITING está implementado y visible en LangSmith.
    
    Busca trazas con "rewrite" en el nombre en los últimos 15 minutos.
    """
    client = get_langsmith_client()
    project_name = os.getenv("LANGSMITH_PROJECT")
    
    # Verificar configuración del proyecto
    if not project_name:
        pytest.skip(
            "LANGSMITH_PROJECT not configured.\n"
            "Add it to your .env file with your LangSmith project name."
        )
    
    # Verificar que tracing esté habilitado
    tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2", "").lower()
    tracing_langsmith = os.getenv("LANGSMITH_TRACING", "").lower()
    tracing_enabled = tracing_v2 == "true" or tracing_langsmith == "true"
    
    if not tracing_enabled:
        pytest.skip(
            f"Tracing not enabled.\n"
            f"  LANGCHAIN_TRACING_V2: '{tracing_v2}'\n"
            f"  LANGSMITH_TRACING: '{tracing_langsmith}'\n"
            "Set one of these to 'true' in your .env"
        )
    
    try:
        # Buscar trazas recientes (últimos 15 minutos)
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=15)
        
        runs = list(client.list_runs(
            project_name=project_name,
            start_time=start_time,
            limit=100
        ))
        
        # Si no hay trazas, dar instrucciones
        if len(runs) == 0:
            pytest.skip(
                f"\n[INFO] No se encontraron trazas en el proyecto '{project_name}' "
                f"en los ultimos 15 minutos.\n\n"
                "Para generar trazas:\n"
                "  1. Ejecuta una consulta RAG:\n"
                "     curl -X POST http://localhost:8000/api/v1/ask \\\n"
                '       -H "Content-Type: application/json" \\\n'
                '       -d \'{"question": "tu pregunta", "collection": "test_collection"}\'\n\n'
                "  2. Espera ~30 segundos y vuelve a ejecutar este test\n"
            )
        
        # Buscar trazas con "rewrite" en el nombre
        rewriting_runs = [
            run for run in runs 
            if run.name and "rewrite" in run.name.lower()
        ]
        
        # Verificación principal
        if len(rewriting_runs) == 0:
            # Mostrar nombres para debugging
            run_names = [run.name for run in runs[:10] if run.name]
            pytest.fail(
                f"\nQUERY REWRITING NO IMPLEMENTADO\n\n"
                f"Se encontraron {len(runs)} trazas, pero NINGUNA con 'rewrite' en el nombre.\n\n"
                f"Verifica:\n"
                f"  1. QueryRewritingService tiene @traceable\n"
                f"  2. RAGGraphService usa query_rewriting_service\n"
                f"  3. El servicio se instancia en app/routers/ask.py\n\n"
                f"Trazas encontradas (primeras 10):\n  {run_names}\n"
            )
        
        # Test pasó
        print(f"\nQuery Rewriting implementado correctamente")
        print(f"   {len(rewriting_runs)} traza(s) encontradas")
        print(f"   Última: {rewriting_runs[0].name}")
        print(f"   Hora: {rewriting_runs[0].start_time}")
        
    except Exception as e:
        pytest.fail(f"Error al consultar LangSmith: {str(e)}")


def test_reranking_implementado():
    """
    TEST 2: Verifica que RERANKING está implementado y visible en LangSmith.
    
    Busca trazas con "rerank" en el nombre en los últimos 15 minutos.
    """
    client = get_langsmith_client()
    project_name = os.getenv("LANGSMITH_PROJECT")
    
    # Verificar configuración del proyecto
    if not project_name:
        pytest.skip(
            "LANGSMITH_PROJECT not configured.\n"
            "Add it to your .env file with your LangSmith project name."
        )
    
    # Verificar que tracing esté habilitado
    tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2", "").lower()
    tracing_langsmith = os.getenv("LANGSMITH_TRACING", "").lower()
    tracing_enabled = tracing_v2 == "true" or tracing_langsmith == "true"
    
    if not tracing_enabled:
        pytest.skip(
            f"Tracing not enabled.\n"
            f"  LANGCHAIN_TRACING_V2: '{tracing_v2}'\n"
            f"  LANGSMITH_TRACING: '{tracing_langsmith}'\n"
            "Set one of these to 'true' in your .env"
        )
    
    try:
        # Buscar trazas recientes
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=15)
        
        runs = list(client.list_runs(
            project_name=project_name,
            start_time=start_time,
            limit=100
        ))
        
        if len(runs) == 0:
            pytest.skip(
                f"\n[INFO] No se encontraron trazas en el proyecto '{project_name}' "
                f"en los ultimos 15 minutos.\n\n"
                "Para generar trazas:\n"
                "  1. Ejecuta una consulta RAG\n"
                "  2. Espera ~30 segundos\n"
                "  3. Vuelve a ejecutar este test\n"
            )
        
        # Buscar trazas con "rerank" en el nombre
        reranking_runs = [
            run for run in runs 
            if run.name and "rerank" in run.name.lower()
        ]
        
        # Verificación principal
        if len(reranking_runs) == 0:
            # Mostrar nombres para debugging
            run_names = [run.name for run in runs[:10] if run.name]
            pytest.fail(
                f"\nRERANKING NO IMPLEMENTADO\n\n"
                f"Se encontraron {len(runs)} trazas, pero NINGUNA con 'rerank' en el nombre.\n\n"
                f"Verifica:\n"
                f"  1. RerankingService tiene @traceable\n"
                f"  2. RAGGraphService usa reranking_service\n"
                f"  3. El servicio se instancia en app/routers/ask.py\n\n"
                f"Trazas encontradas (primeras 10):\n  {run_names}\n"
            )
        
        # Test pasó
        print(f"\nReranking implementado correctamente")
        print(f"   {len(reranking_runs)} traza(s) encontradas")
        print(f"   Última: {reranking_runs[0].name}")
        print(f"   Hora: {reranking_runs[0].start_time}")
        
    except Exception as e:
        pytest.fail(f"Error al consultar LangSmith: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
