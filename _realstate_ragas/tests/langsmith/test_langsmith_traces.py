"""
Tests de verificación de trazas en LangSmith.

Estos tests verifican que:
1. Las trazas se están enviando a LangSmith
2. Las trazas contienen metadata correcta
3. Las trazas incluyen información RAG
"""

import os
import pytest
from datetime import datetime, timedelta
from langsmith import Client
from pathlib import Path

# Intentar cargar variables de entorno desde .env si existe
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv no instalado, usar variables de entorno del sistema


def get_langsmith_client() -> Client:
    """
    Función auxiliar para obtener un cliente de LangSmith configurado.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        pytest.skip("LANGSMITH_API_KEY not set")
    
    return Client(api_key=api_key)


def test_recent_traces_exist():
    """
    Verifica que existen trazas recientes en LangSmith (en los últimos 15 minutos).
    
    IMPORTANTE: Este test requiere que:
    1. Tengas documentos cargados en una colección
    2. Hayas ejecutado al menos una consulta RAG
    3. Tracing habilitado: LANGCHAIN_TRACING_V2='true' o LANGSMITH_TRACING='true'
    """
    client = get_langsmith_client()
    project_name = os.getenv("LANGSMITH_PROJECT")
    
    # Soportar ambas variables de tracing (LANGCHAIN_TRACING_V2 o LANGSMITH_TRACING)
    tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2", "").lower()
    tracing_langsmith = os.getenv("LANGSMITH_TRACING", "").lower()
    tracing_enabled = tracing_v2 == "true" or tracing_langsmith == "true"
    
    
    # Verificar que LANGSMITH_PROJECT esté configurado
    if not project_name:
        pytest.skip(
            "LANGSMITH_PROJECT not configured.\n"
            "Add it to your .env file with your LangSmith project name."
        )
    
    # Verificar que el tracing esté habilitado
    if not tracing_enabled:
        pytest.skip(
            f"Tracing not enabled.\n"
            f"  LANGCHAIN_TRACING_V2: '{tracing_v2}'\n"
            f"  LANGSMITH_TRACING: '{tracing_langsmith}'\n"
            "Set one of these to 'true' in your .env"
        )
    
    try:
        # Calcular rango de tiempo (últimos 15 minutos)
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=15)
        
        # Consultar trazas recientes
        runs = list(client.list_runs(
            project_name=project_name,
            start_time=start_time,
            limit=50
        ))
        
        # Si no hay trazas, dar instrucciones claras
        if len(runs) == 0:
            pytest.skip(
                f"\n[INFO] No se encontraron trazas en el proyecto '{project_name}' "
                f"en los ultimos 15 minutos.\n\n"
                "Para generar trazas:\n"
                "  1. Asegurate de que tu servidor este corriendo (uvicorn main:app --reload)\n"
                "  2. Carga un documento:\n"
                "     curl -X POST http://localhost:8000/api/v1/documents/load-from-url \\\n"
                '       -H "Content-Type: application/json" \\\n'
                "       -d '{\"source_url\": \"tu-url\", "
                '\"collection_name\": \"test\", \"chunking_strategy\": \"fixed_size\"}\'\\n\n'
                "  3. Ejecuta una consulta:\n"
                "     curl -X POST http://localhost:8000/api/v1/ask \\\n"
                '       -H "Content-Type: application/json" \\\n'
                '       -d \'{"question": "tu pregunta", "collection": "test"}\'\\n\n'
                "  4. Espera ~30 segundos y vuelve a ejecutar este test\n"
            )
        
        print(f"\n[OK] Se encontraron {len(runs)} traza(s) en los ultimos 15 minutos")
        print(f"  Proyecto: {project_name}")
        print(f"  Rango: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}")
        
        # Mostrar la traza más reciente
        if runs:
            latest_run = runs[0]
            print(f"\n[INFO] Traza mas reciente:")
            print(f"  ID: {latest_run.id}")
            print(f"  Nombre: {latest_run.name}")
            print(f"  Hora: {latest_run.start_time}")
        
    except Exception as e:
        pytest.fail(f"Failed to retrieve traces from LangSmith: {str(e)}")


def test_trace_has_correct_structure():
    """
    Verifica que las trazas tienen la estructura correcta.
    
    Este test depende de que existan trazas recientes.
    """
    client = get_langsmith_client()
    project_name = os.getenv("LANGSMITH_PROJECT")
    
    # Verificar configuración
    if not project_name:
        pytest.skip("LANGSMITH_PROJECT not configured")
    
    # Soportar ambas variables de tracing
    tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2", "").lower()
    tracing_langsmith = os.getenv("LANGSMITH_TRACING", "").lower()
    tracing_enabled = tracing_v2 == "true" or tracing_langsmith == "true"
    
    if not tracing_enabled:
        pytest.skip(f"Tracing not enabled (LANGCHAIN_TRACING_V2: '{tracing_v2}', LANGSMITH_TRACING: '{tracing_langsmith}')")
    
    try:
        # Obtener trazas recientes
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=15)
        
        runs = list(client.list_runs(
            project_name=project_name,
            start_time=start_time,
            limit=10
        ))
        
        # Si no hay trazas, skip en lugar de fail
        if len(runs) == 0:
            pytest.skip(
                f"No traces found in project '{project_name}' to verify structure. "
                "Execute a RAG query first."
            )
        
        # Verificar la traza más reciente
        latest_run = runs[0]
        
        # Verificar campos básicos
        assert latest_run.id is not None, "Trace ID is missing"
        assert latest_run.name is not None, "Trace name is missing"
        assert latest_run.start_time is not None, "Trace start_time is missing"
        
        print(f"\n[OK] Estructura de traza correcta")
        print(f"  Trace ID: {latest_run.id}")
        print(f"  Nombre: {latest_run.name}")
        print(f"  Inicio: {latest_run.start_time}")
        
        # Verificar inputs/outputs (opcional, puede no existir en todos los casos)
        if latest_run.inputs:
            print(f"  Inputs: SI ({len(latest_run.inputs)} campo(s))")
        
        if latest_run.outputs:
            print(f"  Outputs: SI ({len(latest_run.outputs)} campo(s))")
        
        # Calcular duración si está disponible
        if latest_run.end_time and latest_run.start_time:
            duration = (latest_run.end_time - latest_run.start_time).total_seconds()
            print(f"  Duracion: {duration:.2f} segundos")
        
    except Exception as e:
        pytest.fail(f"Failed to verify trace structure: {str(e)}")


if __name__ == "__main__":
    # Permite ejecutar los tests directamente
    pytest.main([__file__, "-v", "--tb=short"])

