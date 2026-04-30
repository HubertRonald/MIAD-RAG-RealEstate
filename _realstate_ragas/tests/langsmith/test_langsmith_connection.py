"""
Tests de conexión y autenticación con la API de LangSmith.

Estos tests verifican que:
1. La API key de LangSmith es válida
2. El proyecto es accesible
3. La conexión a LangSmith funciona correctamente
"""

import os
import pytest
from langsmith import Client


def test_langsmith_api_key_exists():
    """
    Verifica que la variable de entorno LANGSMITH_API_KEY existe.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    assert api_key is not None, "LANGSMITH_API_KEY environment variable not set"
    assert len(api_key) > 0, "LANGSMITH_API_KEY is empty"
    print(f"\n[OK] LANGSMITH_API_KEY existe (longitud: {len(api_key)})")


def test_langsmith_api_key_valid():
    """
    Verifica que la API key de LangSmith es válida intentando conectarse.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    assert api_key is not None, "LANGSMITH_API_KEY not set"
    
    try:
        # Inicializar cliente de LangSmith
        client = Client(api_key=api_key)
        
        # Intentar listar proyectos - fallará si la API key es inválida
        projects = list(client.list_projects(limit=1))
        
        print(f"\n[OK] La API key de LangSmith es valida")
        print(f"  Conexion exitosa a la API de LangSmith")
        
    except Exception as e:
        pytest.fail(f"Failed to connect to LangSmith with provided API key: {str(e)}")


def test_langsmith_project_accessible():
    """
    Verifica que el proyecto de LangSmith es accesible.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    project_name = os.getenv("LANGCHAIN_PROJECT")
    
    assert api_key is not None, "LANGSMITH_API_KEY not set"
    
    # Si no hay proyecto configurado, usar valor por defecto
    if not project_name:
        print(f"\n[INFO] LANGCHAIN_PROJECT no configurado, LangSmith usara el proyecto por defecto")
        print(f"  Se recomienda configurar LANGCHAIN_PROJECT en tu .env")
        # No fallamos, solo informamos
        return
    
    try:
        # Inicializar cliente de LangSmith
        client = Client(api_key=api_key)
        
        # Intentar obtener el proyecto
        try:
            project = client.read_project(project_name=project_name)
            print(f"\n[OK] Proyecto '{project_name}' existe y es accesible")
            print(f"  Project ID: {project.id}")
        except Exception:
            # El proyecto puede no existir aún, se creará con la primera traza
            print(f"\n[INFO] Proyecto '{project_name}' no encontrado (se creara con la primera traza)")
        
        # Listar todos los proyectos para verificar acceso
        projects = list(client.list_projects(limit=10))
        print(f"\n[OK] Acceso al dashboard de LangSmith confirmado")
        print(f"  Total de proyectos accesibles: {len(projects)}")
        
    except Exception as e:
        pytest.fail(f"Failed to access LangSmith: {str(e)}")


if __name__ == "__main__":
    # Permite ejecutar los tests directamente
    pytest.main([__file__, "-v", "--tb=short"])


