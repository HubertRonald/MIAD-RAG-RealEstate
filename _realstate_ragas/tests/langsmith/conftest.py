"""
Configuración de fixtures para tests de LangSmith.
"""

import os
import pytest


def pytest_report_header(config):
    """
    Header personalizado para los tests de LangSmith.
    """
    return [
        "=" * 70,
        "LANGSMITH OBSERVABILITY TESTS",
        "=" * 70,
        "Verifica la integracion con LangSmith para observabilidad del RAG",
        "=" * 70,
    ]


