"""
Configuración de fixtures para tests de Semana 5.
"""

import os
import pytest


def pytest_report_header(config):
    """
    Header personalizado para los tests de Semana 5.
    """
    return [
        "=" * 70,
        "SEMANA 5: QUERY REWRITING Y RERANKING TESTS",
        "=" * 70,
        "Verifica implementacion de query rewriting y reranking con LangSmith",
        "=" * 70,
    ]
