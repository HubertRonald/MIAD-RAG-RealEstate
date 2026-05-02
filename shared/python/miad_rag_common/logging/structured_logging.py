from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Optional


# Claves internas/reservadas de logging.LogRecord.
# Si se pasan dentro de extra={...}, logging puede lanzar KeyError:
# "Attempt to overwrite 'message' in LogRecord".
_RESERVED_LOG_RECORD_KEYS = set(
    logging.LogRecord(
        name="",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__.keys()
) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Formatter JSON simple para Cloud Run / Cloud Logging."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(record.created),
            ),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _RESERVED_LOG_RECORD_KEYS:
                continue

            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = str(value)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    level: str = "INFO",
    json_logs: bool = True,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """
    Configura logging para apps Cloud Run.

    Uso recomendado en main.py:

        from miad_rag_common.logging.structured_logging import configure_logging

        logger = configure_logging(level="INFO", json_logs=True)

    Si logger_name es None, configura el root logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level.upper())

    # Evita handlers duplicados al recargar en dev o tests.
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        )

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Retorna un logger por módulo.

    Ejemplo:

        logger = get_logger(__name__)
    """
    return logging.getLogger(name)


def _sanitize_extra(extra: dict[str, Any]) -> dict[str, Any]:
    """
    Evita que extra sobrescriba claves internas de LogRecord.

    Si llega una clave reservada, se renombra con prefijo 'extra_'.
    """
    sanitized: dict[str, Any] = {}

    for key, value in extra.items():
        safe_key = f"extra_{key}" if key in _RESERVED_LOG_RECORD_KEYS else key
        sanitized[safe_key] = value

    return sanitized


def log_event(
    logger: logging.Logger,
    message: str,
    severity: str = "info",
    **extra: Any,
) -> None:
    """
    Helper para loggear eventos estructurados.

    Ejemplo:

        log_event(
            logger,
            "recommendation_completed",
            request_id=request_id,
            latency_ms=1234,
            collection="realstate_mvd",
        )
    """
    log_fn = getattr(logger, severity.lower(), logger.info)
    log_fn(message, extra=_sanitize_extra(extra))
