from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Optional


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
            if key.startswith("_"):
                continue

            if key in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }:
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

    Uso:
      from miad_rag_common.logging.structured_logging import configure_logging
      logger = configure_logging()
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level.upper())

    # Evita handlers duplicados al recargar en dev.
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
    return logging.getLogger(name)


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
    log_fn(message, extra=extra)