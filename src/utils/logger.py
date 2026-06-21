"""
Structured JSON logger for HEMA ETL pipeline.

Outputs JSON lines compatible with CloudWatch Logs Insights.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON for CloudWatch ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Merge any extra fields passed via `extra=` kwarg
        for key, value in record.__dict__.items():
            if key not in {
                "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno",
                "funcName", "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process", "name", "message",
            }:
                log_entry[key] = value
        return json.dumps(log_entry, default=str)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Return a named logger with JSON formatting.

    Usage:
        logger = get_logger(__name__)
        logger.info("Job started", extra={"job": "bronze_ingest", "input_path": "s3://..."})
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Avoid duplicate handlers in Glue re-use

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger
