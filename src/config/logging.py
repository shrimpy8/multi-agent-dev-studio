"""Structured logging configuration using structlog.

Usage::

    from src.config.logging import get_logger

    logger = get_logger(__name__)
    logger.info("event_name", key="value")
"""

import logging
import os
import sys

import structlog


def configure_logging() -> None:
    """Configure structlog for the application.

    Reads LOG_LEVEL from the environment (default: INFO).
    Emits JSON-formatted log records to stderr.
    Must be called once at application startup before any logging occurs.
    """
    # Direct env read: logging must be configured before OrchestratorConfig loads
    # (which validates ANTHROPIC_API_KEY), so settings can't be used here.
    log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    _VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level_name not in _VALID_LEVELS:
        import warnings
        warnings.warn(f"Unknown LOG_LEVEL={log_level_name!r}, defaulting to INFO")
        log_level_name = "INFO"
    log_level = getattr(logging, log_level_name)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the given module name.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured structlog BoundLogger instance.
    """
    return structlog.get_logger(name)
