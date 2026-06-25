"""Structured logging setup.

All modules obtain a logger via :func:`get_logger` rather than using ``print``
(BUILD_PLAN §6). The first call configures a single stream handler with a
consistent, timestamped format; subsequent calls reuse it. Standard library
only.
"""

from __future__ import annotations

import logging
import os
import sys

__all__ = ["get_logger", "configure_logging"]

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def configure_logging(level: int | str | None = None, fmt: str = _DEFAULT_FORMAT) -> None:
    """Configure the root ``cloudremoval`` logging handler once (idempotent).

    Args:
        level: Logging level (int or name). Defaults to the ``CLOUDREMOVAL_LOG``
            environment variable, else ``INFO``.
        fmt: ``logging`` format string.
    """
    global _CONFIGURED
    if level is None:
        level = os.environ.get("CLOUDREMOVAL_LOG", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger("cloudremoval")
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(fmt, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger under the ``cloudremoval`` root.

    Args:
        name: Usually ``__name__``. A leading ``cloudremoval`` (or the module's
            full dotted path) is normalised to live under the package root so all
            loggers share one handler/level.

    Returns:
        A configured :class:`logging.Logger`.
    """
    if not _CONFIGURED:
        configure_logging()
    if not name or name == "__main__":
        return logging.getLogger("cloudremoval")
    if name.startswith("cloudremoval"):
        return logging.getLogger(name)
    return logging.getLogger(f"cloudremoval.{name}")
