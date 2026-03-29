"""Zenyx logging utilities."""

from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(
    level: int | str = logging.INFO,
    name: str = "zenyx",
) -> logging.Logger:
    """Set up logging for Zenyx.

    Parameters
    ----------
    level : int | str
        Logging level (e.g., logging.INFO, "DEBUG").
    name : str
        Logger name. Default: "zenyx".

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name.

    Parameters
    ----------
    name : str
        Logger name (typically module name).

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    return logging.getLogger(name)


def set_log_level(name: str, level: int | str) -> None:
    """Set log level for a named logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : int | str
        Logging level.
    """
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
