"""Logger module for the project."""

import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Optional


def setup_logger(
    logger_name: str = "project_template",
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger that logs to both console and file.

    Args:
        logger_name: Name of the logger
        log_level: Logging level (default: INFO)
        log_file: Path to log file (default: logs/project_template.log)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file is provided or use default)
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / f"{logger_name}.log")

    # Ensure log directory exists
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


# Default logger instance
logger = setup_logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Optional name for the logger (default: None)

    Returns:
        Logger instance
    """
    log_level_str = os.environ.get("LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_str, logging.INFO)

    if name:
        return setup_logger(f"project_template.{name}", log_level)
    return logger


class LogLevel(str, Enum):
    """Log levels for the application."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
