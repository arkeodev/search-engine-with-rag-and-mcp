"""Logger module for the project."""

import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

# Dictionary to store configured loggers
_loggers: Dict[str, logging.Logger] = {}
# Root logger name
ROOT_LOGGER_NAME = "project_template"


class LogLevel(str, Enum):
    """Log levels for the application."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def configure_all_loggers(log_level: LogLevel = LogLevel.INFO) -> None:
    """Configure all application loggers and third-party loggers."""
    level = getattr(logging, log_level.value)

    # Reset the root Python logger to avoid duplicate logs
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Configure the project's root logger
    project_root = logging.getLogger(ROOT_LOGGER_NAME)
    project_root.setLevel(level)
    project_root.propagate = False  # Don't propagate to Python's root logger

    # Reset handlers on all child loggers to avoid duplicates
    for name, logger in _loggers.items():
        # Set proper log level
        logger.setLevel(level)

        # Make sure no child logger propagates to the root Python logger
        if name == ROOT_LOGGER_NAME:
            logger.propagate = False
        else:
            # Child loggers should propagate to our project root, not Python's root
            logger.propagate = True

    # Configure third-party loggers to reduce noise at non-debug levels
    if level > logging.DEBUG:
        for library in ["httpx", "httpcore", "urllib3", "requests"]:
            third_party = logging.getLogger(library)
            third_party.setLevel(logging.WARNING)
            third_party.propagate = False  # Don't propagate to root


def setup_logger(
    logger_name: str = ROOT_LOGGER_NAME,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Set up a logger that logs to both console and file."""
    # Check if logger already exists
    if logger_name in _loggers:
        return _loggers[logger_name]

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # If it's not the root project logger, it should propagate to the project root
    is_project_root = logger_name == ROOT_LOGGER_NAME
    logger.propagate = not is_project_root

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    # Console handler - only add for root project logger to avoid duplicates
    if is_project_root:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler (if log_file is provided or use default)
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / f"{logger_name.replace('.', '_')}.log")

    # Ensure log directory exists
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Only add file handler to the project root to avoid duplicates in files
    if is_project_root:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Store the logger in our dictionary
    _loggers[logger_name] = logger

    return logger


# Default logger instance - this is the project root logger
logger = setup_logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance."""
    log_level_str = os.environ.get("LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_str, logging.INFO)

    if not name:
        return _loggers.get(ROOT_LOGGER_NAME, setup_logger(ROOT_LOGGER_NAME, log_level))

    # Format child logger name
    logger_name = f"{ROOT_LOGGER_NAME}.{name}"

    # Return existing logger if available
    if logger_name in _loggers:
        return _loggers[logger_name]

    # Create a new logger
    return setup_logger(logger_name, log_level)
