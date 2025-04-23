"""Shared pytest fixtures and configuration."""
import logging
import os
from typing import Dict, Generator

import pytest


@pytest.fixture(autouse=True)
def setup_logging() -> None:
    """Set up logging for tests."""
    logging.basicConfig(
        level=logging.DEBUG, 
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


@pytest.fixture
def env_vars() -> Generator[Dict[str, str], None, None]:
    """Provide environment variables and clean up after test."""
    # Save original environment
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        "DEBUG": "True",
        "LOG_LEVEL": "DEBUG",
        "APP_ENV": "test",
    }
    
    # Apply test environment variables
    for key, value in test_env.items():
        os.environ[key] = value
    
    # Provide the test environment to the test
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env) 