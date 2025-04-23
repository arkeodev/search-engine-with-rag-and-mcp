"""Environment variable loading utilities."""
import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

from src.utils.logger import logger


def load_env(env_file: Optional[str] = None) -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Path to .env file (default: None, uses .env in project root)
        
    Returns:
        Dictionary of environment variables that were loaded
    """
    if env_file is None:
        # Try to find .env file in project root
        env_path = Path(".env")
        if not env_path.exists():
            logger.warning(".env file not found in current directory")
            return {}
        env_file = str(env_path)
    
    # Load environment variables
    load_dotenv(env_file)
    logger.info(f"Loaded environment variables from {env_file}")
    
    # Return loaded variables (filtering out sensitive ones for logging)
    sensitive_keys = {"API_KEY", "SECRET_KEY", "PASSWORD", "TOKEN"}
    loaded_vars = {}
    
    for key, value in os.environ.items():
        if any(sensitive in key.upper() for sensitive in sensitive_keys):
            loaded_vars[key] = "******"  # Mask sensitive values
        else:
            loaded_vars[key] = value
    
    # Only log non-sensitive variables
    safe_vars = {k: v for k, v in loaded_vars.items() if v != "******"}
    if safe_vars:
        logger.debug(f"Loaded environment variables: {safe_vars}")
    
    return loaded_vars 