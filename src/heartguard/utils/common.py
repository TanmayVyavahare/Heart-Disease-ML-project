"""
HeartGuard Utility Functions
Common helpers used across the ML pipeline.
"""

import os
import json
import joblib
from typing import Any

from src.heartguard.logger import get_logger
from src.heartguard.exception import HeartGuardException

logger = get_logger(__name__)


def save_object(obj: Any, file_path: str) -> None:
    """
    Serialize and save a Python object using joblib.
    
    Args:
        obj: Object to save (model, preprocessor, etc.)
        file_path: Destination path for the .pkl file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object saved to {file_path}")
    except Exception as e:
        raise HeartGuardException(f"Failed to save object to {file_path}: {e}")


def load_object(file_path: str) -> Any:
    """
    Load a serialized Python object using joblib.
    
    Args:
        file_path: Path to the .pkl file
        
    Returns:
        The deserialized Python object
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        obj = joblib.load(file_path)
        logger.info(f"Object loaded from {file_path}")
        return obj
    except FileNotFoundError:
        raise
    except Exception as e:
        raise HeartGuardException(f"Failed to load object from {file_path}: {e}")


def save_json(data: dict, file_path: str) -> None:
    """
    Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Destination path for the .json file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4, default=str)
        logger.info(f"JSON saved to {file_path}")
    except Exception as e:
        raise HeartGuardException(f"Failed to save JSON to {file_path}: {e}")


def load_json(file_path: str) -> dict:
    """
    Load a dictionary from a JSON file.
    
    Args:
        file_path: Path to the .json file
        
    Returns:
        Dictionary with the JSON contents
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HeartGuardException(f"Failed to load JSON from {file_path}: {e}")
