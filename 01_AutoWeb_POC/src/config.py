"""
Configuration management for SeeAct project.

Loads configuration from .env file with fallback to defaults.
"""

import os
from pathlib import Path
from typing import Optional

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    
    # Look for .env file in project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded configuration from {env_path}")
    else:
        print(f"ℹ No .env file found at {env_path}. Using defaults or command-line arguments.")
except ImportError:
    print("ℹ python-dotenv not installed. Install with: pip install python-dotenv")
    print("  Using defaults or command-line arguments only.")


def get_model_path(default: Optional[str] = None) -> Optional[str]:
    """
    Get the model path from environment or default.
    
    Args:
        default: Default path if not found in environment
    
    Returns:
        Path to model directory or None
    """
    return os.getenv("MODEL_PATH", default)


def get_dataset_path(default: Optional[str] = None) -> Optional[str]:
    """
    Get the dataset path from environment or default.
    
    Args:
        default: Default path if not found in environment
    
    Returns:
        Path to dataset directory or None
    """
    return os.getenv("DATASET_PATH", default)


def get_results_path(default: str = "../results") -> str:
    """
    Get the results output path from environment or default.
    
    Args:
        default: Default path if not found in environment
    
    Returns:
        Path to results directory
    """
    return os.getenv("RESULTS_PATH", default)


def print_config():
    """Print current configuration."""
    print("\n" + "=" * 80)
    print("Current Configuration")
    print("=" * 80)
    print(f"MODEL_PATH:    {get_model_path() or '(not set)'}")
    print(f"DATASET_PATH:  {get_dataset_path() or '(not set)'}")
    print(f"RESULTS_PATH:  {get_results_path()}")
    print("=" * 80 + "\n")
