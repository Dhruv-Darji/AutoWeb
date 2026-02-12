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


def get_device(default: str = "auto") -> str:
    """
    Get the device to use for inference from environment or default.
    
    Args:
        default: Default device if not found in environment
    
    Returns:
        Device string: "cuda", "cpu", or "auto"
    """
    device = os.getenv("DEVICE", default)
    if device.lower() == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            # If torch not installed, default to CPU
            return "cpu"
    return device


def get_model_dtype(default: str = "fp16") -> str:
    """
    Get the model dtype from environment or default.
    
    Args:
        default: Default dtype if not found in environment
    
    Returns:
        Dtype string: "fp16" or "float32"
    """
    return os.getenv("MODEL_DTYPE", default)


def get_use_8bit(default: bool = False) -> bool:
    """
    Get whether to use 8-bit quantization from environment or default.
    
    Args:
        default: Default value if not found in environment
    
    Returns:
        Boolean indicating whether to use 8-bit quantization
    """
    value = os.getenv("USE_8BIT", str(default)).lower()
    return value in ("true", "1", "yes")


def print_config():
    """Print current configuration."""
    print("\n" + "=" * 80)
    print("Current Configuration")
    print("=" * 80)
    print(f"MODEL_PATH:    {get_model_path() or '(not set)'}")
    print(f"DATASET_PATH:  {get_dataset_path() or '(not set)'}")
    print(f"RESULTS_PATH:  {get_results_path()}")
    print(f"DEVICE:        {get_device()}")
    print(f"MODEL_DTYPE:   {get_model_dtype()}")
    print(f"USE_8BIT:      {get_use_8bit()}")
    print("=" * 80 + "\n")
