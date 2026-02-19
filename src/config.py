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
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        from AutoWeb.src.logger import logger
        logger.info(f"✓ Loaded configuration from {env_path}")
    else:
        from AutoWeb.src.logger import logger
        logger.info(f"ℹ No .env file found at {env_path}. Using defaults or command-line arguments.")
except ImportError:
    from AutoWeb.src.logger import logger
    logger.warning("ℹ python-dotenv not installed. Install with: pip install python-dotenv")
    logger.info("  Using defaults or command-line arguments only.")


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


def get_deberta_model_path(default: Optional[str] = None) -> Optional[str]:
    return os.getenv("DEBERTA_MODEL_PATH", default)


def get_crossencoder_model_path(default: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> str:
    """Local or HuggingFace path for the pre-trained cross-encoder."""
    return os.getenv("CROSSENCODER_MODEL_PATH", default)


def get_openai_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY")


def get_openai_model(default: str = "gpt-4o-mini") -> str:
    return os.getenv("OPENAI_MODEL", default)


def get_supabase_url() -> Optional[str]:
    return os.getenv("SUPABASE_URL")


def get_supabase_service_role_key() -> Optional[str]:
    return os.getenv("SUPABASE_SERVICE_ROLE_KEY")


def get_supabase_bucket(default: str = "ai-preprocessed-steps") -> str:
    return os.getenv("SUPABASE_BUCKET", default)


def get_policy_hub_path(default: Optional[str] = None) -> Optional[str]:
    """Path to the PolicyHub JSON file.

    Set via ``POLICY_HUB_PATH`` in .env.  Defaults to
    ``src/policyHub/policies.json`` relative to the project root.
    """
    val = os.getenv("POLICY_HUB_PATH", default)
    if val:
        return val
    # Fallback: policies.json next to the policyHub package
    return str(Path(__file__).parent / "policyHub" / "policies.json")


def get_hitl_threshold(default: int = 75) -> int:
    """Composite confidence threshold for HITL triggering (0–100).

    Steps with composite confidence below this value are flagged for
    Human-in-the-Loop review.  Set via ``HITL_THRESHOLD`` in .env.
    """
    try:
        return int(os.getenv("HITL_THRESHOLD", str(default)))
    except (TypeError, ValueError):
        return default


def get_seeact_image_detail(default: str = "low") -> str:
    """Image detail level for action-generation GPT calls.

    Controls OpenAI's vision *detail* parameter:
      - ``"high"`` — full tile analysis (~25K tokens per image, best quality)
      - ``"low"``  — 512×512 thumbnail (85 tokens, ~99% cheaper but lower quality)
      - ``"auto"`` — let OpenAI decide

    Set via `SEEACT_IMAGE_DETAIL` in .env.  Grounding calls always skip the
    image entirely for API models (DeBERTa handles ranking).
    """
    val = os.getenv("SEEACT_IMAGE_DETAIL", default).strip().lower()
    if val not in ("high", "low", "auto"):
        from AutoWeb.src.logger import logger
        logger.warning(f"  ⚠ Unknown SEEACT_IMAGE_DETAIL='{val}', falling back to '{default}'")
        return default
    return val


def print_config():
    """Print current configuration."""
    from AutoWeb.src.logger import logger
    logger.info("\n" + "=" * 80)
    logger.info("Current Configuration")
    logger.info("=" * 80)
    logger.info(f"MODEL_PATH:    {get_model_path() or '(not set)'}")
    logger.info(f"DATASET_PATH:  {get_dataset_path() or '(not set)'}")
    logger.info(f"RESULTS_PATH:  {get_results_path()}")
    logger.info(f"DEVICE:        {get_device()}")
    logger.info(f"MODEL_DTYPE:   {get_model_dtype()}")
    logger.info(f"USE_8BIT:      {get_use_8bit()}")
    logger.info(f"POLICY_HUB:    {get_policy_hub_path() or '(default)'}")
    logger.info(f"HITL_THRESHOLD:{get_hitl_threshold()}")
    logger.info(f"RUNNER_MODE:   {get_runner_mode()}")
    logger.info(f"USE_GPT:       {get_use_gpt()}")
    logger.info(f"LIVE_SITES:    {get_live_sites_path()}")
    logger.info(f"BROWSER_HEADLESS: {get_browser_headless()}")
    logger.info(f"ANNOTATION_ID: {get_annotation_id() or '(not set)'}")
    logger.info(f"DATASET_FILE:  {get_dataset_file() or '(not set)'}")
    logger.info("=" * 80 + "\n")


# ── Runner / live-mode settings ───────────────────────────────────

def get_runner_mode(default: str = "live") -> str:
    """Pipeline run mode: ``live`` or ``dataset``.  Set via ``RUNNER_MODE``."""
    val = os.getenv("RUNNER_MODE", default).strip().lower()
    if val not in ("live", "dataset"):
        return default
    return val


def get_use_gpt(default: bool = False) -> bool:
    """Whether to use GPT-4o (OpenAI API) instead of local Qwen.  Set via ``USE_GPT``."""
    return os.getenv("USE_GPT", str(default)).strip().lower() in ("true", "1", "yes")


def get_live_sites_path(default: Optional[str] = None) -> str:
    """Path to the live_sites.json config.  Set via ``LIVE_SITES_PATH``."""
    val = os.getenv("LIVE_SITES_PATH", default)
    if val:
        return val
    return str(Path(__file__).parent / "live_sites.json")


def get_browser_headless(default: bool = False) -> bool:
    """Run browser headless in live mode.  Set via ``BROWSER_HEADLESS``."""
    return os.getenv("BROWSER_HEADLESS", str(default)).strip().lower() in ("true", "1", "yes")


def get_annotation_id(default: Optional[str] = None) -> Optional[str]:
    """Specific annotation ID for dataset mode.  Set via ``ANNOTATION_ID``."""
    val = os.getenv("ANNOTATION_ID", default)
    return val if val else None


def get_dataset_file(default: Optional[str] = None) -> Optional[str]:
    """Parquet file name for dataset mode.  Set via ``DATASET_FILE``."""
    val = os.getenv("DATASET_FILE", default)
    return val if val else None


def get_force_reprocess(default: bool = False) -> bool:
    """Re-process already-checkpointed tasks.  Set via ``FORCE_REPROCESS``."""
    return os.getenv("FORCE_REPROCESS", str(default)).strip().lower() in ("true", "1", "yes")
