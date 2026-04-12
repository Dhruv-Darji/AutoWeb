"""
Centralized logger for SeeAct pipeline.
Writes logs to a local file (`seeact.log` in the AutoWeb project root) and
also streams to the console. Honor environment variables:
  - SEEACT_LOG_LEVEL  (DEBUG|INFO|WARNING|ERROR)
  - SEEACT_LOG_FILE   (path to log file)

Usage:
    from AutoWeb.src.logger import logger
    logger.info("message")
    logger.warning("warn")

This module keeps configuration minimal and idempotent so importing it
from multiple modules is safe.
"""
from __future__ import annotations
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_LEVEL = os.getenv("SEEACT_LOG_LEVEL", "INFO").upper()
# Default log file placed under the project's `logs/` directory per request
DEFAULT_LOG_PATH = Path(__file__).parent.parent / "logs" / "seeact.log"
LOG_FILE = Path(os.getenv("SEEACT_LOG_FILE", str(DEFAULT_LOG_PATH))).resolve()

# Create logger
logger = logging.getLogger("seeact")
if not logger.handlers:
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # File handler with rotation
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(str(LOG_FILE), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False

# Convenience: expose simple getter
def get_logger(name: str | None = None) -> logging.Logger:
    return logger if not name else logger.getChild(name)
