"""Compatibility shim for HITL gate import path used by AutoWeb."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from shared.hitl.hitl_confidence import HITLConfidenceGate
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from shared.hitl.hitl_confidence import HITLConfidenceGate

__all__ = ["HITLConfidenceGate"]
