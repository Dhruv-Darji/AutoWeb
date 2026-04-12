"""Compatibility wrapper.

This file originally contained auto-extracted notebook cells with mojibake text.
Use the interactive script entrypoint in run_prune4web.py instead.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_prune4web import main


if __name__ == "__main__":
    main()
