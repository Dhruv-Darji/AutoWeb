"""
Persistent processed-task checkpoint store.

Stores a JSON array of processed annotation IDs on disk so batch runs can
skip already-processed tasks across process restarts.

Usage:
    from AutoWeb.src.utils.processed_store import ProcessedStore
    store = ProcessedStore()                  # default path: AutoWeb/state/processed_tasks.json
    store.contains(anno_id)
    store.add(anno_id)

Behavior:
- File written atomically (write temp -> os.replace)
- Safe to call add() repeatedly
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Iterable, Set, Optional

from AutoWeb.src.logger import logger


class ProcessedStore:
    def __init__(self, path: Optional[Path | str] = None):
        # default location: <repo-root>/state/processed_tasks.json
        default = Path(__file__).parent.parent / "state" / "processed_tasks.json"
        self.path = Path(path) if path is not None else default
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ids: Set[str] = set()
        self._load()

    # -----------------------
    # Internal helpers
    # -----------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._ids = set(str(x) for x in data if x)
            else:
                logger.warning("ProcessedStore: unexpected file format, starting fresh: %s", self.path)
                self._ids = set()
        except Exception as e:
            logger.exception("ProcessedStore: failed to load '%s': %s", self.path, e)
            self._ids = set()

    def _atomic_save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(sorted(list(self._ids)), f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception as e:
            logger.exception("ProcessedStore: failed to save '%s': %s", self.path, e)
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

    # -----------------------
    # Public API
    # -----------------------
    def contains(self, annotation_id: str) -> bool:
        return str(annotation_id) in self._ids

    def add(self, annotation_id: str) -> None:
        aid = str(annotation_id)
        if aid in self._ids:
            return
        self._ids.add(aid)
        self._atomic_save()

    def add_many(self, ids: Iterable[str]) -> None:
        changed = False
        for i in ids:
            si = str(i)
            if si not in self._ids:
                self._ids.add(si)
                changed = True
        if changed:
            self._atomic_save()

    def all(self) -> Set[str]:
        return set(self._ids)

    def count(self) -> int:
        return len(self._ids)


# module-level convenience
_default_store: Optional[ProcessedStore] = None

def get_default_store() -> ProcessedStore:
    global _default_store
    if _default_store is None:
        _default_store = ProcessedStore()
    return _default_store
