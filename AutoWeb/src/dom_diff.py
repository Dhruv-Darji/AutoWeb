"""
dom_diff.py – DOM delta algorithm.

Compares two DOMState snapshots and classifies elements as added, removed,
or modified.  A combined ``DOMDeltaProcessor`` class integrates snapshot
capture + diff + relevance filtering into a single pipeline object that the
Prune4Web runner can use.

Usage example (inside run_prune4web_live):
    from src.dom_state import capture_dom_state
    from src.dom_diff import DOMDeltaProcessor

    processor = DOMDeltaProcessor()

    # before action
    before = capture_dom_state(html, url=url, title=title, timestamp=t0)

    # ... execute browser action ...

    # after action
    after = capture_dom_state(html2, url=url2, title=title2, timestamp=t1)

    delta = processor.compute_delta(before, after)
    relevant = processor.get_relevant_elements(delta, sub_task)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from AutoWeb.src.dom_state import DOMState, SnapshotElement, capture_dom_state
except ImportError:
    from src.dom_state import DOMState, SnapshotElement, capture_dom_state


# ---------------------------------------------------------------------------
# Delta data classes
# ---------------------------------------------------------------------------
@dataclass
class ElementChange:
    """Records what changed between two snapshots for a single element."""
    uid: str
    before: Optional[SnapshotElement]   # None if newly added
    after: Optional[SnapshotElement]    # None if removed
    changed_attrs: List[str] = field(default_factory=list)

    @property
    def change_type(self) -> str:
        if self.before is None:
            return "added"
        if self.after is None:
            return "removed"
        return "modified"

    def summary(self) -> str:
        el = self.after or self.before
        typ = self.change_type
        base = el.to_summary() if el else f"[{self.uid[:6]}]"
        if typ == "modified":
            return f"{base}  (changed: {', '.join(self.changed_attrs)})"
        return f"{base}  ({typ})"


@dataclass
class DOMDelta:
    """Full diff between two DOM snapshots."""
    added: List[SnapshotElement] = field(default_factory=list)
    removed: List[SnapshotElement] = field(default_factory=list)
    modified: List[ElementChange] = field(default_factory=list)
    unchanged: List[SnapshotElement] = field(default_factory=list)

    # token reduction bookkeeping
    before_count: int = 0
    after_count: int = 0

    @property
    def changed_count(self) -> int:
        return len(self.added) + len(self.removed) + len(self.modified)

    @property
    def reduction_factor(self) -> float:
        """How many times fewer elements vs full after-state."""
        denom = max(len(self.added) + len(self.modified), 1)
        return self.after_count / denom

    def print_summary(self) -> None:
        print(
            f"DOM delta: before={self.before_count} after={self.after_count} "
            f"added={len(self.added)} removed={len(self.removed)} "
            f"modified={len(self.modified)} unchanged={len(self.unchanged)} "
            f"reduction={self.reduction_factor:.1f}x"
        )


# ---------------------------------------------------------------------------
# Core diff logic
# ---------------------------------------------------------------------------
def _changed_attributes(before: SnapshotElement, after: SnapshotElement) -> List[str]:
    """Return list of attribute names that changed between two snapshots."""
    attrs = [
        "text", "aria_label", "placeholder", "value",
        "disabled", "checked", "selected", "href", "role",
        "elem_class", "input_type",
    ]
    return [a for a in attrs if getattr(before, a) != getattr(after, a)]


def compute_delta(before: DOMState, after: DOMState) -> DOMDelta:
    """
    Diff two DOMState snapshots.

    Uses uid (structural hash) as the stable element identity key.
    Content changes are detected by comparing the element fingerprint.
    """
    delta = DOMDelta(
        before_count=before.element_count,
        after_count=after.element_count,
    )

    before_uids = set(before.elements.keys())
    after_uids = set(after.elements.keys())

    # Added elements (in after, not in before)
    for uid in after_uids - before_uids:
        delta.added.append(after.elements[uid])

    # Removed elements (in before, not in after)
    for uid in before_uids - after_uids:
        delta.removed.append(before.elements[uid])

    # Present in both – check fingerprints
    for uid in before_uids & after_uids:
        b = before.elements[uid]
        a = after.elements[uid]
        if b.fingerprint() != a.fingerprint():
            changed = _changed_attributes(b, a)
            delta.modified.append(ElementChange(uid=uid, before=b, after=a, changed_attrs=changed))
        else:
            delta.unchanged.append(a)

    return delta


# ---------------------------------------------------------------------------
# Relevance filtering (task-context aware)
# ---------------------------------------------------------------------------
_STOP_WORDS = frozenset(
    {"the", "a", "an", "to", "of", "in", "and", "or", "on", "for",
     "is", "it", "at", "by", "be", "do", "so", "if", "as"}
)


def _task_keywords(task_context: str) -> List[str]:
    """Extract lower-cased content words from the task description."""
    words = task_context.lower().split()
    return [w.strip(".,;:'\"()") for w in words if w not in _STOP_WORDS and len(w) > 2]


def _element_matches_task(el: SnapshotElement, keywords: List[str]) -> bool:
    text_blob = " ".join([
        el.text, el.aria_label, el.placeholder,
        el.elem_id, el.name, el.title, el.href,
    ]).lower()
    return any(kw in text_blob for kw in keywords)


def get_relevant_elements(
    delta: DOMDelta,
    task_context: str,
    include_unchanged: bool = False,
    use_embeddings: Optional[bool] = None,
    top_k: int = 20,
) -> List[SnapshotElement]:
    """
    Return elements most relevant to the current task step.

    Priority order:
    1. Added elements that match task keywords (highest signal)
    2. Modified elements (something changed — likely due to previous action)
    3. Unchanged elements that match task keywords (if include_unchanged=True)

    Parameters
    ----------
    delta:             Output of compute_delta().
    task_context:      The planner's sub-task string for this step.
    include_unchanged: Also include unchanged elements that keyword-match the task.
                       Set True for the first step (no before-state) or when
                       delta has very few changed elements.
    use_embeddings:    If True, re-rank the final pool with MiniLM sentence
                       embeddings (SBERT-style). Defaults to the environment
                       flag DOM_DELTA_USE_EMBEDDINGS=1. See
                       `AutoWeb/src/dom_relevance_embed.py`.
    top_k:             Maximum number of elements to return when embedding
                       re-ranking is enabled. Ignored for the pure-keyword path.
    """
    keywords = _task_keywords(task_context)
    relevant: List[SnapshotElement] = []

    # Added elements – filter by task keywords
    for el in delta.added:
        if not keywords or _element_matches_task(el, keywords):
            relevant.append(el)

    # Modified elements – always include (their change is likely task-related)
    for change in delta.modified:
        if change.after:
            relevant.append(change.after)

    # Unchanged elements (optional, keyword-filtered)
    if include_unchanged or not relevant:
        for el in delta.unchanged:
            if not keywords or _element_matches_task(el, keywords):
                relevant.append(el)

    # De-duplicate by uid (modified could also appear in added in edge cases)
    seen: set = set()
    deduped: List[SnapshotElement] = []
    for el in relevant:
        if el.uid not in seen:
            seen.add(el.uid)
            deduped.append(el)

    # Optional embedding re-rank (Improvement B.2)
    if use_embeddings is None:
        use_embeddings = os.getenv("DOM_DELTA_USE_EMBEDDINGS", "").strip() in {"1", "true", "yes"}
    if use_embeddings and deduped and task_context:
        try:
            from AutoWeb.src.dom_relevance_embed import score_elements_semantic
        except ImportError:
            from src.dom_relevance_embed import score_elements_semantic  # type: ignore
        ranked = score_elements_semantic(task_context, deduped, top_k=top_k)
        deduped = [el for el, _ in ranked]

    return deduped


# ---------------------------------------------------------------------------
# DOMDeltaProcessor – high-level pipeline object
# ---------------------------------------------------------------------------
class DOMDeltaProcessor:
    """
    Stateful processor that tracks the previous DOM snapshot and computes
    deltas on each new page state.

    Integrates into the Prune4Web pipeline as a drop-in replacement for
    full-DOM processing.
    """

    def __init__(self) -> None:
        self._prev_state: Optional[DOMState] = None
        self.deltas: List[DOMDelta] = []

    @property
    def has_baseline(self) -> bool:
        return self._prev_state is not None

    def capture_state(
        self,
        html: str,
        url: str = "",
        title: str = "",
        timestamp: float = 0.0,
    ) -> DOMState:
        """Parse *html* and store as the new current state."""
        import time as _time
        ts = timestamp or _time.time()
        return capture_dom_state(html, url=url, title=title, timestamp=ts)

    def update_and_diff(
        self,
        html: str,
        url: str = "",
        title: str = "",
        timestamp: float = 0.0,
    ) -> Tuple[DOMState, Optional[DOMDelta]]:
        """
        Capture a new DOM state, compute the delta against the previous state,
        update the internal baseline, and return both.

        Returns (new_state, delta).  delta is None on the very first call
        (no baseline to compare against).
        """
        new_state = self.capture_state(html, url=url, title=title, timestamp=timestamp)
        delta: Optional[DOMDelta] = None

        if self._prev_state is not None:
            delta = compute_delta(self._prev_state, new_state)
            self.deltas.append(delta)

        self._prev_state = new_state
        return new_state, delta

    def get_relevant_elements(
        self,
        delta: Optional[DOMDelta],
        task_context: str,
        full_state: Optional[DOMState] = None,
        use_embeddings: Optional[bool] = None,
        top_k: int = 20,
    ) -> List[SnapshotElement]:
        """
        Convenience wrapper.  When *delta* is None (first step), falls back
        to keyword-filtering the full_state or previous state.
        """
        if use_embeddings is None:
            use_embeddings = os.getenv("DOM_DELTA_USE_EMBEDDINGS", "").strip() in {"1", "true", "yes"}

        if delta is not None:
            # Always include unchanged on first real delta (few changed nodes)
            include_unch = delta.changed_count < 5
            return get_relevant_elements(
                delta, task_context,
                include_unchanged=include_unch,
                use_embeddings=use_embeddings, top_k=top_k,
            )

        # No delta yet – keyword-filter the entire current state
        state = full_state or self._prev_state
        if state is None:
            return []
        keywords = _task_keywords(task_context)
        pool = list(state.elements.values()) if not keywords else [
            el for el in state.elements.values()
            if _element_matches_task(el, keywords)
        ] or list(state.elements.values())

        if use_embeddings and task_context and pool:
            try:
                from AutoWeb.src.dom_relevance_embed import score_elements_semantic
            except ImportError:
                from src.dom_relevance_embed import score_elements_semantic  # type: ignore
            ranked = score_elements_semantic(task_context, pool, top_k=top_k)
            pool = [el for el, _ in ranked]
        return pool
