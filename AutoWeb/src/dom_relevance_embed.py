"""
dom_relevance_embed.py — Sentence-embedding relevance for DOM elements.

Research basis
--------------
* Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks", EMNLP 2019 (arXiv:1908.10084).
* Wang et al., "MiniLM: Deep Self-Attention Distillation for Task-Agnostic
  Compression of Pre-Trained Transformers", NeurIPS 2020 (arXiv:2002.10957).

The default checkpoint is `sentence-transformers/all-MiniLM-L6-v2` — a
6-layer, 22M-parameter distilled encoder (~80 MB on disk) that runs
comfortably on CPU at ~2 ms per short sentence on a modern laptop.

Why this helps the DOM Delta pipeline
-------------------------------------
The current relevance filter (`AutoWeb/src/dom_diff.py::_element_matches_task`)
is a lowercase substring match + rapidfuzz partial ratio. It misses paraphrase
pairs that are routine on the web:

    task "purchase tickets"  ↔  button "Checkout"
    task "add to cart"       ↔  button "Add to Bag"   (Apple)
    task "sign in"           ↔  link   "Log in"
    task "submit application" ↔  button "Send"

MiniLM cosine similarity picks these up without any hand-written synonym list.

Public API
----------
    from AutoWeb.src.dom_relevance_embed import score_elements_semantic

    ranked = score_elements_semantic(query="Add to cart", elements=candidates)
    # -> list of (SnapshotElement, similarity) sorted desc

The encoder is loaded lazily on first call and cached for the process.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


_DEFAULT_MODEL = os.getenv(
    "DOM_DELTA_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
# Optional: override with a local snapshot path to avoid internet access
_LOCAL_MODEL_DIR = os.getenv("DOM_DELTA_EMBED_MODEL_DIR", "")

_model = None
_device: Optional[str] = None


def _load_model():
    """Load the MiniLM encoder once per process."""
    global _model, _device
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sentence-transformers not installed. Run:\n"
            "    pip install sentence-transformers"
        ) from exc

    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    model_src = _LOCAL_MODEL_DIR if _LOCAL_MODEL_DIR and Path(_LOCAL_MODEL_DIR).exists() else _DEFAULT_MODEL
    print(f"[dom_relevance_embed] Loading {model_src} on {_device} ...")
    _model = SentenceTransformer(model_src, device=_device)
    return _model


def encode(texts: Sequence[str]) -> np.ndarray:
    """Return a (N, D) L2-normalised embedding matrix."""
    model = _load_model()
    # Normalise so dot product == cosine similarity
    embs = model.encode(
        list(texts),
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embs


def score_elements_semantic(
    query: str,
    elements: Sequence,
    top_k: Optional[int] = None,
) -> List[Tuple[object, float]]:
    """
    Rank *elements* by semantic similarity to *query*.

    Each element must expose either:
      * ``.semantic_text()`` method (preferred — `SnapshotElement` has it), or
      * attributes the function can stitch together.

    Returns a list of (element, cosine_similarity) sorted descending. If
    *top_k* is set, the list is truncated.
    """
    if not elements:
        return []

    # Pull a natural-language-ish string for each element
    texts: List[str] = []
    for el in elements:
        if hasattr(el, "semantic_text") and callable(el.semantic_text):
            texts.append(el.semantic_text() or el.tag)
        else:
            # Best-effort fallback for plain ElementNode-style objects
            pieces = [getattr(el, a, "") for a in ("text", "aria_label",
                                                   "placeholder", "title",
                                                   "name", "elem_id")]
            texts.append(" ".join(p for p in pieces if p)[:240] or getattr(el, "tag", ""))

    # One query encode + one batched element encode
    q_emb = encode([query or ""])[0]         # (D,)
    el_embs = encode(texts)                  # (N, D)
    sims = el_embs @ q_emb                   # (N,) cosine (both normalised)

    order = np.argsort(-sims)
    scored = [(elements[i], float(sims[i])) for i in order]
    if top_k is not None:
        scored = scored[:top_k]
    return scored


def hybrid_rank(
    query: str,
    elements: Sequence,
    keyword_scores: Sequence[float],
    alpha: float = 0.6,
    top_k: Optional[int] = None,
) -> List[Tuple[object, float]]:
    """
    Combine keyword scores (already computed upstream) with semantic similarity.

    final_score = alpha * norm(keyword) + (1 - alpha) * cosine(query, element)

    Both components are min-max normalised to [0, 1] before blending, which
    is the late-fusion scheme recommended by SBERT-style retrieval papers.
    """
    if not elements:
        return []
    keyword_scores = np.asarray(keyword_scores, dtype=np.float32)
    k_min, k_max = float(keyword_scores.min()), float(keyword_scores.max())
    k_norm = (keyword_scores - k_min) / max(k_max - k_min, 1e-6)

    semantic = score_elements_semantic(query, elements)
    # Build a lookup: id(element) -> cosine
    sem_map = {id(el): s for el, s in semantic}
    sem_arr = np.asarray([sem_map.get(id(el), 0.0) for el in elements], dtype=np.float32)
    s_min, s_max = float(sem_arr.min()), float(sem_arr.max())
    s_norm = (sem_arr - s_min) / max(s_max - s_min, 1e-6)

    combined = alpha * k_norm + (1.0 - alpha) * s_norm
    order = np.argsort(-combined)
    scored = [(elements[i], float(combined[i])) for i in order]
    if top_k is not None:
        scored = scored[:top_k]
    return scored


if __name__ == "__main__":
    # Smoke test
    class _FakeEl:
        def __init__(self, text):
            self.text = text
            self.aria_label = ""
            self.placeholder = ""
            self.title = ""
            self.name = ""
            self.elem_id = ""
            self.tag = "button"
        def semantic_text(self) -> str:
            return self.text
    cands = [_FakeEl("Sign in"), _FakeEl("Log in"), _FakeEl("Create account"),
             _FakeEl("Search"), _FakeEl("Add to Bag")]
    ranked = score_elements_semantic("login to my account", cands, top_k=3)
    print("Query: 'login to my account'")
    for el, s in ranked:
        print(f"  {s:+.3f}  {el.text}")
