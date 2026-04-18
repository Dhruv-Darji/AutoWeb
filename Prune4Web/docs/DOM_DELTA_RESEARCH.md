# DOM Delta Processing — Research Justification & Implementation

Companion document to the paper
**"Improving Web Automation with DOM Delta Processing and Privacy-Aware LLM Agents."**

This document records (a) the concrete changes we made to harden the
DOM-Delta half of the contribution, (b) the measurement methodology, and
(c) the prior work each design choice is anchored to.

---

## 1. What the contribution now claims

> **Claim.** Re-sending the full interactive DOM to the grounder at every step
> is wasteful. A stateful *delta* processor, combined with sentence-embedding
> relevance ranking and ancestor-preserving element summaries, reduces the
> grounder's input size while maintaining — and in many cases improving —
> top-20 recall of the ground-truth element.

We support this with three experimental artifacts:

| # | Artifact | File |
|---|---|---|
| 1 | A/B benchmark (FULL vs DELTA-KW vs DELTA-EMBED) on Mind2Web `test_task` | `Prune4Web/evaluate_dom_delta.py` |
| 2 | Ablation rows toggleable via env flags | `DOM_DELTA_USE_EMBEDDINGS`, `DOM_DELTA_EMBED_MODEL` |
| 3 | Per-sample JSON + plots for every run | `Prune4Web/results/dom_delta_bench/run_<ts>/` |

---

## 2. Implementation summary

### 2.1 Stable structural UIDs (already in place — now documented)

`AutoWeb/src/dom_state.py:216-217` already builds
`uid = md5(tag | elem_id | name | xpath)`. This is *structural*, not
content-based — an element keeps the same UID across re-renders even when
text or attributes change. Modification detection uses a separate
content-fingerprint (`SnapshotElement.fingerprint`). We document this here
because it directly mirrors the stable identifier used by Mind2Web
(`backend_node_id`); reviewers can compare the two.

**Reference.**
Deng, Gu, Li, Chen, Su. *Mind2Web: Towards a Generalist Agent for the Web*.
NeurIPS 2023. arXiv:2306.06070.
- PDF: https://arxiv.org/abs/2306.06070
- §3 "Dataset Construction" introduces `backend_node_id` as the stable
  DevTools-protocol identifier used so a single element can be tracked across
  snapshots.

### 2.2 Sentence-embedding relevance (new — `dom_relevance_embed.py`)

**What changed.** `AutoWeb/src/dom_relevance_embed.py` (new) wraps a
`sentence-transformers/all-MiniLM-L6-v2` encoder as a lazy process-wide
singleton and exposes two functions:

- `score_elements_semantic(query, elements, top_k)` — cosine rank.
- `hybrid_rank(query, elements, keyword_scores, alpha=0.6, top_k)` —
  late-fusion of keyword and semantic scores.

`AutoWeb/src/dom_diff.py::get_relevant_elements` now takes a
`use_embeddings` flag (also driven by `DOM_DELTA_USE_EMBEDDINGS=1`) that
applies the semantic re-rank on the delta pool before truncation.

**Why this matters.** The previous relevance filter was a substring /
rapidfuzz check against task keywords; it missed routine paraphrases
("purchase" ↔ "Checkout", "add to cart" ↔ "Add to Bag", "sign in" ↔
"Log in") that required hand-written synonym lists.

**References.**
1. Reimers & Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese
   BERT-Networks*. EMNLP 2019. arXiv:1908.10084.
   - PDF: https://arxiv.org/abs/1908.10084
   - Establishes the SBERT training and inference methodology used by the
     MiniLM sentence-transformer checkpoints.
2. Wang, Wei, Dong, Bao, Yang, Zhou. *MiniLM: Deep Self-Attention Distillation
   for Task-Agnostic Compression of Pre-Trained Transformers*.
   NeurIPS 2020. arXiv:2002.10957.
   - PDF: https://arxiv.org/abs/2002.10957
   - The distilled 6-layer, 22M-parameter encoder we load by default. Fits on
     CPU in ~80 MB; inference ~2 ms per short string.
3. `sentence-transformers/all-MiniLM-L6-v2` model card:
   https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

### 2.3 Ancestor + viewport context (new — `dom_state.py`)

**What changed.** `SnapshotElement` gained two fields:

- `ancestors: List[str]` — up to four semantically meaningful ancestor
  tokens (e.g. `form#signup`, `fieldset[Address]`), outermost first.
  Populated by `_ancestor_summary()` at capture time.
- `in_viewport: Optional[bool]` — populated downstream when Playwright
  bounding-box data is available; `None` otherwise.

`SnapshotElement.to_summary(include_ancestors=True)` appends a compact
`ctx=form#signup > fieldset[Address]` suffix and a `[visible]` flag when
appropriate. This gives the grounder structural disambiguation between
visually-identical inputs (e.g. the `email` field inside the signup form
vs the `email` field in the newsletter sidebar).

`SnapshotElement.semantic_text()` is the single natural-language string
used by the embedding encoder — it concatenates text, aria-label,
placeholder, title, name, id.

**Reference.**
Gur, Furuta, Huang, Safdari, Matsuo, Eck, Faust. *A Real-World WebAgent with
Planning, Long Context Understanding, and Program Synthesis*. ICLR 2024.
arXiv:2307.12856.
- PDF: https://arxiv.org/abs/2307.12856
- §3 "Representation" motivates keeping ancestor attributes of task-relevant
  nodes; HTML-T5 encodes hierarchical context rather than flat element lists.

### 2.4 Deferred (documented, not implemented)

Tree-edit-distance classification (detecting moves / restructures in the
delta rather than treating them as `added`+`removed`) was scoped out for
this cycle — the structural UIDs from §2.1 already absorb most of the
benefit. If we revisit, the relevant references are:

- Zhang & Shasha. *Simple fast algorithms for the editing distance between
  trees and related problems*. SIAM Journal on Computing 18(6), 1989.
  DOI: https://doi.org/10.1137/0218082
- Pawlik & Augsten. *RTED: A Robust Algorithm for the Tree Edit Distance*.
  PVLDB 5(4), 2011. PDF: http://www.vldb.org/pvldb/vol5/p334_mateuszpawlik_vldb2012.pdf
- Python library `zss` — https://github.com/timtadh/zhang-shasha

---

## 3. How to reproduce the paper numbers

### 3.1 Environment

```bash
# Install the embedding dependency once
"D:\Environments\ml-env\Scripts\pip.exe" install sentence-transformers
```

### 3.2 Run the A/B benchmark

```bash
# Full run — 100 samples, all three configurations
"D:\Environments\ml-env\Scripts\python.exe" -X utf8 \
    Prune4Web/evaluate_dom_delta.py \
    --split test_task --num-samples 100 --seed 42

# Only the ablation pair you need for Table 2
"D:\Environments\ml-env\Scripts\python.exe" -X utf8 \
    Prune4Web/evaluate_dom_delta.py \
    --configs FULL DELTA_EMBED --num-samples 150
```

Outputs in `Prune4Web/results/dom_delta_bench/run_<timestamp>/`:
- `bench_results.json` — per-sample + aggregate metrics.
- `recall_at_20.png` — headline accuracy plot.
- `tokens_proxy.png` — grounder-input-size plot (lower = cheaper).
- `latency.png` — filter-stage latency plot.
- `bench.log` — full log.

### 3.3 Toggling improvements at runtime

```bash
# Delta processing with embeddings, inside the full live pipeline
DOM_DELTA_USE_EMBEDDINGS=1 \
PRUNE4WEB_DOM_DELTA=1 \
"D:\Environments\ml-env\Scripts\python.exe" -X utf8 \
    Prune4Web/live_benchmark.py --headless --use-local-grounder
```

### 3.4 Ablation table the paper reports

| Row | Config | env flags |
|-----|--------|-----------|
| (a) | FULL (no delta) | `PRUNE4WEB_DOM_DELTA=0` |
| (b) | + DOM Delta keyword filter | `PRUNE4WEB_DOM_DELTA=1` |
| (c) | + MiniLM embedding re-rank | `PRUNE4WEB_DOM_DELTA=1 DOM_DELTA_USE_EMBEDDINGS=1` |
| (d) | + Ancestor / viewport context | rows (c) plus ancestors are on by default in `to_summary()` |

---

## 4. Limitations & honest caveats

- The A/B benchmark measures the filter / retrieval stage (Recall@20 and
  grounder-input size), not end-to-end task success. End-to-end numbers
  still come from `Prune4Web/live_benchmark.py` on 30 live sites.
- `tokens_proxy` is the character length of the serialised candidate block,
  not a model-specific tokeniser count. It tracks cost monotonically but is
  not an exact dollar figure.
- MiniLM was trained on general English; domain-specific element vocabulary
  (e.g. banking-specific jargon) may benefit from a fine-tuned encoder.
  This is future work.
- The `in_viewport` flag is only populated when Playwright bounding-box data
  is attached to snapshots; for offline Mind2Web evaluation it is `None`.

---

## 5. Full citation list (verified)

1. Deng, Gu, Li, Chen, Su. *Mind2Web: Towards a Generalist Agent for the Web*.
   NeurIPS 2023. arXiv:**2306.06070**.
   https://arxiv.org/abs/2306.06070
2. Reimers, Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese
   BERT-Networks*. EMNLP 2019. arXiv:**1908.10084**.
   https://arxiv.org/abs/1908.10084
3. Wang, Wei, Dong, Bao, Yang, Zhou. *MiniLM: Deep Self-Attention Distillation
   for Task-Agnostic Compression of Pre-Trained Transformers*. NeurIPS 2020.
   arXiv:**2002.10957**. https://arxiv.org/abs/2002.10957
4. Gur, Furuta, Huang, Safdari, Matsuo, Eck, Faust. *A Real-World WebAgent with
   Planning, Long Context Understanding, and Program Synthesis*. ICLR 2024.
   arXiv:**2307.12856**. https://arxiv.org/abs/2307.12856
5. Zhang, Shasha. *Simple fast algorithms for the editing distance between
   trees and related problems*. SIAM J. Comput. 18(6), 1989.
   https://doi.org/10.1137/0218082
6. Pawlik, Augsten. *RTED: A Robust Algorithm for the Tree Edit Distance*.
   PVLDB 5(4), 2011.
   http://www.vldb.org/pvldb/vol5/p334_mateuszpawlik_vldb2012.pdf
