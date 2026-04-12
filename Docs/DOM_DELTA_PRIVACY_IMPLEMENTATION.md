# DOM Delta Processing & Privacy-Aware LLM – Implementation Reference

**Date:** April 11, 2026 (Day 3 of 20-day submission plan)
**Branch:** `feature/prune4web`
**Status:** Implementation complete, integration into pipeline done

---

## Overview

Two research contributions implemented on top of the Prune4Web baseline:

| Contribution | Key Claim | Files |
|---|---|---|
| **DOM Delta Processing** | Reduces tokens sent to LLM per step by focusing only on what changed | `src/dom_state.py`, `src/dom_diff.py` |
| **Privacy-Aware LLM** | Masks PII in DOM elements before they reach the LLM | `src/privacy_filters.py`, `src/privacy_aware_llm.py` |

Both are wired into `Prune4Web/run_prune4web.py` and can be toggled via environment variables.

---

## Architecture

```
Browser HTML (each step)
        │
        ▼
┌───────────────────────┐
│   DOM State Capture   │  src/dom_state.py
│   capture_dom_state() │
│   → DOMState snapshot │  hash-based element identity (uid = MD5 of tag|id|name|xpath)
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   DOM Delta Compute   │  src/dom_diff.py
│   compute_delta()     │
│   → DOMDelta          │  added / removed / modified / unchanged
│   get_relevant_elements()  keyword-filtered by planner sub-task
└──────────┬────────────┘
           │  (smaller candidate set – only what changed + task-relevant)
           ▼
┌───────────────────────┐
│  Programmatic Filter  │  Prune4Web keyword scoring (unchanged)
│  score_elements()     │  Works on delta-filtered candidates, not full DOM
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   Privacy Pipeline    │  src/privacy_filters.py
│   PIIDetector         │  regex: email, phone, SSN, credit card, Aadhaar, PAN …
│   ElementAnonymizer   │  hash IDs, mask PII text → [EMAIL_MASKED] etc.
│   DPScorer (optional) │  Laplace noise on scores (ε-DP)
└──────────┬────────────┘
           │  (anonymised candidates)
           ▼
┌───────────────────────┐
│  Privacy-Aware LLM    │  src/privacy_aware_llm.py
│  PrivacyAwareLLM      │  injects [PRIVACY DIRECTIVE] into system prompt
│                       │  scrubs any remaining PII from assembled messages
│                       │  forwards to raw llm_call (OpenAI API)
└──────────┬────────────┘
           │
           ▼
       GPT-4o / model
```

---

## Module Reference

### `src/dom_state.py`

**Purpose:** Hash-based DOM snapshot capture.

#### Key classes

```python
@dataclass
class SnapshotElement:
    uid: str          # MD5 of "tag|id|name|xpath" — stable across re-renders
    tag: str
    text: str
    aria_label: str
    placeholder: str
    elem_id: str
    name: str
    elem_class: str
    href: str
    value: str
    input_type: str
    role: str
    title: str
    disabled: bool
    checked: bool
    selected: bool
    xpath: str        # approximate structural path (display only)

    def fingerprint(self) -> str: ...   # MD5 of all content fields
    def to_summary(self) -> str: ...    # compact one-line repr

@dataclass
class DOMState:
    url: str
    title: str
    timestamp: float
    elements: Dict[str, SnapshotElement]   # uid → element
    order: List[str]                        # document order

    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, raw: str) -> DOMState: ...
```

#### Main function

```python
def capture_dom_state(html: str, url: str = "", title: str = "", timestamp: float = 0.0) -> DOMState:
    """Parse HTML and return a DOMState. Filters interactive elements only."""
```

---

### `src/dom_diff.py`

**Purpose:** Compare two DOMState snapshots and extract what changed.

#### Key classes

```python
@dataclass
class ElementChange:
    uid: str
    before: Optional[SnapshotElement]
    after: Optional[SnapshotElement]
    changed_attrs: List[str]
    change_type: str   # "added" | "removed" | "modified"

@dataclass
class DOMDelta:
    added: List[SnapshotElement]
    removed: List[SnapshotElement]
    modified: List[ElementChange]
    unchanged: List[SnapshotElement]
    before_count: int
    after_count: int
    reduction_factor: float   # after_count / (added + modified) — token savings
```

#### Main functions

```python
def compute_delta(before: DOMState, after: DOMState) -> DOMDelta:
    """Diff two snapshots. Uses uid as stable identity, fingerprint for content change."""

def get_relevant_elements(delta: DOMDelta, task_context: str, include_unchanged: bool = False) -> List[SnapshotElement]:
    """Return elements most relevant to the current step.
    Priority: added (task-keyword match) > modified > unchanged (keyword match)."""
```

#### Pipeline class

```python
class DOMDeltaProcessor:
    """Stateful: tracks previous snapshot, computes delta on each new HTML."""

    def update_and_diff(self, html, url, title, timestamp) -> Tuple[DOMState, Optional[DOMDelta]]:
        """Returns (new_state, delta). delta is None on first call."""

    def get_relevant_elements(self, delta, task_context, full_state=None) -> List[SnapshotElement]:
        """Falls back to keyword-filtering full state when delta is None."""
```

---

### `src/privacy_filters.py`

**Purpose:** PII detection, element anonymisation, and differential privacy scoring.

#### PIIDetector

Detects these patterns with regex:

| Pattern | Example |
|---------|---------|
| `email` | `user@example.com` |
| `phone_us` | `555-123-4567` |
| `ssn` | `123-45-6789` |
| `credit_card` | `4111 1111 1111 1111` |
| `ipv4` | `192.168.1.1` |
| `date_of_birth` | `01/15/1990` |
| `passport` | `A1234567` |
| `aadhaar` | `1234 5678 9012` |
| `pan_india` | `ABCDE1234F` |
| `sensitive_field_name` | input with `name="password"` |

```python
detector = PIIDetector()
report = detector.detect_in_element(element)   # → PIIReport
reports = detector.scan_elements(elements)     # → Dict[uid, PIIReport]
score = detector.compute_privacy_score(elements)  # fraction with PII, 0-1
```

#### ElementAnonymizer

```python
anonymizer = ElementAnonymizer(hash_ids=True, mask_text=True)
clean_el = anonymizer.anonymize(element, report=report)
# elem_id  → "id_3f2a1b7c"
# name     → "name_7c1b4d9e"
# text     → "Contact: [EMAIL_MASKED] or [PHONE_US_MASKED]"

clean_list, n_masked = anonymizer.anonymize_batch(elements, reports)
```

#### DPScorer

Adds calibrated Laplace noise to element relevance scores:

```python
scorer = DPScorer(epsilon=1.0, sensitivity=10.0)
# epsilon: privacy budget — lower = more noise = more privacy
# sensitivity: max possible score change between adjacent inputs

noisy_scores = scorer.add_noise(scores)
reranked_elements = scorer.privatise_ranking(scored_pairs, top_n=20)
```

#### PrivacyPipeline (unified entry point)

```python
config = PrivacyConfig(
    enable_pii_detection=True,
    enable_anonymization=True,
    enable_dp_scoring=False,    # True to add DP noise
    dp_epsilon=1.0,
)
pipeline = config.build()

processed_elements, reports = pipeline.process_elements(elements, scored_pairs, top_n=20)
print(pipeline.privacy_summary(elements))
```

---

### `src/privacy_aware_llm.py`

**Purpose:** Drop-in wrapper around `llm_call` that applies privacy preprocessing to every prompt.

#### What it does to each message

1. Prepends `[PRIVACY DIRECTIVE]` block to the system prompt
2. Scans candidate elements for PII, replaces their summaries in the user message
3. Regex-sweeps the fully assembled prompt for any missed PII
4. Forwards the cleaned messages to the real `llm_call`

#### Usage

```python
from src.privacy_aware_llm import build_privacy_aware_llm

privacy_llm = build_privacy_aware_llm(
    raw_llm_call=llm_call,
    enable_dp=False,     # True to also DP-noise rankings
    dp_epsilon=1.0,
)

# Use exactly like llm_call:
result = privacy_llm.call(messages, stage="grounder", model=GROUNDER_MODEL)

# After the run:
privacy_llm.print_privacy_summary()
```

#### Console output per call (when PII is found)

```
  [privacy:grounder] pii_matches=3 elements_masked=2 scrubbed=True
```

---

## Integration in `Prune4Web/run_prune4web.py`

### Environment flags

| Variable | Default | Effect |
|---|---|---|
| `PRUNE4WEB_DOM_DELTA` | `1` | Enable DOM delta processing |
| `PRUNE4WEB_PRIVACY` | `1` | Enable PII masking |
| `PRUNE4WEB_DP_EPSILON` | `0` | DP epsilon (`0` = DP disabled) |

### Per-step pipeline flow

```
1. Capture HTML → dom_state (capture_dom_state)
2. Compute delta vs. previous step (DOMDeltaProcessor.update_and_diff)
3. Filter to delta-relevant elements (get_relevant_elements)
4. Map SnapshotElements back to ElementNodes (by id/text match)
5. Prune4Web keyword scoring on filtered candidates (score_elements)
6. Privacy pipeline: detect PII, anonymise, optional DP (PrivacyPipeline)
7. Privacy-aware LLM call (PrivacyAwareLLM.call → llm_call)
```

### Console output additions

```
  [enhancement] DOM Delta Processing: ENABLED
  [enhancement] Privacy-Aware LLM: ENABLED
  dom_delta       : delta_relevant=12/87 (reduction=7.3x)
  DOM delta: before=87 after=91 added=4 removed=0 modified=8 unchanged=79 reduction=11.4x
  [privacy:grounder] pii_matches=2 elements_masked=1 scrubbed=False
```

---

## Research Metrics to Track

These numbers feed directly into the paper's results section:

| Metric | How to measure | Target |
|---|---|---|
| **DOM Reduction Factor** | `len(all_elements) / len(delta_candidates)` per step | ≥ 5× |
| **Token savings** | Compare prompt token counts (baseline vs delta) | ≥ 30% |
| **PII mask rate** | `PIIDetector.compute_privacy_score(elements)` | ≥ 90% of PII fields masked |
| **Accuracy delta** | Element accuracy with vs without these enhancements | ≤ 2% drop acceptable |

---

## Disabling Enhancements (Baseline Run)

To run the original Prune4Web baseline without any enhancements:

```bash
PRUNE4WEB_DOM_DELTA=0 PRUNE4WEB_PRIVACY=0 python -m Prune4Web.run_prune4web
```

---

## Next Steps (Day 4 – April 12)

- [ ] `src/prune4web_delta.py` – offline evaluation class (Mind2Web dataset)
- [ ] Terminal CLI with `--mode baseline|delta|privacy|full`
- [ ] Collect Recall@20 and Element Accuracy metrics for all four modes
