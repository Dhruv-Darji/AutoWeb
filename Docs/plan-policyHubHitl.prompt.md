# PolicyHub + HITL Confidence Gate — Implementation Plan

> **Feature:** Policy-Aware Web Agent with Confidence-Triggered Human-in-the-Loop  
> **Branch:** `research/seeact-framework-cleanup`  
> **Date:** 2026-02-19  
> **Model support:** Both GPT (OpenAI API) and local Qwen2-VL-2B

---

## Overview

Add a **PolicyHub** layer that injects global + site-specific policy constraints into Action Generation prompts, force **structured JSON output** from the model (action + confidence 0–100 + HITL reason), compute **composite confidence** using LLM self-score + grounding score gap + policy risk keywords, and **log HITL triggers** when confidence falls below threshold.

### Updated Architecture

```
User Task
    ↓
Screenshot + DOM
    ↓
PolicyHub (global + site-specific)
    ↓
Action Generation (LLM)
    → Action Plan
    → Confidence Score (0–100)
    → HITL Reason
    ↓
Action Grounding (DeBERTa + VL model)
    → Selected Element
    → top1_score, top2_score
    ↓
Composite Confidence Gate
    final = 0.5 * LLM_conf + 0.3 * grounding_gap + 0.2 * policy_risk
    ├── High confidence → Continue execution
    └── Low confidence → Log HITL trigger (reason + score)
    ↓
Evaluation (extended with HITL metrics)
```

---

## Design Decisions

| Decision               | Choice                          | Rationale                                                                                      |
| ---------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------- |
| Model backend          | Both GPT + Qwen                 | Qwen gets conservative fallback (confidence=50) when JSON parse fails                          |
| Confidence method      | Composite from start            | `0.5 * LLM + 0.3 * grounding_gap + 0.2 * policy_risk` — robust against hallucinated confidence |
| HITL mode              | Log-only                        | No interactive pause — suitable for offline research evaluation                                |
| Policy storage         | Python config class + JSON file | `PolicyHub` class with hardcoded defaults, loadable/savable from `policies.json`               |
| Policy injection point | Action Generation prompt only   | Keeps grounding prompt tight and focused on element selection                                  |
| Grounding scores       | Expose existing DeBERTa scores  | Already computed internally, just not returned — avoids extra compute                          |

---

## Phase 1 — PolicyHub Module

### 1.1 Create `src/policyHub/__init__.py`

Empty init to make it a Python package.

### 1.2 Create `src/policyHub/policy_hub.py`

`PolicyHub` class:

- **Constructor** loads from JSON file path (default: `src/policyHub/policies.json`), with hardcoded `DEFAULT_POLICIES` fallback list:
  - "Do not submit payment forms automatically."
  - "Do not click destructive actions like delete or remove without confirmation."
  - "Do not change account passwords."
  - "Do not share sensitive personal information."
  - "Do not approve financial transactions without explicit user confirmation."
- `site_specific` dict loaded from JSON, empty by default.
- **`get_active_policy(domain: Optional[str] = None) -> str`** — merges `default_policy` + `site_specific[domain]` (if domain key exists), returns formatted bullet-point string.
- **`add_site_policy(domain: str, rules: List[str])`** / **`update_default_policy(rules: List[str])`** — runtime mutators + persist to JSON.
- **`get_policy_risk_keywords() -> List[str]`** — returns known risky action keywords extracted from all active policies (e.g., "delete", "remove", "purchase", "payment", "password", "approve", "submit", "transfer").

### 1.3 Create `src/policyHub/policies.json`

```json
{
  "default_policy": [
    "Do not submit payment forms automatically.",
    "Do not click destructive actions like delete or remove without confirmation.",
    "Do not change account passwords.",
    "Do not share sensitive personal information.",
    "Do not approve financial transactions without explicit user confirmation."
  ],
  "site_specific": {
    "amazon.com": ["Do not purchase items.", "Do not modify cart quantity."],
    "banking_portal": [
      "Do not approve or initiate financial transactions.",
      "Do not transfer funds between accounts."
    ]
  }
}
```

At runtime: `active_policy = default_policy + site_specific.get(current_domain, [])`

### 1.4 Add `get_policy_hub_path()` to `src/config.py`

Reads `POLICY_HUB_PATH` env var, defaults to `src/policyHub/policies.json` relative to project root.

---

## Phase 2 — HITL Confidence Module

### 2.1 Create `src/policyHub/hitl_confidence.py`

`HITLConfidenceGate` class:

- **Constructor** takes `threshold: int` from `HITL_THRESHOLD` env var (default: `75`).
- **`compute_composite_confidence(...) -> dict`**:

  ```python
  def compute_composite_confidence(
      self,
      llm_confidence: float,         # 0–100 from model JSON output
      grounding_top1_score: float,    # DeBERTa cross-encoder score (0–1)
      grounding_top2_score: float,    # second-best score (0–1)
      action_text: str,               # the generated action plan text
      policy_risk_keywords: List[str] # from PolicyHub
  ) -> dict:
  ```

  Formula:

  ```
  grounding_gap = (top1 - top2) * 100   # normalized to 0–100
  policy_risk = 100 if no keywords match action_text
               = 50  if 1 keyword matches
               = 20  if 2+ keywords match

  final = 0.5 * llm_confidence
        + 0.3 * grounding_gap
        + 0.2 * policy_risk
  ```

  Returns:

  ```json
  {
    "final_confidence": 72.5,
    "llm_confidence": 85,
    "grounding_gap": 45.0,
    "policy_risk_score": 50,
    "hitl_triggered": true,
    "threshold": 75
  }
  ```

- **`should_trigger_hitl(composite: dict) -> bool`** — `composite["final_confidence"] < self.threshold`

### 2.2 Add `get_hitl_threshold()` to `src/config.py`

Reads `HITL_THRESHOLD` env var, defaults to `75`.

---

## Phase 3 — Prompt Modification (Input Preparation)

### 3.1 Modify `src/Input_Prepration.py`

**`SeeActInputPreparator.prepare_input()`** changes:

- Add parameter: `policy_text: Optional[str] = None`
- Inject new `POLICY CONSTRAINTS` section into prompt (between INPUTS and ACTION SPACE):

```
────────────────────────────────────────────────────────────────────────────
POLICY CONSTRAINTS (you MUST follow these strictly):
{policy_text}

- If an action may violate any policy above, you MUST reduce your confidence score.
- If uncertain whether an action is safe, set confidence below 70 and explain in hitl_reason.
────────────────────────────────────────────────────────────────────────────
```

- Modify OUTPUT FORMAT section to require structured JSON:

```
OUTPUT FORMAT (respond with ONLY this JSON, nothing else):
{
  "action": "[element_type] ELEMENT_TEXT -> CLICK",
  "confidence": <integer 0-100>,
  "hitl_reason": "<empty string if confident, otherwise explain why confidence is low>"
}

Rules for confidence scoring:
- 90-100: Very confident, clear next step, no policy concerns
- 70-89:  Fairly confident, minor ambiguity
- 50-69:  Uncertain — ambiguous elements, possible policy risk
- 0-49:   Very uncertain — likely policy violation or cannot determine action

Rules for hitl_reason:
- Empty string "" if confidence >= 80
- Otherwise explain: policy violation risk, ambiguous target, uncertain action, etc.
```

- If `policy_text` is `None` or empty, omit the POLICY CONSTRAINTS section entirely (backward compatible).

---

## Phase 4 — Action Generation Output Parsing

### 4.1 Modify `src/action_generation.py`

Add method to `SeeActActionGenerator`:

```python
def _parse_structured_output(self, raw_text: str) -> dict:
    """
    Parse structured JSON output from model.

    Fallback chain:
    1. json.loads() on full text
    2. Regex extract JSON block from surrounding text
    3. Regex extract individual fields
    4. Treat entire output as action text with confidence=50

    Returns: {"action": str, "confidence": int, "hitl_reason": str}
    """
```

Modify `generate_plans()`:

- Call `_parse_structured_output()` on model output
- Return dict now includes: `confidence: int`, `hitl_reason: str` alongside existing `output_text`, `raw_text`, `latency`
- For **Qwen local model**: if JSON parsing fails after retry, fall back to `confidence=50, hitl_reason="local model did not produce structured output"`
- Update `_is_valid_dataset_action()` to also accept the `action` field extracted from JSON

---

## Phase 5 — Pipeline Integration

### 5.1 Modify `src/seeact_pipeline.py`

**`SeeActPipeline.__init__()`:**

- Import and instantiate `PolicyHub` and `HITLConfidenceGate`
- Log: `"✓ PolicyHub ready (N default + M site-specific rules)"`
- Log: `"✓ HITL confidence gate ready (threshold: T)"`

**`predict_single_task()`:**

Before action loop:

- Extract domain from task metadata if available (e.g., from `action.get("website")` or URL parsing), fall back to `None`
- Call `policy_hub.get_active_policy(domain)` → `policy_text`
- Get `policy_risk_keywords = policy_hub.get_policy_risk_keywords()`

Step modifications:

```
[2/6] Input Preparation → pass policy_text to prepare_input()
[3/6] Action Generation → extract llm_confidence, hitl_reason from result
[4/6] Action Grounding  → extract top1_score, top2_score from result
[4.5] NEW: Composite Confidence Gate
       → hitl_gate.compute_composite_confidence(
             llm_confidence, top1_score, top2_score,
             output_plan, policy_risk_keywords)
       → Log composite score
       → If triggered: logger.warning("⚠ HITL TRIGGERED — confidence: {}, reason: {}")
[6/6] Evaluation → pass confidence + HITL data to evaluator.record_step()
```

---

## Phase 6 — Grounding Score Exposure

### 6.1 Modify `src/action_grounding.py`

Ensure `SeeActActionGrounding.process()` return dict includes DeBERTa scores:

```python
return {
    "success": True,
    "selected_element": best_element,
    "top1_score": float(sorted_scores[0]),   # NEW
    "top2_score": float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0,  # NEW
    "latency": latency,
}
```

These are already computed internally during DeBERTa cross-encoder ranking — just need to expose them in the return value.

---

## Phase 7 — Evaluation Extension

### 7.1 Modify `src/seeact_evaluation.py`

**`StepEvalResult`** — add fields:

```python
llm_confidence: float = 0.0
composite_confidence: float = 0.0
hitl_triggered: bool = False
hitl_reason: str = ""
```

**`TaskEvalResult`** — add fields:

```python
hitl_trigger_rate: float = 0.0    # fraction of steps where HITL triggered
avg_confidence: float = 0.0       # average composite confidence across steps
```

**`AggregateMetrics`** — add fields:

```python
avg_confidence: float = 0.0
hitl_trigger_rate: float = 0.0
total_hitl_triggers: int = 0
```

**`record_step()`** — accept and store new fields:

```python
def record_step(self, ..., llm_confidence=0.0, composite_confidence=0.0,
                hitl_triggered=False, hitl_reason=""):
```

**`compute_metrics()`** — aggregate confidence and HITL stats across tasks.

**`print_summary()`** — add HITL section:

```
HITL Statistics:
  Avg Composite Confidence: 74.3
  HITL Trigger Rate:        23.1% (6/26 steps)
  Total HITL Triggers:      6
```

---

## Phase 8 — Config & Documentation

### 8.1 Update `.env.example`

Add:

```env
# PolicyHub
POLICY_HUB_PATH=src/policyHub/policies.json

# HITL Confidence Gate
HITL_THRESHOLD=75
```

---

## File Change Summary

| File                               | Action     | Description                                                     |
| ---------------------------------- | ---------- | --------------------------------------------------------------- |
| `src/policyHub/__init__.py`        | **CREATE** | Package init                                                    |
| `src/policyHub/policy_hub.py`      | **CREATE** | PolicyHub class                                                 |
| `src/policyHub/policies.json`      | **CREATE** | Default policy data                                             |
| `src/policyHub/hitl_confidence.py` | **CREATE** | HITL confidence gate                                            |
| `src/config.py`                    | **MODIFY** | Add `get_policy_hub_path()`, `get_hitl_threshold()`             |
| `src/Input_Prepration.py`          | **MODIFY** | Add policy injection + JSON output format to prompt             |
| `src/action_generation.py`         | **MODIFY** | Add `_parse_structured_output()`, return confidence/hitl_reason |
| `src/action_grounding.py`          | **MODIFY** | Expose `top1_score`, `top2_score` in return dict                |
| `src/seeact_pipeline.py`           | **MODIFY** | Integrate PolicyHub + HITL gate into pipeline loop              |
| `src/seeact_evaluation.py`         | **MODIFY** | Add confidence/HITL fields to all dataclasses + aggregation     |
| `.env.example`                     | **MODIFY** | Add new env vars                                                |

---

## Verification Checklist

- [ ] Run pipeline with GPT on a single annotation_id → verify JSON output with `action`, `confidence`, `hitl_reason` parses correctly
- [ ] Add test site policy for the annotation's domain → verify HITL triggers on risky actions and confidence drops
- [ ] Run with local Qwen → verify graceful fallback when JSON parsing fails (confidence defaults to 50)
- [ ] Compare evaluation output JSON before/after — new fields appear (`llm_confidence`, `composite_confidence`, `hitl_triggered`, `hitl_reason`, `hitl_trigger_rate`)
- [ ] Run batch mode → verify aggregate `hitl_trigger_rate` and `avg_confidence` compute correctly
- [ ] Verify backward compatibility — pipeline works without POLICY_HUB_PATH or HITL_THRESHOLD env vars (defaults apply)

---

## Research Evaluation Protocol

Compare three configurations:

| Config           | PolicyHub | HITL Gate | Purpose                                        |
| ---------------- | --------- | --------- | ---------------------------------------------- |
| **Baseline**     | OFF       | OFF       | Original SeeAct performance                    |
| **+Policy**      | ON        | OFF       | Impact of policy constraints on action quality |
| **+Policy+HITL** | ON        | ON        | Full system with confidence gating             |

Metrics to compare:

- Task success rate (Offline0, Offline1)
- Element accuracy, Operation F1, Value accuracy
- Average confidence score
- HITL trigger rate (% of steps flagged)
- Cost per task (token usage)
- Error reduction in policy-sensitive actions

This produces a clear ablation study suitable for the MTech thesis.
