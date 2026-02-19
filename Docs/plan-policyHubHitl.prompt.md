# PolicyHub + Risk-Based HITL Gate — Implementation Plan & Final State

> **Feature:** Policy-Aware Web Agent with Risk-Based Human-in-the-Loop  
> **Branch:** `research/seeact-framework-cleanup`  
> **Date:** June 2025  
> **Model support:** Both GPT-4o (OpenAI API) and local Qwen2-VL-2B  
> **Status:** ✅ Fully implemented and tested on live websites

---

## Overview

We extend the **SeeAct** web automation pipeline with a **PolicyHub** layer and a **risk-based HITL gate**. PolicyHub injects global + site-specific policy constraints into the Action Generation prompt. The model responds with **structured JSON output** (action + confidence 0-100 + `policy_risk` boolean + HITL reason). The HITL gate evaluates three independent risk signals and **blocks dangerous actions before grounding**, saving compute and ensuring safety.

### Architecture

```
User Task
    ↓
Screenshot + DOM
    ↓
PolicyHub (global + site-specific rules, 24 risk keywords)
    ↓
Input Preparation
    → Inject POLICY CONSTRAINTS block into prompt
    → Set JSON output format with policy_risk field
    ↓
Action Generation (GPT-4o / Qwen2-VL-2B)
    → Returns: { action, confidence, policy_risk, hitl_reason }
    → 4-level fallback parse
    ↓
Risk-Based HITL Gate  ◄── BEFORE grounding (saves cost)
    ├── policy_risk_flag == true?     → ⚠️ HITL
    ├── risk keyword in action/reason? → ⚠️ HITL
    ├── confidence < 30?              → ⚠️ HITL
    └── none?                         → ✅ Safe
    ↓
Action Grounding (DeBERTa + CrossEncoder)  ← only if ✅
    → Selected Element
    ↓
Evaluation + Result Storage
    → Per-step JSON + screenshots in liveSiteResults/
```

---

## Design Decisions

| Decision             | Choice                                        | Rationale                                                                       |
| -------------------- | --------------------------------------------- | ------------------------------------------------------------------------------- |
| HITL trigger logic   | Risk-based (3 independent triggers, OR-logic) | Separates uncertainty from risk — safe low-confidence navigation is not blocked |
| Model backend        | Both GPT-4o + Qwen2-VL-2B                     | Qwen gets conservative fallback (`confidence=50`) when JSON parse fails         |
| Confidence scoring   | LLM self-reported (0-100), realistic scale    | No artificial pessimism — model reports honest certainty                        |
| HITL gate position   | Before grounding                              | Saves DeBERTa + CrossEncoder cost on risky actions                              |
| HITL mode            | Log-only (no interactive pause)               | Suitable for offline research evaluation                                        |
| Policy storage       | JSON file (`policies.json`) + Python loader   | Easily extensible with site-specific overrides                                  |
| Policy injection     | Action Generation prompt only                 | Keeps grounding prompt tight and focused on element selection                   |
| Keyword scan scope   | Action text + hitl_reason                     | Catches model contradictions (says safe, describes danger)                      |
| Risk keywords        | 24 curated keywords                           | Covers payments, credentials, CAPTCHA, destructive actions, security            |
| Low confidence floor | 30 (configurable via `.env`)                  | Below this = model is guessing → always flag                                    |

---

## Implementation Phases (Completed)

### Phase 1: PolicyHub Module

**Goal:** Create the policy-awareness engine that loads and merges policy rules.

**Files created:**

- `src/policyHub/__init__.py` — Package init, exports `PolicyHub` and `HITLConfidenceGate`
- `src/policyHub/policy_hub.py` — Loads `policies.json`, merges default + site-specific rules per domain
- `src/policyHub/policies.json` — 15 default policy rules, empty site-specific dict, 24 risk keywords

**Key methods:**

```python
class PolicyHub:
    def get_active_policy(self, domain: str) -> str
        # Returns formatted bullet-point string of merged rules

    def get_policy_risk_keywords(self) -> list[str]
        # Returns 24 risk keywords for the HITL gate
```

---

### Phase 2: Risk-Based HITL Gate

**Goal:** Replace composite confidence formula with risk-based triggering.

**File created:**

- `src/policyHub/hitl_confidence.py`

**Evolution:**

1. ~~Composite formula: `0.5*LLM + 0.3*grounding_gap + 0.2*policy_risk`~~ — Discarded (mixed uncertainty with risk)
2. ~~Paranoid calibration: "DEFAULT confidence MUST be 40-55"~~ — Discarded (artificially pessimistic)
3. ✅ **Risk-based gate with 3 independent triggers** — Current

**Final implementation:**

```python
class HITLConfidenceGate:
    def __init__(self, low_confidence_floor=30, threshold=75):
        # threshold stored for backward compat but NOT used for gating

    def evaluate(self, llm_confidence, policy_risk_flag, action_text,
                 policy_risk_keywords, hitl_reason=""):
        combined_text = f"{action_text} {hitl_reason}"

        # Rule 1: LLM self-flagged the action as policy-risky
        if policy_risk_flag:
            return (True, "llm_flagged_policy_risk", ...)

        # Rule 2: Risk keywords found in action text OR hitl_reason
        matched = self._scan_keywords(combined_text, policy_risk_keywords)
        if matched:
            return (True, f"risk_keywords_matched: {matched}", ...)

        # Rule 3: Extremely low confidence (model is guessing)
        if llm_confidence < self.low_confidence_floor:
            return (True, f"very_low_confidence ({llm_confidence})", ...)

        return (False, "no_risk_detected", ...)

    def compute_composite_confidence(self, llm_confidence, policy_risk_flag,
                                      action_text, policy_risk_keywords,
                                      hitl_reason=""):
        # Backward-compatible wrapper that adds legacy keys:
        # final_confidence, threshold, policy_risk_score
        # Internally calls evaluate()
```

---

### Phase 3: Prompt Engineering (Input Preparation)

**Goal:** Inject policy constraints into LLM prompt and switch to JSON output.

**File modified:** `src/Input_Prepration.py`

**What was added:**

1. `POLICY CONSTRAINTS` block — injected between INPUTS and ACTION SPACE sections
2. `RISK ASSESSMENT RULES` — instructs model on when to set `policy_risk: true`
3. JSON output format — `{action, confidence, policy_risk, hitl_reason}`
4. Realistic confidence instructions — no artificial reduction for safe actions

**Key design choices:**

- `policy_risk` is a **boolean**, not a score — simpler for the model, no ambiguous values
- If PolicyHub is not active, falls back to original SeeAct one-line format
- Risk assessment rules explicitly enumerate what counts as risky: destructive ops, financial transactions, credentials, CAPTCHA, account changes

---

### Phase 4: Action Generation Parsing

**Goal:** Parse the new `policy_risk` boolean from model output.

**File modified:** `src/action_generation.py`

**4-level fallback parse chain:**

| Level | Method                                 | `policy_risk` source                   |
| ----- | -------------------------------------- | -------------------------------------- | --------- |
| 1     | Direct `json.loads()` on full response | `parsed["policy_risk"]`                |
| 2     | Regex extraction of JSON block         | `parsed["policy_risk"]`                |
| 3     | Individual field regex                 | `re.search(r'"policy_risk"\s*:\s*(true | false)')` |
| 4     | Plain text fallback                    | Defaults to `False`                    |

---

### Phase 5: Pipeline Integration

**Goal:** Wire PolicyHub + HITL gate into the SeeAct prediction pipeline.

**File modified:** `src/seeact_pipeline.py`

**Changes in both `predict_single_task()` and `predict_live_step()`:**

```python
# 1. Extract from action generation output
policy_risk_flag = output.get("policy_risk", False)
hitl_reason_from_llm = output.get("hitl_reason", "")

# 2. Get risk keywords from PolicyHub
risk_keywords = self.policy_hub.get_policy_risk_keywords()

# 3. Run HITL gate BEFORE grounding
hitl_result = self.hitl_gate.compute_composite_confidence(
    llm_confidence=confidence,
    policy_risk_flag=policy_risk_flag,
    action_text=action_text,
    policy_risk_keywords=risk_keywords,
    hitl_reason=hitl_reason_from_llm
)

# 4. Check result
if hitl_result["hitl_triggered"]:
    logger.warning(f"🛑 HITL triggered: {hitl_result['hitl_reason']}")
    # SKIP grounding — go straight to result recording
else:
    logger.info("✅ No risk detected — proceeding to grounding")
    # Continue to DeBERTa + CrossEncoder grounding
```

---

### Phase 6: Evaluation Extension

**Goal:** Record policy risk data in per-step evaluation results.

**File modified:** `src/seeact_evaluation.py`

**New field in `StepEvalResult`:**

```python
@dataclass
class StepEvalResult:
    # ... existing fields ...
    policy_risk_flag: bool = False  # NEW
```

**`record_step()` now accepts `policy_risk_flag` parameter.**

---

### Phase 7: Configuration

**File modified:** `src/config.py`

**New config getter:**

```python
def get_low_confidence_floor(default: int = 30) -> int:
    """Read LOW_CONFIDENCE_FLOOR from .env. Default 30."""
    return int(os.getenv("LOW_CONFIDENCE_FLOOR", str(default)))
```

**`.env.example` updated:**

```dotenv
LOW_CONFIDENCE_FLOOR=30
HITL_THRESHOLD=75    # Legacy, stored but not used for gating
```

---

### Phase 8: Browser Driver Stability

**Goal:** Fix "Event loop is closed" warning on browser teardown.

**File modified:** `src/browser_driver.py`

**Fix:** Made `BrowserDriver.close()` idempotent with a `_closed` flag. Each resource cleanup (page, context, browser, playwright) wrapped individually in try/except so one failure doesn't prevent others from closing.

```python
async def close(self):
    if self._closed:
        return
    self._closed = True
    # close page, context, browser, playwright — each in try/except
```

---

## File Change Summary

| File                               | Type         | What Changed                                                          |
| ---------------------------------- | ------------ | --------------------------------------------------------------------- |
| `src/policyHub/__init__.py`        | **New**      | Package init, exports PolicyHub + HITLConfidenceGate                  |
| `src/policyHub/policy_hub.py`      | **New**      | Policy loader — merges default + site-specific rules                  |
| `src/policyHub/policies.json`      | **New**      | 15 policy rules + 24 risk keywords                                    |
| `src/policyHub/hitl_confidence.py` | **New**      | Risk-based HITL gate — 3 trigger rules, keyword scan on action+reason |
| `src/Input_Prepration.py`          | **Modified** | Policy block injection, JSON output format, risk assessment rules     |
| `src/action_generation.py`         | **Modified** | 4-level parse of `policy_risk` boolean                                |
| `src/seeact_pipeline.py`           | **Modified** | HITL gate wired before grounding, passes `hitl_reason`                |
| `src/seeact_evaluation.py`         | **Modified** | `StepEvalResult.policy_risk_flag` field + `record_step()` param       |
| `src/config.py`                    | **Modified** | `get_low_confidence_floor()` getter                                   |
| `src/browser_driver.py`            | **Modified** | Idempotent `close()` with `_closed` flag                              |
| `.env.example`                     | **Modified** | `LOW_CONFIDENCE_FLOOR=30` documented                                  |

---

## Verification Checklist

- [x] PolicyHub loads and merges policies per domain
- [x] Risk keywords extracted (24 keywords including "robot", "captcha", "security")
- [x] Policy constraints injected into action generation prompt
- [x] Model returns structured JSON: `{action, confidence, policy_risk, hitl_reason}`
- [x] `policy_risk` parsed from all 4 fallback levels (JSON, regex JSON, regex fields, plain text)
- [x] HITL gate evaluates 3 independent risk triggers (OR-logic)
- [x] Keyword scan covers both `action_text` AND `hitl_reason`
- [x] HITL gate runs BEFORE grounding (saves compute on risky actions)
- [x] Pipeline passes `hitl_reason` from LLM to HITL gate
- [x] `StepEvalResult` records `policy_risk_flag`
- [x] `LOW_CONFIDENCE_FLOOR` configurable via `.env`
- [x] Browser close is idempotent (no "Event loop is closed" warning)
- [x] Backward compatible — legacy dataset mode works without PolicyHub

---

## Known Limitations

1. **HITL is log-only** — does not actually pause for user input. Designed for research evaluation, not production deployment.
2. **No runtime keyword expansion** — the 24 risk keywords are static. A future version could use LLM-based risk classification.
3. **Model self-contradiction** — GPT-4o sometimes sets `policy_risk: false` while describing a risky action in `hitl_reason`. The keyword scan mitigates this but doesn't solve the root cause (model inconsistency).
4. **Site-specific policies** — the `site_specific` dict is empty. Needs curation per target domain for production use.
5. **No RL/fine-tuning** — Policy awareness comes from prompt injection only. A fine-tuned model could internalize safety constraints rather than relying on in-context instructions.
