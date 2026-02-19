# Improvement Over SeeAct — PolicyHub + Risk-Based HITL Gate

> **Branch:** `research/seeact-framework-cleanup`
> **Last updated:** February 2026

---

## 1. What We Added to SeeAct

The original SeeAct pipeline blindly executes every predicted action — no safety checks, no policy awareness, no human oversight. We introduce three new components:

| Component                | Purpose                                                                                            |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| **PolicyHub**            | Loads policy rules (global + site-specific) that constrain what the model is allowed to do         |
| **Risk-Based HITL Gate** | Decides if an action needs human review before execution — based on **risk**, not just uncertainty |
| **`policy_risk` Flag**   | A boolean the LLM itself reports — "does this action involve a risky operation?"                   |

---

## 2. .env Configuration

```dotenv
# PolicyHub + HITL Risk-Based Gate
POLICY_HUB_PATH="src/policyHub/policies.json"
HITL_THRESHOLD=75                 # Legacy (stored but not used for gating)
LOW_CONFIDENCE_FLOOR=30           # HITL triggers if LLM confidence < this (model guessing)
```

| Variable               | Value                   | Used By              | Purpose                                                                            |
| ---------------------- | ----------------------- | -------------------- | ---------------------------------------------------------------------------------- |
| `POLICY_HUB_PATH`      | path to `policies.json` | `PolicyHub` class    | Loads global + site-specific policy rules                                          |
| `HITL_THRESHOLD`       | `75`                    | **Not used anymore** | Legacy — was used in old composite formula. Stored for backward compat.            |
| `LOW_CONFIDENCE_FLOOR` | `30`                    | `HITLConfidenceGate` | If LLM confidence < 30, HITL triggers regardless — the model is basically guessing |

---

## 3. Complete Pipeline Flow (Live Mode)

```
User runs: python live_runner.py
                │
                ▼
    ┌─── .env: RUNNER_MODE=live ───┐
    │                              │
    ▼                              │
  Load live_sites.json             │
  Open Playwright browser          │
    │                              │
    ▼                              │
  For each site:                   │
    Navigate to URL                │
    User provides instruction      │
    │                              │
    ▼                              │
  capture_page_state()             │
    → full-page screenshot (PIL)   │
    → body HTML                    │
    │                              │
    ▼                              │
  predict_live_step()              │
    │                              │
    ├── [Step 1] PolicyHub         │
    │     Load policies.json       │
    │     Merge: default_policy    │
    │           + site_specific    │
    │             [domain match]   │
    │     Extract risk keywords    │
    │     → policy_text string     │
    │     → risk_keywords list     │
    │                              │
    ├── [Step 2] Input Preparation │
    │     Build prompt with:       │
    │       - User instruction     │
    │       - Screenshot           │
    │       - Action history       │
    │       - POLICY CONSTRAINTS   │◄── policy_text injected here
    │       - JSON output format   │
    │                              │
    ├── [Step 3] Action Generation │
    │     Send to GPT-4o           │
    │     Model returns JSON:      │
    │     {                        │
    │       "action": "...",       │
    │       "confidence": 0-100,   │
    │       "policy_risk": bool,   │◄── model self-reports risk
    │       "hitl_reason": "..."   │
    │     }                        │
    │     Parse → extract all 4    │
    │                              │
    ├── [Step 3.5] HITL Gate       │◄── BEFORE grounding (saves cost)
    │     │                        │
    │     │  Three trigger rules:  │
    │     │                        │
    │     │  1. policy_risk_flag   │
    │     │     = true?            │
    │     │     → HITL ⚠️          │
    │     │                        │
    │     │  2. Risk keywords      │
    │     │     found in action    │
    │     │     text? (delete,     │
    │     │     remove, purchase,  │
    │     │     payment, password, │
    │     │     etc.)              │
    │     │     → HITL ⚠️          │
    │     │                        │
    │     │  3. confidence < 30    │
    │     │     (LOW_CONFIDENCE    │
    │     │      _FLOOR)?          │
    │     │     → HITL ⚠️          │
    │     │                        │
    │     │  None of above?        │
    │     │     → SAFE ✅          │
    │     │                        │
    │     ├── If HITL triggered:   │
    │     │     Log warning        │
    │     │     SKIP grounding     │◄── saves GPT API cost
    │     │     Record to results  │
    │     │                        │
    │     └── If SAFE:             │
    │           Continue ↓         │
    │                              │
    ├── [Step 4] Action Grounding  │
    │     (only if HITL passed)    │
    │     DeBERTa + CrossEncoder   │
    │     Match action text to     │
    │     actual DOM element       │
    │                              │
    └── [Step 5] Save Results      │
          screenshot.png           │
          result.json              │
          → liveSiteResults/       │
            run_YYYYMMDD_HHMMSS/   │
            SiteName/step_001/     │
```

---

## 4. The Key Design Decision: Risk ≠ Ambiguity

### Problem with the old approach

The first HITL implementation used a **composite confidence formula**:

```
composite = 0.7 × LLM_confidence + 0.3 × policy_risk_score

if composite < threshold (75):
    trigger HITL
```

This mixed two separate concerns:

- **UI ambiguity** — "page has many links, I'm 60% sure this is the right one"
- **Policy risk** — "this action deletes data"

Result: Wikipedia (dense, many links, confidence ~60) would always trigger HITL even though clicking a link on Wikipedia is 100% safe. The system was blocking safe navigation actions.

### Current approach: Risk-based gating

```python
# HITL triggers ONLY when:
if policy_risk_flag == True:       # model says it's risky
    → HITL
elif risk_keywords_in_action:      # "delete", "payment", etc. detected
    → HITL
elif confidence < 30:              # model is guessing
    → HITL
else:
    → SAFE, continue to grounding
```

Confidence becomes **informational** — useful for evaluation and logging, but it does NOT block safe actions.

---

## 5. HITL Gate Decision Table

| Scenario                               | confidence | policy_risk | Keywords?  | HITL?   | Why                                  |
| -------------------------------------- | ---------- | ----------- | ---------- | ------- | ------------------------------------ |
| `[link] Wikibooks -> CLICK`            | 72         | `false`     | none       | **No**  | Safe navigation                      |
| `[checkbox] I am not a robot -> CLICK` | 60         | `true`      | none       | **Yes** | Model flagged CAPTCHA as risky       |
| `[button] Delete item -> CLICK`        | 55         | `true`      | "delete"   | **Yes** | Both flag + keyword                  |
| `[button] Search -> CLICK`             | 90         | `false`     | none       | **No**  | Safe, high confidence                |
| `[div] maybe this? -> CLICK`           | 20         | `false`     | none       | **Yes** | Below floor (30) — model is guessing |
| `[button] Buy Now -> CLICK`            | 70         | `false`     | "purchase" | **Yes** | Risk keyword matched                 |
| `[link] About Us -> CLICK`             | 55         | `false`     | none       | **No**  | Low-ish confidence, but safe action  |

---

## 6. PolicyHub Structure

### policies.json format

```json
{
  "default_policy": {
    "rules": [
      "Do not submit payment forms automatically.",
      "Do not change account passwords.",
      "Do not click destructive actions like delete or remove.",
      "Do not interact with CAPTCHA or security mechanisms."
    ],
    "risk_keywords": [
      "delete",
      "remove",
      "purchase",
      "payment",
      "password",
      "approve",
      "submit",
      "transfer",
      "checkout",
      "cancel",
      "terminate",
      "destroy",
      "discard",
      "unsubscribe"
    ]
  },
  "site_specific": {
    "amazon.com": {
      "rules": ["Do not purchase items.", "Do not modify cart quantity."]
    }
  }
}
```

### Runtime behavior

For a request on `amazon.com`:

```
active_policy = default_policy.rules (4) + amazon.com.rules (2) = 6 rules
risk_keywords = default_policy.risk_keywords (14 keywords)
```

Both are injected into the prompt under `POLICY CONSTRAINTS`.

---

## 7. What the Model Sees in the Prompt

When PolicyHub is active, the prompt includes:

```
────────────────────────────────────────────────────────
POLICY CONSTRAINTS (you MUST follow these strictly):
• Do not submit payment forms automatically.
• Do not change account passwords.
• Do not click destructive actions like delete or remove.
• Do not interact with CAPTCHA or security mechanisms.

RISK ASSESSMENT RULES:
- Set "policy_risk" to true ONLY when the action involves:
  destructive operations, financial transactions, credentials,
  CAPTCHA/security mechanisms, or account changes.
- Set "policy_risk" to false for normal navigation, reading,
  searching, clicking links, typing search queries.
- Report REALISTIC confidence based on how certain you are
  about element selection. Do NOT artificially reduce confidence
  for safe actions just because the page is complex.
────────────────────────────────────────────────────────
```

And the output format becomes:

```json
{
  "action": "[element_type] ELEMENT_TEXT -> CLICK",
  "confidence": 75,
  "policy_risk": false,
  "hitl_reason": ""
}
```

When PolicyHub is **not** active (e.g., legacy dataset mode), the prompt uses the original SeeAct one-line format with no JSON output.

---

## 8. Files Involved

| File                               | Role                                                                          |
| ---------------------------------- | ----------------------------------------------------------------------------- |
| `src/policyHub/policies.json`      | Policy rules + risk keywords                                                  |
| `src/policyHub/policy_hub.py`      | Loads/merges policies per domain                                              |
| `src/policyHub/hitl_confidence.py` | Risk-based HITL gate (3 trigger rules)                                        |
| `src/Input_Prepration.py`          | Prompt builder — injects policy + JSON format                                 |
| `src/action_generation.py`         | Parses `{action, confidence, policy_risk, hitl_reason}` from model output     |
| `src/seeact_pipeline.py`           | Wires everything: PolicyHub → Prompt → Generation → HITL Gate → Grounding     |
| `src/seeact_evaluation.py`         | Records `policy_risk_flag`, `hitl_triggered`, `hitl_reason` per step          |
| `src/config.py`                    | `get_low_confidence_floor()`, `get_hitl_threshold()`, `get_policy_hub_path()` |
| `src/browser_driver.py`            | Playwright browser for live mode (full-page screenshots)                      |
| `src/live_result_store.py`         | Saves screenshots + JSON results to `liveSiteResults/`                        |
| `live_runner.py`                   | Entry point — reads `.env`, dispatches live or dataset mode                   |

---

## 9. Comparison: Original SeeAct vs Our Extension

| Aspect               | Original SeeAct    | Our Extension                                   |
| -------------------- | ------------------ | ----------------------------------------------- |
| Safety checks        | None               | PolicyHub + HITL gate                           |
| Risk awareness       | None               | 14 risk keywords + LLM policy_risk flag         |
| Human oversight      | None               | Risk-based HITL triggering                      |
| Confidence scoring   | Not reported       | LLM self-reports 0-100                          |
| Policy constraints   | None               | Global + site-specific rules injected in prompt |
| Output format        | Free-text one-line | Structured JSON (when PolicyHub active)         |
| Grounding cost       | Always paid        | Skipped when HITL triggers (saves API cost)     |
| Live website support | No (Mind2Web only) | Playwright browser + live_runner.py             |
| Result persistence   | Eval files only    | Screenshots + JSON per step in liveSiteResults/ |
| HITL trigger logic   | N/A                | `policy_risk OR keywords OR confidence < 30`    |

---

## 10. Why Risk-Based Over Threshold-Based

**Threshold-based** (old):

- `composite = 0.7 × confidence + 0.3 × keyword_score`
- `if composite < 75: HITL`
- Problem: Wikipedia link click (confidence=60, no risk) → composite=72 → HITL triggered on a perfectly safe action

**Risk-based** (current):

- HITL only fires when there's an actual safety concern
- A confident=60 action on a safe page → **no HITL** (execute normally)
- A confident=90 action on "Delete Account" → **HITL** (keyword match overrides confidence)
- Confidence stays as a diagnostic metric, but doesn't gate execution of safe actions
