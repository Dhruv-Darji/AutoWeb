# Improvement Over SeeAct — PolicyHub + Risk-Based HITL Gate

> **Branch:** `research/seeact-framework-cleanup`  
> **Last updated:** June 2025  
> **Status:** Implemented and tested on live websites

---

## 1. Executive Summary

The original **SeeAct** pipeline (Zheng et al., 2024) performs web task automation by predicting UI actions from screenshots and DOM, then grounding those actions to specific HTML elements. However, it has **no safety layer** — every predicted action is executed blindly, including destructive operations (delete, purchase), credential changes, and CAPTCHA interactions.

We extend SeeAct with two tightly integrated modules:

1. **PolicyHub** — a policy-awareness engine that injects global and site-specific safety constraints into the LLM prompt, forcing the model to reason about risk before predicting actions.
2. **Risk-Based HITL (Human-in-the-Loop) Gate** — a safety checkpoint that runs **before** the expensive grounding step, deciding whether an action requires human oversight based on **actual risk signals**, not just model uncertainty.

### Key Insight: Risk ≠ Ambiguity

Our central design principle separates two independent concerns:

- **Task uncertainty** — "the page has many links and I'm 60% sure this is the right one" → This is normal. Execute.
- **Policy risk** — "this action deletes user data" → This is dangerous. Flag for human review.

Previous confidence-based approaches (including our own first iteration) conflated these, causing safe navigation to be blocked on complex pages. The risk-based approach eliminates this problem.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                   SeeAct + PolicyHub + HITL Pipeline                  │
│                                                                      │
│  ┌─────────────┐     ┌──────────────┐     ┌───────────────────┐     │
│  │  User Task   │────▶│  PolicyHub   │────▶│ Input Preparation │     │
│  │ (instruction)│     │              │     │                   │     │
│  └─────────────┘     │ ● Load rules │     │ ● Inject policy   │     │
│                      │ ● Merge by   │     │   constraints     │     │
│  ┌─────────────┐     │   domain     │     │ ● Build JSON      │     │
│  │ Screenshot  │────▶│ ● Extract    │     │   output format   │     │
│  │ + DOM HTML  │     │   risk       │     │ ● Attach          │     │
│  └─────────────┘     │   keywords   │     │   screenshot      │     │
│                      └──────────────┘     └────────┬──────────┘     │
│                                                     │                │
│                                                     ▼                │
│                                           ┌─────────────────┐       │
│                                           │ Action Generation│       │
│                                           │  (GPT-4o / Qwen)│       │
│                                           │                 │       │
│                                           │ Returns JSON:   │       │
│                                           │  action         │       │
│                                           │  confidence     │       │
│                                           │  policy_risk    │◄─ NEW │
│                                           │  hitl_reason    │       │
│                                           └────────┬────────┘       │
│                                                     │                │
│                              ┌──────────────────────┼───────┐       │
│                              │     HITL Gate         │       │       │
│                              │  (Risk-Based)         ▼       │       │
│                              │                              │       │
│                              │  policy_risk == true? ──▶ ⚠️ │       │
│                              │  risk keyword found?  ──▶ ⚠️ │       │
│                              │  confidence < 30?     ──▶ ⚠️ │       │
│                              │  none of above?       ──▶ ✅ │       │
│                              │                              │       │
│                              │  ⚠️ = HITL triggered        │       │
│                              │       → skip grounding       │       │
│                              │       → log + save result    │       │
│                              │                              │       │
│                              │  ✅ = Safe                   │       │
│                              │       → continue pipeline    │       │
│                              └──────────────────────────────┘       │
│                                                     │                │
│                                                     ▼ (only if ✅)   │
│                                           ┌─────────────────┐       │
│                                           │ Action Grounding │       │
│                                           │ (DeBERTa +       │       │
│                                           │  CrossEncoder)   │       │
│                                           │                  │       │
│                                           │ Match action to  │       │
│                                           │ specific DOM     │       │
│                                           │ element          │       │
│                                           └────────┬─────────┘       │
│                                                     │                │
│                                                     ▼                │
│                                           ┌─────────────────┐       │
│                                           │   Evaluation     │       │
│                                           │ + Result Storage │       │
│                                           └─────────────────┘       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. How PolicyHub Works (Policy-Awareness Layer)

### 3.1 Purpose

PolicyHub makes the LLM **aware of what it should NOT do** before it generates any action. Without this, the model freely predicts dangerous actions like "Delete Account", "Buy Now", or "Submit Payment" if those actions happen to advance the task goal.

### 3.2 Policy Structure

Policies are stored in `src/policyHub/policies.json`:

```json
{
  "default_policy": [
    "Do not submit payment or checkout forms automatically.",
    "Do not complete purchases or financial transactions without explicit user confirmation.",
    "Do not approve transfers, withdrawals, or monetary operations.",
    "Do not click destructive actions such as delete, remove, cancel, terminate, or destroy without confirmation.",
    "Do not modify account credentials including password, email, phone number, or security settings.",
    "Do not submit login forms unless explicitly instructed by the user.",
    "Do not automatically accept Terms of Service, privacy agreements, or legal disclaimers.",
    "Do not bypass CAPTCHA, bot-detection mechanisms, or security verification systems.",
    "If a CAPTCHA or bot-detection screen is detected, stop automation and trigger HITL.",
    "Do not attempt to simulate human verification challenges.",
    "Do not perform actions that appear to evade anti-bot systems or rate limits.",
    "Do not scrape or extract personal or sensitive information.",
    "Do not send messages, emails, or notifications automatically without confirmation.",
    "Do not submit forms containing personal, financial, or medical information without approval.",
    "If uncertainty exists about the safety or legality of an action, reduce confidence and request human intervention."
  ],
  "site_specific": {},
  "risk_keywords": [
    "delete",
    "remove",
    "purchase",
    "payment",
    "password",
    "approve",
    "checkout",
    "cancel",
    "terminate",
    "destroy",
    "discard",
    "unsubscribe",
    "login",
    "verify",
    "captcha",
    "robot",
    "security",
    "confirm",
    "withdraw",
    "order",
    "buy",
    "pay",
    "otp",
    "authentication"
  ]
}
```

### 3.3 Policy Resolution at Runtime

```python
# PolicyHub merges default + site-specific policies

domain = "amazon.com"
active_policy = policy_hub.get_active_policy(domain)
# → Merges default_policy (15 rules) + amazon.com rules (if any)
# → Returns formatted bullet-point string

risk_keywords = policy_hub.get_policy_risk_keywords()
# → ["delete", "remove", "purchase", "payment", ... ]  (24 keywords)
```

### 3.4 How Policies Enter the Prompt

The `POLICY CONSTRAINTS` block is injected between the INPUTS section and the ACTION SPACE section of the prompt:

```
────────────────────────────────────────────────────────────────────────────
POLICY CONSTRAINTS (you MUST follow these strictly):
• Do not submit payment or checkout forms automatically.
• Do not bypass CAPTCHA, bot-detection mechanisms, or security verification systems.
• [... all 15 default rules ...]

RISK ASSESSMENT RULES:
- Set "policy_risk" to true ONLY when the action involves:
  destructive operations, financial transactions, credentials,
  CAPTCHA/security mechanisms, or account changes.
- Set "policy_risk" to false for normal navigation, reading,
  searching, clicking links, typing search queries.
- Report REALISTIC confidence based on how certain you are about
  element selection. Do NOT artificially reduce confidence for safe
  actions just because the page is complex.
────────────────────────────────────────────────────────────────────────────
```

### 3.5 Structured JSON Output Format

When PolicyHub is active, the model is instructed to respond with structured JSON instead of free-text:

```json
{
  "action": "[element_type] ELEMENT_TEXT -> CLICK",
  "confidence": 75,
  "policy_risk": false,
  "hitl_reason": ""
}
```

| Field         | Type      | Description                                                           |
| ------------- | --------- | --------------------------------------------------------------------- |
| `action`      | string    | SeeAct dataset-style action format (e.g., `[button] Search -> CLICK`) |
| `confidence`  | int 0-100 | Model's realistic certainty about element selection                   |
| `policy_risk` | boolean   | `true` if the action involves a policy-sensitive operation            |
| `hitl_reason` | string    | Explanation when `policy_risk` is true; empty otherwise               |

When PolicyHub is **not** active (legacy dataset mode), the prompt uses the original SeeAct one-line format — full backward compatibility.

---

## 4. How HITL Gate Works (Risk-Based Safety Checkpoint)

### 4.1 Purpose

The HITL gate decides whether a predicted action is safe enough to proceed to grounding (which costs additional compute/API calls). It runs **before** the Action Grounding step to save cost on risky actions.

### 4.2 The Three Trigger Rules

The gate uses an **OR-logic** over three independent risk signals. If **any one** fires, HITL is triggered:

```python
class HITLConfidenceGate:

    def evaluate(self, llm_confidence, policy_risk_flag, action_text,
                 policy_risk_keywords, hitl_reason=""):

        # Combine action text + model's reason for broader keyword scan
        combined_text = f"{action_text} {hitl_reason}"

        # Rule 1: Model explicitly flagged the action as risky
        if policy_risk_flag == True:
            # → HITL ("llm_flagged_policy_risk")

        # Rule 2: Risk keywords detected in action text OR hitl_reason
        matched = _scan_keywords(combined_text, risk_keywords)
        if matched:
            # → HITL (f"risk_keywords_matched: {matched}")

        # Rule 3: Model confidence is below floor (guessing)
        if llm_confidence < LOW_CONFIDENCE_FLOOR:  # default: 30
            # → HITL (f"very_low_confidence ({llm_confidence} < 30)")

        # None triggered → SAFE, continue to grounding
```

### 4.3 Why OR-Logic (Not AND)

Any single risk signal is sufficient because:

- **Rule 1 alone**: Model says `policy_risk: true` but forgot to use a keyword → still risky
- **Rule 2 alone**: Model says `policy_risk: false` but action text contains "delete" → model underestimated risk, still risky
- **Rule 3 alone**: No risk keywords but confidence is 15 → model is guessing, unsafe to execute anything

### 4.4 Why Keyword Scan Covers `hitl_reason` Too

We found that GPT-4o sometimes reports `policy_risk: false` but then writes `hitl_reason: "CAPTCHA security mechanism"`. The model contradicts itself. By scanning both the action text AND the reason text, we catch these contradictions:

```
Action text:   [checkbox] I am not a robot -> CLICK
policy_risk:   false          ← model says safe (WRONG)
hitl_reason:   "CAPTCHA security mechanism"  ← but mentions CAPTCHA here!

Keyword scan on combined text:
  "I am not a robot ... CAPTCHA security mechanism"
  → matches: ["robot", "captcha", "security"]
  → HITL triggered ✅ (CORRECT)
```

This was discovered during live testing on Google reCAPTCHA demo. The model accurately described why the action was risky in `hitl_reason` but set `policy_risk: false` — a contradiction that the keyword scanner now catches.

### 4.5 What Happens When HITL Triggers

```
1. Log warning with trigger reasons and matched keywords
2. SKIP the Action Grounding step (saves DeBERTa + CrossEncoder compute)
3. Record the step with hitl_triggered=True in evaluation
4. Save to liveSiteResults/ (screenshot + JSON)
5. Move to next step (in live mode, prompt user again)
```

### 4.6 What Happens When HITL Does NOT Trigger

```
1. Log "No risk detected"
2. Proceed to Action Grounding (DeBERTa + CrossEncoder)
3. Ground action text to specific DOM element
4. Evaluate against ground truth (dataset mode) or save results (live mode)
```

---

## 5. HITL Gate Decision Table

| Scenario                                   | confidence | policy_risk | Keywords matched               | HITL?   | Trigger rule |
| ------------------------------------------ | ---------- | ----------- | ------------------------------ | ------- | ------------ |
| `[link] Wikibooks -> CLICK`                | 72         | `false`     | none                           | **No**  | —            |
| `[checkbox] I am not a robot -> CLICK`     | 70         | `false`     | "robot" (action)               | **Yes** | Rule 2       |
| same + reason "CAPTCHA security mechanism" | 70         | `false`     | "robot", "captcha", "security" | **Yes** | Rule 2       |
| `[button] Delete item -> CLICK`            | 55         | `true`      | "delete"                       | **Yes** | Rule 1 + 2   |
| `[button] Search -> CLICK`                 | 90         | `false`     | none                           | **No**  | —            |
| `[div] maybe this? -> CLICK`               | 20         | `false`     | none                           | **Yes** | Rule 3       |
| `[button] Buy Now -> CLICK`                | 70         | `false`     | "buy"                          | **Yes** | Rule 2       |
| `[link] About Us -> CLICK`                 | 55         | `false`     | none                           | **No**  | —            |
| `[button] Submit Payment -> CLICK`         | 80         | `true`      | "payment"                      | **Yes** | Rule 1 + 2   |
| `[input] Search -> TYPE: laptop`           | 85         | `false`     | none                           | **No**  | —            |
| `[button] Login -> CLICK`                  | 65         | `false`     | "login"                        | **Yes** | Rule 2       |

---

## 6. Privacy & Safety Awareness Architecture

### 6.1 Why Privacy Matters in Web Automation

Autonomous web agents navigate real websites with real user sessions. Without guardrails, the LLM can:

- **Leak credentials** — typing passwords into wrong fields or submitting login forms to phishing look-alikes
- **Exfiltrate data** — scraping personal information visible on screen
- **Trigger irreversible actions** — purchasing items, deleting accounts, sending messages
- **Evade security controls** — attempting to solve CAPTCHAs or bypass bot detections

### 6.2 How Privacy Is Enforced (3 Layers)

```
Layer 1: Prompt-Level Policy Injection (proactive)
├── 15 policy rules injected INTO the LLM prompt
├── Model sees constraints BEFORE generating any action
├── Forces risk reasoning: "policy_risk" boolean in output
└── Model self-reports risk reason in "hitl_reason" field

Layer 2: HITL Gate Keyword Scanner (reactive)
├── 24 risk keywords scanned against action text + hitl_reason
├── Catches model contradictions (says safe, describes danger)
├── Independent of model's self-assessment
└── Pattern: "delete|purchase|captcha|login|password|..."

Layer 3: Structural Pipeline Block (architectural)
├── HITL fires BEFORE grounding → risky action never reaches execution
├── No DOM element is identified for risky actions
├── Screenshot + metadata preserved for audit trail
└── Pipeline continues without executing the flagged action
```

### 6.3 Coverage Matrix

| Privacy Threat         | Policy Rule  | Keywords                                               | Example                 |
| ---------------------- | ------------ | ------------------------------------------------------ | ----------------------- |
| Unauthorized payment   | Rules 1-3    | purchase, payment, checkout, buy, pay, order, withdraw | "Buy Now" button        |
| Credential exposure    | Rules 5-6    | password, login, otp, authentication                   | "Change Password" form  |
| Data destruction       | Rule 4       | delete, remove, cancel, terminate, destroy, discard    | "Delete Account" button |
| Bot detection bypass   | Rules 8-11   | captcha, robot, security, verify                       | reCAPTCHA checkbox      |
| Personal data leakage  | Rules 12, 14 | _(covered by policy rules, not keywords)_              | Scraping profile info   |
| Unsolicited messaging  | Rule 13      | _(covered by policy rules)_                            | Auto-sending an email   |
| Legal agreement bypass | Rule 7       | confirm, approve                                       | "Accept Terms" checkbox |

### 6.4 Audit Trail

Every step is recorded with full context:

```json
{
  "step": 3,
  "action_text": "[button] Delete All -> CLICK",
  "confidence": 65,
  "policy_risk_flag": true,
  "hitl_triggered": true,
  "hitl_reason": "Destructive bulk deletion action",
  "trigger_rules": [
    "llm_flagged_policy_risk",
    "risk_keywords_matched: ['delete']"
  ],
  "screenshot": "step_003/screenshot.png",
  "timestamp": "2025-06-15T14:30:22Z"
}
```

This audit trail supports:

- Post-hoc analysis of agent behavior
- False positive/negative measurement
- Ablation studies (PolicyHub ON vs OFF)

---

## 7. Design Evolution (What Changed and Why)

### Iteration 1: Composite Confidence Formula (Discarded)

```
composite = 0.7 × LLM_confidence + 0.3 × policy_risk_score
if composite < 75: trigger HITL
```

**Problem:** Mixed uncertainty with risk. Wikipedia (many links, confidence ~60) → composite ~72 → HITL triggered on a perfectly safe navigation action.

### Iteration 2: Paranoid Prompt Calibration (Discarded)

```
"Your DEFAULT confidence MUST be 40-55. Start there and adjust."
"Only go above 75 if ALL of: (a) unambiguous, (b) obvious, (c) zero risk, (d) direct goal."
```

**Problem:** Made the model artificially pessimistic. Every action got confidence 40-55 regardless of actual certainty. HITL triggered on 80%+ of steps.

### Iteration 3: Risk-Based Gating (Current)

```
if policy_risk_flag OR risk_keywords_found OR confidence < 30:
    trigger HITL
else:
    safe to execute
```

**Why this works:**

- Confidence is now **realistic** — model reports honest certainty (60-90 for normal navigation)
- Risk is detected by **three independent signals** — even if the model lies about one, the others catch it
- Safe actions with moderate confidence (55-70) proceed without interruption
- Dangerous actions get caught regardless of confidence level

---

## 8. .env Configuration

```dotenv
# PolicyHub + HITL Risk-Based Gate
POLICY_HUB_PATH="src/policyHub/policies.json"
LOW_CONFIDENCE_FLOOR=30           # HITL triggers if LLM confidence < this
HITL_THRESHOLD=75                 # Legacy (stored, not used for gating)
```

| Variable               | Default                       | Used By              | Purpose                                                  |
| ---------------------- | ----------------------------- | -------------------- | -------------------------------------------------------- |
| `POLICY_HUB_PATH`      | `src/policyHub/policies.json` | `PolicyHub`          | Location of policy rules + risk keywords                 |
| `LOW_CONFIDENCE_FLOOR` | `30`                          | `HITLConfidenceGate` | Confidence below this = model is guessing → HITL         |
| `HITL_THRESHOLD`       | `75`                          | _(not used)_         | Retained from earlier composite formula. Can be removed. |

---

## 9. Complete Pipeline Flow

### 9.1 Live Mode (`RUNNER_MODE=live`)

```
python live_runner.py
    │
    ├── Read .env configuration
    ├── Load live_sites.json (target URLs + instructions)
    ├── Initialize SeeActPipeline (model + PolicyHub + HITL gate)
    ├── Launch Playwright browser (headless or headful)
    │
    └── For each site:
         │
         ├── Navigate to URL
         ├── Prompt user for instruction
         │
         └── For each step:
              │
              ├── [1] Capture page state
              │     → Full-page screenshot (PIL.Image)
              │     → Body outerHTML (for grounding)
              │
              ├── [2] PolicyHub resolution
              │     → Merge default + site-specific rules for domain
              │     → Extract 24 risk keywords
              │
              ├── [3] Input Preparation
              │     → Build prompt: instruction + screenshot + history
              │     → Inject POLICY CONSTRAINTS block
              │     → Set JSON output format with policy_risk field
              │
              ├── [4] Action Generation (GPT-4o / Qwen)
              │     → Model returns: {action, confidence, policy_risk, hitl_reason}
              │     → 4-level fallback parse: JSON → regex JSON → regex fields → plain text
              │
              ├── [5] HITL Gate ◄── BEFORE grounding
              │     ├── policy_risk=true?    → ⚠️ HITL
              │     ├── risk keyword found?  → ⚠️ HITL (scans action + hitl_reason)
              │     ├── confidence < 30?     → ⚠️ HITL
              │     └── none?               → ✅ Safe
              │
              ├── [6] Action Grounding (only if ✅)
              │     → DeBERTa + CrossEncoder
              │     → Match to specific DOM element
              │
              └── [7] Save Results
                    → screenshot.png + result.json
                    → liveSiteResults/run_YYYYMMDD_HHMMSS/SiteName/step_NNN/
```

### 9.2 Dataset Mode (`RUNNER_MODE=dataset`)

Same pipeline but reads from Mind2Web parquet files instead of live browser. Ground-truth evaluation is computed per step (Element Accuracy, Operation F1, Value Accuracy, Step Success).

---

## 10. Files Involved

| File                               | Role                                                                           | Status      |
| ---------------------------------- | ------------------------------------------------------------------------------ | ----------- |
| `src/policyHub/policies.json`      | 15 default policy rules + 24 risk keywords                                     | Implemented |
| `src/policyHub/policy_hub.py`      | Loads/merges policies per domain, extracts keywords                            | Implemented |
| `src/policyHub/hitl_confidence.py` | Risk-based HITL gate: 3 trigger rules, keyword scan on action+reason           | Implemented |
| `src/policyHub/__init__.py`        | Package init, exports `PolicyHub` and `HITLConfidenceGate`                     | Implemented |
| `src/Input_Prepration.py`          | Prompt builder — injects policy block + JSON output format + policy_risk field | Implemented |
| `src/action_generation.py`         | Parses `{action, confidence, policy_risk, hitl_reason}` from model output      | Implemented |
| `src/seeact_pipeline.py`           | Wires: PolicyHub → Prompt → Generation → HITL Gate → Grounding                 | Implemented |
| `src/seeact_evaluation.py`         | Records `policy_risk_flag`, `hitl_triggered`, `hitl_reason` per step           | Implemented |
| `src/config.py`                    | Getters: `get_low_confidence_floor()`, `get_policy_hub_path()`, etc.           | Implemented |
| `src/browser_driver.py`            | Playwright browser for live mode (full-page screenshots, idempotent close)     | Implemented |
| `src/live_result_store.py`         | Saves screenshots + JSON results to `liveSiteResults/`                         | Implemented |
| `live_runner.py`                   | Entry point — reads `.env`, dispatches live or dataset mode                    | Implemented |
| `.env.example`                     | Documents all configuration variables                                          | Updated     |

---

## 11. Comparison: Original SeeAct vs Our Extension

| Aspect               | Original SeeAct                               | Our Extension                                                     |
| -------------------- | --------------------------------------------- | ----------------------------------------------------------------- |
| Safety checks        | None — executes all predicted actions blindly | PolicyHub rules + HITL gate blocks risky actions                  |
| Risk awareness       | None                                          | 24 risk keywords + LLM `policy_risk` boolean flag                 |
| Human oversight      | None                                          | Risk-based HITL triggering (log-only mode)                        |
| Policy constraints   | None                                          | 15 global rules + site-specific overrides, injected into prompt   |
| Confidence scoring   | Not reported by model                         | LLM self-reports 0-100 as structured JSON field                   |
| Output format        | Free-text one-line                            | Structured JSON: `{action, confidence, policy_risk, hitl_reason}` |
| HITL trigger logic   | N/A                                           | `policy_risk_flag OR keywords OR confidence < 30`                 |
| Grounding cost       | Always paid                                   | Skipped when HITL triggers — saves DeBERTa + API cost             |
| Keyword scan scope   | N/A                                           | Scans both action text AND hitl_reason text                       |
| Live website support | No (Mind2Web dataset only)                    | Playwright browser automation via `live_runner.py`                |
| Result persistence   | Eval JSON files only                          | Screenshots + JSON per step in `liveSiteResults/`                 |
| Browser management   | N/A                                           | Full-page Playwright screenshots, idempotent close                |

---

## 12. Research Evaluation Protocol

### Ablation Configurations

| Config           | PolicyHub | HITL Gate | Purpose                                        |
| ---------------- | --------- | --------- | ---------------------------------------------- |
| **Baseline**     | OFF       | OFF       | Original SeeAct performance (no safety)        |
| **+Policy**      | ON        | OFF       | Impact of policy constraints on action quality |
| **+Policy+HITL** | ON        | ON        | Full system with risk-based gating             |

### Metrics to Compare

- Task success rate (Offline0 strict, Offline1 tolerant)
- Element accuracy, Operation F1, Value accuracy
- Average LLM confidence score
- HITL trigger rate (% of steps flagged)
- False positive rate (safe actions incorrectly flagged)
- Cost per task (tokens used — lower when HITL skips grounding)
- Risk prevention rate (% of truly risky actions correctly flagged)
