# DOM Delta Benchmark — Frequently Asked Questions

This document explains the key concepts, metrics, and terminology used in
our DOM Delta A/B benchmark (`evaluate_dom_delta.py`).

---

## Q1. What does "Delta" mean in our terminology?

"Delta" = the **difference between two consecutive DOM snapshots**.

Instead of looking at the **entire page** every step, we compare the
**current page** vs the **previous page** and classify every element:

| Classification | Meaning |
|----------------|---------|
| **Added** | New on the page (wasn't there before) |
| **Removed** | Gone from the page (was there, now isn't) |
| **Modified** | Same element, but its text/attributes changed |
| **Unchanged** | Nothing changed about this element |

**Why this matters:** If you clicked "Add to Cart" and a popup appeared,
only the popup elements are *added/modified*. The 150 other elements on the
page are *unchanged*. So why send all 150 to the grounder LLM again? Just
send the changed + task-relevant ones. That's "DOM Delta Processing" —
processing only the delta (change/difference).

---

## Q2. What is Recall@20?

Recall@20 answers one simple question per sample:

> "Is the correct element (the ground-truth target the user should
> click/type on) present somewhere inside our top-20 candidate list?"

- If yes → `recall = 1` (hit)
- If no → `recall = 0` (miss)

The final percentage (e.g., 84%) is: `(number of hits / total samples) × 100`.

### What we're calculating recall ON

The **Mind2Web dataset** gives us thousands of real web automation tasks.
Each task has a known correct element (identified by `backend_node_id`).
Each web page has ~50–200 interactive elements. Our filter picks the top 20
most relevant ones. Recall measures whether the correct element survived
the filtering.

This metric matters because the **grounder (LLM) only sees those top-20
elements**. If the correct element gets filtered out, the grounder can
never pick it — the task fails before the LLM even runs.

### The three Mind2Web test splits

| Split | Samples | What it tests |
|-------|--------:|---------------|
| `test_task` | 1,257 | New tasks on websites seen during training |
| `test_website` | 975 | Completely new websites in known domains |
| `test_domain` | 3,838 | Completely new website domains |

---

## Q3. What is the FULL Baseline?

`FULL` = **no delta processing at all**. This is how a standard web agent
works:

1. Take ALL interactive elements from the page HTML
2. Score them using keyword-weighted matching (based on the task description)
3. Pick the top 20

This is the "before our contribution" baseline. Every element on the page
competes equally — there's no notion of "what changed since last step."

---

## Q4. What are the four configurations?

### FULL (baseline)
- No delta awareness. Score ALL elements by keywords.
- Speed: ~7 ms. No paraphrase handling.

### DELTA_KW (Delta + Keyword filtering)
Two things combined:

1. **Delta awareness** — prioritize added/modified elements from the diff.
2. **Keyword matching** — among those elements, keep ones whose
   text/attributes contain words from the task description.

Example: task is "click the checkout button"
- Keywords extracted: `["click", "checkout", "button"]`
- `<button>Checkout</button>` → matches "checkout" → kept
- `<a>About Us</a>` → matches nothing → dropped

**Strength:** Fast (~2 ms), catches exact word matches.  
**Weakness:** Misses paraphrases. Task says "purchase" but button says
"Checkout" → no keyword overlap → missed.

### DELTA_EMBED (Delta + Embedding re-ranking)
Replaces keywords with **MiniLM sentence embeddings**:

1. Convert the task description into a 384-dimensional vector.
2. Convert each element's text into a 384-dimensional vector.
3. Rank by **cosine similarity** (how semantically close are they?).

Example: task is "purchase this item"
- "Add to Bag" → cosine 0.72 (semantically close to "purchase")
- "About Us" → cosine 0.15 (not related)
- "Checkout" → cosine 0.81 (very related)

**Strength:** Catches paraphrases and synonyms that keywords miss.  
**Weakness:** Sometimes misses exact matches that keywords catch trivially.
Also slower (~112 ms because of the neural network forward pass). That's
why in the results DELTA_EMBED had 81% recall — it *understands meaning*
but occasionally ranks an exact keyword match lower than a semantic
near-miss.

### DELTA_HYBRID (Delta + Keywords + Embeddings fused)
Our best configuration. Fuses both scoring methods:

```
score = 0.6 × normalized_keyword_score + 0.4 × cosine_similarity
```

- "Checkout" with task "click checkout" → high keyword **AND** high
  embedding → top ranked
- "Add to Bag" with task "purchase item" → low keyword **BUT** high
  embedding → still ranked well
- "About Us" → low on both → dropped

That's why DELTA_HYBRID recovered to 84% recall (matching FULL) while
DELTA_EMBED alone dropped to 81%.

### Summary table

| Config | What it does | Speed | Paraphrase handling |
|--------|---|---|---|
| **FULL** | No delta. Score ALL elements by keywords | ~7 ms | No |
| **DELTA_KW** | Delta-aware + keyword filter | ~2 ms | No |
| **DELTA_EMBED** | Delta-aware + MiniLM semantic ranking | ~112 ms | Yes |
| **DELTA_HYBRID** | Delta-aware + keywords + embeddings fused | ~112 ms | Yes |

---

## Q5. What is Token Reduction and how is it calculated?

Token reduction measures: **"How much smaller is the text we send to the
grounder compared to the FULL baseline?"**

### The formula

```
token_reduction_pct = (1 − delta_tokens / full_tokens) × 100
```

### What "tokens" actually means here

It's a **character-length proxy**. For each configuration's top-20
candidates, we sum up `len(element.to_summary())` — the character length of
the text representation that would be sent to the grounder LLM. This is
called `tokens_proxy` in the code.

### Concrete example (from the 100-sample run)

| Config | Mean tokens_proxy | Meaning |
|--------|--:|---|
| FULL | 1,406 chars | Full baseline — all top-20 summaries |
| DELTA_HYBRID | 1,136 chars | Our method's top-20 summaries |
| **Reduction** | **19.2%** | `(1 − 1136/1406) × 100 = 19.2%` |

### Why delta methods produce smaller token counts

The delta-aware filters preferentially pick elements that are more tightly
relevant to the task. Even though both output up to 20 elements, the delta
method's 20 elements tend to have shorter, more focused summaries (less
irrelevant noise). Also, when the delta pool is smaller than 20, fewer
candidates are returned.

### Important caveat for the paper

This is a *character-length proxy*, not actual LLM tokens. We chose this
because actual tokenization depends on which tokenizer you use (GPT-4o vs
Qwen vs Llama all tokenize differently), and `len(str)` is model-agnostic
and reproducible. Character count correlates strongly with token count
(~1 token per 4 characters for English). This should be noted in the paper
as "character-length proxy for grounder input cost."

---

## Q6. How to read the result summary table

Example output:

```
config          recall@20     tokens       ms   Δrecall   tok_red%
FULL               84.00%       1406     7.24
DELTA_KW           84.00%       1087     2.00    +0.00pp      22.7%
DELTA_EMBED        81.00%        853   113.29    -3.00pp      39.3%
DELTA_HYBRID       84.00%       1136   112.03    +0.00pp      19.2%
```

| Column | What it means |
|--------|---------------|
| `recall@20` | % of samples where correct element is in the top-20 (higher = better) |
| `tokens` | Average character-length of the top-20 candidate block (lower = cheaper) |
| `ms` | Filter-stage time per sample in milliseconds (lower = faster) |
| `Δrecall` | Recall change vs FULL baseline in percentage points |
| `tok_red%` | Token reduction vs FULL baseline (higher = more savings) |

### How to interpret

- **DELTA_KW**: Same recall as FULL, 22.7% fewer tokens, 3.5× faster.
  Pure win — delta + keyword filtering drops irrelevant elements without
  losing the correct one.

- **DELTA_EMBED**: 3pp recall drop, but 39.3% token reduction. The
  embedding ranker is aggressive — it understands semantics but sometimes
  misses exact keyword matches.

- **DELTA_HYBRID**: Same recall as FULL, 19.2% token reduction. Best
  balance — fuses keyword exactness with semantic understanding. This is
  the configuration we report in the paper.

---

## Q7. Whose scoring algorithm is used in each config?

The FULL baseline uses **Prune4Web's existing scoring algorithm**, not ours.
Our novel scoring only appears in DELTA_EMBED and DELTA_HYBRID.

| Config | Scoring Algorithm | Whose code? |
|--------|---|---|
| **FULL** | `keywords_from_text_heuristic()` + `score_elements()` — weighted keyword matching with rapidfuzz | **Prune4Web's** existing code |
| **DELTA_KW** | Same Prune4Web scorer, but applied on a keyword-prefiltered pool | **Prune4Web's** scoring + our delta-style prefilter on top |
| **DELTA_EMBED** | `score_elements_semantic()` — MiniLM cosine similarity | **Ours** (new, `dom_relevance_embed.py`) |
| **DELTA_HYBRID** | `hybrid_rank()` — fuses Prune4Web keyword scores + MiniLM cosine | **Ours** (new, `dom_relevance_embed.py`) |

### What this means for the paper

We are comparing **Prune4Web's existing filter** (FULL baseline) against
**our DOM Delta + embedding-enhanced filter** (DELTA_HYBRID). The
contribution is:

1. The **delta-awareness layer** (only surface changed + task-relevant
   elements instead of the full DOM)
2. The **MiniLM hybrid scoring** (semantic paraphrase understanding fused
   with keyword exactness)

The base keyword scorer (`score_elements` with weighted keywords +
rapidfuzz) is Prune4Web's, not our novel contribution.

---

## Remaining Questions

1. "Structural UIDs (an MD5 hash over tag, id, name, xpath)" what is MD5?
- MD5 is a widely used cryptographic hash function that produces a 128-bit (16-byte) hash value, typically represented as a 32-character hexadecimal string. In the context of structural UIDs for DOM elements, an MD5 hash is generated based on the element's tag, id, name, and xpath to create a unique identifier for that element. This allows us to track elements across different DOM snapshots and identify which ones have been added, removed, modified, or unchanged.

2. we contain 6000 arounde samples total in test, but in evaluation why it is showing only 800 steps? what does here meaning of step ? is it like batching? what is the value of it?

● A step = one gradient update on one effective batch, not one sample. With effective batch size 16 (per-device ×
  grad-accum) on 6,626 train samples:

  - 6,626 / 16 ≈ 414 steps per epoch
  - Qwen3-0.6B trained for 2 epochs → 414 × 2 ≈ 828 steps (the ~800 you're seeing)
  - Qwen2.5-0.5B trained for 3 epochs → 414 × 3 ≈ 1,242 steps

  So "step" here is optimizer step (batched), not sample. The 6,626 samples are all being consumed each epoch — just 16
  at a time per update.
  
---

## Q8. How do I run a live browser session with different configurations?

The live runner is `run_prune4web.py` at the repo root. It launches a real
Playwright browser, drives Prune4Web on a URL + natural-language task, and
prints per-step decisions plus a token/cost summary at the end.

### Minimal command

```bash
cd D:/Mtech/Sem\ 3/Codes/WebAgents/Prune4Web
python run_prune4web.py --url "https://www.amazon.com" --task "search for noise cancelling headphones and add the first result to cart"
```

If `--url` or `--task` is omitted, the script will prompt for them.

### CLI flags (defined in `run_prune4web.py`)

| Flag | Default | Purpose |
|---|---|---|
| `--url` | (prompted) | Starting URL the browser opens |
| `--task` | (prompted) | High-level natural-language task |
| `--max-steps` | `5` (or `PRUNE4WEB_MAX_STEPS`) | Hard cap on planner→filter→grounder iterations |
| `--grounder` | `qwen3` (or `GROUNDER_BACKEND`) | Grounder backend: `qwen3` (local Qwen3-0.6B + LoRA, default), `qwen25` (local Qwen2.5-0.5B + LoRA), or `gpt4o` (hosted OpenAI) |

The default grounder is the **locally-fine-tuned Qwen3-0.6B**. The
`OPENAI_API_KEY` is still required because the planner and filter stages
continue to use the hosted model — only the grounder stage is local
unless you also point `PRUNE4WEB_PLANNER_MODEL` / `PRUNE4WEB_FILTER_MODEL`
at a local OpenAI-compatible endpoint. Everything beyond these flags is
configured via **environment variables** (typically in a `.env` at the
repo root).

### Environment variables (live-run knobs)

| Variable | Default | What it controls |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Hosted-LLM key for planner/filter/grounder calls |
| `PRUNE4WEB_PLANNER_MODEL` | `gpt-4o` | Model name used for the planner stage |
| `PRUNE4WEB_FILTER_MODEL` | `gpt-4o` | Model used to emit the filter scoring script |
| `PRUNE4WEB_GROUNDER_MODEL` | `gpt-4o` | Hosted model name used **only when** `--grounder gpt4o` is chosen |
| `GROUNDER_BACKEND` | `qwen3` | Default grounder backend if `--grounder` is omitted (`qwen3` / `qwen25` / `gpt4o`) |
| `GROUNDER_BASE_MODEL` | auto-set by `--grounder` | Local base-model directory; override only for custom paths |
| `GROUNDER_OUTPUT_MODEL` | auto-set by `--grounder` | LoRA adapter directory; override only for custom paths |
| `GROUNDER_DISABLE_THINKING` | `1` for `qwen3`, unset for `qwen25` | Strips empty `<think>` block from Qwen3's chat template |
| `PRUNE4WEB_MAX_STEPS` | `5` | Same as `--max-steps`, used when flag is omitted |
| `PRUNE4WEB_USE_VISION` | `1` | `1` = send screenshot alongside HTML, `0` = HTML only |
| `PRUNE4WEB_DOM_DELTA` | `1` | Toggle DOM Delta layer (structural UID + delta-aware filter) |
| `PRUNE4WEB_PRIVACY` | `1` | Toggle Privacy-Aware LLM wrapper (regex PII masking + directive) |
| `PRUNE4WEB_DP_EPSILON` | `0` | Laplace-noise budget on candidate scores; `0` disables DP |
| `PRUNE4WEB_USE_HITL` | `1` | Toggle the risk-based HITL gate around the planner |
| `PRUNE4WEB_HITL_THRESHOLD` | `75` | Confidence ceiling used by the HITL gate |
| `PRUNE4WEB_HITL_LOW_CONFIDENCE_FLOOR` | `30` | Confidences below this auto-trigger HITL |
| `PRUNE4WEB_POLICY_HUB_PATH` | `AutoWeb/src/policyHub/policies.json` | Path to the PolicyHub rule/keyword JSON |

`HITL_THRESHOLD` and `LOW_CONFIDENCE_FLOOR` (without the `PRUNE4WEB_`
prefix) are also accepted as fallbacks for backward compatibility.

### Common live-run recipes

**1. Vanilla Prune4Web baseline (no DOM Delta, no Privacy, no HITL).**
Useful when you want to compare against a "stock" Prune4Web run.

```bash
PRUNE4WEB_DOM_DELTA=0 \
PRUNE4WEB_PRIVACY=0 \
PRUNE4WEB_USE_HITL=0 \
python run_prune4web.py --url "<URL>" --task "<TASK>"
```

**2. Full proposed pipeline (everything on — the paper config).**

```bash
PRUNE4WEB_DOM_DELTA=1 \
PRUNE4WEB_PRIVACY=1 \
PRUNE4WEB_USE_HITL=1 \
python run_prune4web.py --url "<URL>" --task "<TASK>"
```

**3. Privacy-only ablation (HITL off, Delta on).**

```bash
PRUNE4WEB_DOM_DELTA=1 \
PRUNE4WEB_PRIVACY=1 \
PRUNE4WEB_USE_HITL=0 \
python run_prune4web.py --url "<URL>" --task "<TASK>"
```

**4. Strict HITL (lower the ceiling so more sub-tasks ask for confirmation).**

```bash
PRUNE4WEB_HITL_THRESHOLD=50 \
PRUNE4WEB_HITL_LOW_CONFIDENCE_FLOOR=40 \
python run_prune4web.py --url "<URL>" --task "<TASK>"
```

**5. Cheaper hosted model on the planner/filter, local Qwen3 grounder.**

```bash
PRUNE4WEB_PLANNER_MODEL=gpt-4o-mini \
PRUNE4WEB_FILTER_MODEL=gpt-4o-mini \
python run_prune4web.py --url "<URL>" --task "<TASK>"
# --grounder defaults to qwen3, no flag needed
```

**5b. Pick a different grounder explicitly.**

```bash
# Use the local Qwen2.5-0.5B grounder instead of Qwen3
python run_prune4web.py --grounder qwen25 --url "<URL>" --task "<TASK>"

# Fall back to the hosted gpt-4o grounder (original Prune4Web behaviour)
python run_prune4web.py --grounder gpt4o --url "<URL>" --task "<TASK>"
```

**6. Differential-privacy noise on the candidate scores (ε = 1.0).**

```bash
PRUNE4WEB_DP_EPSILON=1.0 \
python run_prune4web.py --url "<URL>" --task "<TASK>"
```

### Windows PowerShell equivalent

PowerShell does not accept inline `KEY=VAL command` syntax. Set first, then
run:

```powershell
$env:PRUNE4WEB_DOM_DELTA = "1"
$env:PRUNE4WEB_USE_HITL  = "1"
python run_prune4web.py --url "<URL>" --task "<TASK>"
```

Or put the same keys into a `.env` file at the repo root and let
`python-dotenv` load them — the script calls `load_dotenv()` on startup, so
a `.env` is the cleanest way to pin a configuration for repeated runs.

### What you will see during a run

- A real Chromium window opens on the given URL.
- For each step, the console prints the planner's chosen sub-task, the
  filter's top-20 candidates, the HITL decision (`hitl_required: true|false`
  with a reason), and the grounder's chosen element.
- If HITL fires, the run pauses and asks you to perform that sub-task in
  the live browser yourself; once done, the agent resumes from the next
  page state.
- When the run ends (task complete or `--max-steps` hit), a summary block
  prints token counts per stage, a USD cost estimate using
  `MODEL_PRICING_PER_1M`, and the final HITL/Privacy flags that were active.

### Other runnable scripts (not the live runner)

These exist alongside `run_prune4web.py` but do not drive a live browser:

| Script | Purpose |
|---|---|
| `run_full_dom_delta_bench.py` | Offline A/B benchmark that produced the FULL vs DELTA_HYBRID Recall@20 numbers in this FAQ |
| `run_followup_experiments.py` | Sweep harness for additional ablation runs |
| `grounder_finetune/run_qwen3_pipeline.py` | QLoRA fine-tuning entrypoint for the local grounder |

The local QLoRA grounder is now the **default** — `--grounder qwen3` is
applied unless you ask for `qwen25` or `gpt4o`. If you want fully
on-device runs (planner + filter + grounder all local), point
`PRUNE4WEB_PLANNER_MODEL` and `PRUNE4WEB_FILTER_MODEL` at a local
OpenAI-compatible endpoint (e.g. `vllm` or `text-generation-inference`)
and keep `OPENAI_API_KEY` set to any non-empty value so the SDK
initialises. End-to-end local execution has not been validated against
the benchmark numbers in this FAQ.

---

## Q9. What models are used by default for planner / filter / grounder, and what combinations are supported?

### Defaults (no flags, no env overrides)

| Stage | Default | Hosted/Local |
|---|---|---|
| **Planner** | `gpt-4o` | hosted (OpenAI) — no local fine-tune yet |
| **Filter** | deterministic keyword extractor + DOM-Delta (MiniLM) | **local** |
| **DOM Delta** | structural UID hash + MiniLM (22M) hybrid score (α=0.6) | **local** |
| **Grounder** | `Qwen3-0.6B + LoRA` | **local** |
| **PolicyHub / HITL** | rule engine (24 keywords × 3 triggers) | **local** |
| **Privacy-Aware-LLM** | regex PII masking (+ optional ε-DP) | **local** |

Out of the box, **only the planner LLM** still calls GPT-4o. Filter, DOM-Delta, grounder, HITL and privacy all run on-device. `OPENAI_API_KEY` is still required because the planner uses it.

### Filter backends exposed by `--filter`

| Flag value | Backend | Notes |
|---|---|---|
| `local` *(default)* | deterministic tokenizer → keyword weights, then DOM-Delta + MiniLM hybrid scorer | No API call. The keyword bag-of-words feeds the existing Python scorer; DOM Delta narrows the candidate pool ahead of it. |
| `gpt4o` | hosted GPT-4o keyword-weight LLM (the original Prune4Web filter) | Use `--filter gpt4o` to reproduce the legacy paper baseline. |

### Grounder backends exposed by `--grounder`

| Flag value | Backend | Model directory / model id |
|---|---|---|
| `qwen3` *(default)* | local QLoRA | base `D:/Environments/Models/Qwen3-0.6B` + adapter `D:/Environments/Models/Qwen3-0.6B-Prune4Web-Grounder` |
| `qwen25` | local QLoRA | base `D:/Environments/Models/Qwen2.5-0.5B` + adapter `D:/Environments/Models/Qwen2.5-0.5B-Prune4Web-Grounder` |
| `gpt4o` | hosted OpenAI | whatever `PRUNE4WEB_GROUNDER_MODEL` is set to (default `gpt-4o`) |

The runtime banner now prints the resolved value, e.g. `grounder_model : local:Qwen3-0.6B-Prune4Web-Grounder`.

### Planner / Filter overrides

Planner and filter do not have CLI flags; they are controlled by env vars at process start:

```bash
export PRUNE4WEB_PLANNER_MODEL=gpt-4o-mini   # cheaper planner
export PRUNE4WEB_FILTER_MODEL=gpt-4o-mini    # cheaper filter
```

Any model id reachable through the OpenAI-compatible client works (`gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, ...). If `OPENAI_BASE_URL` is pointed at a local OpenAI-compatible server (vLLM, TGI, llama.cpp's server, LM Studio), these names can also be local model ids served by that endpoint — but this path has **not** been validated against the benchmark numbers in this FAQ.

### Supported combinations

The three stages compose independently. The matrix below lists the combinations that are wired up and tested:

| # | Planner | Filter | Grounder | How to run | Notes |
|---|---|---|---|---|---|
| 1 | `gpt-4o` | local (default) | `qwen3` (local) | `python run_prune4web.py --url ... --task ...` | **Default.** Only planner is hosted. |
| 2 | `gpt-4o` | `gpt4o` | `qwen3` (local) | `... --filter gpt4o` | Restores the legacy keyword-weight LLM (paper baseline filter). |
| 3 | `gpt-4o` | `gpt4o` | `gpt4o` (hosted) | `... --filter gpt4o --grounder gpt4o` | Full original Prune4Web baseline (88.28% EA). |
| 3a | `gpt-4o` | local | `qwen25` (local) | `... --grounder qwen25` | 85.0% EA on `test_task`; smaller VRAM footprint. |
| 4 | `gpt-4o-mini` | `gpt-4o-mini` | `qwen3` (local) | `PRUNE4WEB_PLANNER_MODEL=gpt-4o-mini PRUNE4WEB_FILTER_MODEL=gpt-4o-mini python run_prune4web.py ...` | Cheapest hosted-tier combo; expect a small EA drop on harder splits. |
| 5 | `gpt-4o-mini` | `gpt-4o-mini` | `gpt-4o-mini` | `PRUNE4WEB_PLANNER_MODEL=gpt-4o-mini PRUNE4WEB_FILTER_MODEL=gpt-4o-mini PRUNE4WEB_GROUNDER_MODEL=gpt-4o-mini python run_prune4web.py ... --grounder gpt4o` | Pure low-cost API run. |
| 6 | local OpenAI-compatible | local OpenAI-compatible | `qwen3` (local) | Set `OPENAI_BASE_URL=http://localhost:<port>/v1` plus the two planner/filter env vars to your served model id; run with default flags. | Fully on-device. **Not benchmark-validated.** |
| 7 | (legacy) any | any | local | `GROUNDER_USE_LOCAL=1 python run_prune4web.py ...` | Back-compat path; equivalent to `--grounder qwen3`. The CLI flag wins if both are present. |

### Quick decision guide

- **Want the published numbers?** Use combo #1 (default) or #3.
- **Want minimum API cost while keeping the local grounder?** Use combo #4.
- **Tight on VRAM (<6 GB)?** Prefer `qwen25` (combo #2).
- **Cannot use OpenAI at all?** Combo #6 — works but unvalidated.

---

## Q10. Where do I change models without editing code?

A single config file: **`Prune4Web/config/models.json`**.

Open it, edit the `backend` / `model` / model-path values, save, run. No code change needed.

```json
{
  "planner":  { "backend": "openai", "model": "gpt-4o", "use_vision": true },
  "filter":   { "backend": "local",  "gpt4o_model": "gpt-4o" },
  "grounder": {
    "backend": "qwen3",
    "qwen3":  { "base_model": "...", "adapter": "...", "disable_thinking": true },
    "qwen25": { "base_model": "...", "adapter": "..." },
    "gpt4o":  { "model": "gpt-4o" }
  },
  "embeddings": { "minilm_model": "sentence-transformers/all-MiniLM-L6-v2", "alpha_blend": 0.6 },
  "privacy":    { "enabled": true, "dp_epsilon": 0 },
  "hitl":       { "enabled": true, "threshold": 72, "low_confidence_floor": 60, "policy_hub_path": "..." }
}
```

### Precedence

For any setting, the order is **CLI flag > pre-existing env var > config file > built-in default**. So you can:

- Edit `models.json` to change defaults globally.
- Override one stage for a single run with `--grounder qwen25` or `--filter gpt4o`.
- Force a model from a script with `PRUNE4WEB_PLANNER_MODEL=gpt-4o-mini python run_prune4web.py ...`.

### Switching backends — quick examples

| What you want | Edit in `models.json` |
|---|---|
| Use Qwen2.5-0.5B grounder instead of Qwen3 | `"grounder": { "backend": "qwen25", ... }` |
| Restore the original GPT-4o filter | `"filter":   { "backend": "gpt4o", "gpt4o_model": "gpt-4o" }` |
| Run planner on a cheaper API model | `"planner":  { "model": "gpt-4o-mini" }` |
| Move the local model directory | Update `qwen3.base_model` and `qwen3.adapter` paths |
| Turn off the privacy layer | `"privacy":  { "enabled": false }` |
| Tighten HITL gate | `"hitl":     { "threshold": 80 }` |

### Custom config file location

Pass `--config /path/to/your_models.json` or set `PRUNE4WEB_CONFIG=/path/...`. Useful for keeping benchmark and live-run configs separate.

---

*Document created 2026-04-16. Q8 added 2026-04-30. Q9, Q10 added 2026-05-01.*
