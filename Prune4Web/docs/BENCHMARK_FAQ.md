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

*Document created 2026-04-16.*
