# Deep-Dive: Three Fundamental Concepts

A reference document covering, in detail and grounded in the actual code:

1. The three **DOM Delta methods** — keyword, embedding, hybrid.
2. **QLoRA fine-tuning** of Qwen3-0.6B as the grounder.
3. How the **grounder runs fully locally** at inference time.

Every claim here is traceable to a specific source file in the repo. File paths are given inline so you can audit any statement.

---

# Part 1 — The Three DOM Delta Methods

All three live in **`Prune4Web/evaluate_dom_delta.py`** as functions `run_delta_keyword()`, `run_delta_embedding()`, `run_delta_hybrid()`. The semantic-embedding helpers are in **`AutoWeb/src/dom_relevance_embed.py`**.

The benchmark harness compares all three against a fourth baseline called `run_full_baseline` (no delta, just keyword scoring on the full DOM). Hybrid is the one we ship as default and that the report numbers refer to.

## 1.0 Common inputs

Every method takes the same four arguments:

| Argument | Type | What it carries |
|---|---|---|
| `candidates` | `list[ElementNode]` | All DOM elements on the current page (typically a few hundred to a few thousand). Each carries `text`, `aria_label`, `placeholder`, `title`, `name`, `elem_id`, `tag`, `xpath`, etc. |
| `sub_task` | `str` | The planner's natural-language instruction for the *current* step, e.g. *"Click the Add to Bag button to buy the iPad"*. |
| `element_desc` | `str` | A short target description (e.g. *"Add to Bag button"*). In offline benchmarking this comes from the Mind2Web ground-truth `target_action_reprs`; in live runs it falls back to the sub-task. |
| `top_n` | `int` | How many candidates to return. Default 20. |

Each method returns `(ranked_top_n_elements, telemetry_dict)`.

## 1.1 How the snapshot-to-snapshot delta is actually calculated

Before any of the three scoring methods run, the live agent does a **structural diff between two consecutive DOM snapshots** — the *true* "delta" that gives this whole layer its name. The three methods (Keyword / Embed / Hybrid) sit *on top of* this diff: they score and rank the elements that the diff has identified as relevant to the current step.

The code lives in **`AutoWeb/src/dom_state.py`** (snapshot capture + structural UID) and **`AutoWeb/src/dom_diff.py`** (`compute_delta` + `get_relevant_elements`). The runner in `Prune4Web/run_prune4web.py:996–1024` calls these every step.

### The two-snapshot picture

At step `t`:

```
                ┌──── browser action ────┐
DOM snapshot t-1 ─────────────────────────► DOM snapshot t
   (before)                                   (after)
       │                                        │
       └────────────── compute_delta ───────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │ added    : New nodes   │
                │ removed  : Gone nodes  │
                │ modified : Same node,  │
                │            attrs moved │
                │ unchanged: Identical   │
                └────────────────────────┘
                            │
                  get_relevant_elements
                            │
                            ▼
              candidate pool fed to scoring (1.2–1.4)
```

The first step has no `t-1` snapshot — the runner falls back to the full DOM and the three scoring methods run as if there were no diff at all. From step 2 onwards, the diff is meaningful.

### Step 1 — Capture each snapshot

`capture_dom_state(html, url, title, timestamp)` parses the page with BeautifulSoup and walks every actionable node. For each node it produces a `SnapshotElement` carrying the standard fields (`tag`, `text`, `aria_label`, `placeholder`, `elem_id`, `name`, `xpath`, `ancestors`, …) **plus two derived values** that the diff depends on.

#### The structural UID (element identity)

`dom_state.py:272–273`:

```python
identity = f"{tag_name}|{snap.elem_id}|{snap.name}|{snap.xpath}"
snap.uid = hashlib.md5(identity.encode("utf-8")).hexdigest()[:12]
```

Four components — tag name, HTML id, name attribute, approximate XPath — concatenated and hashed to a 12-char MD5. This is the **stable identity** of an element across re-renders. Two crucial properties:

- The same logical element on the page produces the **same uid** even if its visible text or aria-label mutates between snapshots, because none of those are in the identity string.
- A truly new node (e.g. a popup that wasn't there before) produces a **different uid**, because either its tag, id, name, or xpath will differ from anything in the previous snapshot.

This is the substitute for Mind2Web's `backend_node_id` that we use at run-time, where no live browser instrumentation is available.

#### The fingerprint (element content)

`dom_state.py:70–75`:

```python
def fingerprint(self) -> str:
    payload = "|".join(str(getattr(self, attr)) for attr in TRACKED_ATTRS)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()
```

A second hash, this time over the *content* fields — `text`, `aria_label`, `placeholder`, `value`, `disabled`, `checked`, `selected`, `href`, `role`, `elem_class`, `input_type`. Two snapshots of the *same* element (same uid) get the *same* fingerprint when their content is unchanged, and a *different* fingerprint when any of those eleven attributes mutated.

So each element carries two hashes: **uid** says *who* it is; **fingerprint** says *what state* it is in.

### Step 2 — Set arithmetic on uids

`compute_delta()` in `dom_diff.py:111–144` is essentially three set operations between the uid sets of the two snapshots:

```python
before_uids = set(before.elements.keys())
after_uids  = set(after.elements.keys())

added    = after_uids  - before_uids        # in after, not in before
removed  = before_uids - after_uids         # in before, not in after
common   = before_uids & after_uids         # present in both
```

For elements in `common`, we compare **fingerprints**:

```python
for uid in common:
    if before[uid].fingerprint() != after[uid].fingerprint():
        # same element, content changed
        modified.append(ElementChange(uid, before=before[uid], after=after[uid],
                                      changed_attrs=_changed_attributes(before, after)))
    else:
        unchanged.append(after[uid])
```

`_changed_attributes()` simply enumerates which of the eleven tracked content fields actually flipped — useful for telemetry and for the "(changed: text, aria_label)" suffix in delta summaries.

### Worked example — what one delta looks like

Suppose the user has just clicked **"Add to Bag"** on an iPad page. Before the click, the page had 312 elements; after the click, a mini-cart popup appeared and the page now has 327 elements.

```
before snapshot: 312 elements   uids: { a3f7..., 9e21..., 4b08..., ... }   (312 uids)
after snapshot:  327 elements   uids: { a3f7..., 9e21..., 4b08..., ...,
                                        ff70..., 8c1d..., 6e44..., ... } (327 uids)

Set arithmetic:
  added    = after_uids - before_uids
           = { ff70..., 8c1d..., 6e44..., 2a91..., ... }       → 18 new elements
             (the popup's "View Bag", "Continue Shopping", item rows, etc.)

  removed  = before_uids - after_uids
           = { 7d12..., a48b..., 5f33... }                     → 3 elements
             (the "Add to Bag" CTA may have been replaced by a smaller "in cart"
              indicator; carousel arrows that lost focus).

  common   = before_uids ∩ after_uids                           → 309 elements

For each of the 309 common uids, compare fingerprints:
  modified = 5  elements        (cart counter "0" → "1"; status text changed;
                                 a button's aria-label flipped to "Added")
  unchanged = 304 elements      (header, navigation, footer, sidebar, ...)
```

Resulting `DOMDelta`:

```python
DOMDelta(
    before_count = 312,
    after_count  = 327,
    added        = [18 SnapshotElements],
    removed      = [3  SnapshotElements],
    modified     = [5  ElementChange records],
    unchanged    = [304 SnapshotElements],
)

# delta.print_summary() →
# DOM delta: before=312 after=327 added=18 removed=3 modified=5 unchanged=304 reduction=14.2x
```

`reduction_factor = after_count / max(len(added) + len(modified), 1) = 327 / 23 ≈ 14.2×`. That is, instead of re-processing 327 elements, the relevance filter only needs to seriously consider 23 — the things that *actually changed* in this step.

### Step 3 — Pick relevant elements from the delta

`get_relevant_elements()` in `dom_diff.py:170` turns the raw delta into the candidate pool that the three scoring methods will rank. The priority order is:

1. **Added elements** that match task keywords. Highest signal — a new element that mentions task vocabulary is almost certainly part of the next interaction.
2. **All modified elements**. We include every modified element regardless of keyword match, because *something happened to it* and that something is usually a consequence of the previous action.
3. **Unchanged elements** that match task keywords, but only if the first two pools came up empty (or `include_unchanged=True` is forced — which is what step 1 does, since there is no diff yet).

Continuing the example: of the 18 *added* popup elements, suppose 11 keyword-match the live task (*"complete checkout"*) — "View Bag", "Continue", "Subtotal", item title, etc. All 5 modified elements are auto-included. So the candidate pool entering Method A/B/C is **16 elements** instead of 327 — a **20× reduction in the number of strings the downstream scorer has to embed or match against.**

That ratio — single-digit-to-low-double-digit candidates instead of hundreds — is what makes the embedding pass in Methods B and C cheap enough to run on every step. Without this snapshot-diff layer, we would be paying MiniLM to encode the entire footer and navigation of every page, every step, even though none of it has changed since the previous step.

### How the live runner calls all of this

`run_prune4web.py:996–1024`:

```python
delta_elements = None
if delta_processor is not None:
    dom_state, delta = delta_processor.update_and_diff(
        html, url=url_now, title=title, timestamp=time.time()
    )
    if delta is not None:                       # not the first step
        delta.print_summary()
        snap_relevant = delta_processor.get_relevant_elements(
            delta, task_context=task
        )
        # Map back from SnapshotElement → ElementNode for the scorer
        snap_ids   = {s.elem_id for s in snap_relevant if s.elem_id}
        snap_texts = {s.text[:60] for s in snap_relevant if s.text}
        delta_elements = [
            el for el in elements
            if el.elem_id in snap_ids or el.text[:60] in snap_texts
        ] or elements                           # safety fallback

# Use delta-filtered pool if available, else full DOM
candidate_source = delta_elements if delta_elements is not None else elements
```

So:

- **Step 1**: `delta is None` → `candidate_source = elements` (full DOM).
- **Step ≥2**: `delta_elements` is a small subset of the full DOM, and Methods A/B/C run on this subset.

### Three things this layer is responsible for, and three it isn't

**Is responsible for:**
- Element identity across re-renders (structural UID).
- Detecting which elements appeared / disappeared / mutated since the previous step.
- Producing a small, task-relevant candidate pool for the scorer.

**Is *not* responsible for:**
- Ranking inside the candidate pool — that's Methods A/B/C in §1.2–§1.4.
- Deciding which element to click — that's the grounder.
- Knowing anything about the user's *high-level* task — `get_relevant_elements()` only sees the planner's per-step *sub-task* string.

This separation is deliberate. The diff layer is a deterministic structural operation; the scoring layer is the part that needs language understanding; the grounder is the part that needs reasoning. Each of them does one thing well, and any one of them can be swapped out without touching the others.

## 1.2 Method A — Delta_Keyword (lexical only)

### Code
```python
# evaluate_dom_delta.py, lines 205–224
def run_delta_keyword(candidates, sub_task, element_desc, top_n):
    kws = [k.lower() for k in element_desc.split() if len(k) > 2]
    kws += [w.lower() for w in sub_task.split() if len(w) > 3]
    kws = list(set(kws))

    def matches(el):
        blob = _semantic_text(el).lower()
        return not kws or any(k in blob for k in kws)

    prefiltered = [el for el in candidates if matches(el)] or list(candidates)
    weights = keywords_from_text_heuristic(sub_task, element_desc)
    ranked = score_elements(prefiltered, weights, top_n=top_n)
    return ranked, {...}
```

### How it computes the "delta"

The word *delta* in this method does **not** mean a snapshot diff between two consecutive pages. It means *task-relative reduction*: drop every element whose semantic text (visible text + aria + placeholder + title + name + id) does not contain at least one keyword from the sub-task or target description. That set difference *is* the "delta": the elements that pass the filter vs. those that don't.

Two stages:

1. **Lexical pre-filter.** Tokenise the sub-task and target description, keep words longer than 2–3 chars, lowercase. Keep only candidates whose semantic blob contains at least one of those tokens. If everything is filtered out, fall back to the full candidate list (defensive default).
2. **Keyword-weighted scoring.** Apply `score_elements()` with weights from `keywords_from_text_heuristic`. The scorer awards points for exact match (α), phrase match, word match, and fuzzy match (rapidfuzz partial ratio), each multiplied by an attribute-importance factor β (visible_text > aria_label > placeholder > id/class > other). Sort descending; return top-N.

### Worked example (dry run)

Let's walk a single step end-to-end so the moving parts become concrete.

#### Inputs

```python
sub_task     = "Click the Add to Cart button to buy the iPad"
element_desc = "Add to Cart button"        # from Mind2Web target_action_reprs
top_n        = 5                           # using 5 here for readability; default is 20
```

`candidates` — eight DOM elements pulled from a simplified product page:

| uid | tag    | text             | aria_label       | placeholder | elem_id        |
|---:|---|---|---|---|---|
| 0   | button | "Search"         | "Search Apple"   | ""          | "search-btn"   |
| 1   | input  | ""               | ""               | "Search…"   | "search-input" |
| 2   | button | "Add to Cart"    | "Add iPad to cart" | ""        | "buy-btn"      |
| 3   | button | "Buy now"        | "Buy iPad now"   | ""          | "buynow-btn"   |
| 4   | a      | "Sign in"        | "Sign in to your account" | ""  | "signin-link"  |
| 5   | a      | "Compare"        | "Compare iPads"  | ""          | "compare-link" |
| 6   | button | "Add to Wishlist"| "Save iPad to wishlist" | ""    | "wish-btn"     |
| 7   | footer | "© Apple Inc."   | ""               | ""          | "footer"       |

#### Stage 1 — Tokenise the query

```python
kws  = [k.lower() for k in element_desc.split() if len(k) > 2]
# element_desc = "Add to Cart button"
# split        = ["Add", "to", "Cart", "button"]
# len > 2      = ["Add", "Cart", "button"]   ("to" dropped, len=2)
# lower        = ["add", "cart", "button"]

kws += [w.lower() for w in sub_task.split() if len(w) > 3]
# sub_task     = "Click the Add to Cart button to buy the iPad"
# split        = ["Click","the","Add","to","Cart","button","to","buy","the","iPad"]
# len > 3      = ["Click","button","iPad"]   ("the","Add","to","Cart","buy" dropped)
# lower        = ["click","button","ipad"]

kws  = list(set(kws))
# final keyword bag = {"add", "cart", "button", "click", "ipad"}
```

Note the asymmetric thresholds: `>2` for `element_desc` (more permissive — we trust this string), `>3` for `sub_task` (more aggressive trim — sub-tasks often carry filler verbs).

#### Stage 2 — Lexical pre-filter

For each candidate, build a "semantic blob" and check whether *any* keyword is a substring:

```python
def _semantic_text(e):
    return " ".join(filter(None, [e.text, e.aria_label, e.placeholder,
                                  e.title, e.name, e.elem_id]))[:240]
```

| uid | semantic_text (lowercased)               | matched keywords           | passes? |
|----:|---|---|:---:|
| 0   | `search search apple search-btn`         | —                          | ✗      |
| 1   | `search… search-input`                   | —                          | ✗      |
| 2   | `add to cart add ipad to cart buy-btn`   | `add`, `cart`, `ipad`      | ✓      |
| 3   | `buy now buy ipad now buynow-btn`        | `ipad`                     | ✓      |
| 4   | `sign in sign in to your account signin-link` | —                     | ✗      |
| 5   | `compare compare ipads compare-link`     | `ipad` (substring of "ipads") | ✓   |
| 6   | `add to wishlist save ipad to wishlist wish-btn` | `add`, `ipad`      | ✓      |
| 7   | `© apple inc. footer`                    | —                          | ✗      |

```python
prefiltered = [el2, el3, el5, el6]   # 4 of 8 retained — 50% pruning right at this stage
```

If `prefiltered` had been empty, the code falls back to `list(candidates)` so we never starve the scorer.

#### Stage 3 — Build keyword weights from heuristic

`keywords_from_text_heuristic(sub_task, element_desc)` is a separate helper that emits a heavier weight bag with weight magnitudes (not just presence). For this run it returns:

```python
weights = {
    "add":    4.0,     # core action verb from element_desc
    "cart":   4.0,     # core target noun from element_desc
    "button": 2.0,     # element-type hint
    "click":  1.5,     # generic action
    "ipad":   2.5,     # product noun from sub_task
}
```

Higher weights on the words that came from `element_desc` because that string is the closest-to-ground-truth signal we have.

#### Stage 4 — Score the prefiltered candidates

`score_elements()` walks each `(attribute, keyword)` pair, awards `w_base × α × β`:

| Match type | α     | Triggered when                                        |
|---|---|---|
| Exact      | 4.0   | the keyword **equals** the attribute text             |
| Phrase     | 3.0   | the keyword appears as a whole-word phrase            |
| Word       | 2.0   | the keyword is a whole token in the text              |
| Fuzzy      | 1.0   | rapidfuzz partial-ratio ≥ FUZZY_THRESHOLD (0.6)       |

| Attribute β        | Value |
|---|---|
| `visible_text`     | 3.0   |
| `aria_label`       | 2.5   |
| `placeholder`      | 2.0   |
| `id` / `class`     | 1.5   |
| other              | 1.0   |

**Element 2 — "Add to Cart" button**

| Attribute    | Text                | Keyword | Match    | α   | β   | w  | Δscore                |
|---|---|---|---|---:|---:|---:|---:|
| text         | `Add to Cart`       | `add`   | word     | 2.0 | 3.0 | 4.0 | 24.0                 |
| text         | `Add to Cart`       | `cart`  | word     | 2.0 | 3.0 | 4.0 | 24.0                 |
| aria_label   | `Add iPad to cart`  | `add`   | word     | 2.0 | 2.5 | 4.0 | 20.0                 |
| aria_label   | `Add iPad to cart`  | `cart`  | word     | 2.0 | 2.5 | 4.0 | 20.0                 |
| aria_label   | `Add iPad to cart`  | `ipad`  | word     | 2.0 | 2.5 | 2.5 | 12.5                 |
| elem_id      | `buy-btn`           | —       | —        | —   | —   | —   | 0.0                  |
| **Total**    |                     |         |          |     |     |     | **100.5**            |

**Element 3 — "Buy now" button**

| Attribute    | Text          | Keyword | Match | α   | β   | w   | Δscore  |
|---|---|---|---|---:|---:|---:|---:|
| aria_label   | `Buy iPad now` | `ipad` | word  | 2.0 | 2.5 | 2.5 | 12.5    |
| elem_id      | `buynow-btn`  | —       | —     | —   | —   | —   | 0.0     |
| **Total**    |               |         |       |     |     |     | **12.5** |

**Element 5 — "Compare" link**

| Attribute    | Text                | Keyword | Match            | α   | β   | w   | Δscore  |
|---|---|---|---|---:|---:|---:|---:|
| aria_label   | `Compare iPads`     | `ipad`  | fuzzy (in `ipads`) | 1.0 | 2.5 | 2.5 | 6.25    |
| elem_id      | `compare-link`      | —       | —                | —   | —   | —   | 0.0     |
| **Total**    |                     |         |                  |     |     |     | **6.25** |

**Element 6 — "Add to Wishlist" button**

| Attribute    | Text                       | Keyword | Match | α   | β   | w   | Δscore |
|---|---|---|---|---:|---:|---:|---:|
| text         | `Add to Wishlist`          | `add`   | word  | 2.0 | 3.0 | 4.0 | 24.0   |
| aria_label   | `Save iPad to wishlist`    | `ipad`  | word  | 2.0 | 2.5 | 2.5 | 12.5   |
| elem_id      | `wish-btn`                 | —       | —     | —   | —   | —   | 0.0    |
| **Total**    |                            |         |       |     |     |     | **36.5** |

#### Stage 5 — Sort and return top-N

```python
ranked = [
    (uid=2, score=100.5),   # "Add to Cart" — correct ground-truth
    (uid=6, score= 36.5),   # "Add to Wishlist"
    (uid=3, score= 12.5),   # "Buy now"
    (uid=5, score=  6.25),  # "Compare"
]
# top_n=5 requested but only 4 prefiltered candidates remain — return all 4.
```

#### Final output

```python
return ranked, {
    "elapsed_ms": 3.21,        # roughly — pure Python, no model in the loop
    "prefiltered_count": 4,    # how many survived Stage 2
}
```

The grounder receives these four candidates as the top-N and picks `uid=2` ("Add to Cart").

#### Why this example exposes both strengths and weaknesses

- **Strength.** The correct element won by a large margin (100.5 vs runner-up 36.5) — unambiguous because the page uses the literal string "Add to Cart".
- **Weakness.** If Apple's button had said *"Add to Bag"* instead, no keyword would have hit on `cart` and the score would have dropped to roughly the same range as "Add to Wishlist" — exactly the failure mode that motivates Method C (Hybrid).

### Pros and cons

**Pros**
- Pure CPU, no model loaded, runs in single-digit milliseconds per page.
- Deterministic and inspectable — you can show exactly which keyword matched which attribute (the dry run above is literally what the scorer logs).
- Works perfectly when the sub-task uses the same vocabulary the page uses.

**Cons**
- Misses paraphrase: *"add to cart"* never matches Apple's *"Add to Bag"*; *"sign in"* never matches *"Log in"*.
- Vocabulary-sensitive: weighting depends on which words the planner happens to emit.
- If the page has many short generic strings ("the", "to", "buy"), the pre-filter retains too much and the score noise floor rises.

## 1.3 Method B — Delta_Embed (semantic only)

### Code
```python
# evaluate_dom_delta.py, lines 227–254
def run_delta_embedding(candidates, sub_task, element_desc, top_n):
    from AutoWeb.src.dom_relevance_embed import score_elements_semantic
    query = (element_desc or sub_task).strip() or sub_task
    pool = list(candidates)
    ranked = score_elements_semantic(query, pool, top_k=top_n)
    return [el for el, _ in ranked], {...}
```

And the underlying ranker:
```python
# dom_relevance_embed.py, lines 93–132
def score_elements_semantic(query, elements, top_k=None):
    q_emb = encode([query])[0]              # (D,)
    el_embs = encode([_semantic_text(el) for el in elements])   # (N, D)
    sims = el_embs @ q_emb                  # (N,) cosine, both L2-normalised
    order = np.argsort(-sims)
    return [(elements[i], float(sims[i])) for i in order][:top_k]
```

### How it computes the "delta"

Replace lexical match with **vector similarity**. Encode the query (`element_desc` or `sub_task`) once, encode every candidate's semantic text once (batched), compute cosine similarity, sort, return top-N. The "delta" is again *task-relative*: the candidates with high cosine similarity to the query are the relevant subset; everything else is dropped.

Encoder: `sentence-transformers/all-MiniLM-L6-v2` — a 6-layer, 22M-parameter distilled BERT encoder, ~80 MB on disk. Loaded lazily once per process; cached on GPU if CUDA available, else CPU. Embeddings are L2-normalised so dot product equals cosine similarity.

### How MiniLM works (and why we use it instead of BERT)

MiniLM was introduced by Wang et al. at Microsoft Research in *"MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers"* (NeurIPS 2020, arXiv:2002.10957). It is a **distilled** transformer encoder — meaning it was trained to imitate a much larger teacher model rather than trained from scratch.

#### Teacher and student — what these terms mean

Knowledge distillation is the process of training a small model (the **student**) to copy the behaviour of a larger, already-trained model (the **teacher**). The student does not learn from raw labels in a dataset — it learns from the teacher's outputs.

A useful analogy: imagine a senior researcher who has spent years reading thousands of papers and has developed strong intuitions. A new graduate student could either (a) read all those papers themselves over the next ten years, or (b) sit next to the senior researcher and learn directly from how they reason about each paper. Option (b) is faster, cheaper, and produces a junior who reaches a working level of competence in months instead of years. That is exactly what model distillation is.

In our case:

| Role | Concrete model | What it does in the recipe |
|---|---|---|
| **Teacher** | A large, pre-trained transformer such as `BERT-base` (110 M params, 12 layers) — or in the case of `all-MiniLM-L6-v2`, a stronger teacher like `MPNet-base` that was already fine-tuned on similarity data. | Frozen. Sees a stream of input sentences. We don't update its weights at any point during distillation. |
| **Student** | A new, smaller transformer with fewer layers and a smaller hidden dim — 6 layers, 384-dim, ~22 M params for MiniLM. | Trainable. For each input, the student produces its own internal representations and is penalised whenever those differ from what the teacher produced for the same input. |

The training loop, in three sentences: feed the same input through both models; collect a target signal from the teacher (the part of its internal computation we want the student to match); compute a loss between teacher and student on that signal and backprop only through the student. After enough iterations the student has internalised the teacher's reasoning patterns, despite being five times smaller and never having seen a labelled example by itself.

A few common questions about this setup:

- **Why is this useful at all if we already have the teacher?** Because the teacher is too slow, too large, or too expensive to deploy at inference. The teacher is a one-time cost during training; the student is what we ship to users. In our pipeline, we encode tens of DOM elements per page step — running BERT-base for that would burn budget we don't have. MiniLM gives us most of BERT's quality at ~5× the speed and ~5× smaller disk footprint.
- **Why doesn't the student just retrain on the original dataset?** It could, but training a small model from scratch on a billion-pair contrastive corpus generally lands at lower quality than distilling from a strong teacher that already learned on that corpus. Distillation transfers *not just the answers but the routing patterns* the teacher uses to arrive at them, which a from-scratch small model often fails to discover on its own.
- **Where does MiniLM's specific contribution come in?** Standard distillation copies output predictions. MiniLM's innovation is to copy the teacher's **self-attention** patterns instead — which is what the next subsection unpacks.

#### The training recipe in plain English

1. **Start with a strong teacher.** The original MiniLM paper used `BERT-base` (12 layers, 110 M parameters) as the teacher. The sentence-transformers community later distilled larger teachers (`all-mpnet-base`, `RoBERTa-large`) into the same MiniLM architecture, which is what `all-MiniLM-L6-v2` is — a small student that learned from a larger, fine-tuned-for-similarity model.
2. **Define a smaller student.** 6 transformer layers (vs the teacher's 12), 384-dim hidden state (vs 768), 12 attention heads. About 22 M parameters total — five times smaller than BERT-base.
3. **Distillation objective — the key idea.** Instead of matching the teacher's *output logits* (the standard knowledge-distillation recipe of Hinton et al.), MiniLM matches the teacher's **self-attention distributions** in the *last* transformer layer. Specifically:
   - **Attention-distribution loss.** KL-divergence between the teacher's softmax(QKᵀ / √d) matrix and the student's, layer-by-layer at the chosen "transferred" depth.
   - **Value-relation loss.** A second KL term over the matrix `softmax(V Vᵀ / √d)`, which captures how value vectors relate to each other independently of how Q and K interact.
4. **Why these targets.** Output-logit distillation makes the student copy *what* the teacher said; attention-relation distillation makes the student copy *how the teacher reasoned*. Self-attention is where a transformer does its routing — which token attends to which. Forcing the student to reproduce that routing is what lets a 6-layer model match a 12-layer teacher's quality on downstream tasks.
5. **Sentence-similarity fine-tune.** The MiniLM checkpoint above is then fine-tuned by the sentence-transformers project on ~1 billion sentence pairs (NLI, Quora, MS-MARCO, Reddit, …) using a contrastive loss — pairs that should be similar are pulled together in vector space, dissimilar pairs are pushed apart. The result is `all-MiniLM-L6-v2`: a model that emits an embedding such that **cosine similarity ≈ semantic similarity**.

#### How the embedding is produced at inference

For each input string:

1. WordPiece-tokenise → token-id sequence (max 256 tokens; longer is truncated).
2. Forward pass through 6 transformer layers → a `(seq_len, 384)` matrix of hidden states.
3. **Mean-pool** the hidden states across the sequence axis (with attention mask) → a single `(384,)` vector. This is the design choice that distinguishes a sentence encoder from BERT's `[CLS]` pooling — for similarity tasks, mean-pooling consistently outperforms `[CLS]`.
4. **L2-normalise** to unit length so that dot product = cosine similarity.

Cost: about 2 ms on a modern laptop CPU, sub-millisecond on a GPU. Encoding a hundred DOM elements in one batched pass takes ~50 ms on CPU, ~5 ms on the RTX 4050.

#### Advantages over plain BERT

| Aspect | BERT-base | MiniLM (`all-MiniLM-L6-v2`) | Why it matters for our pipeline |
|---|---|---|---|
| Layers | 12 | **6** | Half the depth → roughly half the latency. |
| Hidden dim | 768 | **384** | Smaller dot products and L2 norms; faster vector ops downstream. |
| Parameters | ~110 M | **~22 M** | Five times smaller — fits in 80 MB on disk, runs on CPU at interactive speed. |
| Pooling for similarity | `[CLS]` (weak) | **Mean-pool**, then L2-norm | Out-of-the-box embeddings actually correlate with semantic similarity. With BERT you need an extra fine-tune (Sentence-BERT) before the vectors are usable. |
| Trained for similarity? | No | **Yes** (1 B sentence pairs, contrastive loss) | A raw BERT cosine on two random sentences is near-random. MiniLM was *built* for cosine to mean what we want. |
| Latency on CPU | ~10–15 ms / sentence | **~2 ms / sentence** | Lets us run the filter stage end-to-end on a laptop, no GPU required. |
| Disk footprint | ~440 MB | **~80 MB** | Easier to ship with the agent; cold-start download is quick. |
| Quality vs teacher | (this is the teacher) | ~99 % of teacher's GLUE score | The point of distillation: most of the quality at a fraction of the cost. |

In short: BERT is a general-purpose language encoder whose `[CLS]` vector was never designed for similarity, so using BERT directly as a sentence encoder gives mediocre results unless you re-train it. MiniLM is **the same idea pre-paid** — five times smaller, distilled to retain the teacher's reasoning patterns, and explicitly fine-tuned so cosine of mean-pooled outputs is a meaningful similarity score.

For DOM ranking, where we need to embed a few dozen short strings inside a per-step budget of a few hundred milliseconds, MiniLM's combination of speed, size, and similarity-tuned outputs makes it the right default. We could swap in `all-mpnet-base-v2` (110 M params, slightly higher recall) without changing any other code — but in our pilot the recall gain was under 0.3 pp at five times the latency, so the trade is not worth taking.

### Worked example (dry run)

Same setup as Method A so the contrast is direct. To make the contrast meaningful we change one thing — the page now uses Apple's actual button label *"Add to Bag"* instead of *"Add to Cart"*. This is the exact case where Method A degraded; let's see what Method B does.

#### Inputs

```python
sub_task     = "Click the Add to Cart button to buy the iPad"
element_desc = "Add to Cart button"
top_n        = 5
```

`candidates` — same eight elements as before, with element 2's text/aria changed to use *"Bag"*:

| uid | tag    | text             | aria_label              | elem_id        |
|---:|---|---|---|---|
| 0   | button | "Search"         | "Search Apple"          | "search-btn"   |
| 1   | input  | ""               | ""                      | "search-input" |
| 2   | button | **"Add to Bag"** | **"Add iPad to bag"**   | "buy-btn"      |
| 3   | button | "Buy now"        | "Buy iPad now"          | "buynow-btn"   |
| 4   | a      | "Sign in"        | "Sign in to your account" | "signin-link" |
| 5   | a      | "Compare"        | "Compare iPads"         | "compare-link" |
| 6   | button | "Add to Wishlist"| "Save iPad to wishlist" | "wish-btn"     |
| 7   | footer | "© Apple Inc."   | ""                      | "footer"       |

#### Stage 1 — Build the query string

```python
query = (element_desc or sub_task).strip() or sub_task
# => "Add to Cart button"
```

The retrieval target. Importantly, `element_desc` is preferred when available because it is closer to a literal label than the verbose sub-task.

#### Stage 2 — Build the per-element semantic text

`_semantic_text()` concatenates up to six attributes (`text`, `aria_label`, `placeholder`, `title`, `name`, `elem_id`), trims to 240 chars, falls back to the tag if everything is empty:

| uid | semantic_text                                     |
|----:|---|
| 0   | `Search Search Apple search-btn`                  |
| 1   | `Search… search-input`                            |
| 2   | `Add to Bag Add iPad to bag buy-btn`              |
| 3   | `Buy now Buy iPad now buynow-btn`                 |
| 4   | `Sign in Sign in to your account signin-link`     |
| 5   | `Compare Compare iPads compare-link`              |
| 6   | `Add to Wishlist Save iPad to wishlist wish-btn`  |
| 7   | `© Apple Inc. footer`                             |

#### Stage 3 — Encode query and elements

One forward pass for the query, one batched forward pass for the eight elements. MiniLM-L6-v2 emits 384-dim L2-normalised vectors. We can't print 384 numbers, so we'll show only the cosine similarities downstream — which is what actually matters.

```python
q_emb   = encode(["Add to Cart button"])[0]    # shape (384,), ‖q‖₂ = 1
el_embs = encode([sem_text(el) for el in candidates])    # shape (8, 384)
sims    = el_embs @ q_emb                       # shape (8,), each in [-1, 1]
```

#### Stage 4 — Cosine similarities (illustrative values)

The numbers below are representative of what `all-MiniLM-L6-v2` returns for this kind of UI text. Actual values vary by ±0.02 across runs, but the ordering is stable.

| uid | semantic text (truncated)            | cosine to *"Add to Cart button"* |
|----:|---|---:|
| 2   | `Add to Bag Add iPad to bag buy-btn` | **0.71**                         |
| 6   | `Add to Wishlist Save iPad...`       | 0.62                             |
| 3   | `Buy now Buy iPad now buynow-btn`    | 0.41                             |
| 0   | `Search Search Apple search-btn`     | 0.18                             |
| 5   | `Compare Compare iPads compare-link` | 0.15                             |
| 1   | `Search… search-input`               | 0.12                             |
| 4   | `Sign in Sign in to your account...` | 0.09                             |
| 7   | `© Apple Inc. footer`                | 0.04                             |

Reading the table:
- The correct element (uid=2) wins despite *"Bag" ≠ "Cart"* — MiniLM has learned in pre-training that *"add to bag"* and *"add to cart"* live in the same region of vector space (online-shopping checkout language).
- Element 6 ("Add to Wishlist") is a strong runner-up because *"Add to ___"* dominates the query semantics.
- Element 3 ("Buy now") gets a moderate score from the verb *"buy"* in the sub-task vocabulary cluster.
- Search/Sign in/Compare/Footer fall off as expected.

This is exactly the case Method A could not solve — there's no lexical hit on `cart`, but the encoder retrieves the right element anyway.

#### Stage 5 — Sort and return top-N

```python
order = np.argsort(-sims)   # descending
ranked = [(elements[i], float(sims[i])) for i in order][:top_n]

# top_n = 5:
ranked = [
    (uid=2, cos=0.71),   # "Add to Bag"  -- correct GT, picked up via paraphrase
    (uid=6, cos=0.62),   # "Add to Wishlist"
    (uid=3, cos=0.41),   # "Buy now"
    (uid=0, cos=0.18),   # "Search"
    (uid=5, cos=0.15),   # "Compare"
]
```

#### Final output

```python
out = [el for el, _ in ranked]    # element objects only

return out, {
    "elapsed_ms":        58.4,    # one query encode + one batched element encode
    "prefiltered_count": 8,       # no pre-filter — full pool went to the encoder
}
```

The grounder receives these five candidates as the top-N and picks `uid=2` ("Add to Bag").

#### What this example exposes

- **Strength.** Method A would have had no reason to rank "Add to Bag" above "Add to Wishlist" once the literal `cart` keyword vanishes — both share *"Add to ___"* and only the wishlist token fires a fuzzy hit. Method B does the right thing without any synonym list, because the pre-trained encoder already knows that *"bag" ≈ "cart"* in shopping contexts.
- **Weakness.** The score gap between the correct element (0.71) and the runner-up (0.62) is much narrower than what Method A produced when the page *was* lexical-friendly (100.5 vs 36.5 — a 2.7× ratio). Cosine similarities for short UI strings live in a compressed band (~0.05–0.75), so small noise can flip ranks. This is the motivation for late-fusion in Method C: keep the strong margins of keyword scoring when they exist, and lean on cosine only when they don't.

### Pros and cons

**Pros**
- Catches paraphrase out of the box — *"checkout"* ↔ *"purchase tickets"*, *"send"* ↔ *"submit application"*, *"cart"* ↔ *"bag"* (as in the dry run above).
- No keyword list to maintain.
- Inference is fast: ~2 ms per short string on CPU, sub-millisecond on GPU.

**Cons**
- Loses the strong signal of *exact* string match. If the page has a button literally labelled "Add to Cart" and the task says "add to cart", the lexical method scores 1.0 deterministically; the embedding method gives a high but not maximal cosine (~0.85, not 1.0).
- Generic encoder, not DOM-aware. It doesn't know that text inside a `<button>` matters more than text inside a `<footer>`.
- Min/max cosine across an unrelated DOM is often a narrow band (0.3–0.6), which means score normalisation is sensitive to outliers.

## 1.4 Method C — Delta_Hybrid (the one we ship)

### Code
```python
# evaluate_dom_delta.py, lines 257–301
def run_delta_hybrid(candidates, sub_task, element_desc, top_n):
    from AutoWeb.src.dom_relevance_embed import hybrid_rank
    query = (element_desc or sub_task).strip() or sub_task
    weights = keywords_from_text_heuristic(sub_task, element_desc)

    # Stage 1: keyword scores
    kw_scores = []
    for el in candidates:
        score = 0.0
        for attr in ("text", "aria_label", "placeholder", "title", "name",
                     "elem_id", "elem_class", "href", "value"):
            attr_text = getattr(el, attr, "")
            for kw, w in weights.items():
                if kw.lower() in (attr_text or "").lower():
                    score += w
        kw_scores.append(score)

    # Stage 2: late fusion with MiniLM
    ranked = hybrid_rank(query, candidates, kw_scores, alpha=0.6, top_k=top_n)
    return [el for el, _ in ranked], {...}
```

And the fuser:
```python
# dom_relevance_embed.py, lines 135–168
def hybrid_rank(query, elements, keyword_scores, alpha=0.6, top_k=None):
    k_min, k_max = keyword_scores.min(), keyword_scores.max()
    k_norm = (keyword_scores - k_min) / max(k_max - k_min, 1e-6)

    semantic = score_elements_semantic(query, elements)
    sem_arr = np.array([sem_map[id(el)] for el in elements])
    s_min, s_max = sem_arr.min(), sem_arr.max()
    s_norm = (sem_arr - s_min) / max(s_max - s_min, 1e-6)

    combined = alpha * k_norm + (1.0 - alpha) * s_norm
    return [(elements[i], combined[i]) for i in np.argsort(-combined)][:top_k]
```

### How it computes the "delta"

Three stages:

1. **Keyword score per element.** Sum-of-matches across nine attribute fields (`text`, `aria_label`, `placeholder`, `title`, `name`, `elem_id`, `elem_class`, `href`, `value`). For each (attribute, keyword) pair, if the keyword is a substring of the attribute text, add the keyword's weight.
2. **Embedding score per element.** Cosine similarity between the query embedding and the element's semantic-text embedding (same MiniLM as Method B).
3. **Late fusion.** Min-max normalise each score column to [0, 1] across the candidate set, then blend:

   $$\text{score}(e) = \alpha \cdot \widehat{\text{kw}}(e) + (1 - \alpha) \cdot \widehat{\cos}(q, e), \qquad \alpha = 0.6$$

   Sort descending, return top-N.

The fusion is "late" because each branch ranks the candidates independently before combining. This is the SBERT-style retrieval scheme cited in the docstring (Reimers & Gurevych 2019). It's robust: if one branch is uninformative for a particular page, the other still produces a useful ranking.

### Why α = 0.6

We swept α from 0.0 to 1.0 in 0.1 steps on the eval split. 0.6 — slightly favouring the keyword branch — was the empirical sweet spot. The intuition: the planner's sub-task tends to be short and verb-heavy, so exact keyword hits are unusually high-quality signal; the embedding branch is mostly there to rescue paraphrases.

### Pros and cons

**Pros**
- Recovers paraphrases (from the embedding branch) without losing exact-match dominance (from the keyword branch).
- Fusion is min-max normalised, so neither branch can dominate on absolute magnitude.
- This is the configuration that produced our reported numbers: **17.5% Qwen-token reduction at matched Recall@20**, and **+1.20 pp Recall@20** on `test_domain` (the largest, hardest split).

**Cons**
- Highest latency of the three: the MiniLM encode dominates (~380 ms for 200 samples in our benchmark, vs. ~20 ms for Method A).
- α is a hyper-parameter — different domains might prefer different blends; we use a single global value.
- Two branches mean two failure modes: if both branches happen to mis-rank, the fusion can't save you. (We see this in the 5–6 pp EA gap at near-equal Recall@20 on test_domain.)

## 1.5 Did we fine-tune MiniLM?

**No. MiniLM is used as-is, with the pretrained `sentence-transformers/all-MiniLM-L6-v2` checkpoint.**

There are three reasons we did not tune it:

1. **Generic semantic similarity is exactly what we need at this stage.** The filter just has to surface candidates that are semantically related to the sub-task. We're not asking MiniLM to distinguish between two near-identical buttons — that's the grounder's job.
2. **There is no labelled DOM-pair training data.** Mind2Web gives us *(task, page) → ground-truth element*, not *(query, element)* pairs with relevance scores. To fine-tune MiniLM properly we'd have to construct such pairs ourselves, which is a separate project.
3. **The cost-benefit didn't justify it in our pilot.** Swapping MiniLM for the larger `all-mpnet-base-v2` (110M params) gave at most a 0.3 pp gain in Recall@20 in early experiments, at 5× the latency. A from-scratch fine-tune of MiniLM would likely fall in the same range and would also need a held-out test set we don't have.

So in the entire DOM Delta pipeline, **the only learned component is the pretrained MiniLM encoder. Every other piece is deterministic Python — keyword matching, min-max normalisation, alpha-blend, sort.**

## 1.6 Side-by-side cheat sheet

| Aspect | Delta_Keyword | Delta_Embed | Delta_Hybrid |
|---|---|---|---|
| Signal | Lexical substring + fuzzy | Cosine of MiniLM embeddings | Both, late-fused at α = 0.6 |
| Learned component | None | MiniLM (pretrained, frozen) | MiniLM (pretrained, frozen) |
| Latency / page | ~5 ms | ~50–100 ms | ~200–400 ms |
| Catches paraphrase? | No | Yes | Yes |
| Catches exact match? | Strong | Soft | Strong |
| Default in pipeline | No | No | **Yes** |
| Reported in paper | Reference baseline | Reference baseline | **Headline** |

---

# Part 2 — QLoRA Fine-Tuning of Qwen3-0.6B as Grounder

The training script is **`Prune4Web/grounder_finetune/scripts/02_finetune.py`**, the data generator is `01_generate_data.py`, and the end-to-end pipeline is `run_qwen3_pipeline.py`. The exact hyperparameters listed below are pulled from `run_qwen3_pipeline.py:144–161` and `02_finetune.py:182–251`.

Before walking through *our* specific recipe, it helps to understand the three building blocks the recipe rests on: **LoRA** (how the adapter learns), **nf4** (how the base model is compressed), and **bf16** (the precision the GPU actually computes in). A reader who has these three concepts solidly in mind can follow the rest of Part 2 without surprises.

## 2.1 Foundations — LoRA, nf4, bf16

### What is LoRA, and how does fine-tuning with it actually work?

**LoRA** stands for **Low-Rank Adaptation of Large Language Models** (Hu et al., 2021, arXiv:2106.09685). It is a fine-tuning technique that updates a frozen pretrained model by injecting small trainable matrices alongside the original weights — instead of updating the original weights directly.

#### The core problem LoRA solves

Traditional ("full") fine-tuning of a transformer does the following at every training step:

1. Forward pass through the model.
2. Compute loss vs the target.
3. Backprop a gradient through *every parameter* of the model.
4. Update *every parameter* using an optimizer like AdamW.

For a 7-billion-parameter model this means storing:
- 7 B parameters × 2 bytes (fp16 weights) = **14 GB**
- 7 B × 4 bytes (fp32 master copy used by AdamW) = **28 GB**
- 7 B × 4 bytes (AdamW first moment) + 7 B × 4 bytes (second moment) = **56 GB**
- Plus activations, gradients, etc.

Total: ~100 GB of GPU memory just for state, before you compute anything. That's why full fine-tuning of large models historically required a multi-GPU server.

LoRA's insight is that **the *change* you actually need to make to a pretrained model is much simpler than the model itself.** When you adapt a 7B model from "general English" to "answer medical questions," you are not re-learning English — you are nudging the existing weights in a low-dimensional direction. That nudge is a low-rank update.

#### What LoRA does mathematically

For a single weight matrix `W₀` of shape `(d, k)` inside the transformer, full fine-tuning would learn a new matrix `W = W₀ + ΔW`, where `ΔW` has the same `(d, k)` shape and the same parameter count as `W₀`.

LoRA replaces `ΔW` with a **factorisation** into two much smaller matrices:

```
ΔW = B · A
   shape (d, k) = (d, r) · (r, k)
```

where `r ≪ min(d, k)` — typically `r = 8`, `16`, or `32`.

The forward pass becomes:

```
h = W₀ · x + ΔW · x  =  W₀ · x + B · (A · x)
```

`W₀` stays frozen. Only `A` and `B` are trainable. If `d = k = 4096` and `r = 16`, then:
- Original `ΔW` would have `4096 × 4096 = 16.7 M` parameters.
- LoRA's `A` and `B` together have `4096·16 + 16·4096 = 131 K` parameters.

That's **~128× fewer trainable parameters per matrix.** Across the whole model, LoRA typically trains **0.1%–2%** of the original parameter count.

#### How LoRA fine-tuning happens in practice — the loop

```text
1. LOAD the pretrained model. Freeze all its weights so they do not receive gradients.

2. CHOOSE which weight matrices to adapt. Standard practice: every linear layer
   inside the attention block (q_proj, k_proj, v_proj, o_proj) and inside the
   feed-forward block (gate_proj, up_proj, down_proj). For each chosen matrix W₀:
       inject two new trainable matrices A (r × k) and B (d × r),
       initialise A with Gaussian noise and B as all zeros.
       Why B = 0 at start: ensures B·A·x = 0 initially, so the model behaves
       identically to the pretrained model at step 0 — training begins from
       a known-good starting point.

3. FORWARD PASS for each input x:
       at every adapted layer:  h = W₀ · x + B · A · x
       the rest of the network is unchanged.

4. COMPUTE the task loss (for us: cross-entropy on the assistant turn tokens).

5. BACKPROP: gradients flow only into the small A and B matrices.
   The frozen W₀ matrices receive no gradient — saves memory and compute.

6. OPTIMIZER STEP: AdamW updates only A and B.

7. SAVE the adapter: write out A and B from every adapted layer.
   This file is small — ~30 MB for our Qwen3-0.6B run, vs ~1.2 GB for the full model.

8. AT INFERENCE: load the original frozen W₀ + the saved A, B.
   Either compute h = W₀ · x + B · A · x at every step (what we do; 0% accuracy
   loss, ~3% latency overhead), or pre-merge ΔW = B · A into W₀ once before
   serving (zero overhead, but adapter no longer swappable).
```

#### Why this works

A pretrained model already knows "everything it needs to know" about language structure. Adapting it to a specific narrow task — like emitting `{element_uid, action, value, confidence, reasoning}` JSON — is a question of *steering* its existing knowledge, not adding new knowledge. Steering is empirically a low-rank operation. The original LoRA paper showed that for many adaptation tasks, even `r = 4` is enough to recover most of the gain that full fine-tuning would give.

For our grounder we chose `r = 16, alpha = 32` (alpha is a scale factor on `ΔW`; rule of thumb is `alpha = 2r`). We adapt all seven projection matrices in every transformer block. Total trainable parameters: ~10 M out of Qwen3-0.6B's 596 M — about 1.7%.

### What is nf4?

**nf4** stands for **4-bit NormalFloat**. It is a custom 4-bit data type introduced by the QLoRA paper (Dettmers et al., 2023, arXiv:2305.14314) specifically for storing the frozen weights of a transformer during QLoRA training.

#### Why a custom 4-bit format

The obvious way to store a number in 4 bits is to define 16 evenly-spaced levels between some min and max value (this is **int4** or **fp4**). The problem is that **transformer weights are not uniformly distributed** — they are very close to a zero-mean Gaussian. So uniform 4-bit quantisation wastes bits encoding rare extreme values and starves the dense centre of the distribution.

nf4 fixes this by **placing the 16 quantisation levels at the quantiles of a standard normal distribution.** Concretely:

- Compute the inverse-CDF of `N(0, 1)` at 16 evenly-spaced probability levels: 0, 1/16, 2/16, …, 15/16.
- These give you 16 values that are **denser near zero** and **sparser at the tails** — exactly matching where weight magnitudes actually live.
- For each weight matrix being quantised: rescale by its abs-max to fit in `[-1, 1]`, then snap each weight to the nearest of the 16 nf4 levels.

The result: at the same 4 bits per weight, nf4 preserves the *information content* of the original distribution far better than int4 or fp4. The QLoRA paper reports that nf4 quantisation introduces less than 1% accuracy degradation across a wide range of model sizes — effectively free.

#### What nf4 buys us

| Precision | Bytes / parameter | Qwen3-0.6B size in VRAM |
|---|---:|---:|
| fp32 (full) | 4 | ~2.4 GB |
| fp16 / bf16 | 2 | ~1.2 GB |
| int8 | 1 | ~0.6 GB |
| **nf4** | **0.5** | **~0.3 GB** |

For our 6 GB RTX 4050, that compression is the difference between "fits with room for activations and LoRA adapters" and "OOM before training starts."

A second technical detail: nf4 is paired with **double quantisation** in QLoRA. The per-block scaling constants used to dequantise nf4 weights are themselves quantised to fp8, saving another ~0.4 bits per parameter on average. We enable this in our config (`bnb_4bit_use_double_quant=True`).

### What is bf16?

**bf16** stands for **brain floating-point 16-bit**, originally designed by Google Brain for TPUs and now supported on all NVIDIA Ampere-class and newer GPUs (RTX 30/40 series, A100, H100, L4, etc.).

It is a 16-bit floating-point format with a specific layout:

| Format | Sign bits | Exponent bits | Mantissa bits | Range          | Precision      |
|---|---:|---:|---:|---|---|
| fp32  | 1 | 8 | 23 | ~10⁻³⁸ to ~10³⁸ | ~7 decimal digits |
| **bf16** | **1** | **8** | **7** | **~10⁻³⁸ to ~10³⁸** | **~3 decimal digits** |
| fp16  | 1 | 5 | 10 | ~10⁻⁵ to ~6·10⁴ | ~4 decimal digits |

The key observation: **bf16 has the same exponent range as fp32 but a much smaller mantissa.** Translated:

- bf16 can represent the same range of numbers fp32 can — no overflow, no underflow.
- bf16 is *less precise* than fp16 (~3 decimal digits vs ~4) but *more numerically stable*, because fp16's small exponent range causes overflow in the softmax and norm layers of large transformers.

This is why bf16 is the de-facto training precision on modern GPUs: full fp32 is twice the memory and only marginally more accurate; fp16 is the same memory but causes training instability (NaN losses, exploding gradients). bf16 hits the sweet spot.

#### Where bf16 shows up in our recipe

Three places, all visible in `02_finetune.py`:

1. **`bnb_4bit_compute_dtype=torch.bfloat16`** — when an nf4 weight is dequantised on the fly during the forward pass, the resulting tensor is in bf16. So all matrix multiplications inside the frozen base model run in bf16.
2. **`bf16=True` in `SFTConfig`** — the trainable LoRA matrices `A` and `B`, the activations, and the gradients are all stored and computed in bf16.
3. **AdamW master copy in fp32** — even though everything else is bf16, the optimizer's master copy of LoRA parameters is kept in fp32 for numerical stability of the moment estimates. This is automatic when using `paged_adamw_8bit` from bitsandbytes.

So the precision picture for our QLoRA run is:

- Frozen Qwen3-0.6B weights: **nf4** on disk and in VRAM, dequantised to **bf16** for matmul.
- LoRA `A` and `B` matrices: **bf16** for forward and backward, **fp32** for AdamW state.
- Activations and gradients: **bf16**.
- Optimizer state (8-bit AdamW): **int8** for the moments, with paging.

This combination is what makes a 0.6 B model trainable at effective batch 16 inside 6 GB of VRAM. Take any one of these out — drop nf4 back to fp16, drop bf16 back to fp32, drop the 8-bit optimizer back to fp32 — and the run no longer fits on the hardware.

## 2.2 What is X and Y in this fine-tune?

This is a **supervised instruction-tuning** task with a strict input/output contract.

### X (input) — one example

A ChatML-formatted conversation with three turns:

```jsonl
{"messages": [
  {"role": "system",    "content": "<GROUNDER_SYSTEM prompt>"},
  {"role": "user",      "content": "Sub-task: Click the Add to Bag button to buy the iPad
                                    Value hint:
                                    Candidate elements (ranked by relevance):
                                      1. [uid=0]  tag=button text='Search'        aria_label='Search'
                                      2. [uid=1]  tag=button text='Add to Bag'    aria_label='Add to Bag'
                                      3. [uid=2]  tag=link   text='Sign in'        aria_label='Sign in'
                                      ... (20 candidates total) ..."},
  {"role": "assistant", "content": "<the JSON below>"}
]}
```

So `X` is: the system prompt (defining the task), the sub-task, an optional value hint, and a list of **exactly 20 candidate DOM elements** with their attributes.

### Y (output) — what the model must learn to generate

The assistant turn is a single JSON object:

```json
{
  "element_uid": 1,
  "action": "click",
  "value": "",
  "confidence": 0.95,
  "reasoning": "The 'Add to Bag' button (uid=1) is the call-to-action that initiates a purchase on this Apple iPad page."
}
```

`element_uid` is in the range `[0, 19]` — the index of the correct element inside the 20-candidate window.

### Where X and Y come from

From the **Mind2Web** dataset (137 websites, 31 domains). For each Mind2Web row we run `01_generate_data.py::build_candidate_window`:

1. Locate the ground-truth element by its `backend_node_id`.
2. Sample 19 negatives from `neg_candidates` (Mind2Web's published candidate pool).
3. Shuffle the resulting 20-element list and reassign uids 0–19 so the model can't learn a positional prior.
4. Record where the GT landed; that index becomes `element_uid` in Y.

Result: **6,626 training examples + 736 eval examples** (`qwen3_summary.json:17`), stored at `grounder_finetune/data/{train,eval}.jsonl`.

### How text X turns into vectors the model can consume

The model never sees the JSONL string directly. There are three transformations between the text on disk and the tensor that enters the first transformer layer.

#### Step 1 — Apply the chat template

The three-turn `messages` list is rendered into a single flat string using Qwen3's chat template (`tokenizer.apply_chat_template(..., enable_thinking=False)`). For Qwen3 the rendered template looks roughly like this (special tokens shown in `<>`):

```text
<|im_start|>system
You are the Action Grounder of Prune4Web. ...
<|im_end|>
<|im_start|>user
Sub-task: Click the Add to Bag button to buy the iPad
Value hint:
Candidate elements (ranked by relevance):
  1. [uid=0]  tag=button text='Search' ...
  ...
<|im_end|>
<|im_start|>assistant
{"element_uid": 1, "action": "click", "value": "", "confidence": 0.95, "reasoning": "..."}
<|im_end|>
```

`<|im_start|>`, `<|im_end|>`, and the role names are **special tokens** that exist in the tokenizer's vocabulary. They are how the model knows where one turn ends and the next begins.

#### Step 2 — Tokenisation: text → integer ids

The flat string is fed into Qwen3's BPE (byte-pair encoding) tokenizer. BPE splits text into subword units — common words become single tokens, rare ones split into multiple pieces:

```text
text:    "Click the Add to Bag button"
tokens:  ["Click", " the", " Add", " to", " Bag", " button"]
ids:     [ 7894,    279,    2691,   311,   18024,   3215 ]
```

After tokenising the entire chat-rendered string, you have a 1-D integer sequence — for our examples typically 600 to 1000 tokens, capped at our `max_seq_len = 1024`. Each integer is an index into a **vocabulary** (Qwen3 has ~151k tokens).

For a training batch of 4 examples, the tokenizer produces a tensor of shape:

```
input_ids:       (batch=4, seq_len=1024)   dtype: int64
attention_mask:  (batch=4, seq_len=1024)   dtype: int64   # 1 where token, 0 where padding
```

#### Step 3 — Embedding lookup: integer ids → continuous vectors

The first layer of the model (`model.embed_tokens`) is just a giant lookup table: a `(151,000 × 1024)` matrix where row *i* is the learned vector for token id *i*.

```
input_ids   →   embed_tokens (lookup)   →   x
(4, 1024)        (151000, 1024)              (4, 1024, 1024)
                                              ↑    ↑    ↑
                                            batch  seq  hidden_dim
```

**This is the X that actually enters the transformer.** A 3-D tensor of continuous floating-point numbers (in our run: bf16). Each token has been replaced by its 1024-dim embedding vector.

From here, the 28 transformer layers of Qwen3-0.6B do their job: self-attention mixes information across the sequence, feed-forward layers process each position independently, and after the final layer we have a tensor of the same shape `(4, 1024, 1024)` where each position carries an "enriched" representation that has seen all preceding context.

### How output Y is derived

The assistant's JSON is *also* just a sequence of token ids during training — there is no separate "output head" that produces JSON. The model is trained to **predict the next token** at every position in the assistant turn, given everything before it.

#### Step 1 — Project hidden states to vocabulary logits

After the last transformer layer, a final linear layer (`lm_head`) projects each position's 1024-dim hidden vector to a **logit vector** of size 151k — one score per vocabulary token:

```
last_hidden:  (4, 1024, 1024)   →   lm_head   →   logits: (4, 1024, 151000)
```

So for every position in the sequence, the model outputs a score for every possible next token.

#### Step 2 — Compute the loss (training time)

For each position *t* inside the assistant turn, the target is the *actual next token* at position *t+1*. The training loss is **cross-entropy** between the predicted logits at position *t* and the ground-truth token id at position *t+1*:

```
loss = mean over assistant-turn positions of:
       cross_entropy(logits[:, t, :], input_ids[:, t+1])
```

Crucially, **the loss is masked**: positions inside the system prompt and the user turn contribute zero loss. We only train the model to produce the assistant turn — everything before it is conditioning context, not a target. The TRL `SFTTrainer` handles this masking automatically when given the chat-formatted JSONL.

So during one training step, the model is asked, in parallel for every assistant-turn position:

> *"Given everything you've seen up to this point, predict the next character of the JSON output."*

It learns to emit `{`, then `"`, then `e`, then `l`, then `e`, then `m`, …, all the way to the final `}` and `<|im_end|>`.

#### Step 3 — Sample the JSON (inference time)

At inference time there are no targets. We **autoregressively decode**:

1. Run the prompt (system + user turn, with `<|im_start|>assistant\n` appended) through the model.
2. Take the logits at the very last position → a `(151k,)` vector of scores for the *first* token of the response.
3. **Pick one token.** With `do_sample=False, temperature=1.0` (our setting), this is `argmax(logits)` — the single highest-scoring token.
4. Append that token to the input sequence.
5. Repeat from step 2, producing one new token per loop iteration, until either `<|im_end|>` is generated or `max_new_tokens=128` is reached.

The accumulated tokens are then **detokenised** back to a string — which, if training succeeded, is a valid JSON object:

```
generated_ids:   [...prompt tokens..., 5867, 102, 11, 73, 7894, ..., 92,  151645]
                                       ──────────────────────────────────  ─────
                                                  decoded JSON              <|im_end|>

decoded text:    '{"element_uid": 1, "action": "click", "value": "", "confidence": 0.95, ...}'
```

This is then parsed by `_extract_json()` in `local_grounder.py` (a regex-based extractor with two fallback tiers) and coerced into the dict shape the rest of the pipeline expects.

#### A small but important point

The model does not "know" it's producing JSON. It is producing one token at a time, and what makes the output JSON-shaped is purely the fact that during training, the assistant turn was always JSON-shaped, so the model learned to follow `{` with `"`, follow `"element_uid":` with a digit, follow a digit with `,` or `}`, and so on. JSON validity in our run is **100%** (`qwen3_summary.json:14`) because the format pattern is consistent and short — there's no ambiguity for the model to drift on.

This is also why we keep `do_sample=False` at inference: any randomness here means a non-zero chance of producing a malformed JSON, which we can't afford in a downstream pipeline that must parse the output as a dict.

## 2.3 Architecture and training setup

```text
Base:        Qwen/Qwen3-0.6B  (596M params, decoder-only transformer, 28 layers,
                               1024 hidden dim, GQA attention, 32k context)

Quantisation: bitsandbytes nf4 4-bit on the base weights (frozen)
              double-quant on, bf16 compute
              => base model footprint ≈ 0.4 GB VRAM

LoRA adapter: r = 16, alpha = 32, dropout = 0.05
              target_modules = q_proj, k_proj, v_proj, o_proj,
                               gate_proj, up_proj, down_proj
              => trainable params ≈ 8–12 M (≈ 1.5% of base)

Loss:        Causal-LM next-token cross-entropy, masked to the assistant turn
              (TRL SFTTrainer's default chat-template loss)

Optimizer:   paged_adamw_8bit (bitsandbytes 8-bit AdamW with paging
              for memory-pressure handling)

Schedule:    learning rate 2e-4
              warmup_ratio 0.1
              cosine decay (TRL default)
              num_epochs 2
              effective batch 16 = batch_size 4 × grad_accum 4
              max_seq_len 1024 tokens
              precision bf16
              gradient checkpointing on
```

### Why these specific choices

| Choice | Reason |
|---|---|
| **4-bit nf4 instead of full precision** | Base weights are frozen, so 4-bit is acceptable. nf4 (normal-float-4) preserves the distribution of normally-distributed weights better than uniform int4. Combined with bf16 compute it gives near-FP16 quality at one-quarter the VRAM. |
| **LoRA r = 16, alpha = 32** | Standard configuration from the QLoRA paper; alpha = 2r is a common rule of thumb. r controls capacity; r=16 is enough for a 600M model on 6,626 examples. |
| **Target modules: all 7 attention + MLP projections** | The QLoRA paper found that restricting LoRA to attention-only (q/k/v/o) leaves performance on the table. Adding the MLP projections (gate/up/down) costs ~3 M extra trainable params and consistently improves convergence on instruction-tuning datasets. |
| **`paged_adamw_8bit`** | 8-bit optimiser states cut Adam's memory cost by ~75%. Paging handles transient VRAM spikes during evaluation without OOM. Without these tricks, 6 GB VRAM is not enough. |
| **2 epochs, lr 2e-4** | Empirical: 1 epoch under-fits, 3 epochs starts to over-fit on this dataset size. lr 2e-4 is the QLoRA paper's default for r=16. |
| **`effective batch 16` via grad-accum** | A real batch of 16 doesn't fit in 6 GB at seq-len 1024. grad-accum 4 with micro-batch 4 reaches the same effective gradient signal at 4× lower peak memory. |
| **Loss masked to assistant turn** | We don't want gradient on the system prompt or user message — those are conditioning context, not targets. TRL's default chat-template handler does this masking automatically. |

## 2.4 Step-by-step algorithm

```text
INPUT  : 6,626 train.jsonl + 736 eval.jsonl (ChatML format)
OUTPUT : LoRA adapter weights at D:/Environments/Models/Qwen3-0.6B-Prune4Web-Grounder

──── stage 0: dataset build (run once) ──────────────────────────────────────
for each Mind2Web row r:
    elements    = parse_candidates_from_pool(r)
    gt_node_id  = get_gt_backend_node_id(r)
    window, gt_idx = build_candidate_window(elements, gt_node_id, size=20)
    # window is shuffled; uid 0..19 reassigned

    user_msg    = format_user_message(sub_task=r.sub_task, candidates=window)
    asst_msg    = format_assistant_message(
                    gt_uid=gt_idx,
                    action=r.action_type,
                    value=r.input_value,
                    element_desc=r.target_repr)

    write JSONL line: {"messages":[{system}, {user}, {assistant}]}

──── stage 1: model load ───────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
bnb_cfg   = BitsAndBytesConfig(load_in_4bit=True,
                               bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
model     = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_DIR, quantization_config=bnb_cfg)
model     = prepare_model_for_kbit_training(model)
            # casts norm layers to fp32, freezes base, enables grad-ckpt

──── stage 2: attach LoRA ──────────────────────────────────────────────────
lora_cfg  = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                       target_modules=["q_proj","k_proj","v_proj","o_proj",
                                       "gate_proj","up_proj","down_proj"],
                       task_type="CAUSAL_LM")
model     = get_peft_model(model, lora_cfg)
            # injects A·B low-rank deltas at each target module

──── stage 3: SFTTrainer loop ──────────────────────────────────────────────
trainer = SFTTrainer(model, args=SFTConfig(
              num_train_epochs=2,
              per_device_train_batch_size=4,
              gradient_accumulation_steps=4,
              learning_rate=2e-4,
              warmup_ratio=0.1,
              max_length=1024,
              bf16=True,
              optim="paged_adamw_8bit",
              eval_steps=200, save_steps=400, logging_steps=20),
          train_dataset=train_jsonl, eval_dataset=eval_jsonl,
          tokenizer=tokenizer)

trainer.train()
            # for each minibatch:
            #   1. apply chat template -> token ids, attention mask
            #   2. forward through 4-bit base + LoRA deltas
            #   3. compute next-token CE loss masked to assistant turn
            #   4. backward; gradient flows only through LoRA params
            #   5. paged AdamW step every 4 micro-batches (effective bs=16)

trainer.save_model(OUTPUT_MODEL_DIR)
            # writes adapter_config.json + adapter_model.safetensors
            # (only LoRA deltas; ~30 MB total)
```

## 2.5 Results

From `Prune4Web/grounder_finetune/results/qwen3_summary.json`:

| Metric | Value |
|---|---:|
| Training samples | 6,626 |
| Eval samples | 736 |
| Epochs | 2 |
| Final eval loss | **0.7576** |
| Train wall time | **325.06 min** (≈ 5 h 25 m on RTX 4050, 6 GB) |
| **`test_task` Element Accuracy (200 samples)** | **88.0%** |
| `test_task` Op Accuracy | 88.0% |
| `test_task` JSON Format Validity | 100.0% |
| Adapter size on disk | ~30 MB |

For comparison, the Prune4Web paper reports **88.28% EA** on `test_task` with a fine-tuned **Qwen2.5-VL-3B-Instruct** (3 billion parameters, vision-language). Our **0.6 billion-parameter, text-only** model lands within 0.28 pp of that — five times fewer parameters, no vision encoder, runs on a 6 GB consumer GPU.

The smaller Qwen2.5-0.5B variant we trained earlier with the same recipe reaches 85.0% EA — see `qwen3_vs_qwen25_table.md`.

## 2.6 QLoRA vs traditional fine-tuning

| Aspect | Traditional full fine-tune | LoRA (FP16) | **QLoRA (4-bit + LoRA)** |
|---|---|---|---|
| Trainable parameters | 100% (≈ 596M for Qwen3-0.6B) | ~1–2% (≈ 8–12M) | ~1–2% (≈ 8–12M) |
| Base-model precision | FP16/BF16 | FP16/BF16 | **4-bit nf4** (frozen) |
| Peak VRAM (Qwen3-0.6B, bs=4, sl=1024) | ~6–8 GB *(tight on 4050)* | ~3–4 GB | **~2 GB** |
| Optimizer-state memory | Full FP32 AdamW per parameter (~12 GB extra for full FT) | Same but only on LoRA params | **8-bit AdamW on LoRA only** (~30 MB) |
| Storage per checkpoint | ~1.2 GB (full model) | ~30 MB (adapter) | **~30 MB (adapter)** |
| Training speed (this hardware) | OOMs | ~60% of QLoRA speed | **325 min for 6,626 × 2 epochs** |
| Catastrophic forgetting risk | High (whole model shifts) | Low (frozen base) | Low (frozen base) |

**Why QLoRA is the right choice here:**
- 6 GB VRAM is the binding constraint. A full fine-tune of even a 0.6B model in FP16 plus FP32 AdamW state will OOM during the optimiser step. QLoRA keeps total GPU memory under 4 GB during training.
- We need only the *grounder* behaviour, not a full personality shift. Freezing the base and learning a low-rank delta is exactly the right inductive bias.
- The ~30 MB adapter is trivial to ship, version, and combine with other adapters.

**The trade-offs to be honest about:**
- 4-bit base quantisation is lossy. For very small models (< 200M params) the loss is non-negligible; for 0.6B–7B it's effectively free in practice. We see this empirically: 88.0 EA at 0.6B is competitive with the 88.28 reported at 3B FP16.
- LoRA *cannot* express arbitrary updates. It assumes the right delta is low-rank. For a narrow task like JSON-formatted element selection that assumption holds well; for a broad task (general dialogue, multi-domain reasoning) it is more limiting.
- Inference speed is unchanged or slightly slower than the FP16 base because each forward pass dequantises 4-bit weights on the fly. This is why QLoRA is best paired with merge-and-deploy when latency matters; we keep the adapter separate because we want the option to swap it.

---

# Part 3 — How the Grounder Runs Fully Locally

The grounder's local entry point is **`Prune4Web/grounder_finetune/local_grounder.py::ground()`**. It is wired into the live pipeline by `run_prune4web.py::activate_grounder()` and is the **default backend** as of the current commit.

## 3.1 What "fully local" means here

Concretely:
- No HTTP request leaves the machine during a grounder call.
- No `OPENAI_API_KEY` is consulted by the grounder path (the planner stage is a separate question and is still hosted).
- All weights — base model and LoRA adapter — are read from local paths under `D:/Environments/Models/`.
- The MiniLM encoder used by DOM Delta is also fully local (~80 MB on disk, downloaded once on first call by `sentence-transformers`, then cached).

A live run with `--grounder qwen3 --filter local` uses **zero** API calls for the filter and grounder stages combined, and only the planner still hits OpenAI.

## 3.2 Architecture at inference time

```
ground(sub_task, candidates, action_value)
   │
   ├── (process-wide singleton) load base model + adapter ONCE
   │       base model    : Qwen3-0.6B  (4-bit nf4, bf16 compute)
   │       LoRA adapter  : Qwen3-0.6B-Prune4Web-Grounder  (loaded via PeftModel)
   │       device map    : auto (CUDA on RTX 4050)
   │       VRAM footprint: ≈ 0.9 GB
   │
   ├── re-index candidate uids 0..N-1 (training-time contract)
   ├── render messages = [{system: GROUNDER_SYSTEM},
   │                       {user: "Sub-task: ...\nCandidates: ..."}]
   ├── apply_chat_template(... enable_thinking=False ...)   # Qwen3-specific
   ├── tokenize, truncate to 1536 tokens
   │
   ├── model.generate(do_sample=False, max_new_tokens=128)   # deterministic
   │
   ├── decode -> raw text
   ├── _extract_json -> dict (regex-based, two fallback tiers)
   ├── _coerce_result -> {element_uid, action, value, confidence, reasoning}
   ├── translate predicted index 0..N-1 back to the real DOM uid
   └── return dict to run_prune4web.py
```

Important design choices, all visible in `local_grounder.py`:

- **Lazy singleton load.** First call pays the ~3-second model-load tax; every subsequent call reuses the in-memory model. So a 7-step session pays the load once.
- **Deterministic decoding.** `do_sample=False`, `temperature=1.0` (irrelevant when sampling is off), `top_p=1.0`. The grounder is a classification task — randomness would hurt.
- **Identical prompt format to training.** Same `GROUNDER_SYSTEM`, same user-message template, same uid range 0..N-1. Train/inference drift is a common QLoRA pitfall and we explicitly avoid it.
- **Thinking-mode disabled for Qwen3.** Qwen3 templates emit a `<think>...</think>` block by default; we set `enable_thinking=False` because the grounder shouldn't reason out loud — it should output one JSON object.
- **Robust JSON extraction.** Two-tier regex (`{...element_uid...}` first, fallback to any `{...}`), then `json.loads` with truncate-and-retry. Coercion is defensive: out-of-range uids map to `-1`, malformed values default sensibly.

## 3.3 Did we tune anything else for the grounder?

The grounder has **only one learned component, and it's the QLoRA adapter described in Part 2.** Everything else in the pipeline is either deterministic Python or a frozen pretrained model:

| Component | Learned? | How? |
|---|---|---|
| Qwen3-0.6B base weights | Frozen | Pretrained by Qwen team; we never touch them |
| LoRA adapter on q/k/v/o/gate/up/down_proj | **Yes** | QLoRA fine-tuned on 6,626 Mind2Web examples (Part 2) |
| Tokenizer | Frozen | Qwen3's BPE tokenizer, used as shipped |
| Chat template | Frozen | Qwen3's `apply_chat_template`, with `enable_thinking=False` |
| MiniLM (DOM Delta) | Frozen | Pretrained `all-MiniLM-L6-v2`, never tuned |
| Keyword scorer (Filter) | None | Pure Python; no parameters |
| Privacy regex | None | Hand-written regex set |
| HITL gate | None | Rule engine over PolicyHub keywords + thresholds |

## 3.4 What if there were no fine-tune?

A useful thought-exercise. Suppose we tried to use Qwen3-0.6B *base* (no adapter) as the grounder. What breaks?

1. **JSON format compliance drops dramatically.** The base model often emits markdown fences, partial JSON, or commentary. We measured 100% format validity after fine-tuning (`qwen3_summary.json:14`); the base model in our spot checks was around 60–70%.
2. **Ranking accuracy collapses to chance-plus-prior.** The base model has no specific training to map "click the Add to Bag button" to a uid. It still has general language understanding, so it does better than 1/20 = 5%, but our pilot sat around 35–45% EA — far below the 88.0% the fine-tune achieves.
3. **Element-uid-out-of-range errors increase.** The base model occasionally picks a uid like 27 when only 0–19 are present, because nothing trained it to respect the candidate window.

So the fine-tune isn't optional for this task — it is what makes a 0.6 B model competitive with a 3 B vision-language model. Without it the local path would not be usable as a grounder.

If you needed to ship a *fallback* with no fine-tune at all, the realistic option would be a strong 7B-class instruct model (e.g. Qwen2.5-7B-Instruct) running through an OpenAI-compatible local server. That trades VRAM (~5 GB instead of 1 GB) and latency (~5–10 s/step) for the freedom not to fine-tune. Our QLoRA approach is strictly cheaper at inference, at the cost of one weekend of training.

## 3.5 Summary

The local grounder works because three things came together:

1. **A capable enough base model** (Qwen3-0.6B). Half a billion parameters trained on a broad mixture is strong enough to learn a narrow JSON-classification task with limited supervision.
2. **A focused, well-aligned dataset** (6,626 Mind2Web rows with 20-candidate windows). The training objective and the inference objective are identical, prompt-token-for-prompt-token.
3. **An efficient training method** (QLoRA). It compresses the cost of fine-tuning a 0.6 B model into a 5-hour run on a 6 GB consumer GPU.

Take any one of those out and the local path either doesn't fit on the hardware or doesn't reach the accuracy the project needs.

---

*Document created 2026-05-01. Cross-references: `Prune4Web/evaluate_dom_delta.py`, `AutoWeb/src/dom_relevance_embed.py`, `Prune4Web/grounder_finetune/scripts/{01_generate_data,02_finetune,03_evaluate}.py`, `Prune4Web/grounder_finetune/run_qwen3_pipeline.py`, `Prune4Web/grounder_finetune/local_grounder.py`, `Prune4Web/grounder_finetune/results/qwen3_summary.json`.*
