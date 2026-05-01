# Presentation Speech — Major Project II Defense

**Title:** Improving Web Automation with DOM Delta Processing and Privacy-Aware LLM Agents
**Speaker:** Darji Dhruv S. (24MCD003)
**Mentor:** Dr. Parita Oza
**Target duration:** ~15–18 minutes for the slides + 5 minutes Q&A buffer.

---

## How to use this document

Each slide gets three things:
1. **Speech** — what to actually say, in spoken English. Read it aloud once and tighten the parts that don't sit right in your voice; don't memorise word-for-word.
2. **How to deliver** — pacing, where to point, what to gesture toward, what to skip if you're running short.
3. **Anticipated questions** (only on the slides where the panel is most likely to dig in).

Two general rules for the whole talk:
- **Lead each slide with the headline number, not the setup.** Examiners will give you about ten seconds before they decide whether the slide is worth listening to.
- **Be honest about scope.** When something is implemented and measured, say "we measured." When it's qualitative, say so. The panel notices the difference and respects the second more than over-claiming.

---

## Slide 1 — Title

### Speech
"Good morning. My project is titled *Improving Web Automation with DOM Delta Processing and Privacy-Aware LLM Agents*. The work is supervised by Dr. Parita Oza and is being submitted as Major Project Two for the M.Tech in Data Science programme. The talk covers what the problem is, what existing systems do well and where they fall short, what we built on top of them, and what the experiments say."

### How to deliver
- Spend no more than fifteen seconds here. State your name, the title, and your guide. Move on.
- Make eye contact with both examiners before you click to the next slide.

---

## Slide 2 — Contents

### Speech
"The flow is standard. I'll start with the problem and the literature; then I'll spend most of the time on the two contributions — DOM Delta Processing and the privacy plus human-in-the-loop layer; and I'll close with the experiments on the Mind2Web benchmark and the limitations we still have."

### How to deliver
- Don't read every bullet aloud. Group them: *"Sections 1–5 set up the problem. 6–9 are the implementation. 10 is results. 11 and 12 are limitations and what's next."*
- Maximum 30 seconds.

---

## Slide 3 — Introduction

### Speech
"Web agents — systems that read a webpage and decide which button to click on behalf of a user — are one of the more practical applications of large language models today. The challenge is that a real webpage carries between ten thousand and one hundred thousand DOM tokens, which is more than most LLMs can hold in context, and almost all of it is irrelevant to any single user task. So the engineering problem is one of *selection*: how do you pick the few elements that matter, do it cheaply, and do it without leaking the user's data."

### How to deliver
- One minute, no more.
- If the introduction slide is the one you "kept as-is" from the previous deck, anchor your speech in **the cost numbers** (10⁴–10⁵ tokens) rather than re-reading whatever bullets are on the slide.

---

## Slide 4 — Problem Statement

### Speech
"There are four concrete problems we wanted to address. First, today's agents process the *entire* DOM at every step. Even when the user clicks a button and only a popup appears, the agent re-reads the whole page from scratch — that's pure waste. Second, the strongest grounder models in the literature are three-billion-parameter vision-language models like Qwen2.5VL-3B, or hosted models like GPT-4V. Sub-one-billion-parameter open-weight grounders barely show up in the literature except as ablation footnotes. Third, privacy and human review are usually added on after the fact — the system is built first, then a filter is bolted on. And fourth, putting all of that together: a localised, token-efficient, privacy-aware web agent that runs on a consumer GPU is still missing from the published work."

### How to deliver
- Read the four problems with **a beat of silence between each** so the panel registers them as four separate things.
- Point at the slide for the second problem (the parameter-count one) — that one is the easiest sell to the examiner because it directly motivates the local Qwen grounder.

### Anticipated question
**"Why is full-DOM processing actually a problem if context windows are growing?"**
*Answer:* Cost and latency. Token count multiplies the API bill linearly and the model's wall-clock latency super-linearly. Even with infinite context, a 100k-token prompt is still 100k tokens that you pay for and wait for. And if the page barely changed, you paid for the same information twice.

---

## Slides 5–7 — Literature Review

### Speech
"The literature splits cleanly into three groups. The multimodal-agent line — SeeAct, AdaptAgent, WebExperT — uses screenshots plus DOM and consistently identifies grounding as the bottleneck. The structural-pruning line — Prune4Web, AutoWebGLM, Agent-E, WebAgent — tries to reduce the DOM before sending it to an LLM. And the state-change line — D2Snap, Agentic Compilation — tries to compress consecutive snapshots, but always processes one snapshot at a time. Two pieces of supporting work matter for our implementation: MiniLM, a 22-million-parameter encoder we use for the embedding side of the filter, and QLoRA, which is what makes it possible to fine-tune a half-billion-parameter model on a six-gigabyte consumer GPU. The single most important paper for us is Prune4Web — it's the system we re-implemented and built on top of."

### How to deliver
- The literature table has eight rows. **Don't read the table.** Group them as I just did and let the panel scan the rows themselves.
- Point at Prune4Web specifically — circle it with your laser pointer or finger — and say *"this is the base paper."*
- Spend forty-five seconds maximum on this slide; the panel almost never asks deep questions on the lit review.

### Anticipated question
**"Why didn't you use SeeAct as the base paper, since it's the more famous one?"**
*Answer:* Two reasons. SeeAct depends on GPT-4V which is a paid API, and we could not reproduce it locally on the budget we had. More importantly, Prune4Web reports a stronger, more recent number on Mind2Web (88.28% EA) and has a more modular pipeline — Planner, Filter, Grounder — that we could attach our delta layer to without rewriting. SeeAct is still cited in the literature review.

---

## Slide 8 — Research Gaps

### Speech
"From that survey, three gaps stand out and they map one-to-one to the contributions of this work. First: every existing pruning system reprocesses the full DOM at every step. There's no structural-UID-based delta layer plugged into a real filter. Second: sub-one-billion-parameter grounders are under-studied. The strong baselines all sit at three billion parameters or above. Third: privacy and human-in-the-loop are bolted on, not architected in. None of the published pipelines puts a privacy directive into the grounder's system prompt or runs a regex pass over candidate elements before grounding."

### How to deliver
- This slide is doing the heavy lifting for the rest of the talk. **Slow down here.** Each gap takes about fifteen seconds.
- Use your fingers — count out one, two, three. The panel is looking for a clear gap-to-contribution mapping.

---

## Slide 9 — Objectives

### Speech
"The five objectives are derived directly from the gaps. We reproduce Prune4Web on a single six-gigabyte GPU. We add a DOM Delta layer at the filter stage that uses structural UIDs and a hybrid keyword-plus-MiniLM score. We replace the three-billion-parameter vision-language grounder with a QLoRA-fine-tuned half-billion-parameter text-only model. We add a Privacy Hub and a risk-based human-in-the-loop gate around the planner. And we evaluate the system on all three Mind2Web test splits."

### How to deliver
- Read each objective at half speed — these are what the panel will judge you against.
- Pause on the third objective ("text-only grounder") because it's a non-trivial design choice you'll have to defend later.

### Anticipated question
**"Why text-only, when the literature uses vision-language?"**
*Answer:* The DOM already carries the semantic information that the screenshot would carry — element text, ARIA labels, role attributes, surrounding context. A vision encoder roughly triples the parameter count without giving us new information that isn't already in the DOM. We confirmed this empirically: our text-only Qwen3-0.6B reaches 88.00% EA, within rounding distance of the 88.28% reported by the 3B vision-language baseline.

---

## Slide 10 — Prune4Web: Base Paper

### Speech
"The base paper is *Prune4Web: DOM Tree Pruning Programming for Web Agents*, by Zhang and colleagues, published in late 2025. It introduces a three-stage pipeline. The Planner converts the user's high-level task into a low-level sub-task. The Filter is the interesting piece — instead of asking an LLM to read the DOM directly, it asks the LLM to *write a small Python keyword-weight script*, and then a deterministic Python scorer applies that script to the DOM and ranks the elements. The Grounder, a fine-tuned three-billion-parameter Qwen2.5VL, picks the final element from the top-N. They report 88.28% Element Accuracy on Mind2Web's `test_task` split and a 25-to-50× reduction in DOM tokens at the filter stage."

### How to deliver
- The most important sentence is the one about the Filter. **Slow down on it** — many panels won't have read the paper, and that detail is what makes Prune4Web architecturally different from SeeAct or AutoWebGLM.
- About one minute on this slide.

### Anticipated question
**"What does the Python scorer actually do with the keywords?"**
*Answer:* For each DOM element, the scorer combines two factors: an alpha factor that scores how the keyword matched (exact, phrase, word, or fuzzy), and a beta factor that scores which attribute the keyword landed in (visible text, ARIA label, placeholder, id/class, other). It multiplies them, sums over all keywords, and ranks. It's deterministic and runs in a few milliseconds.

---

## Slide 11 — Dataset: Mind2Web

### Speech
"Mind2Web is the standard benchmark for this work — 137 websites across 31 domains, with 6,626 training samples, 736 evaluation samples, and three test splits totalling 6,070 samples. We use the standard splits exactly as released. The `test_domain` split is the largest at 3,838 samples and the most representative of real-world generalisation, because it tests on domains that don't appear in training at all. Most of the headline numbers we'll discuss come from `test_task` for comparability with Prune4Web, and from `test_domain` because that's where DOM Delta gives the largest gain."

### How to deliver
- Don't read the table. Just state the totals — 6,626 train, 6,070 test — and call out `test_domain` as the hardest split.
- Thirty seconds.

---

## Slide 12 — Novelty 1: DOM Delta Processing

### Speech
"This is the first contribution. The idea is simple: instead of treating each page snapshot independently, we track DOM elements *across* steps using a structural identifier and only re-rank what has changed.

Each element gets a structural UID — an MD5 hash over the tag, id, name, and xpath. This identifier is stable even when the visible text mutates, which is what gives us the cross-step continuity that systems like Prune4Web and AutoWebGLM don't have. We also attach a short ancestor summary — up to four meaningful ancestors like form, nav, section, dialog — so the grounder sees not just the element but the structural context.

The scoring is a hybrid: a normalised keyword score with weight alpha equals 0.6, plus a MiniLM cosine similarity between the sub-task and the element's text, with weight 0.4. We tuned alpha empirically on the eval split.

The result, measured on all three test splits totalling 6,070 samples, is a 16-to-18% reduction in grounder input tokens at matched Recall-at-20. On the largest split, `test_domain`, Recall-at-20 actually *improves* by 1.20 percentage points — so we're paying fewer tokens *and* getting better recall on the hardest split."

### How to deliver
- Spend a full ninety seconds here. This is the slide that demonstrates technical novelty.
- When you say "structural UID," **point at the formula**. When you say "hybrid score," **point at the equation**. The panel needs to physically map the words to the slide.
- The number to leave on their mind is **17.5% token reduction with +1.20 pp recall on test_domain**. Repeat that exact phrase if you have time.

### Anticipated questions
**"Why MiniLM and not a larger encoder?"**
*Answer:* MiniLM is 22M parameters, runs in milliseconds on CPU, and the filter stage doesn't need a deep encoder — it just needs a usable cosine similarity. Larger encoders (like all-mpnet-base) gave at most a 0.3 pp gain in our pilot but ran 5× slower. The cost-benefit didn't justify it.

**"Why alpha = 0.6?"**
*Answer:* We swept alpha from 0.0 to 1.0 in 0.1 increments on the eval split. 0.6 was the empirical sweet spot — keyword signal dominates because the planner's sub-task is short and verb-heavy, but we keep enough semantic signal to handle paraphrase ("Add to Cart" vs "Add to Bag", which is what Apple's site uses).

---

## Slide 13 — Novelty 2: Privacy Hub + Risk-Based HITL

### Speech
"The second contribution sits around the planner instead of around the filter. The PolicyHub is a JSON file with 24 risk keywords — payments, credentials, CAPTCHA, destructive actions, account changes — and is user-editable so a deployer can add their own.

We extended the planner's JSON output with three new fields: a boolean `policy_risk`, a numeric `risk_confidence`, and a free-text `hitl_reason`. The HITL gate then fires on any one of three OR-combined triggers — the LLM self-flagged the action as risky, a risk keyword matched the action text, or the confidence dropped below a configurable threshold.

Crucially, the gate sits *before* the grounder. So a risky action — say, a payment confirmation — short-circuits the expensive grounding call and yields control to the user, instead of grounding first and then asking permission. That's the architectural difference from a bolted-on filter.

The Privacy-Aware-LLM wrapper is a separate layer between Filter and Grounder. It runs a regex pass for ten classes of personally identifiable information — email, phone, SSN, credit card, IPv4, date of birth, passport, Aadhaar, PAN, and password fields — and replaces matches with opaque placeholders like `[EMAIL_MASKED]`. We also inject a short privacy directive into the grounder's system prompt instructing it to treat masked tokens as opaque tokens that must not be unmasked or guessed."

### How to deliver
- Two minutes on this slide is fine. It is the more original of the two contributions and the one the panel is least likely to have seen before.
- When you describe the three OR-triggers, **count them on your fingers** — the panel always asks back about one of these three.
- The single sentence to land cleanly: *"the gate sits before the grounder, so risky actions short-circuit the expensive call."*

### Anticipated questions
**"What's the false-positive rate of the regex PII detector?"**
*Answer:* Honest answer — we don't have a quantitative number yet because Mind2Web has no labelled PII test set. We have a 30-sample pilot planned where we inject synthetic PII into Mind2Web pages and measure precision/recall. The qualitative result so far is that on live login and payment pages the regexes fire correctly. I'd rather report this as a pilot in future work than fabricate a number now.

**"Why three triggers? Why not just confidence?"**
*Answer:* Confidence alone fails on the cases where the LLM is *confidently wrong* — a payment confirmation page that the planner reads as a normal click. The keyword scan catches those. And the LLM's self-flag catches subtle cases where the planner notices something risky that didn't keyword-match. Empirically each trigger fires on different samples, so they're complementary rather than redundant.

---

## Slide 14 — Architecture

### Speech
"This is the architecture diagram. Reading left to right: the Planner takes the user task and the page; the DOM Delta Filter applies structural UIDs and the hybrid score and reduces the candidate set; the PolicyHub HITL gate decides whether to short-circuit; the Privacy-Aware-LLM wrapper masks PII; and the local QLoRA grounder — Qwen3-0.6B by default, with Qwen2.5-0.5B as a smaller fallback — picks the final element. Of these five stages, four run entirely on the local machine. Only the Planner currently calls a hosted LLM, and the system is configured through a single JSON file so the deployer can swap any backend without touching code."

### How to deliver
- Trace the diagram with your finger or laser pointer **left to right** as you read.
- The closing sentence — "four out of five stages run locally" — is the deployability claim. Don't rush it.
- Forty-five seconds.

---

## Slide 15 — Updated Algorithm

### Speech
"This slide formalises what I just described. The Prune4Web original outputs an action `a_t` from the planner, the LLM-emitted keyword script, and the grounder. Our pipeline outputs an *augmented* action `A_t` that carries four extra fields: the action itself, a confidence `c`, a policy-risk bit `r`, and the human-in-the-loop reason. The DOM Delta `Delta_t` is computed against the previous step using structural UIDs. If the policy-risk bit is set, or a keyword matches, or the confidence drops below the threshold, we trigger HITL. Otherwise the top twenty candidates from `Delta_t` go to the local grounder."

### How to deliver
- This is a math-heavy slide. **Do not** read every symbol. Pick three: `Delta_t`, the OR-trigger, and the top-20.
- Thirty seconds.

---

## Slide 16 — Implementation Notes

### Speech
"The implementation lives in a single Python module, `run_prune4web.py`, with three pluggable backends per stage. The grounder uses QLoRA with 4-bit nf4 quantisation — that's what brings the model under one gigabyte of VRAM at inference, on a six-gigabyte RTX 4050. We trained on the Mind2Web train split using the *exact same prompt format* at training and inference time, which avoids the train-inference drift that's a common QLoRA pitfall. The DOM Delta layer and the privacy wrapper run entirely on-device. All model selections — planner, filter, grounder, embedding model, alpha, privacy and HITL toggles — are read from a single JSON config file at startup, so a deployer doesn't have to edit Python to switch backends."

### How to deliver
- The "single JSON config file" sentence is a deployability claim that matters for industrial credibility. Land it cleanly.
- Forty-five seconds.

---

## Slide 17 — Experiments: Filter-Stage Recall@20

### Speech
"This is the first results table — Recall-at-20 on all three Mind2Web test splits, totalling six thousand and seventy samples. Recall-at-20 is the metric the filter stage should be judged on: of the top twenty candidates the filter sends to the grounder, did the *correct* element make the cut? The FULL column is Prune4Web's filter alone. The DELTA-HYBRID column is our filter with the DOM Delta layer added.

On `test_task` and `test_website` we match FULL within 0.23 and 0.10 percentage points respectively — statistically indistinguishable. On `test_domain`, the largest and hardest split, we *improve* by 1.20 percentage points. And on all three splits we reduce grounder input tokens by 16 to 18 percent. So the headline is: same recall or better, with about one-sixth fewer tokens going downstream."

### How to deliver
- Walk down the table column-by-column.
- Repeat the headline twice — *"same recall, 17.5% fewer tokens"* — once at the start and once at the end of the slide.

### Anticipated question
**"How are these statistics? Have you computed confidence intervals?"**
*Answer:* On `test_domain`, with 3,838 samples, a 1.20 pp difference in Recall-at-20 is well outside a binomial 95% CI of roughly ±1.0 pp at this base rate, so the gain is significant. On `test_task` (1,257 samples) the deltas are within the CI and that's why I called them statistically indistinguishable rather than improvements.

---

## Slide 18 — Experiments: Grounder Element Accuracy

### Speech
"This is the second results table — Element Accuracy of the grounder. We compare three configurations. The Prune4Web baseline is Qwen2.5VL-3B-Instruct, three billion parameters, vision-language, reporting 88.28%. Our smaller Qwen2.5-0.5B with QLoRA reaches 85.0% — six times fewer parameters than the baseline. Our Qwen3-0.6B with QLoRA reaches 88.00% — five times fewer parameters and *matches the vision-language baseline within rounding margin*, with no vision encoder, on a 200-sample slice of `test_domain`. The cost we pay for this is one hour of QLoRA training on a 4050."

### How to deliver
- The single number to leave with the panel: **"88.00 from a 0.6B text-only model versus 88.28 from a 3B vision-language model."**
- Pause after that sentence. Let it land.
- Sixty seconds.

### Anticipated questions
**"Why does the 0.6B model match the 3B model? Isn't that suspicious?"**
*Answer:* Two reasons. First, the grounder is a constrained classification task — pick one of N candidates and output a small JSON. That doesn't need broad world knowledge; it needs format compliance and reading ability, both of which QLoRA fine-tuning gives even a small model. Second, the *upstream* filter has already narrowed the candidate set to 20, so the grounder isn't doing the heavy lifting. The vision encoder in the 3B baseline is mostly redundant given that the DOM already contains the semantic information.

**"Why didn't you train a 1.5B or 3B local grounder?"**
*Answer:* Honest answer — VRAM. A 3B model in 4-bit doesn't fit comfortably in 6 GB *during training* with reasonable batch sizes. We picked 0.6B as the largest model we could train end-to-end on the available hardware, and the result happens to be strong enough that the larger model wasn't strictly needed for the headline number.

---

## Slide 19 — Experiments: Training & Efficiency

### Speech
"For deployability: the Qwen3-0.6B QLoRA fine-tune took about five and a half hours on the 4050, the Qwen2.5-0.5B took about ten hours. Inference latency is around 5 seconds per sample for Qwen2.5 and 20 seconds for Qwen3 — and that 4× gap is *not* a model-size gap, it's a chat-template artefact: Qwen3's template emits an empty think-block even when thinking is disabled, and a fix is in progress. The MiniLM filter-stage scoring runs at 220 milliseconds per step. So the dominant cost is the grounder, and Qwen2.5 is the model to use if you care about latency over the last 3 percentage points of accuracy."

### How to deliver
- Be transparent about the Qwen3 latency artefact. The panel will respect "this is a known bug, fix is in progress" more than they will respect "20 seconds is fine."
- Forty-five seconds.

---

## Slide 20 — Privacy & HITL (qualitative pilot)

### Speech
"This slide is deliberately qualitative. Mind2Web has no labelled PII benchmark, and we did not want to fabricate quantitative numbers. What we *can* report is that on live sessions on Apple, login flows, and payment pages, the PolicyHub fires correctly — login forms trigger the credentials keyword, payment buttons trigger the payments keyword, and CAPTCHA pages trigger the bot-detection keyword. The PII regex masks email, phone, date of birth, and password fields before they enter the grounder prompt. A 30-sample injected-PII pilot is planned for the final report — synthetic PII inserted into Mind2Web pages, measuring HITL fire rate, masking rate, and regex precision and recall."

### How to deliver
- This is a **honesty slide**. The panel will respect you for not over-claiming. Say *"qualitative"* clearly.
- Thirty seconds.

### Anticipated question
**"Why didn't you build a synthetic PII benchmark already?"**
*Answer:* Time-boxed. Major Project 2 had a fixed window, and we prioritised the DOM Delta and grounder experiments because those are the load-bearing claims of the paper. The PII pilot is the natural next step and is scoped for the final-report iteration.

---

## Slide 21 — Future Work

### Speech
"Five concrete next steps. One: route the privacy wrapper through all three call sites — planner, filter, and grounder — currently it's instantiated and applied to the grounder only. Two: run the 30-sample injected-PII pilot. Three: investigate why DOM Delta gives a 5–6 percentage-point Element Accuracy gap at near-equal Recall-at-20 — we suspect it's a candidate-ordering sensitivity inside the top 20 that the grounder is picking up on. Four: strip the empty think-block from the Qwen3 chat template to bring its latency down from 20 seconds to 5. Five: evaluate on multi-step trajectory benchmarks like WebVoyager and Mind2Web-Live, where the cross-step delta should pay off the most."

### How to deliver
- Be brisk. Each item is one sentence. Forty-five seconds total.
- The 5–6 pp gap one is the most interesting — be ready to defend it. *"Same recall, lower EA"* sounds contradictory and the panel will ask.

### Anticipated question
**"How can Recall-at-20 be the same but Element Accuracy drop?"**
*Answer:* Recall-at-20 only asks whether the correct element is somewhere in the top 20. Element Accuracy asks whether it's the *one* the grounder picks. If DOM Delta changes the *ordering* within the top 20 — say, the correct element drops from rank 3 to rank 17 — Recall@20 is unchanged but the grounder may pick a different candidate. We have a planned ablation that re-ranks the DELTA top-20 by the original FULL scores to test this hypothesis directly.

---

## Slide 22 — Conclusion

### Speech
"To summarise. We took the Prune4Web pipeline as a base and made three changes. First, DOM Delta Processing replaces the stateless full-DOM filter with a structural-UID-tracked delta view, giving 16-to-18 percent token reduction at matched-or-improved Recall-at-20 across 6,070 Mind2Web test samples. Second, a QLoRA-fine-tuned Qwen3-0.6B local grounder reaches 88.00% Element Accuracy, matching the 3-billion-parameter vision-language baseline within rounding margin while running on under one gigabyte of VRAM on a 4050. Third, a PolicyHub plus risk-based HITL gate, with a regex PII detector and a privacy directive injected into the grounder prompt, brings privacy-awareness *into* the action loop instead of bolting it on. Net result: the pipeline is cheaper at the filter stage, safer at the grounder stage, and deployable on consumer hardware — without sacrificing element accuracy."

### How to deliver
- The conclusion is the slide that the panel remembers. **Slow down and project.**
- Three quantitative anchors to leave on the screen: **17.5% token reduction**, **88.00% EA**, **<1 GB VRAM**. Say each one explicitly.
- Ninety seconds.

---

## Slide 23 — References

### Speech
"References are listed; I won't read them but I'm happy to take questions on any of them."

### How to deliver
- Ten seconds. Move on.

---

## Slide 24 — Thanks!

### Speech
"Thank you. I'm happy to take questions."

### How to deliver
- Smile, hand off to the panel, sit down (or stand quietly).
- **Important:** when the first question comes, repeat the question back before answering. It buys you five seconds to think and signals to the panel that you understood them.

---

# Suggested PPT Updates

After going back over the deck against the current state of the codebase, the following slides should be edited before submission. Most are small.

## Required updates

1. **Slide 8 (Research Gaps) and Slide 14 (Architecture)** — Reflect the *new* state where the **filter stage is now also local**. The deck still implies the filter calls GPT-4o. As of the latest commit, the default `--filter local` skips the GPT-4o keyword-weight call and uses a deterministic keyword extractor + DOM Delta + MiniLM hybrid score. Only the Planner remains hosted. Suggested wording on Slide 14: *"Four of the five pipeline stages run locally; only the Planner calls a hosted LLM."*

2. **Slide 16 (Implementation Notes)** — Add one bullet: *"All backends are selected through a single `config/models.json` file; switching from GPT-4o filter to local filter, or Qwen3 grounder to Qwen2.5, is a one-line config change with no code edit."* This is a deployability point that examiners care about.

3. **Slide 22 (Conclusion)** — Strengthen the deployability sentence to say *"...four out of five stages run on-device, with planner the only remaining hosted call, and the entire backend wiring is config-driven."* This raises the contribution from "the grounder is local" to "the *system* is largely local," which is the more defensible claim now.

## Recommended updates

4. **Slide 17 (Filter-Stage Recall@20)** — Add a footnote line under the table: *"DELTA-HYBRID rows reflect the local filter (deterministic keywords + MiniLM); the original Prune4Web GPT-4o keyword-script filter remains supported via `--filter gpt4o` for the FULL baseline."* This pre-empts the question "wait, what is FULL exactly?"

5. **Slide 19 (Training & Efficiency)** — Add an inference-cost row: *"OpenAI API calls per step: 2 (planner only) — down from 3 in the Prune4Web baseline."* This makes the cost win concrete in numbers the panel can audit.

6. **Slide 21 (Future Work)** — Add one item: *"Train a local QLoRA planner adapter to remove the last hosted dependency and ship a fully on-device pipeline."* This converts the planner-is-still-hosted limitation into stated future work and protects you in Q&A.

## Optional polish

7. **Slide 12 (DOM Delta novelty)** — The hybrid-score equation reads a bit dense. Splitting it across two lines (one for the keyword term, one for the embedding term) makes it easier to point at while explaining.

8. **Slide 18 (Grounder EA)** — Bold the Qwen3-0.6B row. The 88.00 vs 88.28 comparison is the strongest single number in the talk and the formatting should make that obvious at a glance.

9. **Slide 20 (Privacy & HITL pilot)** — Add the line *"N = 30 (pilot, planned for final report)"* in the slide footer so the qualitative scope is visible without you having to say it. Reduces the chance of an examiner thinking we're glossing over missing numbers.

---

*Document created 2026-05-01. Aligned with the current state of `run_prune4web.py` after the introduction of `--filter local` (default) and `config/models.json`.*
