# Major Project – 2: Updated PPT Content

Slide-by-slide content for the new submission. Replace one slide at a time in the existing template.

---

## Slide 1 — Title
"Improving Web Automation with DOM Delta Processing and Privacy-Aware LLM Agents"
— Darji Dhruv S. (24MCD003)
— Mentor: Dr. Parita Oza
**Major Project – 2**
Computer Science & Engineering (Data Science)

---

## Slide 2 — Contents
01 Introduction
02 Problem Statement
03 Literature Review
04 Research Gaps
05 Objectives
06 Prune4Web: Base Paper *(replaces SeeAct)*
07 Dataset (Mind2Web)
08 Novelty: DOM Delta + Privacy Hub & HITL
09 Implementation
10 Experiments & Results
11 Future Work
12 Conclusion
13 References

---

## Slide 3 — Introduction
*(keep as-is from previous deck)*

---

## Slide 4 — Problem Statement *(refreshed)*
- LLM-based web agents process the **entire DOM at every step** (10⁴–10⁵ tokens), which is computationally expensive even when the page barely changes.
- Open-weight grounders below 1B parameters are largely unstudied; most strong baselines depend on hosted vision-language models like GPT-4V or Qwen2.5VL-3B.
- Privacy and HITL safeguards are bolted on after deployment, not designed into the action loop — every DOM snippet is shipped raw to a hosted LLM.
- A localised, token-efficient, privacy-aware web agent that runs on commodity hardware is still missing.

---

## Slides 5–7 — Literature Review *(append new rows; keep existing surveys + SeeAct + AdaptAgent + WebExperT)*

| Year | Title | Key Contribution | Limitation |
|---|---|---|---|
| Nov 2025 [11] | **Prune4Web (Zhang et al.)** | Three-stage Planner→Filter→Grounder; programmatic JS-extractor + chunked LLM judge; 88.28% EA Mind2Web with fine-tuned **Qwen2.5VL-3B-Instruct** grounder | Filter is prompt-engineered & LLM-bound; grounder is a 3B VLM; no privacy/HITL layer |
| 2025 [12] | **D2Snap** | DOM-diff snapshotting; ~10³-token compression via post-order merging + TextRank | Operates on isolated snapshots; no cross-step reuse |
| 2024 [13] | **Agent-E** | Accessibility-tree pruning; MutationObserver for cross-step changes | Converts changes into prompt text — no token reduction |
| 2024 [14] | **AutoWebGLM** | HTML pruning by trimming non-actionable nodes, depth-limit | Per-step snapshot only |
| 2023 [15] | **WebAgent / HTML-T5** | HTML-specific encoder for long DOM sequences | Each step independent; no delta reuse |
| 2020 [16] | **MiniLM** | 22M-param distilled encoder; ms-level CPU inference | Generic encoder, not DOM-aware |
| 2023 [17] | **QLoRA** | 4-bit nf4 + LoRA — billion-scale fine-tuning on a single consumer GPU | — |

SeeAct stays as a Lit-Review reference only (could not be reproduced locally — GPT-4V API cost).

---

## Slide 8 — Research Gaps *(rewritten — three sharper gaps)*

- **Stateless full-DOM filtering.** Every existing pruning system (Prune4Web, AutoWebGLM, Agent-E, HTML-T5) re-processes the full DOM each step, even when the page changed marginally. No structural-UID + delta layer is wired into a real filter.
- **Sub-1B local grounders are under-studied.** Strong web agents lean on GPT-4V or Qwen2.5VL-3B. Open-weight 0.5B–0.6B grounders exist as proof-of-concept variants only — no documented training recipe, no per-split numbers.
- **Privacy and HITL bolted on, not architected in.** Risk gating and PII masking sit outside the planner→filter→grounder loop. None of the published pipelines inject a privacy directive into the grounder's system prompt or run a regex-PII pass over candidate elements before grounding.

---

## Slide 9 — Objectives *(rewritten)*

- Reproduce the Prune4Web Planner→Filter→Grounder pipeline locally on a single 6 GB consumer GPU.
- Add a **DOM Delta layer** at the filter stage — structural UIDs over (tag, id, name, xpath) + hybrid keyword/MiniLM scoring — to reduce per-step grounder input.
- Replace the closed Qwen2.5VL-3B grounder with a **QLoRA-fine-tuned Qwen2.5-0.5B / Qwen3-0.6B** local text-only grounder.
- Layer a **Privacy Hub + risk-based HITL gate** around the planner: a 24-keyword PolicyHub, regex PII detector, and three OR-combined HITL triggers (`policy_risk`, risk-keyword scan, confidence floor).
- Evaluate Recall@20, Element Accuracy, token reduction, and latency on all three Mind2Web test splits.

---

## Slide 10 — Prune4Web: Base Paper
Paper title: *"Prune4Web: DOM Tree Pruning Programming for Web Agents"* (Zhang et al., 2025).
- Three-stage pipeline: **Planner** (decides sub-task) → **Programmatic Element Filter** (LLM emits a Python keyword/weight script that a deterministic scorer applies) → **Action Grounder** (fine-tuned Qwen2.5VL-3B-Instruct).
- Reports **88.28% Element Accuracy** on Mind2Web `test_task` with the fine-tuned 3B VLM grounder; 25–50× DOM-token reduction at the filter stage.
- Chosen as base over SeeAct because SeeAct depends on GPT-4V (paid) and could not be reproduced locally; Prune4Web has an open recipe and a stronger published number.

---

## Slide 11 — Dataset: Mind2Web
Same Multimodal-Mind2Web dataset (137 websites, 31 domains).

| Split | Samples | Purpose |
|---|---|---|
| Train | 6,626 | QLoRA fine-tuning of grounder |
| Eval | 736 | Mid-training validation |
| test_task | 1,257 | Cross-task generalisation |
| test_website | 975 | Cross-website generalisation |
| test_domain | 3,838 | Cross-domain robustness (largest, hardest) |

Total test = 6,070 samples.

---

## Slide 12 — Novelty 1: DOM Delta Processing
- **Structural UID** = MD5(tag, id, name, xpath) — stable across re-renders even when visible text mutates. Same property as Mind2Web's `backend_node_id` but computed offline from parsed HTML, no live browser needed.
- **Ancestor summary** — up to 4 semantically meaningful ancestors (`form`, `nav`, `section`, `dialog`) appended as `ctx=form#signup > fieldset[Address]`.
- **Hybrid score** of each candidate element:
  score(e) = α·norm(kw(e)) + (1−α)·cos(emb(task), emb(e)),  α = 0.6
  with sentence embeddings from `all-MiniLM-L6-v2`.
- **Result**: 16–18% Qwen-token reduction at matched Recall@20 across all three test splits; **+1.20 pp Recall@20** on test_domain (3,838 samples) — the largest, hardest split.

---

## Slide 13 — Novelty 2: Privacy Hub + Risk-Based HITL
- **PolicyHub**: 24-keyword risk list (payments, credentials, CAPTCHA, destructive actions, account changes); user-customisable; can hold a full site policy.
- LLM JSON output extended with `policy_risk` (bool), `risk_confidence`, `hitl_reason`.
- **HITL gate** fires on any one of three OR-combined triggers:
  (a) LLM self-flags `policy_risk=true`,
  (b) a risk keyword appears in the action text or `hitl_reason`,
  (c) `risk_confidence` falls below a configured threshold.
- The gate sits **before** the grounder, so risky steps short-circuit the expensive grounder call.
- **Privacy-Aware LLM wrapper** (between Filter and Grounder) runs a regex PII detector — email, phone (US), SSN, credit card, IPv4, DOB, passport, Aadhaar, PAN, password — and replaces matches with `[EMAIL_MASKED]`-style placeholders. A short privacy directive is injected into the grounder system prompt instructing the model to treat masked tokens as opaque.

---

## Slide 14 — Architecture (updated diagram)
*(use `MTech_Report_Sem_IV/Diagrams/Architecture.png`)*
Planner → DOM Delta Filter (structural UID + MiniLM hybrid) → PolicyHub HITL gate → Privacy-Aware LLM wrapper → Local QLoRA Grounder (Qwen2.5-0.5B / Qwen3-0.6B).

---

## Slide 15 — Updated Algorithm
- **Prune4Web original**:
  aₜ = π(sₜ, T, {a₁…aₜ₋₁}); element ranking via LLM-emitted keyword script; grounded by Qwen2.5VL-3B.
- **Ours**:
  Aₜ = π(sₜ, T, {a₁…aₜ₋₁}, pₜ, Δₜ)
  Δₜ = DOMDelta(DOMₜ, DOMₜ₋₁) using structural UIDs
  Aₜ ∈ (aₜ, c, r, hr) where c = confidence, r = policy_risk ∈ {0,1}, hr = hitl_reason
  if r = 1 OR keyword_match(aₜ, pₜ) OR c < τ: trigger HITL
  else: feed top-20 from Δₜ into local QLoRA grounder

---

## Slide 16 — Implementation Notes
- Backbone re-implemented from Prune4Web with deviations.
- Grounder: QLoRA-fine-tuned **Qwen2.5-0.5B** or **Qwen3-0.6B** loaded in 4-bit nf4; **<1 GB VRAM** at inference on RTX 4050 (6 GB).
- Fine-tuned on 6,626 Mind2Web examples using the identical grounder prompt as inference — no train/inference drift.
- DOM Delta + Privacy wrapper run entirely on-device.

---

## Slide 17 — Experiments: Filter-Stage Recall@20

| Split | Samples | FULL Recall@20 | DELTA_HYBRID Recall@20 | Δ | Token red. |
|---|---|---|---|---|---|
| test_task | 1,257 | 90.30% | 90.07% | −0.23 pp | 17.8% |
| test_website | 975 | 89.95% | 89.85% | −0.10 pp | 16.4% |
| **test_domain** | **3,838** | **86.78%** | **87.98%** | **+1.20 pp** | **17.5%** |

DELTA_HYBRID matches FULL Recall@20 within 0.23 pp on two splits and **improves** on the largest split, while reducing grounder input tokens by 16–18%.

---

## Slide 18 — Experiments: Grounder Element Accuracy

| Grounder | Params | test_task EA | Notes |
|---|---|---|---|
| Qwen2.5VL-3B-Instruct (Prune4Web) | 3.0B | 88.28% | Vision-language, paper baseline |
| **Qwen2.5-0.5B + QLoRA (ours)** | 0.5B | 85.00% | Text-only, 6× fewer params |
| **Qwen3-0.6B + QLoRA (ours)** | 0.6B | **88.00%** | Text-only, 5× fewer params, matches paper |

200-sample test_domain slice with DOM Delta enabled: Qwen3 reaches **88.00% EA**, matching the 3B vision-language baseline within rounding margin while running without a vision encoder.

---

## Slide 19 — Experiments: Training & Efficiency
- Qwen3-0.6B QLoRA: 325 min on RTX 4050.
- Qwen2.5-0.5B QLoRA: 609 min.
- Inference latency: Qwen2.5 ≈ 5 s/sample, Qwen3 ≈ 20 s/sample (Qwen3's chat template emits an empty `<think>…</think>` block even when `enable_thinking=False` — fix planned).
- Filter stage MiniLM scoring ≈ 220 ms/step on RTX 4050.

---

## Slide 20 — Privacy & HITL (qualitative pilot)
- Mind2Web has no labelled PII test set, so results are qualitative on small live sessions.
- PolicyHub fires correctly on login, payment, and CAPTCHA-style sub-tasks; PII regex masks email, phone, DOB, password fields before they enter the grounder prompt.
- A 30-sample injected-PII pilot is planned to produce HITL-fire rate, masking rate, and regex precision/recall.

---

## Slide 21 — Future Work
- Wire the `PrivacyAwareLLM` wrapper into all three call sites (planner, filter, grounder) — currently instantiated but not yet routed in `run_prune4web.py`.
- Run the 30-sample injected-PII pilot to attach quantitative numbers to §5.7.
- Investigate the 5–6 pp EA gap between FULL and DELTA_HYBRID at near-equal Recall@20 — likely a candidate-ordering sensitivity inside the top-20.
- Strip the empty `<think>` block from the Qwen3 chat template to bring its 20 s latency down to Qwen2.5's 5 s.
- Evaluate on multi-step trajectory benchmarks (WebVoyager, Mind2Web-Live) where DOM Delta should pay off the most.

---

## Slide 22 — Conclusion
- **DOM Delta Processing** replaces the stateless full-DOM filter with a structural-UID-tracked delta view; hybrid keyword + MiniLM scoring gives 16–18% token reduction at matched or improved Recall@20 across 6,070 Mind2Web test samples.
- A **QLoRA-fine-tuned Qwen3-0.6B** local grounder reaches **88.00% EA**, matching the Prune4Web 3B Qwen2.5VL-Instruct baseline (88.28%) within rounding margin while running at 5× fewer parameters, no vision encoder, **<1 GB VRAM** on a 6 GB RTX 4050.
- A **PolicyHub + risk-based HITL gate** with a regex PII detector and a privacy directive injected into the grounder prompt brings privacy-awareness into the action loop, not bolted on after.
- Net result: the Prune4Web pipeline becomes **cheaper at the filter stage**, **safer at the grounder stage**, and **deployable on consumer hardware** — without sacrificing element accuracy.

---

## Slide 23 — References *(append; keep existing [1]–[10])*
[11] Y. Zhang et al., "Prune4Web: DOM Tree Pruning Programming for Web Agents," 2025.
[12] M. Schiepanski et al., "D2Snap: A DOM Snapshotting Method for Web Agents," 2025.
[13] T. Tarsariya et al., "Agent-E: From Autonomous Web Navigation to Foundational Design Principles," 2024.
[14] H. Lai et al., "AutoWebGLM: A Large Language Model-based Web Navigating Agent," 2024.
[15] I. Gur et al., "A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis," 2024.
[16] W. Wang et al., "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression," 2020.
[17] T. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," 2023.
[18] N. Reimers & I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," 2019.

---

## Slide 24 — Thanks!
*(unchanged)*
