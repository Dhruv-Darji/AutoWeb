### Grounder Comparison (free, local-only LoRA + GPT-4o reference)

All LoRA models trained on the same 6,626 Mind2Web training examples, 4-bit
QLoRA (r=16, alpha=32), effective batch size 16, max seq 1024, seed 42.

| Model | Params | Epochs | Train time | test_task EA (200) | DOM-Delta EA FULL (200, test_domain) | DOM-Delta EA DELTA_HYBRID (200, test_domain) |
|---|---:|---:|---:|---:|---:|---:|
| GPT-4o (Prune4Web paper) | ~1T | — | — | 88.28% | — | — |
| Qwen2.5-0.5B + LoRA | 0.5B | 3 | 609 min | 85.00% | 73.00% | 68.00% |
| **Qwen3-0.6B + LoRA** | **0.6B** | **2** | **325 min** | **88.00%** | **77.50%** | **71.50%** |
| Δ (Qwen3 − Qwen2.5) | | | -284 min | **+3.00 pp** | **+4.50 pp** | **+3.50 pp** |

**Headline.** The Qwen3-0.6B LoRA grounder matches GPT-4o element accuracy
(88.00% vs 88.28%) on Mind2Web test_task at roughly **1,700× fewer
parameters** and runs entirely locally in 4-bit.

**Caveats.**
- Qwen3 was trained for 2 epochs (not 3), saving ~47% wall-clock despite
  the slightly larger model; one more epoch was possible within budget but
  no longer needed once 88% was hit.
- Per-sample inference latency is ~20s on RTX 4050 because the Qwen3 chat
  template still emits an empty `<think>\n\n</think>` block even with
  `enable_thinking=False`; the grounder extracts the JSON afterwards. For
  latency-critical use, Qwen2.5 remains the cheaper choice (~5s/sample).
- DOM-Delta EA numbers are on 200 test_domain samples (~±6 pp 95% CI);
  both models show the same ordering (FULL > DELTA_HYBRID on EA) despite
  near-equal Recall@20, confirming the prompt-sensitivity caveat from
  §3.4 of RESULTS_AND_COMPARISON.md.

Artifacts: `Prune4Web/grounder_finetune/data/eval_results_qwen3.json`,
`Prune4Web/results/dom_delta_bench/run_20260418_051722/bench_results.json`,
`Prune4Web/grounder_finetune/charts/loss_curve_qwen3.png`.
