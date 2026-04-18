"""
run_followup_experiments.py — Execute all 5 follow-up experiments in sequence.

Designed to be launched via nohup. Everything runs on free, local-only
compute (MiniLM embeddings + local Qwen2.5-0.5B LoRA grounder).

Experiments
-----------
1. Per-domain breakdown of the existing 6,070-sample results (post-processing).
2. Actual Qwen token counts (benchmark run with --count-actual-tokens).
3. Vary top-K values (Recall@10 / 20 / 50).
4. Element Accuracy via local Qwen grounder (top-20).
5. Latency breakdown for DELTA_HYBRID.

Launch
------
    nohup "D:/Environments/ml-env/Scripts/python.exe" -X utf8 \
        Prune4Web/run_followup_experiments.py \
        > Prune4Web/results/dom_delta_bench/nohup_followup.log 2>&1 &
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
ROOT = Path(__file__).resolve().parent
EVAL = str(ROOT / "evaluate_dom_delta.py")
PER_DOMAIN = str(ROOT / "analyze_per_domain.py")

RESULTS_DIR = ROOT / "results" / "dom_delta_bench"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FOLLOWUP_DIR = RESULTS_DIR / "followup_experiments"
FOLLOWUP_DIR.mkdir(parents=True, exist_ok=True)


def banner(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72 + "\n", flush=True)


def run(cmd: list[str], label: str) -> dict:
    banner(f"RUN: {label}")
    print(f"  cmd: {' '.join(cmd)}\n", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0
    print(f"\n  [{label}] exit={result.returncode}  time={elapsed:.1f}s",
          flush=True)
    return {"label": label, "returncode": result.returncode,
            "elapsed_s": round(elapsed, 1)}


def main():
    t_start = time.time()
    banner(f"FOLLOW-UP EXPERIMENTS — started {datetime.datetime.now().isoformat()}")

    summary = []

    # ── Experiment 1 — Per-domain breakdown (post-processing, seconds) ──
    summary.append(run(
        [PYTHON, "-X", "utf8", PER_DOMAIN],
        "Experiment 1: per-domain breakdown"))

    # ── Experiment 2 — Actual Qwen token counts on test_domain (500) ──
    # Seed 42, all 4 configs, --count-actual-tokens. ~3 min.
    summary.append(run(
        [PYTHON, "-X", "utf8", EVAL,
         "--split", "test_domain",
         "--num-samples", "500",
         "--seed", "42",
         "--configs", "FULL", "DELTA_KW", "DELTA_EMBED", "DELTA_HYBRID",
         "--count-actual-tokens"],
        "Experiment 2: actual Qwen tokens (500 samples, test_domain)"))

    # ── Experiment 3 — Vary top-K on test_domain (500) ──
    # We already have top-20 numbers; run top-10 and top-50.
    for K in (10, 50):
        summary.append(run(
            [PYTHON, "-X", "utf8", EVAL,
             "--split", "test_domain",
             "--num-samples", "500",
             "--seed", "42",
             "--top-n", str(K),
             "--configs", "FULL", "DELTA_HYBRID"],
            f"Experiment 3: top-{K} Recall (500 samples, test_domain)"))

    # ── Experiment 5 — Latency breakdown (DELTA_HYBRID) on test_domain (200) ──
    # Runs quickly; the latency breakdown is emitted by the log.
    summary.append(run(
        [PYTHON, "-X", "utf8", EVAL,
         "--split", "test_domain",
         "--num-samples", "200",
         "--seed", "42",
         "--configs", "FULL", "DELTA_HYBRID"],
        "Experiment 5: latency breakdown (200 samples, test_domain)"))

    # ── Experiment 4 — Element Accuracy with local Qwen grounder ──
    # Runs last because it is the slowest (model load + per-sample inference).
    # 200 samples, FULL + DELTA_HYBRID only (expensive).
    summary.append(run(
        [PYTHON, "-X", "utf8", EVAL,
         "--split", "test_domain",
         "--num-samples", "200",
         "--seed", "42",
         "--configs", "FULL", "DELTA_HYBRID",
         "--with-grounder",
         "--grounder-configs", "FULL", "DELTA_HYBRID"],
        "Experiment 4: Element Accuracy via local Qwen (200 samples)"))

    total = time.time() - t_start
    banner("ALL FOLLOW-UP EXPERIMENTS COMPLETE")
    for r in summary:
        status = "OK" if r["returncode"] == 0 else "FAIL"
        print(f"  [{status}] {r['label']:<55}  {r['elapsed_s']}s")
    print(f"\n  Total elapsed: {total:.0f}s ({total / 60:.1f} min)")

    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiments": summary,
        "total_elapsed_s": round(total, 1),
    }
    meta_path = FOLLOWUP_DIR / "followup_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  meta -> {meta_path}", flush=True)


if __name__ == "__main__":
    main()
