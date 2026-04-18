"""
run_full_dom_delta_bench.py — Run the DOM Delta A/B benchmark across ALL
Mind2Web test splits with full sample counts for the research paper.

Launch via nohup:
    nohup "D:\Environments\ml-env\Scripts\python.exe" -X utf8 \
        Prune4Web/run_full_dom_delta_bench.py > Prune4Web/results/dom_delta_bench/nohup_full.log 2>&1 &

Runs:
  1. test_task    (1,257 samples) — unseen tasks, seen websites
  2. test_website (  975 samples) — unseen websites, seen domains
  3. test_domain  (3,838 samples) — completely unseen domains

Total: ~6,070 samples across all three generalization axes.
"""
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
SCRIPT = str(Path(__file__).resolve().parent / "evaluate_dom_delta.py")
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "dom_delta_bench"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = [
    {"split": "test_task",    "num_samples": 0},   # 0 = all
    {"split": "test_website", "num_samples": 0},
    {"split": "test_domain",  "num_samples": 0},
]

CONFIGS = ["FULL", "DELTA_KW", "DELTA_EMBED", "DELTA_HYBRID"]

def run_split(split: str, num_samples: int) -> dict:
    args = [
        PYTHON, "-X", "utf8", SCRIPT,
        "--split", split,
        "--seed", "42",
        "--configs", *CONFIGS,
    ]
    if num_samples > 0:
        args += ["--num-samples", str(num_samples)]
    else:
        args += ["--num-samples", "99999"]  # effectively "all"

    print(f"\n{'='*72}")
    print(f"  RUNNING: {split}  (max_samples={'ALL' if num_samples == 0 else num_samples})")
    print(f"  Command: {' '.join(args)}")
    print(f"{'='*72}\n")

    t0 = time.time()
    result = subprocess.run(args, capture_output=False, text=True)
    elapsed = time.time() - t0

    return {
        "split": split,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
    }

def main():
    print("=" * 72)
    print("FULL DOM DELTA BENCHMARK — ALL TEST SPLITS")
    print(f"Started: {datetime.datetime.now().isoformat()}")
    print(f"Configs: {CONFIGS}")
    print("=" * 72)

    all_results = []
    t_total = time.time()

    for spec in SPLITS:
        r = run_split(spec["split"], spec["num_samples"])
        all_results.append(r)
        print(f"\n  [{r['split']}] exit={r['returncode']}  time={r['elapsed_s']}s")

    total_elapsed = time.time() - t_total

    print(f"\n{'='*72}")
    print("ALL SPLITS COMPLETE")
    print(f"{'='*72}")
    for r in all_results:
        status = "OK" if r["returncode"] == 0 else "FAIL"
        print(f"  [{status}] {r['split']:<15}  {r['elapsed_s']}s")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Results in: {RESULTS_DIR}")
    print(f"{'='*72}")

    # Save a meta-summary
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "configs": CONFIGS,
        "splits": all_results,
        "total_elapsed_s": round(total_elapsed, 1),
    }
    meta_path = RESULTS_DIR / "full_bench_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta -> {meta_path}")


if __name__ == "__main__":
    main()
