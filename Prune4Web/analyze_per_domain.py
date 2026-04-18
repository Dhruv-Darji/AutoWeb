"""
analyze_per_domain.py — Per-domain breakdown (Experiment 1, free/no re-run).

Reads the three existing bench_results.json files from the full 6,070-sample
run and re-aggregates by the Mind2Web `domain` / `subdomain` field. The
benchmark itself does not need to be re-run; this script only does
post-processing on results we already have.

Output: Prune4Web/results/dom_delta_bench/per_domain_breakdown.json
        + a bar chart PNG per split.

Usage:
    python -X utf8 Prune4Web/analyze_per_domain.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "dom_delta_bench"
DATASET_PATH = Path(os.getenv("DATASET_PATH",
                              "D:/Environments/Datasets/multimodal-mind2web")) / "data"

# The full-split runs from 2026-04-16 (timestamps from full_bench_meta.json)
SPLIT_RUNS = {
    "test_task":    "run_20260416_175849",
    "test_website": "run_20260416_180712",
    "test_domain":  "run_20260416_181318",
}
CONFIGS = ["FULL", "DELTA_KW", "DELTA_EMBED", "DELTA_HYBRID"]


def load_samples(run_dir: Path) -> List[Dict]:
    path = run_dir / "bench_results.json"
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    return report["samples"]


def load_domain_map(split: str) -> Dict[str, Dict[str, str]]:
    """Build action_uid -> {domain, subdomain, website} from parquet files."""
    paths = sorted(DATASET_PATH.glob(f"{split}-*.parquet"))
    if not paths:
        return {}
    dfs = []
    for p in paths:
        df = pd.read_parquet(str(p), columns=["action_uid", "domain", "subdomain", "website"])
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    mp: Dict[str, Dict[str, str]] = {}
    for _, row in full.iterrows():
        mp[str(row["action_uid"])] = {
            "domain": str(row.get("domain", "") or ""),
            "subdomain": str(row.get("subdomain", "") or ""),
            "website": str(row.get("website", "") or ""),
        }
    return mp


def group_by_domain(samples: List[Dict], domain_map: Dict[str, Dict[str, str]]
                    ) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for s in samples:
        uid = s.get("action_uid", "")
        meta = domain_map.get(uid, {})
        key = s.get("domain", "") or meta.get("domain", "") or "unknown"
        groups.setdefault(key, []).append(s)
    return groups


def aggregate_subset(samples: List[Dict]) -> Dict:
    """Recompute recall/tokens per config for a subset."""
    out: Dict = {"n": len(samples), "configs": {}}
    for cfg in CONFIGS:
        recalls = [s["results"][cfg]["recall_at_20"]
                   for s in samples if cfg in s["results"]]
        tokens = [s["results"][cfg]["tokens_proxy"]
                  for s in samples if cfg in s["results"]]
        if not recalls:
            continue
        out["configs"][cfg] = {
            "samples": len(recalls),
            "recall_at_20_pct": round(sum(recalls) / len(recalls) * 100, 2),
            "mean_tokens_proxy": round(sum(tokens) / len(tokens), 1),
        }
    # Deltas vs FULL
    if "FULL" in out["configs"]:
        base = out["configs"]["FULL"]
        for cfg, a in out["configs"].items():
            if cfg == "FULL":
                continue
            a["delta_recall_pp"] = round(
                a["recall_at_20_pct"] - base["recall_at_20_pct"], 2)
            a["token_reduction_pct"] = round(
                (1 - a["mean_tokens_proxy"] / max(base["mean_tokens_proxy"], 1)) * 100, 1)
    return out


def plot_split_by_domain(split: str, per_domain: Dict, out_path: Path):
    """Grouped bar chart: one group per domain, bars per config (recall)."""
    domains = sorted(per_domain.keys())
    if not domains:
        return
    n_cfg = len(CONFIGS)
    width = 0.8 / n_cfg
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(domains)), 5))
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e"]
    for i, cfg in enumerate(CONFIGS):
        ys = [per_domain[d]["configs"].get(cfg, {}).get("recall_at_20_pct", 0)
              for d in domains]
        xs = [j + i * width for j in range(len(domains))]
        ax.bar(xs, ys, width=width, label=cfg, color=colors[i % len(colors)])
    ax.set_xticks([j + width * (n_cfg - 1) / 2 for j in range(len(domains))])
    ax.set_xticklabels([f"{d}\n(n={per_domain[d]['n']})" for d in domains],
                       rotation=0, fontsize=9)
    ax.set_ylabel("Recall@20 (%)")
    ax.set_title(f"{split} — Recall@20 by Mind2Web domain")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    out_root = RESULTS_DIR / "per_domain_breakdown"
    out_root.mkdir(parents=True, exist_ok=True)

    grand: Dict = {"splits": {}}

    for split, run_name in SPLIT_RUNS.items():
        run_dir = RESULTS_DIR / run_name
        if not (run_dir / "bench_results.json").exists():
            print(f"[skip] missing: {run_dir}")
            continue
        print(f"[load] {split} <- {run_name}")
        samples = load_samples(run_dir)
        print(f"  joining with parquet domain info ...")
        domain_map = load_domain_map(split)
        groups = group_by_domain(samples, domain_map)

        per_domain_agg: Dict = {}
        for dom, subset in groups.items():
            per_domain_agg[dom] = aggregate_subset(subset)

        grand["splits"][split] = {
            "n_total": len(samples),
            "by_domain": per_domain_agg,
        }

        # Plot
        plot_path = out_root / f"{split}_by_domain.png"
        plot_split_by_domain(split, per_domain_agg, plot_path)
        print(f"  -> plot: {plot_path}")

    # Save the full JSON
    out_json = out_root / "per_domain_breakdown.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(grand, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {out_json}")

    # Pretty summary
    print("\n" + "=" * 72)
    print("PER-DOMAIN BREAKDOWN — Recall@20 per Mind2Web domain")
    print("=" * 72)
    for split, s in grand["splits"].items():
        print(f"\n{split}  (total n={s['n_total']})")
        for dom, agg in sorted(s["by_domain"].items()):
            if "FULL" not in agg["configs"] or "DELTA_HYBRID" not in agg["configs"]:
                continue
            full = agg["configs"]["FULL"]
            hyb = agg["configs"]["DELTA_HYBRID"]
            print(f"  {dom:<20} n={agg['n']:<4}  "
                  f"FULL={full['recall_at_20_pct']:>6.2f}%  "
                  f"HYBRID={hyb['recall_at_20_pct']:>6.2f}%  "
                  f"Δ={hyb['delta_recall_pp']:>+5.2f}pp  "
                  f"tok_red={hyb['token_reduction_pct']:>5.1f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
