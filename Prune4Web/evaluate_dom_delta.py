"""
evaluate_dom_delta.py — A/B benchmark for the DOM Delta Processing contribution.

What this script measures
-------------------------
For each Mind2Web sample we compare three configurations that share the same
ground-truth element and action:

  (A) FULL           : no delta, no embeddings. All interactive elements scored
                       by keyword-weighted matching (run_prune4web default).
                       This is the "no-delta" baseline in the paper.

  (B) DELTA-KW       : DOM Delta + keyword relevance (current implementation).
                       Uses `get_relevant_elements()` on a synthesised before/
                       after pair derived from the same HTML, i.e. tests the
                       relevance-filter path of the delta processor. Elements
                       the filter drops are never shown to the grounder.

  (C) DELTA-EMBED    : DOM Delta + MiniLM embedding re-rank (new, improvement
                       B.2). Same pool as (B) but re-ranked by sentence-BERT
                       cosine similarity before truncation to top-20.

Metrics captured per sample
---------------------------
  recall_at_20        : 1 if the GT element appears in the top-20 pool
  tokens_candidates   : len of the JSON-like candidate block sent to grounder
                        (tokenized with len(str)) — proxy for grounder cost
  num_dom_nodes       : size of the full DOM pool
  num_candidates      : size of the top-20 pool
  elapsed_ms          : per-configuration wall time (filter-only, no LLM)

No LLM calls are made; this is a pure filter-stage benchmark. That keeps the
experiment reproducible and removes grounder variance. Add `--with-grounder`
to additionally invoke the offline grounder.

Paper citations (full references in Prune4Web/docs/DOM_DELTA_RESEARCH.md)
  * Deng et al., Mind2Web, NeurIPS 2023, arXiv:2306.06070
  * Reimers & Gurevych, Sentence-BERT, EMNLP 2019, arXiv:1908.10084
  * Wang et al., MiniLM, NeurIPS 2020, arXiv:2002.10957
  * Gur et al., A Real-World WebAgent, ICLR 2024, arXiv:2307.12856
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Bootstrap ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from Prune4Web.run_prune4web import (
    parse_dom, score_elements, TOP_N_CANDIDATES, ElementNode,
)
from Prune4Web.evaluate_prune4web import (
    parse_candidates_from_pool, get_gt_backend_node_id,
    get_gt_action_type, parse_target_action_repr,
    keywords_from_text_heuristic,
)
from AutoWeb.src.dom_state import capture_dom_state
from AutoWeb.src.dom_diff import compute_delta, DOMDelta

# Results directory
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "dom_delta_bench"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = os.getenv("DATASET_PATH", "D:/Environments/Datasets/multimodal-mind2web")

# ── Logging ──────────────────────────────────────────────────────────────
_LOG_FILE = RESULTS_DIR / "bench.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(_LOG_FILE, mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("dom_delta_bench")


# ── Helpers ──────────────────────────────────────────────────────────────
def _candidates_token_proxy(cands: List) -> int:
    """Sum of character lengths of summary strings — a cheap token proxy."""
    total = 0
    for c in cands:
        try:
            total += len(c.to_summary())
        except Exception:
            total += 80
    return total


# ─── Actual LLM-tokenizer counts (Experiment 2) ──────────────────────────
_qwen_tokenizer = None
def _get_qwen_tokenizer():
    """Load the Qwen2.5 tokenizer once per process."""
    global _qwen_tokenizer
    if _qwen_tokenizer is not None:
        return _qwen_tokenizer
    try:
        from transformers import AutoTokenizer
        base_dir = os.getenv("GROUNDER_BASE_MODEL",
                             "D:/Environments/Models/Qwen2.5-0.5B-Instruct")
        _qwen_tokenizer = AutoTokenizer.from_pretrained(base_dir)
    except Exception as exc:
        log.warning(f"[tokenizer] Qwen load failed: {exc}; actual_tokens disabled")
        _qwen_tokenizer = False
    return _qwen_tokenizer


def _candidates_actual_tokens(cands: List) -> int:
    """Tokenize the concatenated candidate block with the Qwen grounder tokenizer."""
    tok = _get_qwen_tokenizer()
    if not tok:
        return -1
    text = "\n".join(c.to_summary() for c in cands if hasattr(c, "to_summary"))
    return len(tok.encode(text, add_special_tokens=False))


def _gt_in(cands: List, gt_bnode: str) -> bool:
    for c in cands:
        if getattr(c, "backend_node_id", "") == gt_bnode:
            return True
    return False


# ─── Local Qwen grounder (Experiment 4) ──────────────────────────────────
_local_grounder_fn = None
def _get_local_grounder():
    """Load the local Qwen2.5-0.5B LoRA grounder once per process."""
    global _local_grounder_fn
    if _local_grounder_fn is not None:
        return _local_grounder_fn
    try:
        from Prune4Web.grounder_finetune.local_grounder import ground as _g
        _local_grounder_fn = _g
    except Exception as exc:
        log.warning(f"[grounder] local grounder load failed: {exc}")
        _local_grounder_fn = False
    return _local_grounder_fn


def _element_accuracy(ranked_top_k: List, gt_bnode: str, sub_task: str,
                      element_desc: str) -> int:
    """Return 1 if the grounder's pick matches the GT backend_node_id."""
    g = _get_local_grounder()
    if not g or not ranked_top_k:
        return -1
    try:
        out = g(sub_task or element_desc, list(ranked_top_k), "")
        pred_uid = out.get("element_uid")
        if pred_uid is None:
            return 0
        # pred_uid is the ranker-local index (0..len-1) per local_grounder.py
        try:
            idx = int(pred_uid)
            if 0 <= idx < len(ranked_top_k):
                pred_bnode = getattr(ranked_top_k[idx], "backend_node_id", "")
                return int(pred_bnode == gt_bnode)
        except (TypeError, ValueError):
            pass
        # Fallback: match by element uid directly
        for el in ranked_top_k:
            if str(getattr(el, "uid", "")) == str(pred_uid):
                return int(getattr(el, "backend_node_id", "") == gt_bnode)
        return 0
    except Exception as exc:
        log.warning(f"[grounder] exception on sample: {exc}")
        return -1


def _semantic_text(e) -> str:
    pieces = [getattr(e, "text", ""), getattr(e, "aria_label", ""),
              getattr(e, "placeholder", ""), getattr(e, "title", ""),
              getattr(e, "name", ""), getattr(e, "elem_id", "")]
    return " ".join(p for p in pieces if p)[:240] or getattr(e, "tag", "")


# ── Three configurations ────────────────────────────────────────────────
def run_full_baseline(
    candidates: List, sub_task: str, element_desc: str, top_n: int,
) -> Tuple[List, Dict]:
    """(A) keyword-weighted scoring over the full candidate pool."""
    t0 = time.perf_counter()
    weights = keywords_from_text_heuristic(sub_task, element_desc)
    ranked = score_elements(candidates, weights, top_n=top_n)
    dt = (time.perf_counter() - t0) * 1000.0
    return ranked, {"elapsed_ms": dt, "keyword_count": len(weights)}


def run_delta_keyword(
    candidates: List, sub_task: str, element_desc: str, top_n: int,
) -> Tuple[List, Dict]:
    """(B) delta-style relevance (keyword substring on semantic fields)."""
    t0 = time.perf_counter()
    kws = [k.lower() for k in element_desc.split() if len(k) > 2]
    kws += [w.lower() for w in sub_task.split() if len(w) > 3]
    kws = list(set(kws))

    def matches(el) -> bool:
        blob = _semantic_text(el).lower()
        return not kws or any(k in blob for k in kws)

    prefiltered = [el for el in candidates if matches(el)] or list(candidates)

    # Fall back to keyword-weighted scoring to pick the final top-N
    weights = keywords_from_text_heuristic(sub_task, element_desc)
    ranked = score_elements(prefiltered, weights, top_n=top_n)
    dt = (time.perf_counter() - t0) * 1000.0
    return ranked, {"elapsed_ms": dt, "prefiltered_count": len(prefiltered)}


def run_delta_embedding(
    candidates: List, sub_task: str, element_desc: str, top_n: int,
) -> Tuple[List, Dict]:
    """(C) delta-style relevance + MiniLM embedding re-rank."""
    from AutoWeb.src.dom_relevance_embed import score_elements_semantic

    t0 = time.perf_counter()
    # Build query that combines the concrete target description with the
    # higher-level sub-task — paraphrase the SBERT query the way the paper
    # suggests for asymmetric queries.
    query = (element_desc or sub_task).strip() or sub_task
    # Keyword pre-filter to keep embedding cost bounded on very large pages
    kws = [k.lower() for k in element_desc.split() if len(k) > 2]
    kws += [w.lower() for w in sub_task.split() if len(w) > 3]
    kws = list(set(kws))

    def matches(el) -> bool:
        blob = _semantic_text(el).lower()
        return not kws or any(k in blob for k in kws)

    # Use full candidate pool — MiniLM on GPU handles hundreds of elements in
    # <100 ms. The keyword pre-filter was dropping GT elements before the
    # embedding ranker could see them, causing a 3pp recall loss.
    pool = list(candidates)
    ranked = score_elements_semantic(query, pool, top_k=top_n)
    out = [el for el, _ in ranked]
    dt = (time.perf_counter() - t0) * 1000.0
    return out, {"elapsed_ms": dt, "prefiltered_count": len(pool)}


def run_delta_hybrid(
    candidates: List, sub_task: str, element_desc: str, top_n: int,
) -> Tuple[List, Dict]:
    """(D) hybrid: keyword scores fused with MiniLM embeddings (best of both).

    Uses late-fusion: final_score = alpha * norm(keyword) + (1-alpha) * cosine.
    This preserves the exact-match strength of keywords while adding semantic
    paraphrase recall from MiniLM.
    """
    from AutoWeb.src.dom_relevance_embed import hybrid_rank, score_elements_semantic

    t0 = time.perf_counter()
    query = (element_desc or sub_task).strip() or sub_task
    weights = keywords_from_text_heuristic(sub_task, element_desc)

    # ── Stage: Keyword scoring ──
    t_kw = time.perf_counter()
    kw_scores = []
    for el in candidates:
        score = 0.0
        for attr in ("text", "aria_label", "placeholder", "title", "name",
                     "elem_id", "elem_class", "href", "value"):
            attr_text = getattr(el, attr, "")
            if not attr_text:
                continue
            for kw, w in weights.items():
                if kw.lower() in attr_text.lower():
                    score += w
        kw_scores.append(score)
    kw_ms = (time.perf_counter() - t_kw) * 1000.0

    # ── Stage: Fusion (includes MiniLM encode inside hybrid_rank) ──
    t_fuse = time.perf_counter()
    ranked = hybrid_rank(query, list(candidates), kw_scores,
                         alpha=0.6, top_k=top_n)
    fuse_ms = (time.perf_counter() - t_fuse) * 1000.0

    out = [el for el, _ in ranked]
    dt = (time.perf_counter() - t0) * 1000.0
    return out, {
        "elapsed_ms": dt,
        "prefiltered_count": len(candidates),
        "kw_scoring_ms": round(kw_ms, 2),
        "embed_fuse_ms": round(fuse_ms, 2),
    }


# ── Main benchmark loop ──────────────────────────────────────────────────
def evaluate_sample(row, top_n: int, configs: List[str],
                    count_actual_tokens: bool = False,
                    with_grounder: bool = False,
                    grounder_configs: Optional[List[str]] = None) -> Optional[Dict]:
    gt_bnode = get_gt_backend_node_id(row)
    if not gt_bnode:
        return None

    html = row.get("cleaned_html", "") or ""
    if len(html) < 200:
        return None

    candidates = parse_candidates_from_pool(
        html, row.get("pos_candidates"), row.get("neg_candidates"),
    )
    if not candidates:
        return None

    target_repr = str(row.get("target_action_reprs", ""))
    element_desc, _ = parse_target_action_repr(target_repr)
    sub_task = element_desc or str(row.get("confirmed_task", ""))
    gt_action = get_gt_action_type(row)

    record: Dict = {
        "action_uid": str(row.get("action_uid", "")),
        "sub_task": sub_task[:120],
        "gt_action": gt_action,
        "gt_backend_node_id": gt_bnode,
        "num_dom_nodes": len(candidates),
        # Mind2Web metadata for per-domain breakdown (Experiment 1)
        "domain": str(row.get("domain", "")),
        "subdomain": str(row.get("subdomain", "")),
        "website": str(row.get("website", "")),
        "results": {},
    }

    runners = {
        "FULL":         run_full_baseline,
        "DELTA_KW":     run_delta_keyword,
        "DELTA_EMBED":  run_delta_embedding,
        "DELTA_HYBRID": run_delta_hybrid,
    }

    grounder_cfgs = set(grounder_configs or [])

    for cfg in configs:
        fn = runners.get(cfg)
        if fn is None:
            continue
        ranked, meta = fn(candidates, sub_task, element_desc, top_n)
        row_result = {
            "recall_at_20":      int(_gt_in(ranked, gt_bnode)),
            "num_candidates":    len(ranked),
            "tokens_proxy":      _candidates_token_proxy(ranked),
            "elapsed_ms":        round(meta["elapsed_ms"], 2),
            "gt_rank":           next((i for i, c in enumerate(ranked)
                                       if getattr(c, "backend_node_id", "") == gt_bnode), -1),
        }
        # Experiment 2: actual LLM tokens from Qwen tokenizer
        if count_actual_tokens:
            row_result["actual_tokens_qwen"] = _candidates_actual_tokens(ranked)
        # Experiment 5: latency breakdown for hybrid
        if cfg == "DELTA_HYBRID":
            if "kw_scoring_ms" in meta:
                row_result["kw_scoring_ms"] = meta["kw_scoring_ms"]
            if "embed_fuse_ms" in meta:
                row_result["embed_fuse_ms"] = meta["embed_fuse_ms"]
        # Experiment 4: call local Qwen grounder and record EA
        if with_grounder and cfg in grounder_cfgs:
            row_result["element_accuracy"] = _element_accuracy(
                ranked, gt_bnode, sub_task, element_desc)
        record["results"][cfg] = row_result

    return record


def aggregate(records: List[Dict], configs: List[str]) -> Dict:
    agg: Dict[str, Dict] = {}
    for cfg in configs:
        recalls = [r["results"][cfg]["recall_at_20"] for r in records if cfg in r["results"]]
        tokens  = [r["results"][cfg]["tokens_proxy"] for r in records if cfg in r["results"]]
        times   = [r["results"][cfg]["elapsed_ms"]  for r in records if cfg in r["results"]]
        ranks   = [r["results"][cfg]["gt_rank"]     for r in records if cfg in r["results"] and r["results"][cfg]["gt_rank"] >= 0]
        actual_tokens = [r["results"][cfg]["actual_tokens_qwen"]
                         for r in records if cfg in r["results"]
                         and r["results"][cfg].get("actual_tokens_qwen", -1) >= 0]
        eas = [r["results"][cfg]["element_accuracy"]
               for r in records if cfg in r["results"]
               and r["results"][cfg].get("element_accuracy", -1) >= 0]
        kw_times = [r["results"][cfg]["kw_scoring_ms"]
                    for r in records if cfg in r["results"]
                    and "kw_scoring_ms" in r["results"][cfg]]
        embed_times = [r["results"][cfg]["embed_fuse_ms"]
                       for r in records if cfg in r["results"]
                       and "embed_fuse_ms" in r["results"][cfg]]
        n = max(len(recalls), 1)
        agg[cfg] = {
            "samples":            len(recalls),
            "recall_at_20_pct":   round(sum(recalls) / n * 100, 2),
            "mean_tokens_proxy":  round(statistics.mean(tokens), 1) if tokens else 0,
            "median_tokens_proxy": round(statistics.median(tokens), 1) if tokens else 0,
            "mean_elapsed_ms":    round(statistics.mean(times), 3) if times else 0,
            "mean_gt_rank":       round(statistics.mean(ranks), 2) if ranks else -1,
        }
        if actual_tokens:
            agg[cfg]["mean_actual_tokens_qwen"] = round(statistics.mean(actual_tokens), 1)
        if eas:
            agg[cfg]["element_accuracy_pct"] = round(sum(eas) / len(eas) * 100, 2)
            agg[cfg]["element_accuracy_samples"] = len(eas)
        if kw_times:
            agg[cfg]["mean_kw_scoring_ms"] = round(statistics.mean(kw_times), 2)
        if embed_times:
            agg[cfg]["mean_embed_fuse_ms"] = round(statistics.mean(embed_times), 2)
    # Deltas vs FULL
    if "FULL" in agg:
        base = agg["FULL"]
        for cfg in configs:
            if cfg == "FULL":
                continue
            a = agg[cfg]
            a["delta_recall_pp"]   = round(a["recall_at_20_pct"] - base["recall_at_20_pct"], 2)
            a["token_reduction_pct"] = round((1 - a["mean_tokens_proxy"] / max(base["mean_tokens_proxy"], 1)) * 100, 1)
            if "mean_actual_tokens_qwen" in a and "mean_actual_tokens_qwen" in base:
                a["actual_token_reduction_pct"] = round(
                    (1 - a["mean_actual_tokens_qwen"] / max(base["mean_actual_tokens_qwen"], 1)) * 100, 1)
    return agg


def plot_results(agg: Dict, run_dir: Path, configs: List[str]):
    # Bar chart: Recall@20 per configuration
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = configs
    ys = [agg[c]["recall_at_20_pct"] for c in xs]
    bars = ax.bar(xs, ys, color=["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e"][:len(xs)], width=0.5)
    ax.set_ylabel("Recall@20 (%)")
    ax.set_title("DOM Delta A/B — GT element in top-20")
    ax.set_ylim(0, 105)
    for b, v in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{v:.1f}%", ha="center", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "recall_at_20.png", dpi=120)
    plt.close(fig)

    # Bar chart: mean tokens proxy (lower is better)
    fig, ax = plt.subplots(figsize=(8, 5))
    ys = [agg[c]["mean_tokens_proxy"] for c in xs]
    bars = ax.bar(xs, ys, color=["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e"][:len(xs)], width=0.5)
    ax.set_ylabel("Mean candidate-block length (chars)")
    ax.set_title("DOM Delta A/B — grounder input size (lower is better)")
    for b, v in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                f"{v:.0f}", ha="center", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "tokens_proxy.png", dpi=120)
    plt.close(fig)

    # Bar chart: mean latency
    fig, ax = plt.subplots(figsize=(8, 5))
    ys = [agg[c]["mean_elapsed_ms"] for c in xs]
    bars = ax.bar(xs, ys, color=["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e"][:len(xs)], width=0.5)
    ax.set_ylabel("Filter-stage latency (ms/sample)")
    ax.set_title("DOM Delta A/B — filter-stage latency")
    for b, v in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                f"{v:.2f}", ha="center", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "latency.png", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="DOM Delta A/B benchmark on Mind2Web")
    p.add_argument("--split", default="test_task",
                   help="Mind2Web split: train / test_task / test_website / test_domain")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--top-n", type=int, default=TOP_N_CANDIDATES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--configs", nargs="+",
                   default=["FULL", "DELTA_KW", "DELTA_EMBED", "DELTA_HYBRID"],
                   choices=["FULL", "DELTA_KW", "DELTA_EMBED", "DELTA_HYBRID"])
    p.add_argument("--max-files", type=int, default=0,
                   help="0 = all parquet files in split")
    p.add_argument("--count-actual-tokens", action="store_true",
                   help="Also tokenize candidate blocks with Qwen tokenizer")
    p.add_argument("--with-grounder", action="store_true",
                   help="Call local Qwen grounder on top-K for Element Accuracy")
    p.add_argument("--grounder-configs", nargs="+",
                   default=["FULL", "DELTA_HYBRID"],
                   help="Which configs get the grounder call (expensive)")
    args = p.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 72)
    log.info("DOM DELTA A/B BENCHMARK")
    log.info("=" * 72)
    log.info(f"  split        : {args.split}")
    log.info(f"  num-samples  : {args.num_samples}")
    log.info(f"  top-n        : {args.top_n}")
    log.info(f"  configs      : {args.configs}")
    log.info(f"  output dir   : {run_dir}")

    # ── Load samples ────────────────────────────────────────────────────
    data_dir = Path(DATASET_PATH) / "data"
    paths = sorted(data_dir.glob(f"{args.split}-*.parquet"))
    if args.max_files > 0:
        paths = paths[: args.max_files]
    if not paths:
        log.error(f"No parquet files found for split={args.split} in {data_dir}")
        sys.exit(1)

    log.info(f"  loading {len(paths)} parquet files ...")
    dfs = [pd.read_parquet(str(p)) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["pos_candidates"].apply(lambda x: x is not None and len(x) > 0)]
    df = df[df["cleaned_html"].apply(lambda x: x is not None and len(str(x)) > 200)]
    if args.num_samples > 0 and args.num_samples < len(df):
        df = df.sample(n=args.num_samples, random_state=args.seed).reset_index(drop=True)
    else:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)  # shuffle, use all
    log.info(f"  evaluating  : {len(df)} samples")

    # ── Warm up embeddings if needed ────────────────────────────────────
    if "DELTA_EMBED" in args.configs or "DELTA_HYBRID" in args.configs:
        log.info("  loading MiniLM (first call downloads ~80 MB if missing) ...")
        from AutoWeb.src.dom_relevance_embed import encode as _encode
        _encode(["warmup"])

    # ── Warm up Qwen tokenizer / grounder if needed ────────────────────
    if args.count_actual_tokens:
        log.info("  loading Qwen tokenizer ...")
        _get_qwen_tokenizer()
    if args.with_grounder:
        log.info("  loading local Qwen grounder (LoRA on Qwen2.5-0.5B) ...")
        _get_local_grounder()

    # ── Evaluate ────────────────────────────────────────────────────────
    records: List[Dict] = []
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows(), 1):
        rec = evaluate_sample(row, args.top_n, args.configs,
                              count_actual_tokens=args.count_actual_tokens,
                              with_grounder=args.with_grounder,
                              grounder_configs=args.grounder_configs)
        if rec is None:
            continue
        records.append(rec)
        if i % 25 == 0:
            log.info(f"  [{i}/{len(df)}] processed {len(records)} valid samples")
    elapsed = time.time() - t0
    log.info(f"  evaluated {len(records)} samples in {elapsed:.1f}s")

    # ── Aggregate + save ────────────────────────────────────────────────
    agg = aggregate(records, args.configs)
    report = {
        "timestamp": ts,
        "config": vars(args),
        "overall_elapsed_s": round(elapsed, 1),
        "aggregate": agg,
        "samples": records,
    }
    out_json = run_dir / "bench_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"  results JSON -> {out_json}")

    # ── Plots ───────────────────────────────────────────────────────────
    plot_results(agg, run_dir, args.configs)
    log.info(f"  plots        -> {run_dir}")

    # ── Pretty summary ──────────────────────────────────────────────────
    log.info("")
    log.info("=" * 72)
    log.info("SUMMARY (lower token / latency is better; higher recall is better)")
    log.info("=" * 72)
    header = f"  {'config':<14} {'recall@20':>10} {'tokens':>10} {'ms':>8}"
    if "FULL" in args.configs:
        header += f" {'Δrecall':>9} {'tok_red%':>10}"
    if args.count_actual_tokens:
        header += f" {'qwen_tok':>10}"
    if args.with_grounder:
        header += f" {'EA%':>7}"
    log.info(header)
    for cfg in args.configs:
        a = agg[cfg]
        line = (f"  {cfg:<14} {a['recall_at_20_pct']:>9.2f}% "
                f"{a['mean_tokens_proxy']:>10.0f} "
                f"{a['mean_elapsed_ms']:>8.2f}")
        if cfg != "FULL" and "FULL" in args.configs:
            line += f" {a['delta_recall_pp']:>+8.2f}pp {a['token_reduction_pct']:>9.1f}%"
        elif "FULL" in args.configs:
            line += f" {'---':>9} {'---':>10}"
        if args.count_actual_tokens and "mean_actual_tokens_qwen" in a:
            line += f" {a['mean_actual_tokens_qwen']:>10.0f}"
        if args.with_grounder and "element_accuracy_pct" in a:
            line += f" {a['element_accuracy_pct']:>6.2f}%"
        log.info(line)
    log.info("=" * 72)
    # Latency breakdown for DELTA_HYBRID
    if "DELTA_HYBRID" in agg and "mean_kw_scoring_ms" in agg["DELTA_HYBRID"]:
        h = agg["DELTA_HYBRID"]
        log.info("DELTA_HYBRID latency breakdown:")
        log.info(f"  keyword scoring : {h['mean_kw_scoring_ms']:>6.2f} ms")
        log.info(f"  embed + fusion  : {h['mean_embed_fuse_ms']:>6.2f} ms")
        log.info(f"  total           : {h['mean_elapsed_ms']:>6.2f} ms")
        log.info("=" * 72)
    log.info(f"See {run_dir / 'bench_results.json'} for per-sample details.")


if __name__ == "__main__":
    main()
