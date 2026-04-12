"""
evaluate_prune4web.py – Offline evaluation of the Prune4Web pipeline on Mind2Web.

Metrics (matching the paper, arXiv 2511.21398):
  Recall@20        : fraction of steps where the GT element is in top-20 candidates
  Element Accuracy : fraction of steps where the grounder picks the GT element
  Op Accuracy      : fraction of steps where both element AND action type are correct

Usage
-----
# Quick mode: programmatic filter only (no grounder LLM calls, very cheap)
python Prune4Web/evaluate_prune4web.py --num-samples 100 --no-grounder

# Full mode: filter + grounder (uses LLM for keyword generation AND grounding)
python Prune4Web/evaluate_prune4web.py --num-samples 50 --split test_task

# Specify dataset file explicitly
python Prune4Web/evaluate_prune4web.py --parquet test_task-00000-of-00005-431389419142b606.parquet --num-samples 200

Environment
-----------
OPENAI_API_KEY, OPENAI_MODEL (or defaults from .env) must be set.
DATASET_PATH in .env points to the multimodal-mind2web root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Bootstrap: load .env, add project root to sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
load_dotenv(Path(_PROJECT_ROOT) / ".env")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Re-use constants and scoring from run_prune4web.py
# ---------------------------------------------------------------------------
from Prune4Web.run_prune4web import (
    ALPHA_EXACT, ALPHA_PHRASE, ALPHA_WORD, ALPHA_FUZZY,
    BETA_VISIBLE_TEXT, BETA_ARIA_LABEL, BETA_PLACEHOLDER, BETA_ID_CLASS, BETA_OTHER,
    FILTER_SYSTEM, GROUNDER_SYSTEM, INTERACTIVE_TAGS, INTERACTIVE_ROLES,
    TOP_N_CANDIDATES, FUZZY_THRESHOLD,
    ElementNode, score_elements, _match_alpha,
    llm_call, parse_json_from_llm, USAGE_TRACKER,
)

# ---------------------------------------------------------------------------
# OpenAI client init (same as run_prune4web)
# ---------------------------------------------------------------------------
from openai import OpenAI
import Prune4Web.run_prune4web as _runner

FILTER_MODEL  = os.getenv("PRUNE4WEB_FILTER_MODEL",  "gpt-4o-mini")  # cheaper for eval
GROUNDER_MODEL = os.getenv("PRUNE4WEB_GROUNDER_MODEL", "gpt-4o-mini")

DATASET_PATH = os.getenv("DATASET_PATH", "D:/Environments/Datasets/multimodal-mind2web")

# ---------------------------------------------------------------------------
# Extended ElementNode that carries backend_node_id
# ---------------------------------------------------------------------------
@dataclass
class EvalElementNode(ElementNode):
    """ElementNode extended with backend_node_id for evaluation."""
    backend_node_id: str = ""


def parse_dom_eval(html: str) -> List[EvalElementNode]:
    """
    Like parse_dom() but also captures backend_node_id attribute.
    This is needed to match Prune4Web candidates against Mind2Web ground-truth.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "meta", "noscript", "head"]):
        tag.decompose()

    nodes: List[EvalElementNode] = []
    uid = 0
    for el in soup.find_all(True):
        tag_name = (el.name or "").lower()
        role = el.get("role", "").lower()
        is_interactive = (
            tag_name in INTERACTIVE_TAGS
            or role in INTERACTIVE_ROLES
            or el.get("onclick")
            or el.get("tabindex") not in (None, "-1", -1)
        )
        if not is_interactive:
            continue
        style = el.get("style", "").replace(" ", "")
        if "display:none" in style or "visibility:hidden" in style:
            continue

        cls = el.get("class", [])
        cls_str = cls if isinstance(cls, str) else " ".join(cls)

        nodes.append(
            EvalElementNode(
                uid=uid,
                tag=tag_name,
                text=el.get_text(separator=" ", strip=True)[:200],
                aria_label=el.get("aria-label", ""),
                placeholder=el.get("placeholder", ""),
                elem_id=el.get("id", ""),
                name=el.get("name", ""),
                elem_class=cls_str,
                href=el.get("href", ""),
                value=el.get("value", ""),
                input_type=el.get("type", ""),
                role=role,
                title=el.get("title", ""),
                backend_node_id=el.get("backend_node_id", ""),
            )
        )
        uid += 1
    return nodes


def parse_candidates_from_pool(html: str, pos_candidates, neg_candidates) -> List[EvalElementNode]:
    """
    Build the element pool from pos_candidates + neg_candidates (the pre-identified
    elements in Mind2Web dataset), looked up in the cleaned HTML by backend_node_id.

    This matches the paper's evaluation setup exactly: the filter stage operates on
    the pre-identified candidate pool, not the full re-parsed DOM.
    """
    soup = BeautifulSoup(html, "lxml")
    # Build a lookup: backend_node_id -> BeautifulSoup element
    bnode_map = {}
    for el in soup.find_all(attrs={"backend_node_id": True}):
        bnode_map[el.get("backend_node_id")] = el

    all_cand_ids: List[str] = []
    if pos_candidates is not None:
        for raw in pos_candidates:
            try:
                c = json.loads(raw) if isinstance(raw, str) else raw
                all_cand_ids.append(str(c.get("backend_node_id", "")))
            except Exception:
                pass
    if neg_candidates is not None:
        for raw in neg_candidates:
            try:
                c = json.loads(raw) if isinstance(raw, str) else raw
                all_cand_ids.append(str(c.get("backend_node_id", "")))
            except Exception:
                pass

    nodes: List[EvalElementNode] = []
    uid = 0
    seen: set = set()
    for node_id in all_cand_ids:
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)

        el = bnode_map.get(node_id)
        if el is None:
            # Element in candidate list but not found in HTML – add a stub
            nodes.append(EvalElementNode(
                uid=uid, tag="", text="", aria_label="", placeholder="",
                elem_id="", name="", elem_class="", href="", value="",
                input_type="", role="", title="", backend_node_id=node_id,
            ))
            uid += 1
            continue

        tag_name = (el.name or "").lower()
        role = el.get("role", "").lower()
        cls = el.get("class", [])
        cls_str = cls if isinstance(cls, str) else " ".join(cls)

        nodes.append(EvalElementNode(
            uid=uid,
            tag=tag_name,
            text=el.get_text(separator=" ", strip=True)[:200],
            aria_label=el.get("aria-label", el.get("aria_label", "")),
            placeholder=el.get("placeholder", ""),
            elem_id=el.get("id", ""),
            name=el.get("name", ""),
            elem_class=cls_str,
            href=el.get("href", ""),
            value=el.get("value", ""),
            input_type=el.get("type", ""),
            role=role,
            title=el.get("title", ""),
            backend_node_id=node_id,
        ))
        uid += 1

    return nodes


# ---------------------------------------------------------------------------
# Ground-truth extraction from Mind2Web rows
# ---------------------------------------------------------------------------
def get_gt_backend_node_id(row) -> Optional[str]:
    """Extract the ground-truth backend_node_id from pos_candidates."""
    pos = row.get("pos_candidates")
    if pos is None or (hasattr(pos, "__len__") and len(pos) == 0):
        return None
    try:
        cand = json.loads(pos[0])
        return cand.get("backend_node_id")
    except Exception:
        return None


def get_gt_action_type(row) -> str:
    """Extract ground-truth action type from operation column."""
    op = row.get("operation", "{}")
    if isinstance(op, str):
        try:
            op = json.loads(op)
        except Exception:
            return "click"
    op_type = op.get("op", op.get("original_op", "CLICK")).upper()
    mapping = {"CLICK": "click", "TYPE": "type", "SELECT": "select", "SCROLL": "scroll"}
    return mapping.get(op_type, "click")


def parse_target_action_repr(target_repr: str) -> Tuple[str, str]:
    """
    Parse Mind2Web target_action_reprs like:
      '[span]  Six Flags Magic Mountain -> CLICK'
    Returns (element_description, action_type)
    """
    if not target_repr:
        return "", "click"
    # Split on ' -> '
    parts = target_repr.rsplit(" -> ", 1)
    action = parts[1].strip().lower() if len(parts) > 1 else "click"
    desc = parts[0].strip()
    # Remove tag prefix like '[span]  '
    desc = re.sub(r"^\[.*?\]\s*", "", desc).strip()
    return desc, action


# ---------------------------------------------------------------------------
# Keyword generation
# ---------------------------------------------------------------------------
def keywords_from_text_heuristic(task: str, element_desc: str) -> Dict[str, float]:
    """
    Fast heuristic keyword extraction (no LLM cost).
    Splits task and element description into tokens and assigns weights.
    """
    # Element description keywords are most specific → high weight
    desc_tokens = re.split(r"[\s\-_/,\.]+", element_desc.lower())
    desc_tokens = [t for t in desc_tokens if len(t) > 2]

    # Task keywords (lower weight, broader context)
    task_tokens = re.split(r"[\s\-_/,\.]+", task.lower())
    task_tokens = [t for t in task_tokens if len(t) > 3 and t not in {
        "this", "that", "with", "from", "into", "click", "type", "select", "find",
        "search", "open", "page", "site", "website", "button", "link", "form",
    }]

    weights: Dict[str, float] = {}
    # Full element description as phrase → highest weight
    if element_desc.strip():
        weights[element_desc.lower().strip()] = 3.0
    for tok in desc_tokens:
        weights[tok] = max(weights.get(tok, 0.0), 2.0)
    for tok in task_tokens:
        weights[tok] = max(weights.get(tok, 0.0), 0.8)

    return weights


def keywords_from_llm(sub_task: str) -> Dict[str, float]:
    """Call LLM filter to generate keyword weights. One LLM call per sample."""
    messages = [
        {"role": "system", "content": FILTER_SYSTEM},
        {"role": "user", "content": f"Sub-task: {sub_task}"},
    ]
    raw = llm_call(messages=messages, stage="eval-filter", model=FILTER_MODEL, max_tokens=256)
    try:
        keywords = json.loads(raw)
    except Exception:
        keywords = parse_json_from_llm(raw)
    return {str(k): float(v) for k, v in keywords.items()}


# ---------------------------------------------------------------------------
# Grounder (offline version – no browser, pure LLM)
# ---------------------------------------------------------------------------
def grounder_llm(sub_task: str, candidates: List[EvalElementNode], action_value: str = "") -> Dict:
    """Call LLM grounder with top-20 candidates. Returns grounding result dict."""
    candidate_lines = [f"  {i + 1}. {el.to_summary()}" for i, el in enumerate(candidates)]
    user_msg = (
        f"Sub-task: {sub_task}\nValue hint: {action_value}\n\n"
        f"Candidate elements (ranked by relevance):\n" + "\n".join(candidate_lines)
    )
    messages = [
        {"role": "system", "content": GROUNDER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    raw = llm_call(messages=messages, stage="eval-grounder", model=GROUNDER_MODEL, max_tokens=256)
    try:
        return json.loads(raw)
    except Exception:
        return parse_json_from_llm(raw)


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------
@dataclass
class SampleResult:
    action_uid: str
    confirmed_task: str
    gt_backend_node_id: str
    gt_action: str
    dom_size: int
    num_candidates: int
    gt_in_top20: bool                # Recall@20
    gt_rank: int                     # rank of GT in scored list (-1 if not found)
    # Grounder results (only when --with-grounder)
    grounder_ran: bool = False
    grounder_uid: int = -1
    grounder_backend_node_id: str = ""
    grounder_action: str = ""
    element_correct: bool = False    # grounder picked right element
    op_correct: bool = False         # element AND action both correct


def evaluate_sample(
    row,
    use_llm_filter: bool,
    use_grounder: bool,
    top_n: int = TOP_N_CANDIDATES,
) -> Optional[SampleResult]:
    """Evaluate one Mind2Web sample. Returns None if sample should be skipped."""

    action_uid = str(row.get("action_uid", ""))
    confirmed_task = str(row.get("confirmed_task", ""))
    target_repr = str(row.get("target_action_reprs", ""))

    # Ground truth
    gt_node_id = get_gt_backend_node_id(row)
    if not gt_node_id:
        return None  # skip rows without GT

    gt_action = get_gt_action_type(row)
    element_desc, _ = parse_target_action_repr(target_repr)

    # Parse candidate pool (pos + neg candidates)  ← paper's approach
    html = row.get("cleaned_html", "") or row.get("raw_html", "") or ""
    if not html:
        return None

    all_elements = parse_candidates_from_pool(
        html,
        row.get("pos_candidates"),
        row.get("neg_candidates"),
    )
    if not all_elements:
        # Fallback: re-parse full DOM
        all_elements = parse_dom_eval(html)
    if not all_elements:
        return None

    dom_size = len(all_elements)

    # Build sub_task for keyword generation
    sub_task = element_desc if element_desc else confirmed_task

    # Generate keyword weights
    if use_llm_filter:
        try:
            kw = keywords_from_llm(sub_task)
        except Exception as e:
            print(f"    [warn] LLM filter failed: {e}, falling back to heuristic")
            kw = keywords_from_text_heuristic(confirmed_task, element_desc)
    else:
        kw = keywords_from_text_heuristic(confirmed_task, element_desc)

    # Score all elements → ranked list
    scored_all = _score_elements_ranked(all_elements, kw)

    # Check if GT is in top-N
    top_n_elements = scored_all[:top_n]
    top_n_node_ids = {el.backend_node_id for el in top_n_elements}
    gt_in_top20 = gt_node_id in top_n_node_ids

    # GT rank in full scored list
    gt_rank = -1
    for rank, el in enumerate(scored_all):
        if el.backend_node_id == gt_node_id:
            gt_rank = rank + 1  # 1-indexed
            break

    result = SampleResult(
        action_uid=action_uid,
        confirmed_task=confirmed_task,
        gt_backend_node_id=gt_node_id,
        gt_action=gt_action,
        dom_size=dom_size,
        num_candidates=len(top_n_elements),
        gt_in_top20=gt_in_top20,
        gt_rank=gt_rank,
    )

    # --- Grounder stage (optional) ---
    if use_grounder and gt_in_top20:
        try:
            ground_res = grounder_llm(sub_task, top_n_elements)
            grounded_uid = int(ground_res.get("element_uid", -1))
            grounded_action = str(ground_res.get("action", "click")).lower()

            grounded_el = next(
                (el for el in top_n_elements if el.uid == grounded_uid), None
            )
            grounded_node_id = grounded_el.backend_node_id if grounded_el else ""

            element_correct = grounded_node_id == gt_node_id
            op_correct = element_correct and (grounded_action == gt_action)

            result.grounder_ran = True
            result.grounder_uid = grounded_uid
            result.grounder_backend_node_id = grounded_node_id
            result.grounder_action = grounded_action
            result.element_correct = element_correct
            result.op_correct = op_correct
        except Exception as e:
            print(f"    [warn] Grounder failed: {e}")

    return result


def _score_elements_ranked(
    elements: List[EvalElementNode],
    keyword_weights: Dict[str, float],
) -> List[EvalElementNode]:
    """Score all elements and return them sorted high→low. Keeps ALL elements."""
    scores: Dict[int, float] = {}
    for element in elements:
        score = 0.0
        for attr_text, beta in element.attribute_tuples():
            for keyword, w_base in keyword_weights.items():
                alpha = _match_alpha(keyword, attr_text)
                if alpha > 0.0:
                    score += w_base * alpha * beta
        scores[element.uid] = score

    sorted_uids = sorted(scores, key=lambda uid: -scores[uid])
    uid_map = {el.uid: el for el in elements}
    return [uid_map[uid] for uid in sorted_uids]


# ---------------------------------------------------------------------------
# Load dataset rows
# ---------------------------------------------------------------------------
def load_rows(dataset_path: str, split: str, parquet_file: Optional[str], num_samples: int) -> pd.DataFrame:
    """Load Mind2Web rows for evaluation."""
    data_dir = Path(dataset_path) / "data"

    if parquet_file:
        paths = [data_dir / parquet_file]
    else:
        paths = sorted(data_dir.glob(f"{split}-*.parquet"))

    if not paths:
        raise FileNotFoundError(f"No parquet files found for split '{split}' in {data_dir}")

    dfs = []
    for p in paths:
        dfs.append(pd.read_parquet(str(p)))

    df = pd.concat(dfs, ignore_index=True)

    # Drop rows without ground truth HTML or candidates
    df = df[df["pos_candidates"].apply(lambda x: x is not None and len(x) > 0)]
    df = df[df["cleaned_html"].apply(lambda x: x is not None and len(str(x)) > 100)]

    if num_samples > 0 and num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
def compute_and_print_metrics(results: List[SampleResult], elapsed: float) -> Dict:
    n = len(results)
    if n == 0:
        print("No results to report.")
        return {}

    recall_at_20 = sum(1 for r in results if r.gt_in_top20) / n * 100
    avg_dom_size = sum(r.dom_size for r in results) / n
    avg_gt_rank = sum(r.gt_rank for r in results if r.gt_rank > 0) / max(1, sum(1 for r in results if r.gt_rank > 0))

    grounder_results = [r for r in results if r.grounder_ran]
    element_accuracy = (
        sum(1 for r in grounder_results if r.element_correct) / len(grounder_results) * 100
        if grounder_results else None
    )
    op_accuracy = (
        sum(1 for r in grounder_results if r.op_correct) / len(grounder_results) * 100
        if grounder_results else None
    )

    print("\n" + "=" * 72)
    print("PRUNE4WEB EVALUATION RESULTS")
    print("=" * 72)
    print(f"Samples evaluated      : {n}")
    print(f"Avg DOM size (elements): {avg_dom_size:.1f}")
    print(f"Top-N candidates       : {TOP_N_CANDIDATES}")
    print(f"Elapsed time           : {elapsed:.1f}s")
    print()
    print(f"{'Recall@20':<30}: {recall_at_20:.2f}%  (paper: 97.6%)")
    print(f"{'Avg GT rank in scored list':<30}: {avg_gt_rank:.1f}")

    if element_accuracy is not None:
        print(f"{'Element Accuracy':<30}: {element_accuracy:.2f}%  (paper: 88.28%)")
    else:
        print(f"{'Element Accuracy':<30}: N/A  (run with --with-grounder)")

    if op_accuracy is not None:
        print(f"{'Op Accuracy (EA + action)':<30}: {op_accuracy:.2f}%")

    # Rank distribution
    rank_bins = {"1": 0, "2-5": 0, "6-10": 0, "11-20": 0, "21+": 0, "not found": 0}
    for r in results:
        if r.gt_rank == -1:
            rank_bins["not found"] += 1
        elif r.gt_rank == 1:
            rank_bins["1"] += 1
        elif r.gt_rank <= 5:
            rank_bins["2-5"] += 1
        elif r.gt_rank <= 10:
            rank_bins["6-10"] += 1
        elif r.gt_rank <= 20:
            rank_bins["11-20"] += 1
        else:
            rank_bins["21+"] += 1

    print()
    print("GT element rank distribution:")
    for label, count in rank_bins.items():
        pct = count / n * 100
        bar = "#" * int(pct / 2)
        print(f"  rank {label:<10}: {count:>4} ({pct:>5.1f}%)  {bar}")

    print("=" * 72)

    return {
        "n": n,
        "recall_at_20": recall_at_20,
        "avg_dom_size": avg_dom_size,
        "avg_gt_rank": avg_gt_rank,
        "element_accuracy": element_accuracy,
        "op_accuracy": op_accuracy,
        "rank_distribution": rank_bins,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Prune4Web on Mind2Web benchmark")
    parser.add_argument("--split", default="test_task", help="Dataset split (test_task, test_domain, test_website, train)")
    parser.add_argument("--parquet", default=None, help="Specific parquet filename to use (e.g. test_task-00000-of-00005-*.parquet)")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate (0 = all)")
    parser.add_argument("--top-n", type=int, default=TOP_N_CANDIDATES, help="Top-N candidates for filter stage (default 20)")
    parser.add_argument("--no-llm-filter", action="store_true", help="Skip LLM filter, use heuristic keyword extraction instead")
    parser.add_argument("--with-grounder", action="store_true", help="Also run LLM grounder to compute Element Accuracy")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    parser.add_argument("--dataset-path", default=None, help="Override DATASET_PATH env var")
    args = parser.parse_args()

    dataset_path = args.dataset_path or DATASET_PATH
    top_n = args.top_n
    use_llm_filter = not args.no_llm_filter
    use_grounder = args.with_grounder

    # Init OpenAI client
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Set it in .env or environment.")
        sys.exit(1)

    _runner._CLIENT = OpenAI(api_key=api_key)

    print(f"Prune4Web Evaluation")
    print(f"  Dataset      : {dataset_path}")
    print(f"  Split        : {args.split}")
    print(f"  Samples      : {args.num_samples or 'ALL'}")
    print(f"  LLM filter   : {'YES (' + FILTER_MODEL + ')' if use_llm_filter else 'NO (heuristic)'}")
    print(f"  Grounder     : {'YES (' + GROUNDER_MODEL + ')' if use_grounder else 'NO'}")
    print(f"  Top-N        : {top_n}")
    print()

    # Load dataset
    print("Loading dataset...")
    df = load_rows(dataset_path, args.split, args.parquet, args.num_samples)
    print(f"Loaded {len(df)} samples.\n")

    # Evaluate
    results: List[SampleResult] = []
    t_start = time.time()
    skipped = 0

    for i, (_, row) in enumerate(df.iterrows()):
        print(f"[{i+1:>4}/{len(df)}] task={str(row.get('confirmed_task',''))[:60]!r}", end=" ", flush=True)

        try:
            result = evaluate_sample(row, use_llm_filter, use_grounder, top_n)
        except Exception as e:
            print(f"ERROR: {e}")
            skipped += 1
            continue

        if result is None:
            print("skip (no GT)")
            skipped += 1
            continue

        status = f"dom={result.dom_size} rank={result.gt_rank}"
        if result.gt_in_top20:
            status += " [IN TOP20]"
        else:
            status += " [MISSED]"
        if result.grounder_ran:
            status += " EA=" + ("OK" if result.element_correct else "X")
        print(status)

        results.append(result)

    elapsed = time.time() - t_start

    if skipped:
        print(f"\n(Skipped {skipped} samples due to missing data or errors)")

    # Print metrics
    metrics = compute_and_print_metrics(results, elapsed)

    # Usage summary
    USAGE_TRACKER.print_summary()

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "config": {
                "split": args.split,
                "num_samples": len(results),
                "top_n": top_n,
                "use_llm_filter": use_llm_filter,
                "use_grounder": use_grounder,
                "filter_model": FILTER_MODEL,
                "grounder_model": GROUNDER_MODEL,
            },
            "metrics": metrics,
            "samples": [
                {
                    "action_uid": r.action_uid,
                    "task": r.confirmed_task,
                    "gt_backend_node_id": r.gt_backend_node_id,
                    "gt_action": r.gt_action,
                    "dom_size": r.dom_size,
                    "gt_rank": r.gt_rank,
                    "gt_in_top20": r.gt_in_top20,
                    "grounder_ran": r.grounder_ran,
                    "element_correct": r.element_correct,
                    "op_correct": r.op_correct,
                }
                for r in results
            ],
        }
        with open(str(out_path), "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
