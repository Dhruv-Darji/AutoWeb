"""
01_generate_data.py — Training data generator for Prune4Web grounder fine-tuning.

Strategy
--------
For each Mind2Web row we construct a single grounder training example:
  Input : sub_task (from target_action_reprs) + 20 candidate elements
          (1 GT from pos_candidates + 19 random negatives from neg_candidates)
  Output: {"element_uid": <int>, "action": ..., "value": ..., "confidence": 0.95,
           "reasoning": ...}

The 20 candidates are shuffled so the model can't memorise a positional prior —
it must ground semantically.

Output format: ChatML-style JSONL that TRL SFTTrainer can read directly.
One line per example, each with a "messages" field:
  {"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}

All activity goes to logs/data_generation.log and stdout.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# -------------------- Bootstrap --------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env")
sys.path.insert(0, str(_PROJECT_ROOT))

_FINETUNE_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = _FINETUNE_ROOT / "logs"
DATA_DIR = _FINETUNE_ROOT / "data"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Logging --------------------
LOG_FILE = LOGS_DIR / "data_generation.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("data_gen")

# -------------------- Imports from Prune4Web --------------------
from Prune4Web.run_prune4web import GROUNDER_SYSTEM  # same system prompt as eval grounder
from Prune4Web.evaluate_prune4web import (
    EvalElementNode,
    parse_candidates_from_pool,
    get_gt_backend_node_id,
    get_gt_action_type,
    parse_target_action_repr,
)

DATASET_PATH = os.getenv("DATASET_PATH", "D:/Environments/Datasets/multimodal-mind2web")

# -------------------- Helpers --------------------
def build_candidate_window(
    all_elements: List[EvalElementNode],
    gt_node_id: str,
    window_size: int = 20,
    rng: Optional[random.Random] = None,
) -> Tuple[List[EvalElementNode], int]:
    """
    Build a shuffled list of `window_size` candidates that always includes GT.
    Returns (shuffled_elements, gt_index_in_window).
    """
    rng = rng or random.Random()

    gt_el = next((e for e in all_elements if e.backend_node_id == gt_node_id), None)
    if gt_el is None:
        return [], -1

    neg_pool = [e for e in all_elements if e.backend_node_id != gt_node_id]

    if len(neg_pool) <= window_size - 1:
        negs = neg_pool
    else:
        negs = rng.sample(neg_pool, window_size - 1)

    window = [gt_el] + negs
    rng.shuffle(window)

    # Re-assign contiguous uids 0..len-1 so the model sees clean indices
    for i, el in enumerate(window):
        el.uid = i

    gt_index = next(i for i, el in enumerate(window) if el.backend_node_id == gt_node_id)
    return window, gt_index


def format_user_message(sub_task: str, candidates: List[EvalElementNode], value_hint: str = "") -> str:
    """Mirror the exact format used by action_grounder() at inference time."""
    candidate_lines = [f"  {i + 1}. {el.to_summary()}" for i, el in enumerate(candidates)]
    return (
        f"Sub-task: {sub_task}\nValue hint: {value_hint}\n\n"
        f"Candidate elements (ranked by relevance):\n" + "\n".join(candidate_lines)
    )


def format_assistant_message(gt_uid: int, action: str, value: str, element_desc: str) -> str:
    """Produce the JSON the grounder should emit."""
    payload = {
        "element_uid": int(gt_uid),
        "action": action,
        "value": value,
        "confidence": 0.95,
        "reasoning": f"Element {gt_uid} matches the sub-task target: {element_desc[:80]}",
    }
    return json.dumps(payload, ensure_ascii=False)


def value_from_operation(operation_raw) -> str:
    """Extract the TYPE/SELECT value if present."""
    if operation_raw is None:
        return ""
    try:
        op = json.loads(operation_raw) if isinstance(operation_raw, str) else operation_raw
    except Exception:
        return ""
    return str(op.get("value") or op.get("action_input") or "").strip()


def row_to_training_example(
    row,
    rng: random.Random,
    window_size: int = 20,
) -> Optional[Dict]:
    gt_node_id = get_gt_backend_node_id(row)
    if not gt_node_id:
        return None

    html = row.get("cleaned_html", "") or ""
    if not html:
        return None

    all_elements = parse_candidates_from_pool(
        html, row.get("pos_candidates"), row.get("neg_candidates")
    )
    if not all_elements:
        return None

    window, gt_index = build_candidate_window(all_elements, gt_node_id, window_size, rng)
    if not window or gt_index < 0:
        return None

    # Sub-task: use target_action_reprs if present, else confirmed_task
    target_repr = str(row.get("target_action_reprs", ""))
    element_desc, repr_action = parse_target_action_repr(target_repr)
    sub_task = element_desc or str(row.get("confirmed_task", ""))

    action = get_gt_action_type(row) or repr_action or "click"
    value = value_from_operation(row.get("operation"))

    user_msg = format_user_message(sub_task, window, value)
    assistant_msg = format_assistant_message(
        gt_uid=gt_index,
        action=action,
        value=value,
        element_desc=element_desc,
    )

    return {
        "messages": [
            {"role": "system", "content": GROUNDER_SYSTEM},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "_meta": {
            "action_uid": str(row.get("action_uid", "")),
            "gt_backend_node_id": gt_node_id,
            "gt_index": gt_index,
            "action": action,
            "window_size": len(window),
            "dom_size": len(all_elements),
        },
    }


# -------------------- Main --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train")
    p.add_argument("--max-files", type=int, default=4, help="How many parquet files to load (0 = all)")
    p.add_argument("--max-samples", type=int, default=5000, help="Cap total training examples")
    p.add_argument("--window-size", type=int, default=20)
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-train", default=str(DATA_DIR / "train.jsonl"))
    p.add_argument("--out-eval", default=str(DATA_DIR / "eval.jsonl"))
    args = p.parse_args()

    rng = random.Random(args.seed)

    log.info("=" * 72)
    log.info("Prune4Web grounder — training-data generation")
    log.info("=" * 72)
    log.info(f"Dataset path   : {DATASET_PATH}")
    log.info(f"Split          : {args.split}")
    log.info(f"Max files      : {args.max_files}")
    log.info(f"Max samples    : {args.max_samples}")
    log.info(f"Window size    : {args.window_size}")
    log.info(f"Eval fraction  : {args.eval_frac}")
    log.info(f"Seed           : {args.seed}")
    log.info(f"Log file       : {LOG_FILE}")

    # Load parquet files
    data_dir = Path(DATASET_PATH) / "data"
    paths = sorted(data_dir.glob(f"{args.split}-*.parquet"))
    if args.max_files > 0:
        paths = paths[: args.max_files]
    log.info(f"Loading {len(paths)} parquet files...")

    dfs = []
    for pth in paths:
        df = pd.read_parquet(str(pth))
        dfs.append(df)
        log.info(f"  {pth.name}: {len(df)} rows")
    df = pd.concat(dfs, ignore_index=True)

    # Filter out rows with missing essentials
    df = df[df["pos_candidates"].apply(lambda x: x is not None and len(x) > 0)]
    df = df[df["cleaned_html"].apply(lambda x: x is not None and len(str(x)) > 100)]
    log.info(f"After filtering: {len(df)} usable rows")

    # Shuffle and cap
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.max_samples > 0:
        df = df.head(args.max_samples * 2)  # oversample, we'll drop failed rows

    # Generate training examples
    examples: List[Dict] = []
    skipped = 0
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            ex = row_to_training_example(row, rng, args.window_size)
        except Exception as e:
            log.debug(f"  row {i}: ERROR {e}")
            skipped += 1
            continue
        if ex is None:
            skipped += 1
            continue
        examples.append(ex)
        if len(examples) >= args.max_samples:
            break
        if (len(examples)) % 500 == 0:
            log.info(f"  generated {len(examples)}/{args.max_samples} examples (skipped {skipped})")

    log.info(f"Total examples generated: {len(examples)} (skipped {skipped}) in {time.time()-t0:.1f}s")

    # Train/eval split
    rng.shuffle(examples)
    n_eval = max(1, int(len(examples) * args.eval_frac))
    eval_set = examples[:n_eval]
    train_set = examples[n_eval:]
    log.info(f"Split: train={len(train_set)} eval={len(eval_set)}")

    # Write JSONL
    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                # Strip internal _meta from the chat record but keep it in a sibling field
                out = {"messages": it["messages"], "meta": it.get("_meta", {})}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    write_jsonl(args.out_train, train_set)
    write_jsonl(args.out_eval, eval_set)
    log.info(f"Wrote train -> {args.out_train}")
    log.info(f"Wrote eval  -> {args.out_eval}")

    # Stats
    gt_positions = [ex["_meta"]["gt_index"] for ex in examples]
    actions = [ex["_meta"]["action"] for ex in examples]
    from collections import Counter
    act_counter = Counter(actions)
    log.info(f"GT-index distribution: min={min(gt_positions)} max={max(gt_positions)} "
             f"mean={sum(gt_positions)/len(gt_positions):.1f}")
    log.info(f"Action distribution: {dict(act_counter)}")

    # Sanity-check a sample
    if examples:
        log.info("\nSample example (first 400 chars of user content):")
        sample = examples[0]
        log.info(f"  user: {sample['messages'][1]['content'][:400]}...")
        log.info(f"  assistant: {sample['messages'][2]['content']}")

    log.info("=" * 72)
    log.info("Data generation complete.")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
