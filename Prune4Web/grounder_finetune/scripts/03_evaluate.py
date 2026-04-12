"""
03_evaluate.py — Evaluate the fine-tuned Prune4Web grounder on Mind2Web.

What it measures
----------------
For each Mind2Web test sample we build the SAME candidate window used at
training time (1 GT + 19 negatives, shuffled), run the fine-tuned model, parse
its JSON output, and compute:

  Element Accuracy   : model picked the correct element uid
  Op Accuracy        : element AND action both correct
  Format Validity    : output was valid JSON with required fields

Two comparison points
---------------------
  * baseline (GPT-4o-mini grounder from evaluate_prune4web.py) — already run,
    Element Accuracy = 72.22% on 25 samples
  * paper (Qwen2.5-0.5B fine-tuned on full Mind2Web) — 88.28% on full test set

Outputs
-------
  logs/evaluation.log           — full per-sample log
  charts/eval_results.png       — bar chart (format validity, EA, Op Acc)
  charts/confusion_by_action.png — EA broken down by action type
  data/eval_results.json        — machine-readable per-sample details
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# -------------------- Bootstrap --------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env")
sys.path.insert(0, str(_PROJECT_ROOT))

_FINETUNE_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = _FINETUNE_ROOT / "logs"
CHARTS_DIR = _FINETUNE_ROOT / "charts"
DATA_DIR = _FINETUNE_ROOT / "data"
for d in (LOGS_DIR, CHARTS_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / "evaluation.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("eval")

# -------------------- Imports from Prune4Web --------------------
from Prune4Web.run_prune4web import GROUNDER_SYSTEM
from Prune4Web.evaluate_prune4web import (
    parse_candidates_from_pool,
    get_gt_backend_node_id,
    get_gt_action_type,
    parse_target_action_repr,
)
from Prune4Web.grounder_finetune.scripts import __init__ as _pkg_init  # ensure package
# Reuse helpers from 01_generate_data.py
sys.path.insert(0, str(_FINETUNE_ROOT / "scripts"))
from importlib.machinery import SourceFileLoader
_gen = SourceFileLoader("gen01", str(_FINETUNE_ROOT / "scripts" / "01_generate_data.py")).load_module()
build_candidate_window = _gen.build_candidate_window
format_user_message = _gen.format_user_message
value_from_operation = _gen.value_from_operation


DATASET_PATH = os.getenv("DATASET_PATH", "D:/Environments/Datasets/multimodal-mind2web")
BASE_MODEL_DIR = Path(os.getenv("GROUNDER_BASE_MODEL",
                                "D:/Environments/Models/Qwen2.5-0.5B-Instruct"))
ADAPTER_DIR = Path(os.getenv("GROUNDER_OUTPUT_MODEL",
                             "D:/Environments/Models/Qwen2.5-0.5B-Prune4Web-Grounder"))


# -------------------- Parsing LLM output --------------------
def extract_json(text: str) -> Optional[Dict]:
    """Extract a JSON object from the model output."""
    if not text:
        return None
    match = re.search(r"\{[^{}]*?\"element_uid\"[^{}]*\}", text, re.DOTALL)
    if not match:
        match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    raw = match.group(0)
    try:
        return json.loads(raw)
    except Exception:
        # Best effort: strip trailing garbage
        try:
            return json.loads(raw.rsplit("}", 1)[0] + "}")
        except Exception:
            return None


# -------------------- Main --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test_task")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--window-size", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-base-only", action="store_true", help="Eval the base model without adapters (baseline)")
    p.add_argument("--output", default=str(DATA_DIR / "eval_results.json"))
    args = p.parse_args()

    log.info("=" * 72)
    log.info("Prune4Web grounder evaluation")
    log.info("=" * 72)
    log.info(f"Base model   : {BASE_MODEL_DIR}")
    log.info(f"Adapter dir  : {'(disabled — base only)' if args.use_base_only else ADAPTER_DIR}")
    log.info(f"Split        : {args.split}")
    log.info(f"Num samples  : {args.num_samples}")
    log.info(f"Window size  : {args.window_size}")
    log.info(f"Log file     : {LOG_FILE}")

    # ---------------- Load model ----------------
    log.info("Loading base model in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    if not args.use_base_only:
        log.info(f"Loading LoRA adapter from {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))

    model.eval()
    log.info(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ---------------- Load dataset ----------------
    log.info("Loading dataset...")
    data_dir = Path(DATASET_PATH) / "data"
    paths = sorted(data_dir.glob(f"{args.split}-*.parquet"))
    dfs = [pd.read_parquet(str(p)) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["pos_candidates"].apply(lambda x: x is not None and len(x) > 0)]
    df = df[df["cleaned_html"].apply(lambda x: x is not None and len(str(x)) > 100)]
    if args.num_samples > 0 and args.num_samples < len(df):
        df = df.sample(n=args.num_samples, random_state=args.seed).reset_index(drop=True)
    log.info(f"Loaded {len(df)} samples.")

    rng = random.Random(args.seed)

    # ---------------- Run eval ----------------
    results = []
    format_ok = 0
    element_correct = 0
    op_correct = 0
    by_action_total = Counter()
    by_action_correct = Counter()

    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        gt_node_id = get_gt_backend_node_id(row)
        if not gt_node_id:
            continue

        html = row.get("cleaned_html", "")
        all_elements = parse_candidates_from_pool(
            html, row.get("pos_candidates"), row.get("neg_candidates")
        )
        if not all_elements:
            continue

        window, gt_index = build_candidate_window(all_elements, gt_node_id, args.window_size, rng)
        if not window or gt_index < 0:
            continue

        target_repr = str(row.get("target_action_reprs", ""))
        element_desc, _ = parse_target_action_repr(target_repr)
        sub_task = element_desc or str(row.get("confirmed_task", ""))

        gt_action = get_gt_action_type(row)
        value_hint = value_from_operation(row.get("operation"))

        user_msg = format_user_message(sub_task, window, value_hint)
        messages = [
            {"role": "system", "content": GROUNDER_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        parsed = extract_json(gen_text)

        pred_uid = -1
        pred_action = ""
        if parsed is not None:
            format_ok += 1
            try:
                pred_uid = int(parsed.get("element_uid", -1))
            except Exception:
                pred_uid = -1
            pred_action = str(parsed.get("action", "")).lower()

        el_correct = (pred_uid == gt_index)
        op_correct_here = el_correct and (pred_action == gt_action)
        by_action_total[gt_action] += 1
        if el_correct:
            element_correct += 1
            by_action_correct[gt_action] += 1
        if op_correct_here:
            op_correct += 1

        results.append({
            "action_uid": str(row.get("action_uid", "")),
            "sub_task": sub_task[:100],
            "gt_index": gt_index,
            "pred_uid": pred_uid,
            "gt_action": gt_action,
            "pred_action": pred_action,
            "el_correct": el_correct,
            "op_correct": op_correct_here,
            "format_ok": parsed is not None,
            "raw_output": gen_text[:200],
        })

        mark = "OK" if el_correct else "X"
        log.info(f"[{i+1:>4}/{len(df)}] gt={gt_index} pred={pred_uid} "
                 f"gt_act={gt_action} pred_act={pred_action} [{mark}]")

    elapsed = time.time() - t0
    n = len(results)
    if n == 0:
        log.error("No samples evaluated.")
        return

    ea = element_correct / n * 100
    opa = op_correct / n * 100
    fmt = format_ok / n * 100

    log.info("=" * 72)
    log.info("FINAL METRICS")
    log.info("=" * 72)
    log.info(f"Samples evaluated : {n}")
    log.info(f"Elapsed           : {elapsed:.1f}s  ({elapsed/n:.2f}s/sample)")
    log.info(f"Format validity   : {fmt:.2f}%")
    log.info(f"Element Accuracy  : {ea:.2f}%   (paper: 88.28% ; gpt-4o-mini baseline: 72.22%)")
    log.info(f"Op Accuracy       : {opa:.2f}%")
    log.info("")
    log.info("Element Accuracy by action type:")
    for act, total in by_action_total.most_common():
        correct = by_action_correct[act]
        pct = correct / total * 100 if total else 0
        log.info(f"  {act:<8}: {correct}/{total} ({pct:.1f}%)")

    # Save results
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "split": args.split,
                "num_samples": n,
                "window_size": args.window_size,
                "use_base_only": args.use_base_only,
                "base_model": str(BASE_MODEL_DIR),
                "adapter_dir": str(ADAPTER_DIR) if not args.use_base_only else None,
            },
            "metrics": {
                "element_accuracy": ea,
                "op_accuracy": opa,
                "format_validity": fmt,
                "paper_element_accuracy": 88.28,
                "gpt_4o_mini_baseline": 72.22,
            },
            "by_action": {act: {"total": by_action_total[act], "correct": by_action_correct[act]}
                          for act in by_action_total},
            "results": results,
        }, f, indent=2)
    log.info(f"Results JSON -> {out_path}")

    # ---------------- Charts ----------------
    # 1) Headline bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["Format\nvalidity", "Element\nAccuracy", "Op\nAccuracy"]
    ours = [fmt, ea, opa]
    bars = ax.bar(metrics, ours, color=["#2ca02c", "#1f77b4", "#ff7f0e"], width=0.55)
    ax.axhline(88.28, color="#d62728", linestyle="--", linewidth=1.5, label="paper EA 88.28%")
    ax.axhline(72.22, color="#7f7f7f", linestyle=":", linewidth=1.5, label="gpt-4o-mini 72.22%")
    ax.set_ylim(0, 105)
    ax.set_ylabel("percentage")
    ax.set_title(f"Prune4Web grounder — fine-tuned Qwen2.5-0.5B ({n} samples on {args.split})")
    for b, v in zip(bars, ours):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{v:.1f}%",
                ha="center", fontsize=11, weight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "eval_results.png", dpi=120)
    plt.close(fig)

    # 2) Per-action breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    actions = list(by_action_total.keys())
    accs = [by_action_correct[a] / by_action_total[a] * 100 if by_action_total[a] else 0 for a in actions]
    totals = [by_action_total[a] for a in actions]
    bars = ax.bar(actions, accs, color="#1f77b4", width=0.55)
    ax.set_ylabel("Element Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Element Accuracy by action type")
    for b, v, t in zip(bars, accs, totals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{v:.1f}%\n(n={t})", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "eval_by_action.png", dpi=120)
    plt.close(fig)

    log.info(f"Charts    -> {CHARTS_DIR / 'eval_results.png'}")
    log.info(f"          -> {CHARTS_DIR / 'eval_by_action.png'}")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
