"""
Script: merge_mind2web_results.py
Purpose:
  - Load a specified Mind2Web parquet shard
  - Load step-level result JSONs produced by the pipeline (eval_results)
  - Join predicted outputs to dataset rows (by annotation_id + action_uid)
  - Emit a single CSV with one row per step (action_uid)

Usage (example):
  python Scripts/merge_mind2web_results.py \
    --dataset "D:\\Environments\\Datasets\\multimodal-mind2web\\data\\train-00000-of-00027-4d11798d7219186d.parquet" \
    --results-dir "D:\\Mtech\\Sem 3\\Codes\\eval_results" \
    --out "eval_results/merged_steps_train-00000-of-00027-4d11798d7219186d.csv"

Notes:
  - The script looks for a file named `step_results_batch_{parquet_basename}.json` inside --results-dir
    (falls back to loading all `step_results_*.json` and filtering by annotation_id if not found).
  - Output CSV contains: action_uid, annotation_id, dataset_true_action (target_action_reprs),
    gt_element_repr, pred_plan_text, pred_action_type, grounding flags, match flags, latency,
    plus available dataset columns (confirmed_task, website, domain, operation).

"""
from __future__ import annotations
import json
import os
import glob
from typing import Optional
import pandas as pd

# --------------------------
# Configuration (edit here)
# --------------------------
# Dataset parquet shard to process (Mind2Web)
DATASET_PATH = r"D:\\Environments\\Datasets\\multimodal-mind2web\\data\\train-00000-of-00027-4d11798d7219186d.parquet"
# Directory containing pipeline `eval_results` (step_results_*.json etc.)
RESULTS_DIR = r"D:\\Mtech\\Sem 3\\Codes\\eval_results"
# Output CSV path (default placed inside RESULTS_DIR)
PARQUET_BASENAME = os.path.basename(DATASET_PATH)
OUT_CSV = os.path.join(RESULTS_DIR, f"merged_steps_{PARQUET_BASENAME}.csv")# How many characters to keep from raw/cleaned HTML for CSV-friendly snippets
HTML_SNIPPET_LEN = 300# --------------------------



def load_dataset_parquet(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    # Keep relevant columns and ensure action_uid + annotation_id present
    required = ["action_uid", "annotation_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Dataset parquet missing required columns: {missing}")

    # Normalize columns we will export
    # We keep full html internally but will export only short snippets to the CSV
    keep_cols = [c for c in [
        "action_uid", "annotation_id", "confirmed_task", "website", "domain",
        "operation", "action_reprs", "target_action_reprs", "target_action_index",
        "raw_html", "cleaned_html"
    ] if c in df.columns]

    df_small = df[keep_cols].copy()

    # string-ify complex fields to keep CSV clean
    if "operation" in df_small.columns:
        df_small["operation_str"] = df_small["operation"].apply(lambda x: json.dumps(x, ensure_ascii=False) if pd.notna(x) else "")
    else:
        df_small["operation_str"] = ""

    # Add short HTML snippets (single-line, truncated) so CSV remains one-row-per-step
    if "raw_html" in df_small.columns:
        df_small["raw_html_snippet"] = df_small["raw_html"].fillna("").astype(str).str.replace(r"[\r\n]+", " ", regex=True).str.slice(0, HTML_SNIPPET_LEN)
    else:
        df_small["raw_html_snippet"] = ""

    if "cleaned_html" in df_small.columns:
        df_small["cleaned_html_snippet"] = df_small["cleaned_html"].fillna("").astype(str).str.replace(r"[\r\n]+", " ", regex=True).str.slice(0, HTML_SNIPPET_LEN)
    else:
        df_small["cleaned_html_snippet"] = ""
    # Ensure action_uid and annotation_id are strings
    df_small["action_uid"] = df_small["action_uid"].astype(str)
    df_small["annotation_id"] = df_small["annotation_id"].astype(str)

    return df_small


def load_step_results(results_dir: str, parquet_basename: str) -> pd.DataFrame:
    # Preferred filename
    preferred = os.path.join(results_dir, f"step_results_batch_{parquet_basename}.json")
    if os.path.exists(preferred):
        with open(preferred, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        df = pd.json_normalize(data)
        return df

    # Fallback: load all step_results_*.json and concatenate, then filter
    files = glob.glob(os.path.join(results_dir, "step_results_*.json"))
    if not files:
        raise FileNotFoundError(f"No step_results_*.json files found in {results_dir}")

    dfs = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            dfs.append(pd.json_normalize(data))
        except Exception:
            continue
    if not dfs:
        raise RuntimeError("No parsable step_results files found")
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def merge_and_emit(dataset_df: pd.DataFrame, results_df: pd.DataFrame, out_csv: str) -> None:
    # Normalize keys
    results_df["action_uid"] = results_df["action_uid"].astype(str)
    results_df["annotation_id"] = results_df["annotation_id"].astype(str)

    # Merge on (annotation_id, action_uid)
    merged = results_df.merge(dataset_df, on=["annotation_id", "action_uid"], how="left", suffixes=("_pred", "_ds"))

    # Sanitize heavy HTML / JSON fields so each CSV row stays single-line
    if "operation_str" in merged.columns:
        merged["operation_str"] = merged["operation_str"].astype(str).str.replace(r"[\r\n]+", " ", regex=True)
    # create safe snippets (truncate) for raw_html / cleaned_html
    if "raw_html_snippet" in merged.columns:
        merged["raw_html_snippet"] = merged["raw_html_snippet"].astype(str).str.replace(r"[\r\n]+", " ", regex=True)
    if "cleaned_html_snippet" in merged.columns:
        merged["cleaned_html_snippet"] = merged["cleaned_html_snippet"].astype(str).str.replace(r"[\r\n]+", " ", regex=True)

    # Pick or rename columns for CSV (omit full raw_html / cleaned_html to avoid huge multiline cells)
    cols = [
        "action_uid",
        "annotation_id",
        "step_index",
        "confirmed_task",
        "website",
        "domain",
        "gt_operation",
        "gt_value",
        # ground-truth textual representation (from results file) and dataset target
        "gt_element_repr",
        "target_action_reprs",
        # predicted plan / type
        "pred_plan_text",
        "pred_action_type",
        "pred_value",
        "pred_backend_id",
        # grounding / match flags
        "grounding_success",
        "operation_match",
        "element_match",
        "value_match",
        "step_success",
        "latency",
        # dataset operation string for debugging
        "operation_str",
        # safe HTML snippets (single-line)
        "raw_html_snippet",
        "cleaned_html_snippet",
    ]

    existing = [c for c in cols if c in merged.columns]

    # Remove full HTML columns (too large / multiline) if present
    for drop_col in ("raw_html", "cleaned_html", "screenshot"):
        if drop_col in merged.columns:
            merged.drop(columns=[drop_col], inplace=True)

    # Add any additional columns present in results but not in our list (except removed ones)
    extras = [c for c in merged.columns if c not in existing]

    final_cols = existing + extras

    merged.to_csv(out_csv, index=False, columns=final_cols, encoding="utf-8")

    # Quick summary
    total_results = len(results_df)
    matched = merged[merged["target_action_reprs"].notna()].shape[0]
    unmatched_results = total_results - merged[merged["target_action_reprs"].notna()].shape[0]
    print(f"Wrote {out_csv} ({len(merged)} rows)")
    print(f"Dataset rows matched with results: {matched} / {len(dataset_df)} (dataset rows) / {total_results} (results)")
    if unmatched_results > 0:
        print(f"Warning: {unmatched_results} result rows had no matching dataset row (action_uid/annotation_id mismatch)")


def main():
    # Use constants defined at the top of this file (no CLI arguments)
    dataset_path = DATASET_PATH
    results_dir = RESULTS_DIR
    parquet_basename = PARQUET_BASENAME

    ds = load_dataset_parquet(dataset_path)
    print(f"Loaded dataset parquet: {dataset_path} ({len(ds)} rows)")

    results_df = load_step_results(results_dir, parquet_basename)
    print(f"Loaded step results: {results_df.shape[0]} rows from results directory {results_dir}")

    out_csv = OUT_CSV

    merge_and_emit(ds, results_df, out_csv)


if __name__ == "__main__":
    main()
