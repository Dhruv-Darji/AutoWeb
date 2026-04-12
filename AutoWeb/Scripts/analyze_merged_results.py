"""
analyze_merged_results.py

- Loads the merged CSV produced by `merge_mind2web_results.py`
- Computes confusion matrix (GT operation vs predicted action type)
- Computes per-class and overall summary metrics (operation_match, element_match, step_success, grounding_success)
- Writes outputs to `results/analysis` (CSV, JSON, PNG if matplotlib available)

Edit the constants below to point at the merged CSV and output folder.
"""
from __future__ import annotations
import os
import json
from pathlib import Path
import pandas as pd

# --------------------------
# Configuration (edit here)
# --------------------------
MERGED_CSV = r"D:\Mtech\Sem 3\Codes\eval_results\merged_steps_train-00000-of-00027-4d11798d7219186d.parquet.csv"
OUTPUT_DIR = r"D:\Mtech\Sem 3\Codes\eval_results\analysis"
# Whether to save confusion matrix image (requires matplotlib)
SAVE_PLOT = True
# --------------------------

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # normalize string columns
    for c in ["gt_operation", "pred_action_type"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    return df


def compute_confusion(df: pd.DataFrame, gt_col: str = "gt_operation", pred_col: str = "pred_action_type") -> pd.DataFrame:
    # confusion counts (rows = GT, cols = Pred)
    cm = pd.crosstab(df[gt_col].fillna(""), df[pred_col].fillna(""), dropna=False)
    return cm


def compute_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    summary = {"total_steps": int(total)}

    # boolean metrics (if available)
    bool_metrics = ["operation_match", "element_match", "value_match", "step_success", "grounding_success"]
    for m in bool_metrics:
        if m in df.columns:
            # coerce to boolean (handles True/False or 0/1)
            series = pd.to_numeric(df[m], errors="coerce")
            summary[f"{m}_rate"] = float(series.fillna(0).mean())
            summary[f"{m}_count"] = int(series.fillna(0).sum())

    # latency
    if "latency" in df.columns:
        summary["avg_latency_s"] = float(df["latency"].dropna().astype(float).mean())
        summary["median_latency_s"] = float(df["latency"].dropna().astype(float).median())

    # per-gt-operation breakdown
    if "gt_operation" in df.columns:
        per = []
        grouped = df.groupby("gt_operation")
        for name, g in grouped:
            row = {
                "gt_operation": name,
                "count": int(len(g)),
                "pred_top5": g["pred_action_type"].fillna("").value_counts().head(5).to_dict()
            }
            for m in ["operation_match", "element_match", "step_success", "grounding_success"]:
                if m in df.columns:
                    row[f"{m}_rate"] = float(pd.to_numeric(g[m], errors="coerce").fillna(0).mean())
            per.append(row)
        summary["per_gt_operation"] = per

    return summary


def save_confusion_table(cm: pd.DataFrame, out_dir: str) -> str:
    out_csv = os.path.join(out_dir, "confusion_gtop_predtype.csv")
    cm.to_csv(out_csv)
    return out_csv


def save_summary_json(summary: dict, out_dir: str) -> str:
    out_json = os.path.join(out_dir, "analysis_summary.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return out_json


def plot_confusion(cm: pd.DataFrame, out_dir: str) -> str:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return ""  # plotting libraries not available

    plt.figure(figsize=(10, max(4, len(cm) * 0.4)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.ylabel("GT operation")
    plt.xlabel("Predicted action type")
    plt.title("Confusion matrix: GT operation vs predicted action type")
    out_png = os.path.join(out_dir, "confusion_gtop_predtype.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


def main():
    df = load_df(MERGED_CSV)
    print(f"Loaded merged CSV: {MERGED_CSV} ({len(df)} rows)")

    cm = compute_confusion(df)
    cm_csv = save_confusion_table(cm, OUTPUT_DIR)
    print(f"Saved confusion matrix CSV: {cm_csv}")

    summary = compute_summary(df)
    summary_json = save_summary_json(summary, OUTPUT_DIR)
    print(f"Saved summary JSON: {summary_json}")

    img_path = "(skipped)"
    if SAVE_PLOT:
        img_path = plot_confusion(cm, OUTPUT_DIR) or "(plot libs unavailable)"
        if img_path:
            print(f"Saved confusion matrix image: {img_path}")

    # print concise console summary
    print("\n=== Quick summary ===")
    print(f"Total steps: {summary.get('total_steps', 0)}")
    for k, v in summary.items():
        if k.endswith("_rate"):
            print(f"{k}: {v:.3f}")
    print("Per-GT operation (top 5 predicted types):")
    for r in summary.get("per_gt_operation", []):
        print(f" - {r['gt_operation']}: count={r['count']}, top_preds={r['pred_top5']}")


if __name__ == "__main__":
    main()
