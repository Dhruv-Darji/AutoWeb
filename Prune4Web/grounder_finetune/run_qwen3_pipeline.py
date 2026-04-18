"""
run_qwen3_pipeline.py — End-to-end Qwen3-0.6B grounder pipeline.

Steps (chained via subprocess, all env-gated for thinking-mode off):
  0. Preflight — CUDA, free VRAM, training data present.
  1. Download Qwen/Qwen3-0.6B -> D:/Environments/Models/Qwen3-0.6B
  2. Smoke-test the non-thinking chat template.
  3. QLoRA fine-tune (2 epochs, matches Qwen2.5 training recipe).
  4. test_task Element Accuracy (200 samples).
  5. DOM-Delta EA (test_domain 200 samples, FULL + DELTA_HYBRID).
  6. Aggregate: JSON summary + Qwen3-vs-Qwen2.5 markdown table for §3.4.

Designed for ≤10 h on RTX 4050 (6 GB VRAM).

Launch:
    nohup "D:/Environments/ml-env/Scripts/python.exe" -X utf8 -u \
        Prune4Web/grounder_finetune/run_qwen3_pipeline.py \
        > Prune4Web/grounder_finetune/logs/qwen3_pipeline.log 2>&1 &
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
ROOT = Path(__file__).resolve().parents[2]  # WebAgents/
FT_ROOT = ROOT / "Prune4Web" / "grounder_finetune"
SCRIPTS = FT_ROOT / "scripts"

BASE_MODEL_DIR = Path("D:/Environments/Models/Qwen3-0.6B")
ADAPTER_DIR = Path("D:/Environments/Models/Qwen3-0.6B-Prune4Web-Grounder")

LOG_DIR = FT_ROOT / "logs"
RESULTS_DIR = FT_ROOT / "results"
DATA_DIR = FT_ROOT / "data"
CHARTS_DIR = FT_ROOT / "charts"
DOM_DELTA_BENCH_DIR = ROOT / "Prune4Web" / "results" / "dom_delta_bench"

for d in (LOG_DIR, RESULTS_DIR, BASE_MODEL_DIR.parent, ADAPTER_DIR.parent):
    d.mkdir(parents=True, exist_ok=True)


def banner(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72, flush=True)


def run_cmd(cmd, label, extra_env=None, cwd=None):
    banner(f"RUN: {label}")
    print(f"  cmd: {' '.join(str(c) for c in cmd)}", flush=True)
    if extra_env:
        shown = {k: v for k, v in extra_env.items()
                 if k.startswith(("GROUNDER_", "HF_", "TRANSFORMERS_"))}
        if shown:
            print(f"  env: {shown}", flush=True)
    full_env = os.environ.copy()
    if extra_env:
        full_env.update(extra_env)
    t0 = time.time()
    result = subprocess.run(cmd, env=full_env, cwd=str(cwd) if cwd else None)
    elapsed = time.time() - t0
    print(f"\n  [{label}] exit={result.returncode}  time={elapsed:.1f}s",
          flush=True)
    return {"label": label, "returncode": result.returncode,
            "elapsed_s": round(elapsed, 1)}


# ---------------------- Step 0: preflight ----------------------
def preflight():
    banner("STEP 0: Preflight")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    free_vram, total_vram = torch.cuda.mem_get_info()
    print(f"  GPU        : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  Free VRAM  : {free_vram/1e9:.2f} / {total_vram/1e9:.2f} GB",
          flush=True)
    if free_vram / 1e9 < 4.0:
        print(f"  [warn] Low free VRAM — close other GPU apps before long runs",
              flush=True)

    train_file = DATA_DIR / "train.jsonl"
    eval_file = DATA_DIR / "eval.jsonl"
    if not train_file.exists() or not eval_file.exists():
        raise FileNotFoundError(
            f"Training data missing: {train_file} or {eval_file}. "
            "Run Prune4Web/grounder_finetune/scripts/01_generate_data.py first.")
    tr = sum(1 for _ in open(train_file, "r", encoding="utf-8"))
    ev = sum(1 for _ in open(eval_file, "r", encoding="utf-8"))
    print(f"  train.jsonl: {tr} samples", flush=True)
    print(f"  eval.jsonl : {ev} samples", flush=True)


# ---------------------- Step 1: download ----------------------
def download_model():
    banner("STEP 1: Download Qwen/Qwen3-0.6B")
    if BASE_MODEL_DIR.exists():
        files = list(BASE_MODEL_DIR.iterdir())
        has_config = any(f.name == "config.json" for f in files)
        has_weights = any(f.suffix in (".safetensors", ".bin") for f in files)
        if has_config and has_weights:
            print(f"  [skip] Qwen3-0.6B already present at {BASE_MODEL_DIR}",
                  flush=True)
            return {"label": "download Qwen/Qwen3-0.6B",
                    "returncode": 0, "elapsed_s": 0.0}
    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snippet = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id='Qwen/Qwen3-0.6B', "
        f"local_dir=r'{BASE_MODEL_DIR}', "
        "allow_patterns=['*.json','*.safetensors','*.txt','*.py',"
        "'tokenizer*','vocab*','merges*','chat_template*','generation*'])"
    )
    cmd = [PYTHON, "-X", "utf8", "-c", snippet]
    return run_cmd(cmd, "download Qwen/Qwen3-0.6B")


# ---------------------- Step 2: smoke test ----------------------
def smoke_test_template():
    banner("STEP 2: Smoke-test non-thinking chat template")
    snippet = (
        "from transformers import AutoTokenizer; "
        f"tok = AutoTokenizer.from_pretrained(r'{BASE_MODEL_DIR}'); "
        "msgs = [{'role':'system','content':'test'},"
        "{'role':'user','content':'hello'}]; "
        "out = tok.apply_chat_template(msgs, tokenize=False, "
        "add_generation_prompt=True, enable_thinking=False); "
        "print('[smoke] template length:', len(out)); "
        "print('[smoke] contains <think>:', '<think>' in out); "
        "print('[smoke] OK')"
    )
    cmd = [PYTHON, "-X", "utf8", "-c", snippet]
    return run_cmd(cmd, "smoke_test_template")


# ---------------------- Step 3: train ----------------------
def train():
    cmd = [
        PYTHON, "-X", "utf8", "-u",
        str(SCRIPTS / "02_finetune.py"),
        "--num-epochs", "2.0",
        "--batch-size", "4",
        "--grad-accum", "4",
        "--lr", "2e-4",
        "--max-seq-len", "1024",
        "--lora-r", "16",
        "--lora-alpha", "32",
        "--lora-dropout", "0.05",
        "--save-steps", "400",
        "--eval-steps", "200",
        "--log-steps", "20",
        "--warmup-ratio", "0.1",
        "--seed", "42",
    ]
    env = {
        "GROUNDER_BASE_MODEL": str(BASE_MODEL_DIR),
        "GROUNDER_OUTPUT_MODEL": str(ADAPTER_DIR),
        "GROUNDER_DISABLE_THINKING": "1",
    }
    return run_cmd(cmd, "train Qwen3-0.6B QLoRA (2 epochs)",
                   extra_env=env, cwd=ROOT)


# ---------------------- Step 4: test_task EA ----------------------
def eval_test_task():
    cmd = [
        PYTHON, "-X", "utf8", "-u",
        str(SCRIPTS / "03_evaluate.py"),
        "--split", "test_task",
        "--num-samples", "200",
        "--seed", "42",
    ]
    env = {
        "GROUNDER_BASE_MODEL": str(BASE_MODEL_DIR),
        "GROUNDER_OUTPUT_MODEL": str(ADAPTER_DIR),
        "GROUNDER_DISABLE_THINKING": "1",
    }
    r = run_cmd(cmd, "test_task EA (200 samples)", extra_env=env, cwd=ROOT)
    src = DATA_DIR / "eval_results.json"
    dst = DATA_DIR / "eval_results_qwen3.json"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  [copy] {src.name} -> {dst.name}", flush=True)
    return r


# ---------------------- Step 5: DOM-Delta EA ----------------------
def eval_dom_delta():
    cmd = [
        PYTHON, "-X", "utf8", "-u",
        str(ROOT / "Prune4Web" / "evaluate_dom_delta.py"),
        "--split", "test_domain",
        "--num-samples", "200",
        "--seed", "42",
        "--configs", "FULL", "DELTA_HYBRID",
        "--with-grounder",
        "--grounder-configs", "FULL", "DELTA_HYBRID",
    ]
    env = {
        "GROUNDER_BASE_MODEL": str(BASE_MODEL_DIR),
        "GROUNDER_OUTPUT_MODEL": str(ADAPTER_DIR),
        "GROUNDER_DISABLE_THINKING": "1",
    }
    return run_cmd(cmd, "DOM-Delta EA (test_domain 200, FULL+DELTA_HYBRID)",
                   extra_env=env, cwd=ROOT)


# ---------------------- Step 6: aggregate ----------------------
def aggregate_results(summary):
    banner("STEP 6: Aggregate results")
    agg = {
        "timestamp": datetime.datetime.now().isoformat(),
        "base_model": str(BASE_MODEL_DIR),
        "adapter_dir": str(ADAPTER_DIR),
        "steps": summary,
    }

    qwen3_eval = DATA_DIR / "eval_results_qwen3.json"
    if qwen3_eval.exists():
        try:
            er = json.loads(qwen3_eval.read_text(encoding="utf-8"))
            m = er.get("metrics", {})
            agg["test_task_EA"] = m.get("element_accuracy")
            agg["test_task_op_accuracy"] = m.get("op_accuracy")
            agg["test_task_format_validity"] = m.get("format_validity")
        except Exception as e:
            print(f"  [warn] eval_results_qwen3 parse failed: {e}", flush=True)

    tmeta = ADAPTER_DIR / "training_meta.json"
    if tmeta.exists():
        try:
            tm = json.loads(tmeta.read_text(encoding="utf-8"))
            agg["train_runtime_min"] = tm.get("train_runtime_min")
            agg["final_eval_loss"] = tm.get("final_eval_loss")
            agg["num_train_examples"] = tm.get("num_train_examples")
            agg["num_epochs"] = tm.get("num_epochs")
        except Exception as e:
            print(f"  [warn] training_meta parse failed: {e}", flush=True)

    # Pull most-recent DOM-Delta run that has element_accuracy_pct
    runs = sorted(DOM_DELTA_BENCH_DIR.glob("run_*"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        bench = r / "bench_results.json"
        if not bench.exists():
            continue
        try:
            data = json.loads(bench.read_text(encoding="utf-8"))
        except Exception:
            continue
        agg_root = data.get("aggregate", {})
        cfgs = agg_root.get("configs", agg_root)
        full = cfgs.get("FULL", {})
        hyb = cfgs.get("DELTA_HYBRID", {})
        if "element_accuracy_pct" in full:
            agg["dom_delta_run"] = r.name
            agg["dom_delta_EA_FULL"] = full.get("element_accuracy_pct")
            agg["dom_delta_EA_DELTA_HYBRID"] = hyb.get("element_accuracy_pct")
            agg["dom_delta_recall_FULL"] = full.get("recall_at_20_pct")
            agg["dom_delta_recall_DELTA_HYBRID"] = hyb.get("recall_at_20_pct")
            break

    out = RESULTS_DIR / "qwen3_summary.json"
    out.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"  summary -> {out}", flush=True)

    # Markdown comparison table for §3.4 of RESULTS_AND_COMPARISON.md
    q3_ea = agg.get("test_task_EA", "—")
    q3_dd_full = agg.get("dom_delta_EA_FULL", "—")
    q3_dd_hyb = agg.get("dom_delta_EA_DELTA_HYBRID", "—")
    md_path = RESULTS_DIR / "qwen3_vs_qwen25_table.md"
    md = [
        "### Grounder Comparison (free, local-only)",
        "",
        "| Model | Params | Test-task EA (200) | DOM-Delta EA FULL (200) "
        "| DOM-Delta EA DELTA_HYBRID (200) |",
        "|---|---:|---:|---:|---:|",
        "| GPT-4o (paper) | ~1T | 88.28% | — | — |",
        "| Qwen2.5-0.5B + LoRA | 0.5B | 85.00% | 73.00% | 68.00% |",
        f"| Qwen3-0.6B + LoRA  | 0.6B | {q3_ea}% | {q3_dd_full}% | "
        f"{q3_dd_hyb}% |",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  table   -> {md_path}", flush=True)

    src = CHARTS_DIR / "loss_curve.png"
    dst = CHARTS_DIR / "loss_curve_qwen3.png"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  [copy] loss_curve.png -> loss_curve_qwen3.png", flush=True)


# ---------------------- Main ----------------------
def main():
    t_start = time.time()
    banner(f"QWEN3 PIPELINE — started {datetime.datetime.now().isoformat()}")
    summary = []
    try:
        preflight()

        r = download_model()
        summary.append(r)
        if r["returncode"] != 0:
            print("[fatal] download failed; aborting", flush=True)
            return

        r = smoke_test_template()
        summary.append(r)
        if r["returncode"] != 0:
            print("[fatal] template smoke-test failed; aborting", flush=True)
            return

        r = train()
        summary.append(r)
        if r["returncode"] != 0:
            print("[fatal] training failed; aborting eval steps", flush=True)
            return

        summary.append(eval_test_task())
        summary.append(eval_dom_delta())

    finally:
        aggregate_results(summary)
        total = time.time() - t_start
        banner("PIPELINE COMPLETE")
        for r in summary:
            status = "OK" if r.get("returncode") == 0 else "FAIL"
            print(f"  [{status}] {r['label']:<55}  {r['elapsed_s']}s",
                  flush=True)
        print(f"\n  Total: {total:.0f}s ({total/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
