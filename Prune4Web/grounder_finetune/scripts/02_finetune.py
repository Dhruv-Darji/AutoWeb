"""
02_finetune.py — QLoRA fine-tuning of Qwen2.5-0.5B-Instruct as Prune4Web grounder.

Configuration
-------------
  * 4-bit quantised base model (bitsandbytes nf4, bf16 compute)
  * LoRA adapters r=16 alpha=32 on q_proj, k_proj, v_proj, o_proj,
    gate_proj, up_proj, down_proj
  * TRL SFTTrainer with chat-formatted JSONL
  * AdamW 8-bit optimiser, bf16 training
  * batch_size=4, grad_accum=4 -> effective batch 16
  * Loss plotted to charts/loss_curve.png at the end of training
  * All training output appended to logs/training.log

Outputs
-------
  adapter weights -> D:/Environments/Models/Qwen2.5-0.5B-Prune4Web-Grounder/
  merged model    -> (optional) same dir
  loss curve      -> Prune4Web/grounder_finetune/charts/loss_curve.png
  log file        -> Prune4Web/grounder_finetune/logs/training.log
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

# -------------------- Paths --------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FINETUNE_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(_PROJECT_ROOT / ".env")

DATA_DIR = _FINETUNE_ROOT / "data"
LOGS_DIR = _FINETUNE_ROOT / "logs"
CHARTS_DIR = _FINETUNE_ROOT / "charts"
for d in (LOGS_DIR, CHARTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

BASE_MODEL_DIR = Path(os.getenv("GROUNDER_BASE_MODEL",
                                "D:/Environments/Models/Qwen2.5-0.5B-Instruct"))
OUTPUT_MODEL_DIR = Path(os.getenv("GROUNDER_OUTPUT_MODEL",
                                  "D:/Environments/Models/Qwen2.5-0.5B-Prune4Web-Grounder"))
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / "training.log"

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("finetune")


# -------------------- Loss-tracking callback --------------------
class LossTrackerCallback(TrainerCallback):
    """Captures train/eval losses from trainer logs for later plotting."""

    def __init__(self):
        self.train_steps, self.train_losses = [], []
        self.eval_steps, self.eval_losses = [], []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        if "loss" in logs and "eval_loss" not in logs:
            self.train_steps.append(step)
            self.train_losses.append(float(logs["loss"]))
            log.info(f"  step {step:>4}  train_loss={logs['loss']:.4f}"
                     + (f"  lr={logs.get('learning_rate', 0):.2e}" if 'learning_rate' in logs else ""))
        if "eval_loss" in logs:
            self.eval_steps.append(step)
            self.eval_losses.append(float(logs["eval_loss"]))
            log.info(f"  step {step:>4}  eval_loss ={logs['eval_loss']:.4f}")


def plot_loss_curve(cb: LossTrackerCallback, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if cb.train_steps:
        ax.plot(cb.train_steps, cb.train_losses, label="train loss",
                color="#1f77b4", linewidth=1.6, marker="o", markersize=3)
    if cb.eval_steps:
        ax.plot(cb.eval_steps, cb.eval_losses, label="eval loss",
                color="#d62728", linewidth=2.0, marker="s", markersize=5)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title("Prune4Web grounder fine-tuning — Qwen2.5-0.5B QLoRA")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info(f"Loss curve saved to {out_path}")


# -------------------- Main --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", default=str(DATA_DIR / "train.jsonl"))
    p.add_argument("--eval-file", default=str(DATA_DIR / "eval.jsonl"))
    p.add_argument("--num-epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=1536)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--log-steps", type=int, default=20)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    log.info("=" * 72)
    log.info("Prune4Web grounder fine-tune — Qwen2.5-0.5B-Instruct")
    log.info("=" * 72)
    log.info(f"Base model   : {BASE_MODEL_DIR}")
    log.info(f"Output dir   : {OUTPUT_MODEL_DIR}")
    log.info(f"Train file   : {args.train_file}")
    log.info(f"Eval file    : {args.eval_file}")
    log.info(f"Epochs       : {args.num_epochs}")
    log.info(f"Batch size   : {args.batch_size} x grad_accum {args.grad_accum} (effective {args.batch_size*args.grad_accum})")
    log.info(f"LR           : {args.lr}")
    log.info(f"LoRA r/alpha : {args.lora_r}/{args.lora_alpha}")
    log.info(f"Max seq len  : {args.max_seq_len}")
    log.info(f"Seed         : {args.seed}")
    log.info(f"GPU          : {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM free    : {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

    # ---------------- Load tokenizer ----------------
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Qwen3-family: force non-thinking template so training targets stay flat JSON.
    # No-op for Qwen2.5 (tokenizer ignores the kwarg via TypeError fallback).
    if os.getenv("GROUNDER_DISABLE_THINKING", "0") == "1":
        _orig_apply = tokenizer.apply_chat_template
        def _apply_no_think(messages, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            try:
                return _orig_apply(messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                return _orig_apply(messages, **kwargs)
        tokenizer.apply_chat_template = _apply_no_think
        log.info("  chat template: non-thinking mode (enable_thinking=False)")

    # ---------------- Load model (4-bit) ----------------
    log.info("Loading base model in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # avoid flash-attn requirement
    )
    model.config.use_cache = False  # disable during training
    model = prepare_model_for_kbit_training(model)
    log.info(f"  base params : {sum(p.numel() for p in model.parameters())/1e6:.1f} M")
    log.info(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ---------------- LoRA ----------------
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"  trainable   : {trainable/1e6:.2f} M ({100*trainable/total:.2f}%)")

    # ---------------- Dataset ----------------
    log.info("Loading dataset...")
    ds = load_dataset(
        "json",
        data_files={"train": args.train_file, "eval": args.eval_file},
    )
    log.info(f"  train: {len(ds['train'])}")
    log.info(f"  eval : {len(ds['eval'])}")

    # TRL wants a "messages" column; drop "meta" so it doesn't confuse the collator
    for split in ("train", "eval"):
        if "meta" in ds[split].column_names:
            ds[split] = ds[split].remove_columns("meta")

    # ---------------- SFTConfig ----------------
    sft_cfg = SFTConfig(
        output_dir=str(OUTPUT_MODEL_DIR),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=args.log_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_length=args.max_seq_len,
        dataset_num_proc=1,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    # ---------------- Trainer ----------------
    loss_cb = LossTrackerCallback()
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        processing_class=tokenizer,
        callbacks=[loss_cb],
    )

    log.info("Starting training...")
    t0 = time.time()
    try:
        trainer.train()
    except Exception as e:
        log.error(f"Training crashed: {e}")
        raise
    elapsed = time.time() - t0
    log.info(f"Training finished in {elapsed/60:.1f} min")

    # ---------------- Final eval ----------------
    log.info("Running final eval...")
    metrics = trainer.evaluate()
    log.info(f"Final eval metrics: {metrics}")

    # ---------------- Save adapters & tokenizer ----------------
    log.info(f"Saving adapters to {OUTPUT_MODEL_DIR}")
    trainer.save_model(str(OUTPUT_MODEL_DIR))
    tokenizer.save_pretrained(str(OUTPUT_MODEL_DIR))

    # Also save training args for reproducibility
    with open(OUTPUT_MODEL_DIR / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "base_model": str(BASE_MODEL_DIR),
            "num_train_examples": len(ds["train"]),
            "num_eval_examples": len(ds["eval"]),
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "lr": args.lr,
            "max_seq_len": args.max_seq_len,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "final_eval_loss": metrics.get("eval_loss"),
            "train_runtime_min": round(elapsed / 60, 2),
        }, f, indent=2)

    # ---------------- Plot loss curve ----------------
    plot_loss_curve(loss_cb, CHARTS_DIR / "loss_curve.png")

    log.info("=" * 72)
    log.info("Fine-tune complete.")
    log.info(f"Adapter dir : {OUTPUT_MODEL_DIR}")
    log.info(f"Log         : {LOG_FILE}")
    log.info(f"Loss chart  : {CHARTS_DIR / 'loss_curve.png'}")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
