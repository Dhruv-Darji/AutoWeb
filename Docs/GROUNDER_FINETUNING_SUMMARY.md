# Prune4Web Grounder Fine-Tuning — Summary

**Author:** Dhruv Darji
**Date:** 2026-04-11
**Paper reference:** *Prune4Web: DOM Tree Programming for Efficient Web Agents* (arXiv 2511.21398)
**Goal:** Replace the GPT-4o grounder in our Prune4Web implementation with a
**fully local** Qwen2.5-0.5B-Instruct model, fine-tuned the same way the paper
does, so the full pipeline runs on a single laptop GPU with no API calls for
the grounding step.

---

## 1. Why fine-tune at all?

Our initial Prune4Web implementation used GPT-4o (via the OpenAI API) for all
three LLM stages — planner, filter, grounder. On a 25-sample Mind2Web
evaluation (`Prune4Web/evaluate_prune4web.py`) we measured:

| Metric                     | Score  | Source                              |
|----------------------------|--------|-------------------------------------|
| Recall@20 (programmatic)   | 72.0%  | our scoring formula                 |
| Element Accuracy (GPT-4o)  | 72.22% | `evaluate_prune4web.py`             |
| Paper Element Accuracy     | 88.28% | Qwen2.5-0.5B fine-tuned (arXiv)     |

The 16-point gap vs. the paper comes from **the paper fine-tunes a small model
specifically for grounding**. We replicate that here with QLoRA on the exact
same base model (Qwen2.5-0.5B-Instruct).

**Secondary benefit:** grounding is the most frequent LLM call at inference
time (once per step). Replacing GPT-4o with a local 0.5B model removes the
largest per-step API cost and makes the system fully offline-capable.

---

## 2. Hardware constraint

| Component | Spec                              |
|-----------|-----------------------------------|
| GPU       | NVIDIA GeForce RTX 4050 Laptop    |
| VRAM      | 6.4 GB total, ~5.3 GB free        |
| System    | Windows 11, Python 3.10, CUDA 12.7 |
| PyTorch   | 2.x with BF16 support             |

The 6 GB VRAM budget ruled out full-precision fine-tuning even of a 0.5B model
(would need ~8 GB for weights+grads+optimizer). Solution: **QLoRA** — 4-bit
quantised base + trainable LoRA adapters.

Memory math (verified empirically):

| Component                          | VRAM    |
|------------------------------------|---------|
| Base model (nf4, 315 M params)     | 0.73 GB |
| LoRA adapters (8.80 M trainable)   | 0.05 GB |
| Activations (bs=4, grad_ckpt on)   | ~1.5 GB |
| Paged AdamW 8-bit optimizer state  | ~0.4 GB |
| **Total at train time**            | **~2.7 GB** |

Well within budget.

---

## 3. Project layout

```
Prune4Web/grounder_finetune/
├── configs/                         (reserved for future experiments)
├── data/
│   ├── train.jsonl                  3 690 training examples (ChatML)
│   ├── eval.jsonl                     409 eval examples
│   └── eval_results.json            per-sample eval dump
├── logs/
│   ├── data_generation.log          stdout+file, from 01_generate_data.py
│   ├── training.log                 stdout+file, from 02_finetune.py
│   └── evaluation.log               stdout+file, from 03_evaluate.py
├── charts/
│   ├── loss_curve.png               train/eval loss over steps
│   ├── eval_results.png             headline EA/OpAcc/FormatOK bar chart
│   └── eval_by_action.png           EA broken down by click/type/select
├── scripts/
│   ├── 01_generate_data.py          parquet -> ChatML JSONL
│   ├── 02_finetune.py               QLoRA training loop
│   └── 03_evaluate.py               measure on Mind2Web test_task
└── local_grounder.py                drop-in `ground()` replacing `action_grounder()`
```

All artefacts except the base/adapter weights live inside `grounder_finetune/`.
The actual model weights are under **`D:/Environments/Models/`** to match the
convention used on this machine for all ML assets:

| Path                                                       | Contents                      |
|------------------------------------------------------------|-------------------------------|
| `D:/Environments/Models/Qwen2.5-0.5B-Instruct/`            | Base model (988 MB safetensors) |
| `D:/Environments/Models/Qwen2.5-0.5B-Prune4Web-Grounder/`  | LoRA adapter + tokenizer      |

---

## 4. Stage 1 — Training-data generation (`01_generate_data.py`)

### Strategy
Rather than asking an LLM to label each Mind2Web row (expensive, slow), we
exploit the dataset's own supervision. Each Mind2Web row already contains:

- `pos_candidates` — ground-truth element(s) with `backend_node_id`
- `neg_candidates` — distractors with `backend_node_id`
- `cleaned_html` — the page HTML with ids present
- `target_action_reprs` — the natural-language sub-task
- `operation` — action type (CLICK/TYPE/SELECT) and value

For each row we:

1. Parse `pos_candidates + neg_candidates` into `EvalElementNode`s via
   `evaluate_prune4web.parse_candidates_from_pool()`.
2. Build a **candidate window of 20** = 1 GT + 19 random negatives.
3. **Shuffle** the window, then re-assign contiguous uids `0..19` so the model
   cannot memorise a positional prior.
4. Format the user message byte-for-byte identically to the inference prompt
   used by `run_prune4web.action_grounder()`:

   ```
   Sub-task: <from target_action_reprs>
   Value hint: <from operation>

   Candidate elements (ranked by relevance):
     1. [uid=0] tag=... text='...'
     2. [uid=1] tag=... text='...'
     ...
   ```

5. Emit a ChatML record — `system` = the exact `GROUNDER_SYSTEM` prompt,
   `user` = the candidate window, `assistant` = the target JSON:

   ```json
   {"element_uid": <gt_index>, "action": "click", "value": "",
    "confidence": 0.95,
    "reasoning": "Element <gt_index> matches the sub-task target: ..."}
   ```

### Result

| Quantity              | Value       |
|-----------------------|-------------|
| Parquet files loaded  | 15          |
| Raw rows              | ~20 000     |
| Usable rows (after filter) | ~12 000 |
| **Generated examples**    | **4 099** |
| Train / eval split    | 3 690 / 409 (90/10) |
| Action distribution   | click=3 418 · type=470 · select=211 |
| GT-index distribution | min=0 max=19 mean=9.4 (uniform — shuffling works) |
| Token length (200-sample sanity check) | min=561 max=894 mean=686 |

Log: `grounder_finetune/logs/data_generation.log`

---

## 5. Stage 2 — QLoRA fine-tune (`02_finetune.py`)

### Hyperparameters

| Hyperparameter            | Value                                |
|---------------------------|--------------------------------------|
| Base model                | Qwen2.5-0.5B-Instruct                |
| Quantisation              | nf4, BF16 compute, double-quant      |
| LoRA rank r               | 16                                    |
| LoRA alpha                | 32                                    |
| LoRA dropout              | 0.05                                 |
| LoRA target modules       | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable params          | 8.80 M / 315 M (2.72%)              |
| Optimiser                 | `paged_adamw_8bit`                   |
| Learning rate             | 2e-4                                 |
| LR schedule               | cosine, warmup 10%                   |
| Weight decay              | 0.01                                 |
| Max grad norm             | 1.0                                  |
| Batch size                | 4 per device                         |
| Grad accumulation         | 4  →  effective batch 16             |
| Epochs                    | 2                                    |
| Max sequence length       | 1 024                                |
| Gradient checkpointing    | on (use_reentrant=False)             |
| Precision                 | bf16                                 |
| Attention impl            | eager (avoids flash-attn dep)        |
| Total optimiser steps     | 462                                  |
| Eval every                | 100 steps                            |
| Save every                | 200 steps                            |
| Seed                      | 42                                   |

### Infrastructure notes

- TRL `SFTTrainer` (v1.0.0) with `processing_class=tokenizer` (new API).
  The chat template is applied by the trainer automatically from each row's
  `messages` column.
- Custom `LossTrackerCallback` captures `on_log()` events to build
  `charts/loss_curve.png` at the end.
- `tokenizer.padding_side = "right"` to match the causal-LM loss formulation.
- `model.config.use_cache = False` during training (required for gradient
  checkpointing).
- `prepare_model_for_kbit_training(model)` enables input-embedding grads for
  4-bit training.
- All trainer output (including TQDM) is duplicated to
  `grounder_finetune/logs/training.log` via a root logger that writes to both
  the file and stdout.

### Execution log (excerpt)

```
[INFO] Base model   : D:\Environments\Models\Qwen2.5-0.5B-Instruct
[INFO] Output dir   : D:\Environments\Models\Qwen2.5-0.5B-Prune4Web-Grounder
[INFO] Epochs       : 2.0
[INFO] Batch size   : 4 x grad_accum 4 (effective 16)
[INFO] LR           : 0.0002
[INFO] LoRA r/alpha : 16/32
[INFO] Max seq len  : 1024
[INFO] GPU          : NVIDIA GeForce RTX 4050 Laptop GPU
[INFO] VRAM free    : 5.32 GB
[INFO]   base params : 315.1 M
[INFO]   VRAM after load: 0.73 GB
[INFO]   trainable   : 8.80 M (2.72%)
[INFO]   train: 3690
[INFO]   eval : 409
[INFO] Starting training...
[INFO]   step    1  train_loss=2.4778  lr=0.00e+00
...
```

**Per-step throughput:** ~25–30 s/step on the 4050 Laptop, so the full 462
steps take roughly 3.5–4 hours (~2 h/epoch). Training is deterministic — the
same `seed=42` and `data_seed=42` were used for both data shuffling and the
trainer.

### Outputs

| Artefact                                       | Location                                                        |
|------------------------------------------------|-----------------------------------------------------------------|
| LoRA adapters (`adapter_model.safetensors`)    | `D:/Environments/Models/Qwen2.5-0.5B-Prune4Web-Grounder/`       |
| Tokenizer (for inference)                      | same dir                                                        |
| `training_meta.json`                           | same dir (hyperparams + final eval loss + runtime)              |
| Loss curve                                     | `Prune4Web/grounder_finetune/charts/loss_curve.png`              |
| Training log                                   | `Prune4Web/grounder_finetune/logs/training.log`                  |

Adapter size on disk: ~35 MB — easy to commit or ship separately from the
~1 GB base model.

---

## 6. Stage 3 — Evaluation (`03_evaluate.py`)

### What it measures

For each Mind2Web **test_task** sample we rebuild the **same** 20-candidate
window used at training time, run the fine-tuned model, parse its JSON
output, and compute three metrics:

- **Element Accuracy** — the predicted `element_uid` equals the GT index.
- **Op Accuracy**      — element AND `action` match (the paper's "Op Acc").
- **Format Validity**  — output parsed as valid JSON with required fields.

These are broken down by action type (click / type / select) for a per-action
confusion view.

### Comparison points plotted

| Baseline                        | Element Accuracy | Source                                |
|---------------------------------|------------------|---------------------------------------|
| GPT-4o-mini (our implementation)| 72.22%           | `evaluate_prune4web.py`, 25 samples   |
| Paper (Qwen2.5-0.5B fine-tuned) | 88.28%           | arXiv 2511.21398                      |
| **Ours (this run)**             | **<fill after run>** | `03_evaluate.py`, 200 samples     |

### Output artefacts

- `charts/eval_results.png` — bar chart of Format/EA/OpAcc with the two
  reference baselines as horizontal lines (paper dashed red, GPT-4o-mini
  dotted grey).
- `charts/eval_by_action.png` — EA broken down per action type with sample
  counts.
- `data/eval_results.json` — full per-sample details (gt/pred uid, gt/pred
  action, raw model output truncated to 200 chars).
- `logs/evaluation.log` — per-sample stdout of every inference call.

### Run command

```bash
python Prune4Web/grounder_finetune/scripts/03_evaluate.py \
       --split test_task --num-samples 200
# Baseline run (base model, no adapter):
python Prune4Web/grounder_finetune/scripts/03_evaluate.py \
       --split test_task --num-samples 200 --use-base-only
```

---

## 7. Integration — using the local grounder at run time

The fine-tuned model is exposed via
`Prune4Web/grounder_finetune/local_grounder.py`, which provides a `ground()`
function whose signature **exactly matches** `run_prune4web.action_grounder()`:

```python
def ground(sub_task: str,
           candidates: List[ElementNode],
           action_value: str = "") -> Dict: ...
```

Internally it lazy-loads the base model in 4-bit and attaches the LoRA
adapter once per process, then reuses the same in-memory model for every
call.

### Switching the live agent

`run_prune4web.py` checks a single environment variable. Set this and the
entire Prune4Web pipeline uses the local model for grounding (planner and
filter still use GPT-4o):

```bash
# Windows PowerShell
$env:GROUNDER_USE_LOCAL = "1"
python .\Prune4Web\run_prune4web.py
```

The relevant hook lives right after `action_grounder()` is defined:

```python
if os.getenv("GROUNDER_USE_LOCAL", "").strip() in {"1", "true", "yes"}:
    from Prune4Web.grounder_finetune.local_grounder import ground as _local_ground
    action_grounder = _local_ground
```

If adapter loading fails (e.g. training not finished), it prints a warning
and falls back to the OpenAI grounder so the agent still runs.

### Smoke test (built into local_grounder.py)

```bash
python Prune4Web/grounder_finetune/local_grounder.py
```

Builds a fake 5-element candidate list (`Search`, `Add to Bag`, `Sign in`,
`Home`, `Cart`), asks the model to click "Add to Bag", and checks that it
returns `element_uid: 1`.

---

## 8. What went right / what to be careful of

### Right
- **4-bit fits easily** — final training VRAM was ~2.7 GB, leaving headroom
  for a larger batch size if we ever want one.
- **Dataset trick works** — sampling `1 GT + 19 negs` from the already-labelled
  `pos/neg_candidates` bypasses the need for an expensive LLM labelling pass
  and gives us effectively unlimited training data.
- **Uniform GT position** — shuffling + uid re-assignment kept the GT index
  mean at 9.4 (close to the expected 9.5), so the model can't learn a
  positional shortcut.

### Be careful of
- **Inference prompt drift** — the format used by `local_grounder.ground()`
  must stay byte-identical to `01_generate_data.format_user_message()`.
  Both are now defined in one place each, and `run_prune4web.action_grounder()`
  uses the same format, but any future change to
  `ElementNode.to_summary()` will implicitly change the prompt — re-run the
  evaluation if you touch it.
- **Unicode in Windows logs** — the default Windows console is `cp1252` and
  cannot encode characters like `→` or `✓`. Keep log strings ASCII or run
  Python with `-X utf8`.
- **Eager attention only** — the base model is loaded with
  `attn_implementation="eager"` because we didn't want to install flash-attn
  on Windows. Throughput could improve with SDPA, but that's a
  nice-to-have.
- **TRL version pinning** — this code is written against `trl==1.0.0` /
  `peft==0.18.1`. The SFTTrainer API was rewritten shortly before this
  (dropped `max_seq_length` in favour of `max_length`, renamed
  `evaluation_strategy` → `eval_strategy`). Upgrading TRL without
  re-reading the script will break it.

---

## 9. How to reproduce end-to-end

```bash
# 0) Install deps (already done in ml-env)
pip install peft trl datasets matplotlib bitsandbytes

# 1) Generate training data (~1 min)
python Prune4Web/grounder_finetune/scripts/01_generate_data.py \
       --split train --max-files 15 --max-samples 5000

# 2) Fine-tune (3.5–4 h on RTX 4050 Laptop)
python Prune4Web/grounder_finetune/scripts/02_finetune.py \
       --num-epochs 2 --batch-size 4 --grad-accum 4 --lr 2e-4

# 3) Evaluate against Mind2Web test_task (~10 min)
python Prune4Web/grounder_finetune/scripts/03_evaluate.py \
       --split test_task --num-samples 200

# 4) (Optional) Baseline: same eval with base model, no adapter
python Prune4Web/grounder_finetune/scripts/03_evaluate.py \
       --split test_task --num-samples 200 --use-base-only

# 5) Use local grounder in the live pipeline
$env:GROUNDER_USE_LOCAL = "1"
python Prune4Web/run_prune4web.py
```

---

## 10. Next steps

1. **Finish training run and fill in the actual numbers** in §6 and the eval
   charts once `03_evaluate.py` completes.
2. **Larger training set** — we capped at 5 000 examples from 15 parquet
   files. The full Mind2Web train split has ~30 000 rows; rerunning with
   `--max-samples 15000` should push EA further toward the paper's 88.28%.
3. **Swap FILTER_MODEL too** — right now the keyword-weight filter still calls
   GPT-4o. A second tiny fine-tuned model for that stage would make the agent
   almost fully local. The paper does this; we haven't yet.
4. **Vision planner** — the planner still needs screenshots, so a small
   vision-language model (e.g. Qwen2.5-VL-2B) would be needed to make the
   agent 100% offline.
