# Grounder Fine-Tuning Instructions (for overnight run)

## Goal
Re-train the Qwen2.5-0.5B grounder with MORE data to close the 5.78% gap
(current: 82.50% EA vs paper: 88.28%).

## Environment
- Python: `D:\Environments\ml-env\Scripts\python.exe` (ALWAYS use this, never system Python)
- Models dir: `D:\Environments\Models\`
- Working dir: `D:\Mtech\Sem 3\Codes\AutoWeb`
- GPU: RTX 4050 Laptop, 6.4 GB VRAM

## Step 1: Generate larger training set (~5 min)

```bash
cd "D:\Mtech\Sem 3\Codes\AutoWeb"
"D:\Environments\ml-env\Scripts\python.exe" -X utf8 Prune4Web/grounder_finetune/scripts/01_generate_data.py --split train --max-files 0 --max-samples 15000 --window-size 20 --seed 42
```

- `--max-files 0` = load ALL parquet files (not just 15)
- `--max-samples 15000` = 4x more than current 3690
- Output: `Prune4Web/grounder_finetune/data/train.jsonl` and `eval.jsonl` (overwritten)
- Log: `Prune4Web/grounder_finetune/logs/data_generation.log`

## Step 2: Fine-tune with 3 epochs (~8-10 hours estimated)

```bash
"D:\Environments\ml-env\Scripts\python.exe" -X utf8 Prune4Web/grounder_finetune/scripts/02_finetune.py --num-epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4 --max-seq-len 1024 --save-steps 500 --eval-steps 200 --log-steps 20 --seed 42
```

Changes from previous run:
- 3 epochs (was 2) -- more passes over the larger dataset
- `--save-steps 500` (was 200) -- fewer checkpoints since we have more steps
- `--eval-steps 200` (was 100) -- eval less frequently to save time
- Previous adapter at `D:\Environments\Models\Qwen2.5-0.5B-Prune4Web-Grounder\` will be OVERWRITTEN

## Step 3: Evaluate (~15 min)

```bash
"D:\Environments\ml-env\Scripts\python.exe" -X utf8 Prune4Web/grounder_finetune/scripts/03_evaluate.py --split test_task --num-samples 200 --seed 42
```

Also run baseline comparison (base model without adapter):
```bash
"D:\Environments\ml-env\Scripts\python.exe" -X utf8 Prune4Web/grounder_finetune/scripts/03_evaluate.py --split test_task --num-samples 200 --seed 42 --use-base-only
```

## Expected results
- Previous run: 3,690 train examples, 2 epochs, 462 steps -> 82.50% EA
- This run: ~13,500 train examples, 3 epochs, ~2,531 steps -> target 85-88% EA
- Training time: ~8-10 hours at ~30s/step on RTX 4050

## Output locations
- Adapter: `D:\Environments\Models\Qwen2.5-0.5B-Prune4Web-Grounder\`
- Loss curve: `Prune4Web\grounder_finetune\charts\loss_curve.png`
- Eval charts: `Prune4Web\grounder_finetune\charts\eval_results.png`, `eval_by_action.png`
- Eval JSON: `Prune4Web\grounder_finetune\data\eval_results.json`
- Logs: `Prune4Web\grounder_finetune\logs\training.log`, `evaluation.log`

## Important notes
- Run Step 1 first and verify the log shows >10,000 examples generated before starting Step 2
- Training will use ~2.7 GB VRAM -- close other GPU-heavy apps
- Do NOT interrupt training midway; if needed, the last checkpoint can be resumed
- The adapter overwrites the previous one -- the current 82.50% model will be replaced
