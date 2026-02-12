# 📊 Visual Guide: How to Get Research Results

## 🎯 Your Goal
Get publication-quality research results from your SeeAct implementation.

---

## 📋 Complete Workflow (Visual)

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: VERIFY INSTALLATION (5 minutes)                    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
        cd 01_AutoWeb_POC
        python tests/test_core.py
                          │
                          ▼
                    ✅ 5/5 tests pass?
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
       YES                                 NO
        │                                   │
        ▼                                   ▼
   Continue                    Fix issues (see troubleshooting)


┌─────────────────────────────────────────────────────────────┐
│ STEP 2: INSTALL DEPENDENCIES (15 minutes)                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
                          │
                          ▼
              ✅ No errors?
                          │
                         YES
                          │
                          ▼


┌─────────────────────────────────────────────────────────────┐
│ STEP 3: DOWNLOAD MODEL (30-60 minutes)                     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
        pip install huggingface_hub
                          │
                          ▼
        python -c "
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id='Qwen/Qwen2-VL-2B-Instruct',
            cache_dir='./models'
        )"
                          │
                          ▼
        ls models/Qwen2-VL-2B-Instruct/
                          │
                          ▼
        ✅ See config.json and model files?
                          │
                         YES
                          │
                          ▼


┌─────────────────────────────────────────────────────────────┐
│ STEP 4: GET DATASET (Optional - 10-30 minutes)             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
    Download Mind2Web from:
    https://github.com/OSU-NLP-Group/Mind2Web
                          │
                          ▼
    Place in: data/Mind2Web/
                          │
                          ▼
    ls data/Mind2Web/data/
                          │
                          ▼
    ✅ See .parquet files?
                          │
                         YES
                          │
                          ▼


┌─────────────────────────────────────────────────────────────┐
│ STEP 5: RUN SMALL TEST (5-10 minutes)                      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
        python enhanced_eval.py \
          --dataset ../data/Mind2Web \
          --model ../models/Qwen2-VL-2B-Instruct \
          --num-samples 10
                          │
                          ▼
    ✅ Gets to 10/10 samples? (even if accuracy is low)
                          │
                         YES
                          │
                          ▼


┌─────────────────────────────────────────────────────────────┐
│ STEP 6: RUN RESEARCH EVALUATION (30-60 minutes)            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
        python enhanced_eval.py \
          --dataset ../data/Mind2Web \
          --model ../models/Qwen2-VL-2B-Instruct \
          --num-samples 100 \
          --output ../results
                          │
                          ▼
    Progress: [##########] 100/100 (100%)
                          │
                          ▼
    ✅ Completed successfully!
                          │
                          ▼


┌─────────────────────────────────────────────────────────────┐
│ STEP 7: ANALYZE RESULTS (10 minutes)                       │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
    ls ../results/run_*/
                          │
                          ▼
    ├── detailed_results.json
    ├── metrics_summary.json
    ├── failure_cases.json
    └── report.txt
                          │
                          ▼
    cat ../results/run_*/report.txt
                          │
                          ▼
    📊 YOUR RESEARCH RESULTS! 📊
```

---

## 📊 Expected Output Example

```
================================================================================
SeeAct Baseline Evaluation Report
================================================================================

Timestamp: 2026-02-06T12:00:00
Samples Evaluated: 100
Total Time: 1234.5s
Avg Time per Sample: 12.35s

--------------------------------------------------------------------------------
Overall Metrics
--------------------------------------------------------------------------------

Action Accuracy (AA):  68.00% (68/100)
Target Accuracy (TA):  53.00% (53/100)
Value Accuracy (VA):   78.00% (18/23)

--------------------------------------------------------------------------------
Per-Action-Type Breakdown
--------------------------------------------------------------------------------

CLICK:
  Samples: 45
  Action Accuracy: 75.56%
  Target Accuracy: 60.00%
  Value Accuracy:  N/A
  Overall Accuracy: 57.78%

TYPE:
  Samples: 23
  Action Accuracy: 65.22%
  Target Accuracy: 47.83%
  Value Accuracy:  78.00%
  Overall Accuracy: 43.48%

SCROLL:
  Samples: 20
  Action Accuracy: 60.00%
  Target Accuracy: 50.00%
  Value Accuracy:  N/A
  Overall Accuracy: 45.00%

NOOP:
  Samples: 12
  Action Accuracy: 50.00%
  Target Accuracy: N/A
  Value Accuracy:  N/A
  Overall Accuracy: 50.00%
```

---

## 🎓 For Your Research Paper

### Method Section

```
We evaluate the SeeAct baseline using Qwen2-VL-2B-Instruct (2B parameters)
on the Mind2Web test set. Following the original SeeAct approach, we use
a vision-only setup with no DOM or OCR information. Images are resized to
1280×720 while maintaining aspect ratio. We use greedy decoding (temperature=0)
for deterministic predictions.
```

### Results Table

```
Table 1: Baseline Performance on Mind2Web Test Set (n=100)

Action Type    Count    AA      TA      VA      Overall
-------------------------------------------------------
Click          45      75.6%   60.0%    -      57.8%
Type           23      65.2%   47.8%   78.0%   43.5%
Scroll         20      60.0%   50.0%    -      45.0%
Noop           12      50.0%    -       -      50.0%
-------------------------------------------------------
Overall        100     68.0%   53.0%   78.0%   51.0%

AA = Action Accuracy, TA = Target Accuracy, VA = Value Accuracy
```

### Key Findings

```
1. Click actions achieve highest accuracy (75.6%)
2. Type actions show good value accuracy (78%) despite lower target accuracy
3. Vision-only approach shows clear limitations (53% target accuracy overall)
4. Further improvements needed for reliable web automation
```

---

## 🚨 Common Issues & Quick Fixes

### Issue: Out of Memory

```bash
# Symptom: Process killed during inference

# Quick Fix: Use 8-bit quantization
# Edit enhanced_eval.py, line ~50:
pipeline = SeeActPipeline(
    model_folder=model_path,
    target_width=1280,
    target_height=720,
    use_8bit=True  # ← Add this line
)
```

### Issue: Model Not Found

```bash
# Symptom: "FileNotFoundError: config.json not found"

# Quick Fix: Check model path
ls -la models/Qwen2-VL-2B-Instruct/

# Should see:
# - config.json
# - model.safetensors (or pytorch_model.bin)
# - tokenizer files

# If missing, re-download
```

### Issue: Dataset Not Loading

```bash
# Symptom: "No parquet files found"

# Quick Fix: Verify dataset structure
ls data/Mind2Web/data/

# Should see: test_task_*.parquet files

# If wrong split name, try:
python enhanced_eval.py --split test_website
# or
python enhanced_eval.py --split train
```

### Issue: Slow Inference

```bash
# Symptom: >5 seconds per sample

# Quick Fix 1: Use GPU
nvidia-smi  # Check if GPU available

# Quick Fix 2: Reduce image size
# Edit enhanced_eval.py:
pipeline = SeeActPipeline(
    model_folder=model_path,
    target_width=640,   # ← Lower from 1280
    target_height=360   # ← Lower from 720
)

# Quick Fix 3: Use 8-bit quantization
use_8bit=True
```

---

## ⏱️ Time Estimates

### With GPU (NVIDIA 6GB+)
- 10 samples: ~1-2 minutes
- 20 samples: ~2-4 minutes
- 50 samples: ~5-10 minutes
- 100 samples: ~10-20 minutes
- 200 samples: ~20-40 minutes

### With CPU (slower)
- 10 samples: ~5-10 minutes
- 20 samples: ~10-20 minutes
- 50 samples: ~25-50 minutes
- 100 samples: ~50-100 minutes
- 200 samples: ~100-200 minutes

---

## ✅ Verification Checklist

Before running full experiments:

```
Setup:
☐ Virtual environment created and activated
☐ Dependencies installed (pip list shows torch, transformers, etc.)
☐ Tests passing (python tests/test_core.py → 5/5)
☐ Mock demo working (python mock_demo.py → 8/8)

Model:
☐ Model downloaded to models/Qwen2-VL-2B-Instruct/
☐ config.json exists
☐ Model files exist (*.safetensors or *.bin)

Dataset:
☐ Mind2Web downloaded to data/Mind2Web/
☐ Parquet files exist in data/Mind2Web/data/
☐ At least 100 samples available in chosen split

Resources:
☐ At least 8GB RAM available
☐ At least 10GB disk space free
☐ GPU available (optional but recommended)

Ready to run!
```

---

## 📈 Result Quality Indicators

### Good Results ✅
- Action Accuracy: 60-75%
- Completes without errors
- Per-action breakdown makes sense
- Latency is consistent

### Need Investigation ⚠️
- Action Accuracy: <50%
- Many JSON parsing errors
- Wildly varying latencies
- Specific action types all failing

### Known Limitations 📝
- Target accuracy usually 10-15% lower than action accuracy
- Noop often has lower accuracy (harder to predict when to do nothing)
- Type actions: high value accuracy but lower target accuracy
- Small model (2B params) has inherent limitations

---

## 🎯 Success Criteria

You've successfully run research experiments when:

1. ✅ Evaluation completes on 100+ samples
2. ✅ Results saved to timestamped directory
3. ✅ All 4 output files generated (JSON + text)
4. ✅ Metrics are in expected ranges (50-75%)
5. ✅ Per-action breakdown is reasonable
6. ✅ You can explain the results

---

## 📚 Quick Reference Links

- **Full Setup**: SETUP_AND_EXECUTION_GUIDE.md
- **API Reference**: QUICK_REFERENCE.md
- **What Changed**: PROJECT_UPDATE_SUMMARY.md
- **Version History**: CHANGELOG.md
- **Troubleshooting**: SETUP_AND_EXECUTION_GUIDE.md (section 6)

---

**Next**: Run your first experiment and get results! 🚀
