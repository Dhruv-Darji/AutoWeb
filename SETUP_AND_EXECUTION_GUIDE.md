# 🚀 Complete Setup & Execution Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Guide](#installation-guide)
3. [Quick Start](#quick-start)
4. [Running Research Experiments](#running-research-experiments)
5. [Collecting Results](#collecting-results)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows with WSL
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional but recommended (NVIDIA with CUDA support)
- **Storage**: 10GB free space (for model and data)

### Recommended Setup for Research
- **OS**: Ubuntu 20.04+ or similar Linux distribution
- **Python**: 3.9 or 3.10
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with 6GB+ VRAM (for faster inference)
- **Storage**: 20GB free space

---

## Installation Guide

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Dhruv-Darji/AutoWeb.git
cd AutoWeb
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Step 4: Download Model

#### Option A: Using Hugging Face CLI (Recommended)

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login to Hugging Face (if model requires authentication)
huggingface-cli login

# Download Qwen2-VL-2B-Instruct
python - << 'EOF'
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="Qwen/Qwen2-VL-2B-Instruct",
    cache_dir="./models",
    resume_download=True
)
print(f"Model downloaded to: {model_path}")
EOF
```

#### Option B: Manual Download

1. Visit: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
2. Download all files to: `./models/Qwen2-VL-2B-Instruct/`
3. Verify structure:
   ```
   models/
   └── Qwen2-VL-2B-Instruct/
       ├── config.json
       ├── model.safetensors (or pytorch_model.bin)
       ├── tokenizer_config.json
       └── ...
   ```

### Step 5: Download Dataset (Optional - for evaluation)

```bash
# Create data directory
mkdir -p data

# Download Mind2Web dataset
# Visit: https://github.com/OSU-NLP-Group/Mind2Web
# Follow their instructions to download the dataset
# Place in: ./data/Mind2Web/
```

### Step 6: Verify Installation

```bash
# Run tests
cd 01_AutoWeb_POC
python tests/test_core.py

# Should see: "✅ All tests passed!"
```

---

## Quick Start

### Test 1: Run Mock Demo (No Model Required)

This demonstrates the pipeline without needing the actual model:

```bash
cd 01_AutoWeb_POC
python mock_demo.py
```

**Expected Output**:
- 8/8 examples should succeed
- Shows JSON parsing, validation, coordinate normalization

### Test 2: Single Prediction (Requires Model)

```bash
# Create a test image (or use your own screenshot)
cd 01_AutoWeb_POC

# Run single prediction
python src/seeact_pipeline.py \
  --model ../models/Qwen2-VL-2B-Instruct \
  --image ../data/test_screenshot.jpg \
  --instruction "Click the login button"
```

**Expected Output**:
```json
{
  "action_type": "click",
  "target": {"selector": "#login-btn"},
  "value": null,
  "confidence": 0.95
}
```

### Test 3: Demo with Mind2Web (Requires Dataset)

```bash
cd 01_AutoWeb_POC

python run_demo.py \
  --dataset ../data/Mind2Web \
  --model ../models/Qwen2-VL-2B-Instruct \
  --num-samples 5
```

---

## Running Research Experiments

### Experiment 1: Baseline Evaluation (Small Scale)

Run evaluation on a small subset to verify everything works:

```bash
cd 01_AutoWeb_POC

python run_demo.py \
  --dataset ../data/Mind2Web \
  --model ../models/Qwen2-VL-2B-Instruct \
  --num-samples 20
```

**What to Expect**:
- Runtime: ~2-5 minutes (depends on GPU)
- Output: Metrics (AA, TA, VA) printed to console
- Logs: Predictions and errors

### Experiment 2: Full Baseline Evaluation

For research-quality results, run on 100-200 samples:

```bash
cd 01_AutoWeb_POC

# Create results directory
mkdir -p ../results/baseline

# Run evaluation with logging
python run_demo.py \
  --dataset ../data/Mind2Web \
  --model ../models/Qwen2-VL-2B-Instruct \
  --num-samples 100 \
  2>&1 | tee ../results/baseline/run_log.txt
```

**Timeline**:
- 100 samples: ~10-20 minutes with GPU, ~45-60 minutes with CPU
- 200 samples: ~20-40 minutes with GPU, ~1.5-2 hours with CPU

### Experiment 3: Action Type Breakdown

Analyze performance by action type:

```bash
# This will be added in the enhanced evaluation script
python enhanced_eval.py \
  --dataset ../data/Mind2Web \
  --model ../models/Qwen2-VL-2B-Instruct \
  --num-samples 100 \
  --output ../results/baseline/detailed_results.json
```

---

## Collecting Results

### Result Files

After running experiments, you'll have:

```
results/
├── baseline/
│   ├── run_log.txt              # Console output
│   ├── detailed_results.json    # Per-sample results
│   ├── metrics_summary.json     # Aggregate metrics
│   └── failure_cases.json       # Failed predictions
```

### Metrics to Report

1. **Action Accuracy (AA)**
   - Percentage of correct action type predictions
   - Formula: `correct_actions / total_samples`

2. **Target Accuracy (TA)**
   - Percentage of correct target predictions
   - Formula: `correct_targets / total_samples`

3. **Value Accuracy (VA)**
   - For type actions only
   - Formula: `correct_values / total_type_actions`

4. **Per-Action-Type Breakdown**
   - AA, TA, VA for each action type (click, type, scroll, noop)

### Analyzing Results

```bash
# View summary
cat ../results/baseline/metrics_summary.json

# View failure cases
python -m json.tool ../results/baseline/failure_cases.json | less

# Count failures by type
grep '"action_match": false' ../results/baseline/detailed_results.json | wc -l
```

---

## Troubleshooting

### Issue 1: Out of Memory (OOM)

**Symptoms**: Process killed, "CUDA out of memory" error

**Solutions**:
1. Use 8-bit quantization:
   ```python
   pipeline = SeeActPipeline(
       model_folder="...",
       use_8bit=True  # Add this
   )
   ```

2. Reduce image size:
   ```python
   pipeline = SeeActPipeline(
       model_folder="...",
       target_width=640,   # Lower from 1280
       target_height=360   # Lower from 720
   )
   ```

3. Use CPU instead of GPU:
   ```python
   pipeline = SeeActPipeline(
       model_folder="...",
       device="cpu"
   )
   ```

### Issue 2: Model Not Loading

**Symptoms**: "Model file not found", "Config not found"

**Solutions**:
1. Verify model path:
   ```bash
   ls -la models/Qwen2-VL-2B-Instruct/
   # Should see: config.json, model files, etc.
   ```

2. Check file permissions:
   ```bash
   chmod -R 755 models/
   ```

3. Re-download model (see Installation Step 4)

### Issue 3: Dataset Loading Fails

**Symptoms**: "No parquet files found", "FileNotFoundError"

**Solutions**:
1. Verify dataset structure:
   ```bash
   ls -la data/Mind2Web/data/
   # Should see: test_task*.parquet files
   ```

2. Check dataset split name:
   ```python
   # In your code, try different splits:
   dataset = Mind2WebDataset(
       root_dir="data/Mind2Web",
       split="test_task"  # or "test_website", "train"
   )
   ```

### Issue 4: Slow Inference

**Symptoms**: Takes >5 seconds per sample

**Solutions**:
1. Use GPU if available:
   ```bash
   # Check GPU
   nvidia-smi
   
   # Verify PyTorch sees GPU
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. Use 8-bit quantization (faster but slightly less accurate):
   ```python
   use_8bit=True
   ```

3. Reduce max_new_tokens:
   ```python
   result = pipeline.predict(
       image=image,
       instruction=instruction,
       max_new_tokens=128  # Lower from 256
   )
   ```

### Issue 5: Import Errors

**Symptoms**: "ModuleNotFoundError", "ImportError"

**Solutions**:
1. Verify virtual environment is activated:
   ```bash
   which python
   # Should point to venv/bin/python
   ```

2. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

3. Check Python version:
   ```bash
   python --version
   # Should be 3.8+
   ```

---

## Performance Tips

### For Faster Inference
1. Use GPU (10-20x faster than CPU)
2. Use 8-bit quantization (2x faster, minimal accuracy loss)
3. Batch processing (if implementing)
4. Lower image resolution (2x faster, some accuracy loss)

### For Better Accuracy
1. Use full precision (no 8-bit)
2. Higher image resolution (1280x720 or higher)
3. More tokens for generation (max_new_tokens=512)
4. Temperature=0 for greedy decoding (deterministic)

### For Research Quality
1. Run on at least 100-200 samples
2. Report confidence intervals
3. Analyze failure cases
4. Compare against oracle actions
5. Document all hyperparameters

---

## Next Steps

After successful setup:

1. ✅ Run tests to verify installation
2. ✅ Run mock demo to understand pipeline
3. ✅ Run small evaluation (5-10 samples)
4. ✅ Run full baseline (100+ samples)
5. ✅ Analyze results
6. ✅ Document findings
7. ✅ Plan improvements (if needed)

---

## Getting Help

- **Documentation**: See README.md, QUICK_REFERENCE.md
- **Architecture**: See ARCHITECTURE.md for component details
- **Issues**: Check GitHub Issues
- **Tests**: Run `python tests/test_core.py` to verify setup

---

**Version**: 1.1  
**Last Updated**: 2026-02-06  
**Status**: Production-ready
