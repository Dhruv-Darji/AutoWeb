# GPU Configuration Guide

## Overview

The SeeAct project supports GPU acceleration for faster inference. This guide explains how to configure and optimize GPU usage, especially for NVIDIA RTX 4050 (6GB VRAM) and similar GPUs.

## Prerequisites

### Check GPU Availability

```bash
# Check if NVIDIA GPU is detected
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### Install CUDA-enabled PyTorch

For NVIDIA RTX 4050 (6GB VRAM), you need PyTorch with CUDA support:

```bash
# For CUDA 11.8 (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Configuration

### Option 1: Using .env File (Recommended)

Edit your `.env` file with GPU settings:

```env
# Enable GPU
DEVICE=cuda

# Use half precision (fp16) for better performance on 6GB VRAM
MODEL_DTYPE=fp16

# Disable 8-bit quantization (not needed for 6GB VRAM)
USE_8BIT=false
```

**For RTX 4050 (6GB VRAM) - Recommended Settings:**
```env
DEVICE=cuda
MODEL_DTYPE=fp16
USE_8BIT=false
```

### Option 2: Manual Configuration

You can also pass GPU settings via command-line (not implemented in current version, but .env works automatically).

## Memory Requirements

### Qwen2-VL-2B Model

| Configuration | VRAM Usage | Speed | Quality |
|---------------|------------|-------|---------|
| **fp16 (recommended for 6GB)** | ~4-5 GB | Fast | High |
| fp32 (full precision) | ~8-10 GB | Slower | High |
| 8-bit quantization | ~2-3 GB | Moderate | Good |

### For Your RTX 4050 (6GB VRAM)

✅ **Recommended**: Use `fp16` (half precision)
- Fits comfortably in 6GB VRAM
- Fast inference (~2-5 seconds per prediction)
- No quality loss for this model

❌ **Not Recommended**: `fp32` (full precision)
- Requires 8-10GB VRAM
- Will cause out-of-memory errors on 6GB GPU

⚠️ **Optional**: 8-bit quantization
- Only needed if you face memory issues
- Slightly slower than fp16
- Minimal quality difference

## Usage Examples

### With .env Configured

Once your `.env` file is set up with GPU settings, all scripts automatically use GPU:

```bash
# Enhanced evaluation - uses GPU automatically
python enhanced_eval.py --num-samples 100

# Demo - uses GPU automatically
python run_demo.py --num-samples 5

# Single prediction - uses GPU automatically
python src/seeact_pipeline.py --image screenshot.jpg --instruction "Click login"
```

### Performance Comparison

**RTX 4050 (6GB) with fp16:**
- Single prediction: ~2-5 seconds
- 100 samples: ~10-15 minutes
- Memory usage: ~4-5 GB VRAM

**CPU (for comparison):**
- Single prediction: ~15-30 seconds
- 100 samples: ~45-60 minutes
- Memory usage: ~8-12 GB RAM

## Monitoring GPU Usage

### During Inference

```bash
# In another terminal, monitor GPU in real-time
watch -n 1 nvidia-smi
```

### Expected Output

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.2    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... On   | 00000000:01:00.0  On |                  N/A |
| 40%   65C    P2    45W /  80W |   4500MiB /  6144MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

## Troubleshooting

### Out of Memory Error

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solution:**
1. Make sure you're using `fp16` in `.env`:
   ```env
   MODEL_DTYPE=fp16
   ```

2. If still failing, enable 8-bit quantization:
   ```env
   USE_8BIT=true
   ```

3. Install bitsandbytes:
   ```bash
   pip install bitsandbytes
   ```

### GPU Not Detected

**Check CUDA:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**If False:**
- Reinstall PyTorch with CUDA support (see Prerequisites above)
- Update NVIDIA drivers
- Verify GPU is not in use by another process

### Slow Performance on GPU

**Possible causes:**
1. Not using fp16:
   ```env
   MODEL_DTYPE=fp16  # Add this to .env
   ```

2. GPU throttling due to temperature
   - Check GPU temperature with `nvidia-smi`
   - Ensure good ventilation

3. Other processes using GPU
   - Check with `nvidia-smi` and close unnecessary applications

## Advanced: 8-bit Quantization

If you need to save more memory (though not necessary for 6GB RTX 4050):

### Install bitsandbytes

```bash
# Windows users may need specific wheel
pip install bitsandbytes
```

### Enable in .env

```env
USE_8BIT=true
```

### Trade-offs

- **Memory**: Reduces VRAM usage by ~50% (~2-3 GB total)
- **Speed**: Slightly slower than fp16 (~10-20% slower)
- **Quality**: Minimal difference for Qwen2-VL-2B

## Optimal Settings Summary

### For NVIDIA RTX 4050 (6GB VRAM)

**Recommended .env configuration:**
```env
# Paths (update with your paths)
MODEL_PATH=D:\Environments\Models\Qwen2-VL-2B
DATASET_PATH=D:\Environments\Datasets\multimodal-mind2web

# GPU Settings (optimal for RTX 4050)
DEVICE=cuda
MODEL_DTYPE=fp16
USE_8BIT=false
```

**Expected Performance:**
- ✅ VRAM usage: 4-5 GB (fits comfortably in 6GB)
- ✅ Speed: 2-5 seconds per prediction
- ✅ Quality: Full quality (no degradation)
- ✅ Stability: No memory errors

### Verification

After setting up, run this to verify GPU is being used:

```bash
cd 01_AutoWeb_POC
python tests/test_core.py

# Then check the pipeline initialization shows GPU
python -c "
import sys
sys.path.insert(0, 'src')
from config import get_device, get_model_dtype, get_use_8bit
print(f'Device: {get_device()}')
print(f'Dtype: {get_model_dtype()}')
print(f'8-bit: {get_use_8bit()}')
"
```

## Support

If you encounter GPU-related issues:

1. Check this guide's troubleshooting section
2. Verify your .env settings match the recommendations
3. Monitor GPU with `nvidia-smi` during inference
4. Check PyTorch CUDA installation

## Quick Reference

| Setting | Value for RTX 4050 | Purpose |
|---------|-------------------|---------|
| `DEVICE` | `cuda` | Enable GPU acceleration |
| `MODEL_DTYPE` | `fp16` | Half precision for 6GB VRAM |
| `USE_8BIT` | `false` | Not needed for 6GB VRAM |

---

**Last Updated**: 2026-02-08  
**Tested On**: NVIDIA RTX 4050 (6GB VRAM)
