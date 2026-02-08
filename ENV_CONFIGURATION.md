# Environment Configuration Guide

## Overview

The project now supports loading model and dataset paths from a `.env` file, making it easier to configure paths once and use them across all scripts.

## Setup Instructions

### 1. Create .env file

Copy the example file and customize it:

```bash
# Copy the example
cp .env.example .env

# Edit with your paths
# Windows: notepad .env
# Linux/Mac: nano .env
```

### 2. Configure Your Paths

Edit `.env` with your actual paths:

**Example for Windows:**
```env
MODEL_PATH=D:\Environments\Models\Qwen2-VL-2B
DATASET_PATH=D:\Environments\Datasets\multimodal-mind2web
```

**Example for Linux/Mac:**
```env
MODEL_PATH=/home/user/models/Qwen2-VL-2B-Instruct
DATASET_PATH=/home/user/datasets/multimodal-mind2web
```

### 3. Install python-dotenv

```bash
pip install python-dotenv
# or
pip install -r requirements.txt
```

## Usage

### With .env File (Recommended)

Once configured, you can run scripts without specifying paths:

```bash
# Enhanced evaluation - no paths needed!
python enhanced_eval.py --num-samples 100

# Demo
python run_demo.py --num-samples 5

# Single prediction
python src/seeact_pipeline.py --image screenshot.jpg --instruction "Click login"
```

### Without .env File (Manual)

You can still provide paths manually:

```bash
# Enhanced evaluation
python enhanced_eval.py \
  --model "D:\Environments\Models\Qwen2-VL-2B" \
  --dataset "D:\Environments\Datasets\multimodal-mind2web" \
  --num-samples 100

# Demo
python run_demo.py \
  --model "D:\Environments\Models\Qwen2-VL-2B" \
  --dataset "D:\Environments\Datasets\multimodal-mind2web"
```

## Configuration Options

| Variable | Description | Required |
|----------|-------------|----------|
| `MODEL_PATH` | Path to Qwen2-VL-2B model directory | Yes |
| `DATASET_PATH` | Path to Multimodal Mind2Web dataset | Yes (for evaluation scripts) |
| `RESULTS_PATH` | Path for results output (default: `../results`) | No |

## Notes

- **Windows Users**: Use forward slashes `/` or escaped backslashes `\\` in paths
  - Good: `D:/Environments/Models/Qwen2-VL-2B`
  - Good: `D:\\Environments\\Models\\Qwen2-VL-2B`
  - Also works: `D:\Environments\Models\Qwen2-VL-2B` (as shown in .env.example)

- **Security**: The `.env` file is automatically ignored by git (listed in `.gitignore`)

- **Priority**: Command-line arguments override `.env` values if both are provided

## Troubleshooting

### "ℹ No .env file found"
- Create a `.env` file by copying `.env.example`
- Make sure it's in the project root directory

### "python-dotenv not installed"
- Install it: `pip install python-dotenv`

### Paths not working
- Check that paths exist on your system
- On Windows, you can use forward slashes or double backslashes
- Remove quotes around paths in the .env file

## Example Workflow

1. **First time setup:**
   ```bash
   cp .env.example .env
   # Edit .env with your paths
   pip install python-dotenv
   ```

2. **Run evaluation:**
   ```bash
   python enhanced_eval.py --num-samples 10
   ```

3. **Success!** The script automatically loads paths from `.env`

## Benefits

✅ Configure paths once, use everywhere  
✅ No need to type long paths every time  
✅ Easy to switch between different environments  
✅ Secure (`.env` is git-ignored)  
✅ Team-friendly (each developer has their own `.env`)
