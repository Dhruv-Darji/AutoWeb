# SeeAct-Style Single-Step UI Action Predictor

A faithful implementation of SeeAct-style single-step UI action prediction that runs locally.

## 🎯 Goal

> **"Run a faithful SeeAct-style single-step UI action predictor locally."**

This system takes a webpage screenshot + task instruction and outputs a valid JSON action prediction.

## ✅ What This Is

- **Single-step** action prediction (not multi-step planning)
- **Vision-only** approach (no DOM, no OCR)
- **Local inference** using Qwen2-VL-2B
- **Strict JSON output** with schema validation
- **Baseline implementation** for future improvements

## ❌ What This Is NOT

- Not a full agent (no planning, no memory, no multi-step execution)
- Not improved or optimized (intentionally kept minimal per SeeAct)
- Not executing actions (prediction only)

## 🏗️ Architecture

```
Screenshot + Instruction
         ↓
   Image Preprocessor (minimal, SeeAct-style)
         ↓
   Prompt Engine (strict JSON enforcement)
         ↓
   Model Interface (Qwen2-VL-2B local)
         ↓
   JSON Repair Layer (handle malformed output)
         ↓
   Action Decoder (parse & validate)
         ↓
   ActionPrediction (structured output)
```

## 📦 Components

### 1. Action Schema (`action_schema.py`)

Defines the strict output contract:

```json
{
  "action_type": "click|type|scroll|noop",
  "target": {"selector": "..."} OR {"coords": [x, y]} OR {"bbox": [x,y,w,h]} OR null,
  "value": "text" OR null,
  "confidence": 0.0 to 1.0
}
```

### 2. JSON Repair (`json_repair.py`)

Handles common model output issues:

- Extracts JSON from surrounding text
- Fixes trailing commas
- Removes markdown wrappers
- Adds missing fields with defaults

### 3. Action Decoder (`action_decoder.py`)

Converts raw model output to validated ActionPrediction:

- Repairs JSON
- Normalizes coordinates
- Validates schema
- Handles errors gracefully

### 4. Image Preprocessor (`image_preprocessor.py`)

Minimal preprocessing per SeeAct principles:

- Resize with aspect ratio preservation
- NO OCR (vision-only)
- NO DOM parsing (intentionally limited)

### 5. Dataset Adapter (`mind2Web_Loader.py`)

Loads Mind2Web samples with oracle actions:

- Clean format: image, instruction, oracle_action
- Ready for evaluation

### 6. Evaluator (`evaluator.py`)

Prepares metrics (structure only, doesn't run yet):

- Action Accuracy (AA)
- Target Accuracy (TA)
- Value Accuracy (VA)

### 7. Pipeline (`seeact_pipeline.py`)

End-to-end integration of all components.

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Qwen2-VL-2B Model

```bash
# From Hugging Face
# Model: Qwen/Qwen2-VL-2B-Instruct
# Save to: /path/to/models/Qwen2-VL-2B-Instruct
```

### 3. Download Mind2Web Dataset (Optional)

For evaluation with real data:

```bash
# Download from: https://github.com/OSU-NLP-Group/Mind2Web
# Save to: /path/to/datasets/Mind2Web
```

## 🧪 Usage

### Run Unit Tests

```bash
cd 01_AutoWeb_POC
python tests/test_components.py
```

### Single Prediction Example

```bash
cd 01_AutoWeb_POC/src
python seeact_pipeline.py \
  --model /path/to/Qwen2-VL-2B-Instruct \
  --image /path/to/screenshot.jpg \
  --instruction "Click the login button"
```

### Demo with Mind2Web

```bash
cd 01_AutoWeb_POC
python run_demo.py \
  --dataset /path/to/Mind2Web \
  --model /path/to/Qwen2-VL-2B-Instruct \
  --num-samples 5
```

### Python API

```python
from src.seeact_pipeline import SeeActPipeline
from PIL import Image

# Initialize pipeline
pipeline = SeeActPipeline(
    model_folder="/path/to/Qwen2-VL-2B-Instruct",
    target_width=1280,
    target_height=720
)

# Load image
image = Image.open("screenshot.jpg")

# Predict action
result = pipeline.predict(
    image=image,
    instruction="Click the login button"
)

if result["success"]:
    prediction = result["prediction"]
    print(f"Action: {prediction.action_type}")
    print(f"Target: {prediction.target}")
    print(f"Confidence: {prediction.confidence}")
```

## ✅ Stopping Criterion

The system meets the stopping criterion when:

> "I can give a screenshot + instruction and my system outputs a valid JSON action prediction locally."

This is achieved when the pipeline:

1. ✅ Takes an image and instruction as input
2. ✅ Preprocesses the image (minimal, SeeAct-style)
3. ✅ Generates a strict prompt for JSON output
4. ✅ Runs local model inference (Qwen2-VL-2B)
5. ✅ Repairs and validates JSON output
6. ✅ Returns a structured ActionPrediction object

## 📊 Evaluation (Prepared, Not Yet Run)

The evaluator is prepared but not executed (as per requirements):

- **Action Accuracy (AA)**: Does predicted action_type match oracle?
- **Target Accuracy (TA)**: Does predicted target match oracle selector/coords?
- **Value Accuracy (VA)**: For type actions, does value match?

Evaluation will be run after the baseline is confirmed working.

## 🔧 Configuration

Key parameters in `seeact_pipeline.py`:

- `target_width`: Image resize width (default: 1280)
- `target_height`: Image resize height (default: 720)
- `max_new_tokens`: Model generation limit (default: 256)
- `temperature`: Sampling temperature (default: 0.0 = greedy)

## 📝 Project Structure

```
AutoWeb/
├── src/
│   ├── action_schema.py       # Output contract definition
│   ├── json_repair.py         # JSON fixing utilities
│   ├── action_decoder.py      # Output parser & validator
│   ├── image_preprocessor.py  # Minimal image preprocessing
│   ├── mind2Web_Loader.py     # Dataset adapter
│   ├── prompt_engine.py       # Prompt builder
│   ├── model_interface.py     # Qwen2-VL interface
│   ├── evaluator.py          # Metrics (structure only)
│   ├── seeact_pipeline.py    # End-to-end pipeline
│   └── preProcessor.py       # (Legacy, not used)
├── tests/
│   └── test_components.py    # Unit tests
├── run_demo.py               # Demo script
└── notebooks/                # Jupyter notebooks (legacy)
```

## 🎯 Design Principles

1. **Scope Lock**: Single-step prediction only (no planning, no memory)
2. **Output Contract**: Strict JSON schema enforced at all layers
3. **Minimal Preprocessing**: Vision-only (no OCR, no DOM)
4. **Strict Prompts**: JSON-only output, no explanations
5. **JSON Repair**: Handle model output issues gracefully
6. **Local Inference**: No API calls, fully local
7. **Evaluation Ready**: Metrics prepared but not executed yet

## 🔬 Next Steps (Future Work)

After the baseline is confirmed working:

1. Run evaluation on larger Mind2Web subset
2. Analyze failure patterns
3. Consider improvements (but baseline comes first!)

## 📚 References

- **SeeAct**: OSU-NLP-Group/SeeAct
- **Mind2Web**: OSU-NLP-Group/Mind2Web
- **Qwen2-VL**: Qwen/Qwen2-VL-2B-Instruct

## 📄 License

[Add license information]

## 👥 Contributors

[Add contributors]
