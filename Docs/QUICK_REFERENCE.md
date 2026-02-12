# SeeAct Pipeline - Quick Reference

## 🎯 Core Principle

**Input**: Screenshot + Instruction  
**Output**: Valid JSON Action Prediction  
**Method**: Single-step, vision-only (no DOM, no OCR)

---

## 📋 Action Schema

Every prediction follows this exact format:

```json
{
  "action_type": "click|type|scroll|noop",
  "target": {
    "selector": "CSS selector"  // OR
    "coords": [x, y]           // OR
    "bbox": [x, y, w, h]       // OR
    null
  },
  "value": "text to type" OR null,
  "confidence": 0.0 to 1.0
}
```

### Action Types

- **click**: Click on an element
- **type**: Type text into an input field
- **scroll**: Scroll the page
- **noop**: No operation (when uncertain)

### Target Types

1. **Selector**: CSS selector string
   ```json
   {"selector": "#login-button"}
   ```

2. **Coordinates**: Absolute pixel position [x, y]
   ```json
   {"coords": [640, 360]}
   ```

3. **Bounding Box**: [x, y, width, height]
   ```json
   {"bbox": [100, 200, 50, 30]}
   ```

4. **Null**: For scroll/noop actions
   ```json
   null
   ```

---

## 🔧 Component Quick Reference

### 1. Action Schema (`action_schema.py`)

```python
from action_schema import create_action_prediction

# Create a validated action
action = create_action_prediction(
    action_type="click",
    target={"selector": "#button"},
    value=None,
    confidence=0.95
)

# Export to JSON
json_str = action.to_json()
```

### 2. JSON Repair (`json_repair.py`)

```python
from json_repair import ActionJSONRepair

# Repair malformed JSON
raw = '```json\n{"action_type": "click"}\n```'
parsed, error = ActionJSONRepair.repair_action_json(raw)
# Returns: {"action_type": "click", "target": null, "value": null, "confidence": 0.5}
```

### 3. Action Decoder (`action_decoder.py`)

```python
from action_decoder import ActionDecoder

decoder = ActionDecoder(image_width=1280, image_height=720)

# Decode raw model output
raw_output = '{"action_type": "click", "target": {"coords": [0.5, 0.5]}}'
prediction, error = decoder.decode(raw_output)

# Coordinates automatically scaled: [0.5, 0.5] → [640, 360]
```

### 4. Image Preprocessor (`image_preprocessor.py`)

```python
from image_preprocessor import SeeActImagePreprocessor
from PIL import Image

preprocessor = SeeActImagePreprocessor(
    target_width=1280,
    target_height=720
)

image = Image.open("screenshot.jpg")
result = preprocessor.preprocess(image)

# Access processed image and scale info
processed_img = result["image"]
scale_info = result["scale_info"]
```

### 5. Complete Pipeline (`seeact_pipeline.py`)

```python
from seeact_pipeline import SeeActPipeline
from PIL import Image

# Initialize
pipeline = SeeActPipeline(
    model_folder="/path/to/Qwen2-VL-2B",
    target_width=1280,
    target_height=720
)

# Predict
image = Image.open("screenshot.jpg")
result = pipeline.predict(
    image=image,
    instruction="Click the login button"
)

if result["success"]:
    pred = result["prediction"]
    print(f"Action: {pred.action_type}")
    print(f"Target: {pred.target}")
    print(f"Confidence: {pred.confidence}")
```

### 6. Evaluator (`evaluator.py`)

```python
from evaluator import ActionEvaluator

evaluator = ActionEvaluator()

# Add predictions
evaluator.add_result(
    predicted_action={"action_type": "click", "target": {...}},
    oracle_action={"action_type": "click", "target": {...}}
)

# Compute metrics
metrics = evaluator.compute_metrics()
print(f"Action Accuracy: {metrics.action_accuracy:.2%}")
print(f"Target Accuracy: {metrics.target_accuracy:.2%}")
```

---

## 🚀 Quick Start Commands

### Run Tests
```bash
# Core tests (no dependencies)
python 01_AutoWeb_POC/tests/test_core.py

# Full tests (requires dependencies)
python 01_AutoWeb_POC/tests/test_components.py
```

### Run Mock Demo
```bash
# Demonstrate pipeline without model
python 01_AutoWeb_POC/mock_demo.py
```

### Run Real Pipeline
```bash
# Single prediction
python 01_AutoWeb_POC/src/seeact_pipeline.py \
  --model /path/to/model \
  --image screenshot.jpg \
  --instruction "Click login"

# Multiple samples from Mind2Web
python 01_AutoWeb_POC/run_demo.py \
  --dataset /path/to/Mind2Web \
  --model /path/to/model \
  --num-samples 5
```

---

## 📊 Evaluation Metrics

### Action Accuracy (AA)
Percentage of predictions where `action_type` matches oracle.

### Target Accuracy (TA)
Percentage of predictions where target (selector/coords/bbox) matches oracle.

Rules:
- Selector: Exact string match
- Coords: Distance ≤ 24px from oracle
- BBox: Centroid distance ≤ 24px from oracle

### Value Accuracy (VA)
For `type` actions only: percentage where typed value matches oracle.

---

## 🐛 Common Issues & Solutions

### Issue: JSON parsing fails
**Solution**: The JSON repair layer handles this automatically. It:
- Extracts JSON from surrounding text
- Fixes trailing commas
- Removes markdown wrappers
- Adds missing defaults

### Issue: Coordinates out of range
**Solution**: Decoder automatically normalizes:
- [0-1] range → scaled to pixel coordinates
- Uses image dimensions from preprocessor

### Issue: Invalid action type
**Solution**: Validation catches this:
- Strict mode: Returns error
- Lenient mode: Falls back to "noop"

### Issue: Missing target for click/type
**Solution**: Schema validation rejects:
- Returns error message
- Can be caught and handled

---

## 📝 Best Practices

1. **Always validate**: Use `ActionSchema.validate_prediction()`
2. **Handle errors**: Check `result["success"]` and `result["error"]`
3. **Use lenient mode**: Set `strict_validation=False` for production
4. **Normalize coordinates**: Let decoder handle scaling automatically
5. **Log failures**: Use evaluator to track and analyze failures

---

## 🔗 File Structure

```
01_AutoWeb_POC/
├── src/                        # Core components
│   ├── action_schema.py        # Output contract
│   ├── json_repair.py          # JSON fixing
│   ├── action_decoder.py       # Output parser
│   ├── image_preprocessor.py   # Image processing
│   ├── mind2Web_Loader.py      # Dataset loader
│   ├── prompt_engine.py        # Prompt builder
│   ├── model_interface.py      # Model wrapper
│   ├── evaluator.py           # Metrics
│   └── seeact_pipeline.py     # End-to-end
├── tests/                      # Unit tests
│   ├── test_core.py           # Lightweight
│   └── test_components.py     # Full
├── run_demo.py                # Demo script
└── mock_demo.py               # Mock demo
```

---

## 📚 References

- **Paper**: SeeAct (OSU-NLP-Group)
- **Dataset**: Mind2Web / Multimodal-Mind2Web
- **Model**: Qwen2-VL-2B-Instruct
- **Docs**: See README.md for full documentation

---

## ✅ Validation Checklist

Before running on real data:

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Model downloaded (Qwen2-VL-2B-Instruct)
- [ ] Dataset downloaded (Mind2Web, if using)
- [ ] Tests passing (`python tests/test_core.py`)
- [ ] Mock demo works (`python mock_demo.py`)
- [ ] Single prediction works (with real model)

---

**Version**: 1.0  
**Status**: ✅ Core implementation complete  
**Next**: Run with real model and evaluate
