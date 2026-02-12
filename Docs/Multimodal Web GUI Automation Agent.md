# 📘 Multimodal Web GUI Automation Agent

# 1. **Project Title**

**Local Multimodal GUI Automation Agent using Vision-Language Models (Qwen2-VL-2B)**

---

# 2. **Introduction / Motivation**

Modern GUI automation relies heavily on DOM inspection, brittle selectors, and manual scripting.

Such approaches fail when:

- UI design changes frequently
- HTML structure is inconsistent
- Visual-only interfaces exist (screenshots, legacy systems, remote desktops)

To overcome these limitations, **multimodal Vision-Language Models (VLMs)** enable a new kind of automation:

👉 *“Read a screenshot like a human, understand the instruction, and predict the next UI action.”*

This project builds a **complete, local, multimodal automation pipeline** using

**Qwen2-VL-2B** running on a **6GB GPU**, evaluated on real research datasets.

---

# 3. **Goal**

The goal is:

> Given a webpage screenshot + DOM + natural-language task, predict the correct UI action (click/type/scroll), decode it to a selector/coords, execute it through Selenium (Optional), and evaluate accuracy using research datasets.
> 

Your system should act like an agent:

**Observe → Understand → Act → Evaluate.**

---

# 4. **High-Level Architecture**

Here is the overall architecture diagram:

```
                   ┌────────────────────────┐
                   │   Dataset (Mind2Web,   │
                   │     GUI-Robust)        │
                   └──────────┬─────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   Preprocessor    │
                    │ (image, DOM, OCR) │
                    └──────────┬────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │   Prompt Engine(DOM) │
                   │ (strict/few-shot)    │
                   └──────────┬───────────┘
                              │ prompt + image
                              ▼
                    ┌─────────────────────┐
                    │  Qwen2-VL-2B Model  │
                    │ (local inference)   │
                    └──────────┬──────────┘
                              │ raw output
                              ▼
                 ┌────────────────────────┐
                 │   Action Decoder        │
                 │ (JSON parsing + target │
                 │  grounding via DOM/OCR)│
                 └───────────┬────────────┘
                             │ structured action
                             ▼
                   ┌─────────────────────┐
                   │   Selenium Executor │
                   │ (click/type/scroll) │
                   └───────────┬─────────┘
                               │ execution result
                               ▼
                ┌──────────────────────────┐
                │       Evaluator          │
                │ (accuracy + E2E success) │
                └──────────────────────────┘

```

---

# 5. **Component Breakdown (Inputs, Outputs, Purpose, Skeleton Code)**

---

## 5.1 **Dataset Loader**

**Purpose:**

Sample dataset entries → return screenshot, task text, DOM snippet, oracle action.

**Input:**

- Dataset directory path
- Number of samples
- File formats: JSON + PNG

**Output:**

Python dictionary:

```python
{
 "id": "mmw_001",
 "screenshot": "path/to/img.png",
 "task": "Click the login button",
 "dom": "<html> ... </html>",
 "oracle": {
    "action": "click",
    "target": {"selector": "#login"},
    "value": None
 }
}

```

**Skeleton Code:**

```python
class DatasetLoader:
    def __init__(self, root):
        self.root = root

    def load_sample(self, sample_id):
        # read json
        # read image path
        # return dict

    def sample_batch(self, n=100):
        # return random n samples
        pass

```

---

## 5.2 **Preprocessor**

**Purpose:**

- Load screenshot
- Resize, normalize
- Extract DOM elements (if available)
- Optional OCR
- Produce tensors + metadata

**Input:** screenshot path, dom snippet

**Output:**

```python
{
 "image_tensor": torch.Tensor,
 "ocr_tokens": [...],
 "dom_elements": [...],
 "meta": {"scale_x":..., "scale_y":...}
}

```

**Skeleton Code:**

```python
def preprocess_image(path, resize_w=1024):
    # open image
    # resize
    # convert to tensor
    # compute scale factors
    return image_tensor, meta

def parse_dom(dom_html):
    # extract text, selectors, bounding boxes if available
    return dom_elements

def run_ocr(image):
    # OCR tokens with bbox
    return ocr_tokens

```

---

## 5.3 **Prompt Engine**

**Purpose:**

Generate multimodal input prompt for Qwen2-VL-2B.

**Input:**

image path, task, DOM snippet, few-shot examples

**Output:**

prompt text ready for the model.

**Skeleton Code:**

```python
class PromptEngine:
    def __init__(self, template):
        self.template = template

    def build_prompt(self, task_text, dom=None, examples=None):
        # fill template
        return prompt

```

**Example Prompt Structure:**

```
You are a UI action predictor.
Output a JSON with keys: action, target, value, confidence.
Screenshot: <image>
Task: "Click Login"
Output:

```

---

## 5.4 **Model Interface (Qwen2-VL)**

**Purpose:**

Load multimodal model → perform inference → return raw text.

**Input:**

image tensor, prompt text

**Output:**

string (model-generated JSON-like text)

**Skeleton Code:**

```python
class VLModel:
    def __init__(self, model_path, device='cuda'):
        # load model + tokenizer

    def infer(self, image_tensor, prompt):
        # forward pass
        return {"raw_text": output, "latency": ms}

```

---

## 5.5 **Action Decoder**

**Purpose:**

Extract valid JSON from model output and ground target to DOM/OCR/coords.

**Input:**

raw_text, dom_elements, ocr_tokens, meta

**Output:**

sanitized structured JSON:

```python
{
 "action": "click",
 "target": {"selector":"#login", "coords":[x,y]},
 "value": None,
 "confidence": 0.88
}

```

**Skeleton Code:**

```python
def extract_json(raw_text):
    # regex to find first {...}
    # json.loads
    return parsed

def ground_target(parsed, dom_elements, ocr_tokens, meta):
    # if selector exists: return as is
    # if text label: match with DOM or OCR
    # if coords: scale them
    return parsed

```

---

## 5.6 **Selenium Executor**

**Purpose:**

Execute predicted action on a browser (mock local HTML pages).

**Input:** parsed action JSON

**Output:** execution result (success/failure + screenshot)

**Skeleton Code:**

```python
def execute_action(driver, action_json):
    if action_json["action"] == "click":
        # find element by selector
        # or fallback to coords
    elif action_json["action"] == "type":
        # find element & send_keys
    elif action_json["action"] == "scroll":
        # execute JS scroll
    return {"status": "success", "after_screenshot": path}

```

---

## 5.7 **Evaluator**

**Purpose:**

Compute quantitative metrics.

**Metrics:**

- **Action Accuracy**
- **Target Accuracy** (selector match OR coords within 24px)
- **Value Accuracy**
- **Step Success Rate (SSR)**
- **End-to-End Task Success**

**Input:**

oracle action + predicted action + exec result

**Output:**

CSV + summary dictionary

**Skeleton Code:**

```python
def evaluate(oracle, predicted, exec_result):
    # compare action
    # compare target
    # compare value
    # check success criterion
    return metrics_dict

```

---

# 6. **Evaluation Strategy**

### **Dataset-Based Evaluation**

Using Multimodal-Mind2Web (or GUI-Robust):

1. Run zero-shot prediction on N samples
2. Compare: action, target, value
3. Compute structured accuracy metrics
4. Report failure cases (hallucinated element, wrong target, invalid JSON)

---

### **End-to-End Execution Evaluation**

Using local mock webpages:

1. Screenshot + task → model prediction
2. Decoder → Selenium execution
3. Measure:
    - Was the click executed?
    - Was the input typed correctly?
    - Did URL or DOM change match expected?

---

### **Robustness Evaluation (GUI-Robust)**

1. Apply layout shifts
2. Test same tasks
3. Measure **robustness drop**

```
robustness_drop = (orig_accuracy - shifted_accuracy) / orig_accuracy

```

---

### **Human-in-Loop Evaluation**

1. Show predicted action
2. User accepts or edits
3. Measure **Human Correction Rate (HCR)**

```
HCR = edits / total_predictions

```

---

# 7. **Future Work (After POC)**

This section goes in your report under “Phase 2 / 3”.

### **1. Prompt Engineering Improvements**

- Few-shot prompts
- DOM-augmented prompts
- Retrieval-augmented few-shot examples

### **2. Lightweight Fine-Tuning (Optional, not initial goal)**

- LoRA-based fine-tuning on 500–2000 samples
- Only fine-tune small adapters
- Improve accuracy on domain-specific UIs

### **3. Agent Loop (multi-step automation)**

- After executing each step, take a new screenshot
- Feed it to the model as next observation
- Repeat until task success

### **4. Visual Grounding via Region Proposal**

- Use CLIP-style similarity to improve mapping
- Improve fallback when selectors missing

---

# 8. **Conclusion**

This documentation describes a **complete, modular, practical architecture** for a local multimodal UI automation agent capable of performing:

- screenshot understanding
- natural-language instruction following
- action prediction
- grounding
- real browser execution
- dataset-based scientific evaluation

It combines ideas from the latest research (SeeAct, Mind2Web, GUI-Robust) but builds a **resource-friendly version** running on a 6GB GPU using Qwen2-VL-2B.