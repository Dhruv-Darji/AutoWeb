# 🌍 **1. PROJECT GOAL (Simple & Clear): Mtech Sem 3 Project -1 (Research based)**

Build a **local, multimodal GUI Automation Agent** that:

- Looks at a **webpage screenshot**
- Reads the **task instruction**
- Predicts the **correct UI action** (click/type/scroll + target element)
- And optionally **executes** it using Selenium
- Then we **evaluate** its accuracy using a real dataset (Multimodal-Mind2Web)

This is a **vision–language → action** pipeline.

---

# 📦 **2. DATASET: Multimodal-Mind2Web**

We will use **Multimodal-Mind2Web** as our _evaluation dataset_.

### What it contains (ground truth for evaluation):

Each sample provides:

| Field                                   | Meaning                                     |
| --------------------------------------- | ------------------------------------------- |
| **action_type**                         | Oracle action (“click”, “type”, “scroll”… ) |
| **action_input**                        | Value to type (if action is “type”)         |
| **action_element.element_css_selector** | Ground-truth target selector                |
| **action_element.element_xpath**        | Target’s XPath                              |
| **action_element.text**                 | Element label                               |
| **screenshot**                          | Image of the webpage                        |
| **task description**                    | Natural language step instruction           |

📌 **This gives us everything we need to compute accuracy.**

We will use _only a small slice_ (e.g., 200–300 samples) for POC.

---

# 🧠 **3. MODEL: Qwen2-VL-2B (local inference)**

Your model is:

- Vision-Language multimodal
- Runs locally on 6GB
- Can take **image + text prompt**
- Can output **structured JSON** describing actions

We do **zero-shot**, **few-shot**, and optionally **LoRA light fine-tuning**.

---

# 🏛️ **4. END-TO-END ARCHITECTURE**

Here’s the exact pipeline you will build:

```
Dataset Sample
(screenshot + task + oracle action)
        ↓
Preprocessor
(resize image, parse DOM if available, run OCR optionally)
        ↓
Prompt Engine
(build strict JSON prompt or few-shot prompt)
        ↓
Model Interface (Qwen2-VL-2B)
(image + text → raw prediction)
        ↓
Action Decoder
(parse JSON, validate fields, map text → selector/coords)
        ↓
Executor (Selenium)
(run action on local mock webpage)
        ↓
Evaluator
(compare predicted vs oracle; compute metrics)
        ↓
Logger
(store results, screenshots, run metadata)
```

### **Component Summary**

- **Preprocessor** → Converts screenshot to tensor, extracts DOM/OCR.
- **Prompt Engine** → Converts each sample into a JSON-generation prompt.
- **Model Interface** → Calls Qwen2-VL-2B locally (fp16/8bit).
- **Action Decoder** →
  - Fix malformed JSON
  - Map predicted target → correct selector
  - Convert coords back using scale info

- **Executor** → Runs predicted action in Selenium on test pages.
- **Evaluator** → Computes accuracy & success metrics.
- **(Optional) Human-in-loop** → Correct actions manually and log corrections.

---

# 📊 **5. WHAT WE WILL EVALUATE (Metrics)**

Evaluation compares **model prediction** vs **dataset ground-truth**.

This is the _heart_ of your project.

---

## 🔹 **A. Step-Level Metrics (from dataset)**

### **1. Action Accuracy (AA)**

Checks if predicted action matches `action_type`.

### **2. Target Accuracy (TA)**

Predicted selector matches oracle:

```
predicted_selector == action_element.element_css_selector
```

OR
Coordinate distance ≤ 24px from true element bbox centroid.

### **3. Value Accuracy (VA)**

If typing:

```
predicted_value == action_input
```

### **4. Structured Step F1**

Partial correctness measure based on fields (action, target, value).

---

## 🔹 **B. Trajectory-Level Metrics (optional for POC)**

### **5. Step Success Rate (SSR)**

Selenium executed step successfully (element found, click event triggered).

### **6. End-to-End Task Success Rate (TSR)**

Whether success_criterion is met:

- URL change
- DOM change
- Expected text appears

Even if dynamic tasks are recreated on local mock pages, this metric works.

---

## 🔹 **C. Efficiency & Robustness**

### **7. Latency per step**

Inference + decode + execution.

### **8. Robustness Drop**

Run the agent on perturbed layout (GUI-Robust).
Compare AA/TA/TSR before and after:

```
robustness_drop = (original - perturbed) / original
```

---

## 🔹 **D. Human-Centered**

### **9. Human Correction Rate (HCR)**

Fraction of model outputs that a human needed to edit.

---

# 🛠️ **6. DEVELOPMENT PLAN (End-to-End)**

### **Phase 1: Zero-Shot Pipeline (baseline)**

- Load 200 samples ⇒ build preprocessor ⇒ build prompt ⇒ run model ⇒ decode ⇒ compare with ground-truth
- Metrics: AA, TA, VA

### **Phase 2: Few-Shot Prompting**

- Add 2–3 examples inside prompt
- Add DOM snippet
- Try retrieval-based prompting

Compare metrics again.

### **Phase 3: Selenium Execution (for demo)**

- Build 3–5 mock webpages (login page, search page)
- Run predicted actions through Selenium
- Compute SSR + TSR
- Capture before/after screenshots for PPT

### **Phase 4: (Optional) LoRA Fine-tuning**

- Small slice (500–2000 samples)
- Low-rank adapter training (fp16/8-bit)
- Re-run metrics

### **Phase 5: Robustness Tests**

- Use GUI-Robust or apply small CSS perturbations
- Compare AA/TA/TSR drop

### **Phase 6: Final Report + PPT**

- Architecture diagram
- Dataset explanation
- Metrics tables
- Failure case images
- 3-minute demo video
- Future work (Phase-2: full agent loop)

---

# 🎬 **7. WHAT YOU WILL SHOW IN SUBMISSION**

### **In Report:**

- Clear problem statement: “mapping webpage screenshot + task → UI action”
- Architecture diagram
- Dataset details
- Metrics + tables
- Zero-shot, few-shot, LoRA comparison
- Robustness evaluation
- Human-in-loop idea
- Next-semester roadmap

### **In PPT:**

- Vision-language + GUI automation motivation
- Dataset sample visualization
- Model prompt example
- Architecture block diagram
- Evaluation graphs
- 1-minute Selenium demo
