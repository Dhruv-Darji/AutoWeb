# 🧠 **SeeAct Architecture — Visual & Block Diagram**

Below is the **end-to-end pipeline** — from dataset loading to execution and evaluation.

```
                       ┌──────────────────────────────┐
                       │        Dataset Loader        │
                       │  Multimodal-Mind2Web (raw)   │
                       └──────────────────────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────┐
                      │   Input Preparation Module     │
                      │  - Prompt Template + Task      │
                      │  - Screenshot Image (i)        │
                      │  - History of Previous Actions │
                      │  - [Optional] HTML DOM (h)     │
                      └────────────────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────────┐
                  │          Action Generation (LMM)           │
                  │  GPT-4V Vision takes:                      │
                  │  • Screenshot                              │
                  │  • Task description                        │
                  │  • Previous step history                   │
                  │                                            │
                  │  Outputs:                                  │
                  │  ˜a = Textual action description           │
                  │  Example: “Click the ‘Find Your Truck’ btn”│
                  └────────────────────────────────────────────┘
                                      │
                                      ▼
                ┌───────────────────────────────────────────────┐
                │               Action Grounding Block          │
                │     (User chooses one grounding strategy)     │
                └───────────────────────────────────────────────┘
                                      │
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          Grounding via:                                 │
    │                                                                         │
    │   1) Element Attributes      2) Textual Choices      3) Image           │
    │      (LLM + DOM search)        (Ranker + LLM)        Annotation         │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
       │                              │                                │
       ▼                              ▼                                ▼
┌─────────────────┐         ┌────────────────────┐          ┌────────────────────┐
│ Full DOM        │         │ Top-K Candidate    │          │ Image + Bounding   │
│ + ˜a →          │         │ elements + ˜a →    │          │ boxes + labels +   │
│ LLM predicts:   │         │ LLM selects choice │          │ ˜a → LLM picks box │
│  ELEMENT TEXT   │         │ (multi-choice)     │          │ label matches text │
│  ELEMENT TYPE   │         │                    │          │ description        │
└─────────────────┘         └────────────────────┘          └────────────────────┘
       │                              │                                │
       ▼                              ▼                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                    Unified Grounding Output Parser (merges into one final action)     │
│ Outputs:                                                                            │
│   e = HTML element (id, selector, coords)                                            │
│   o = operation (click/type/select)                                                  │
│   v = value (if operation requires input)                                            │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────────┐
                       │       Execution (Playwright)      │
                       │   Sends (e,o,v) to Browser         │
                       │   Triggers actual click/type/ etc  │
                       └──────────────────────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────────┐
                       │    New Webpage State Updates      │
                       │   • Screenshot + new DOM          │
                       │   • Repeat if task not complete    │
                       └──────────────────────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────────┐
                       │      Evaluation Module            │
                       │   • Offline Metrics (Ele Acc,     │
                       │     Op F1, Step SR)               │
                       │   • Online Evaluation (live)     │
                       └──────────────────────────────────┘
```

---

# 🧠 **Block-by-Block Explanation**

---

## 📌 **1) Dataset Loader — Multimodal Mind2Web**

- The paper uses the **Multimodal-Mind2Web dataset** for training and evaluation. This includes:
  - Task descriptions
  - Screenshots aligned with HTML DOM
  - Step-by-step action annotations (e, o, v)

- It supports generalization tests across tasks, websites, and domains. ([OpenReview][2])

---

## 📌 **2) Input Preparation**

This module prepares the agent inputs:

- **Prompt template** defines what the LMM should think about (analyze page, previous actions, etc.)
- **Screenshot image** gives the visual state
- **Previous actions history** provides context
- **HTML DOM** may be optionally made available but _not always used directly in Generation_. ([Moonlight][3])

---

## 📌 **3) Action Generation Module**

**Key Purpose:**
Produce a _textual plan_ describing the next action (not directly executable yet).

**Input:**

- Screenshot image
- Task instruction
- History steps

**Output:**

```
˜a = “Move cursor to the ‘Find Your Truck’ button and click.”
```

**Note:**
The model does _not_ emit a selector or element id — just a natural language description. ([Moonlight][3])

---

## 📌 **4) Action Grounding Block (User selects method)**

This is where the model’s textual plan is **mapped to an actual HTML element and operation**.

---

### ⭐ **Grounding Method 1 — via Element Attributes**

**Input:**

- ˜a (the textual description)
- Full DOM

**Process:**

1. Model is prompted to output:

   ```
   ELEMENT: <what element they intend>
   ELEMENT TYPE: BUTTON
   ELEMENT TEXT: Find Your Truck
   ```

2. **Heuristic DOM search** is then run on HTML to find matching elements by type + text. ([OpenReview][2])

**Output Example:**

```
Element matched: <button id="6">Find Your Truck</button>
Operation: CLICK
Value: None
```

---

### ⭐ **Grounding Method 2 — via Textual Choices**

**Input:**

- ˜a
- Top-k elements (selected via a ranking model like DeBERTa)

**Process:**

1. Prepare candidates represented by HTML text
2. Present choices to LMM as a multi-choice question.
3. LMM selects the best matching choice. ([OpenReview][2])

**Output Example:**

```
ELEMENT: G (<button id="6">Find Your Truck</button>)
Action: CLICK
Value: None
```

---

### ⭐ **Grounding Method 3 — via Image Annotation**

**Input:**

- ˜a
- Screenshot with labeled bounding boxes (using Supervision lib)

**Process:**

1. Each candidate element gets a number label and a bounding box
2. LMM outputs the label corresponding to target element description. ([OpenReview][2])

**Output Example:**

```
ELEMENT: 3 (the label nearest to Find Your Truck box)
Action: CLICK
Value: None
```

---

### ⭐ **(Oracle Action Grounding)**

- Only used for evaluation
- Human annotator picks the correct grounding
- Helps compare automated methods to ideal behaviour. ([Moonlight][4])

---

## 📌 **5) Unified Grounding Output Parser**

All grounding branch outputs are normalized into the format:

```
(e, o, v)
```

Examples:

- e = button#6
- o = CLICK
- v = None

This is passed onto the **Execution** module.

---

## 📌 **6) Execution (Playwright)**

- Given (e, o, v), the execution engine performs:
  - Click
  - Type
  - Select

- Automates the task in a real browser environment. ([GitHub][5])

---

## 📌 **7) New Web State Loop**

- After action execution:
  - New screenshot
  - New HTML DOM
  - Task continues until complete

This loop enables multi-step tasks to be handled sequentially.

---

## 📌 **8) Evaluation Block**

Evaluates performance using metrics like:

- Element Accuracy
- Operation F1
- Step Success Rate
- Online evaluation success rates on live websites
  SeeAct uses both offline (cached) and online evaluations. ([OpenReview][2])

---

# 🧠 **Block Details — Action Generation vs. Action Grounding**

---

### 📦 **Action Generation**

- Uses **GPT-4V or another LMM**
- Consumes:
  - Screenshot
  - Task text
  - Previous actions

- Produces:
  - Natural language action description only

- Does _not_ use HTML DOM for grounding, only visuals + context
- Output is a _plan statement_, not actionable on its own.

**Example prompt snippet:**

```
Task: ...
Screenshot: ...
History: ...
---
Describe the next action step.
```

Produces:

```
“Scroll down to the ‘Search’ button and click it.”
```

---

### 🔗 **Action Grounding**

- Converts that description into concrete references
- Each method handles grounding differently:
  - **Element Attributes** → infer attributes → DOM search
  - **Textual Choices** → rank + choose among candidates
  - **Image Annotation** → link bounding boxes to labels

- The goal is to output:

  ```
  element_id, operation, value
  ```

---

# 📌 **Summary — why this matters**

- **Separation of concerns**:
  - Generation = _planning_
  - Grounding = _execution mapping_

- **Multiple grounding strategies** highlight how challenging it is to map text → real webpage actions. ([Moonlight][4])
- This architecture supports both offline analysis and live execution.
