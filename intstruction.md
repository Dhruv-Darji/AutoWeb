```markdown
# 🌍 **PROJECT GOAL (Research + Implementation)**

Build a **local, multimodal GUI Automation Agent** using SeeAct as the base system that:

- Takes a **webpage screenshot + HTML snapshot + task instruction**
- Uses a multimodal LLM (GPT-4V or Qwen2-VL) to **generate an action plan**
- Implements a **grounding module** to map the plan to executable browser operations
- Executes actions via **Playwright (Python)** in offline and online settings
- Evaluates results (success/failure) and analyzes grounding gaps

---

# 📦 **DATASET: Multimodal-Mind2Web / Mind2Web**

We will use a subset (e.g., ~200–300 samples) for offline evaluation.  
Each sample includes:

| Field                                 | Meaning                          |
| ------------------------------------- | -------------------------------- |
| `screenshot`                          | Webpage screenshot               |
| `raw_html` / `cleaned_html`           | Webpage HTML snapshot            |
| `task_description`                    | Natural language instruction     |
| `oracle_action` (type/input/selector) | Ground-truth step for evaluation |

---

# 🧠 **MODEL & TOOLS**

- **SeeAct** (OSU-NLP-Group) as base agent framework
- **LMMs:**
  - GPT-4V API (final evaluation)
  - Qwen2-VL-2B (local/in-development)
- **Playwright (Python)** for browser automation
- Logging & analysis scripts for metrics

---

# 🏗️ **END-TO-END ARCHITECTURE**
```

Dataset Sample (screenshot + html + task)
↓
SeeAct Preprocessor (input formatting)
↓
SeeAct Action Planner (LMM → textual plan)
↓
Grounding Module (selector + bounding box resolution)
↓
Executor (Playwright)
↓
Trace Collector (logs, screenshots)
↓
Evaluator (AA, TA, TA gap, SSR, TSR)
↓
Logger & Analysis (CSV/JSON results)

````

---

# 📊 **METRICS**

* **Action Accuracy (AA)**
* **Target Accuracy (TA)**
* **Value Accuracy (VA)**
* **Structured Step F1**
* **Step Success Rate (SSR)**
* **Task Success Rate (TSR)**
* **Grounding Gap (key research metric)**
* **Latency / Efficiency**
* **Error Taxonomy / Failure Patterns**

---

# 🛠️ **IMPLEMENTATION PLAN (Day-wise)**

**Day 1:** Setup environment, clone SeeAct, install dependencies, verify minimal script.
**Day 2:** Ingest Multimodal-Mind2Web dataset, write loader.
**Day 3:** Run SeeAct offline baseline on 50–100 samples, collect logs.
**Day 4:** Configure and test online execution via Playwright (10 tasks).
**Day 5:** Run full baseline (~200 tasks), record results.
**Day 6:** Analyze baseline, categorize failures.
**Day 7:** Implement a lightweight grounding improvement (text+attr+bbox+Playwright validation).
**Day 8:** Test improved grounding on 50 tasks.
**Day 9:** Run full improved set (~200 tasks), collect results.
**Day 10:** Compare results, compute grounding gap, generate plots.
**Day 11:** Draft results section (tables, graphs, error analysis).
**Day 12:** Final polish: thesis write-up, PPT + demo.

---

# 📌 **GROUNDING IMPROVEMENT STRATEGY**

1. Extract candidate elements via fuzzy text/attribute matching.
2. Use visual proximity (bounding boxes) for refined ranking.
3. Validate candidates using Playwright (`element.isVisible()` + dry check).
4. Scoring:
   `score = w1*text_match + w2*attr_match + w3*bbox_proximity + w4*validation`
5. Persist stable selector signatures for reuse.

---

# 📝 **RESULT LOGGING & ANALYSIS**

For each task store:

* LMM plan
* Grounded selector + bbox + validation status
* Execution success/failure
* Screenshots before/after
* Failure reason

Export results as CSV/JSON for plots and tables in thesis.

---

# 📍 **EXAMPLE COMMANDS**

```bash
# Clone SeeAct
git clone https://github.com/OSU-NLP-Group/SeeAct.git
cd SeeAct

# Setup venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Playwright browsers
playwright install

# Run offline eval (example script)
python scripts/run_offline_eval.py --data_path /data/multimodal_mind2web_slice --out results/offline.json
````

---

# 📎 **DELIVERABLES**

- Baseline (offline) results
- Improved grounding results
- Metrics tables & graphs
- Error analysis & failure taxonomy
- 1-min demo video
- Thesis chapters + PPT slides

---

# 📚 **REFERENCES**

- SeeAct (OSU-NLP-Group) — multimodal web agent project
- Mind2Web / Multimodal-Mind2Web datasets
- Qwen2-VL / GPT-4V for multimodal planning
- Playwright Python for browser execution

```

---

If you want, I can generate a **matching one-page poster / summary slide** that you can paste straight into your thesis PPT as well (with metric definitions and component diagrams).
::contentReference[oaicite:0]{index=0}
```
