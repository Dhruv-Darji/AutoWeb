# Implementation Gap Analysis
## Comparing Your Implementation vs Prune4Web Paper

**Date:** April 8, 2026  
**Purpose:** Identify why your results differ from Prune4Web paper and what gaps exist

---

## 1. Current Implementation Overview

### Your SeeAct-Based Implementation (src/)

| Component | Your Implementation |
|-----------|---------------------|
| **Pipeline** | SeeAct-style: Input Prep → Action Generation → Action Grounding → Evaluation |
| **Grounding Method** | DeBERTa cross-encoder ranks all elements → Top-K → LLM selects from candidates |
| **Models** | Qwen2-VL-2B (local) or GPT-4o (API) |
| **Filtering** | DeBERTa cross-encoder scoring (bi-encoder not used, cross-encoder only) |
| **Evaluation** | SeeAct metrics: Element Acc, Op F1, Step SR, Offline0/Offline1 |
| **Risk Management** | PolicyHub + HITL Confidence Gate |

### Your Prune4Web Implementation (Prune4Web/)

| Component | Your Implementation |
|-----------|---------------------|
| **Pipeline** | Planner → Programmatic Filter → Action Grounder |
| **Planner** | GPT-4o generates sub-task from screenshot + history |
| **Filter** | LLM generates keyword→weight JSON from sub-task ONLY (no DOM) |
| **Scoring** | Python script scores all elements using α/β weights (Equation 1) |
| **Grounder** | GPT-4o selects from top-20 candidates |
| **Execution** | Playwright for live browser automation |

---

## 2. Key Gaps: Why Results Differ

### Gap 1: Candidate Filtering Method (CRITICAL)

**Prune4Web Paper Approach:**
```
Planner → LLM generates {keyword: weight} from sub-task ONLY (NO DOM)
         ↓
Python scoring script traverses full DOM (no LLM)
         ↓
Top-20 candidates
```

**Your DeBERTa Approach (action_grounding.py:291-322):**
```python
# Cross-encoder ranking: feed (plan, element) pairs through DeBERTa
element_reprs = [el["representation"] for el in elements]
scores = self.deberta_loader.compute_cross_scores_batch(textual_plan, element_reprs)
scored_elements.sort(key=lambda x: x["score"], reverse=True)
top_k_elements = scored_elements[:effective_top_k]
```

**Difference:**
| Aspect | Prune4Web | Your DeBERTa |
|--------|-----------|--------------|
| **Scoring Model** | Custom α/β weights from LLM-generated keywords | DeBERTa cross-encoder |
| **LLM Token Usage** | Minimal (keywords only) | High (cross-encoder inference) |
| **Speed** | Fast (Python dict lookup) | Slower (neural inference) |
| **Accuracy** | 88.28% element acc (paper) | Unknown (need to measure) |

**Why Results Differ:**
- Prune4Web's keyword-weight approach is optimized for semantic matching with interpretable weights
- DeBERTa may struggle with certain semantic matches that keyword matching handles well
- Different recall characteristics: keyword matching may preserve more true positives

---

### Gap 2: Architecture Flow

**Prune4Web (Two-Turn):**
```
Screenshot + Task + History
         ↓
┌─────────────────┐
│ Turn 1: Planner │ → Sub-task St
└────────┬────────┘
         ↓
┌─────────────────────┐
│ Turn 1: Filter      │ → Keywords (LLM call)
│ Python Scoring      │ → Top-20 (no LLM)
└────────┬────────────┘
         ↓
┌─────────────────┐
│ Turn 2: Grounder│ → Element UID (LLM call)
└─────────────────┘
```

**Your SeeAct (Sequential):**
```
Screenshot + Task + History + Policy
         ↓
┌─────────────────────────┐
│ Action Generation       │ → Action Plan (LLM call)
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│ DeBERTa Ranking         │ → Top-50 (neural scoring)
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│ LLM Element Selection   │ → Element (LLM call)
└─────────────────────────┘
```

**Key Differences:**
1. **Prune4Web** explicitly separates planning from element selection
2. **Your approach** combines planning and element identification in action generation
3. **Prune4Web** uses fewer LLM calls per step (2 vs potentially more)

---

### Gap 3: Evaluation Metrics

**Prune4Web Paper Claims:**
- Element Accuracy: **88.28%** (with perfect sub-tasks)
- Recall@20: **97.6%**
- Step Success Rate: **52.4%** (end-to-end)

**Your Evaluation (seeact_evaluation.py):**
- Element Accuracy: (calculated via backend_node_id match)
- Operation F1
- Step Success Rate
- Offline0/Offline1 (task-level)

**Critical Issue:**
Your evaluation compares `pred_backend_id` against `gt_pos_backend_ids` from the dataset. Make sure:
1. Ground truth extraction is correct (lines 211-260 in seeact_evaluation.py)
2. Backend node ID matching is working properly
3. You're testing on the same dataset split as Prune4Web

---

### Gap 4: Privacy Implementation

**Current State:**
- PolicyHub exists (policy_hub.py) but focuses on **action policies** (don't delete, payment, etc.)
- **NO DOM-level PII masking**
- **NO differential privacy on element attributes**
- **NO privacy-aware LLM prompting**

**What You Need for Privacy-Aware Research:**

```python
# src/privacy_filters.py (NEEDED)
- detect_pii_in_dom(dom_elements)  # regex/NER for emails, phones, names
- mask_sensitive_attributes(element)  # hash IDs, obfuscate text
- apply_differential_privacy(scores)  # add noise to element scores
- privacy_prompt_wrapper(prompt)  # inject privacy instructions
```

**Current PolicyHub vs Needed Privacy:**
| Feature | PolicyHub (exists) | Privacy Filters (needed) |
|---------|-------------------|-------------------------|
| Action risk detection | ✓ | ✗ |
| PII detection in DOM | ✗ | ✓ |
| Element anonymization | ✗ | ✓ |
| Differential privacy | ✗ | ✓ |
| Privacy budget tracking | ✗ | ✓ |

---

## 3. DOM Delta Processing Gap

**Current State:**
- No DOM delta processing exists in your codebase
- Each step processes full DOM independently
- No comparison between before/after states

**What DOM Delta Processing Should Do:**

```python
# src/dom_delta_processor.py (NEEDED)

class DOMDeltaProcessor:
    def capture_state(self, html: str) -> DOMState:
        """Serialize DOM to comparable state"""
        
    def compute_delta(self, state_before: DOMState, state_after: DOMState) -> DOMDelta:
        """
        Returns:
        - added_nodes: List[ElementNode]
        - removed_nodes: List[ElementNode]
        - modified_nodes: List[(ElementNode, ElementNode)]  # before, after
        """
        
    def get_relevant_elements(self, delta: DOMDelta, task_context: str) -> List[ElementNode]:
        """Return only elements relevant to current task step"""
```

**Benefits for Your Research:**
1. **Token Reduction:** Only changed elements go to LLM
2. **Accuracy:** Focus LLM attention on what changed
3. **Privacy:** Can mask unchanged elements more aggressively

---

## 4. Action Plan to Match/Get Prune4Web Results

### Option A: Fix Prune4Web Implementation (Recommended)

Your `Prune4Web/run_prune4web.py` looks correct! But you need:

1. **Run it on Mind2Web dataset** (currently only runs on live URLs)
2. **Add evaluation metrics** matching paper (Recall@N, Element Accuracy)
3. **Compare with SeeAct baseline** using same dataset split
4. **Debug differences** in scoring function

**Specific Code to Add:**
```python
# In Prune4Web/ - add evaluation module

class Prune4WebEvaluator:
    def evaluate_on_mind2web(self, dataset_split: str):
        """Run Prune4Web on Mind2Web and compute metrics"""
        
    def compute_recall_at_n(self, n: int = 20) -> float:
        """Recall@N metric from paper"""
        # Percentage of ground-truth elements in top-N candidates
```

### Option B: Enhance Your SeeAct with Prune4Web Filtering

Replace DeBERTa ranking with Prune4Web-style programmatic filter:

```python
# In action_grounding.py - new method

def _ground_using_prune4web_filter(self, annotation_id, textual_plan, step_info):
    # 1. Generate keywords from plan (LLM call)
    keywords = self.generate_keyword_weights(textual_plan)
    
    # 2. Score elements using α/β weights (Python only)
    elements = extract_interactive_elements(html)
    candidates = score_elements(elements, keywords, top_n=20)
    
    # 3. LLM selects from candidates
    selected = self.select_element_from_candidates(textual_plan, candidates, ...)
```

---

## 5. Privacy-Aware Implementation Path

To add Privacy-Aware LLM to your research:

### Step 1: PII Detection (Day 1-2)
```python
# src/privacy/pii_detector.py

class PIIDetector:
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
    }
    
    def detect_in_element(self, element: Dict) -> List[str]:
        """Return list of PII types found in element"""
```

### Step 2: Element Anonymization (Day 3)
```python
# src/privacy/element_anonymizer.py

class ElementAnonymizer:
    def anonymize(self, element: Dict, sensitivity: float = 0.5) -> Dict:
        """
        - Hash element IDs
        - Replace text with semantic category
        - Preserve structure for interaction
        """
```

### Step 3: Differential Privacy (Day 4)
```python
# src/privacy/dp_scorer.py

class DPScorer:
    def add_noise(self, scores: List[float], epsilon: float = 1.0) -> List[float]:
        """Add Laplace noise to element scores"""
```

### Step 4: Integration (Day 5-6)
- Modify `action_grounding.py` to apply privacy filters before scoring
- Add privacy metrics to evaluation
- Compare: No Privacy vs Prune4Web-style vs Privacy-Aware

---

## 6. Quick Debugging Checklist

### To Verify Prune4Web Implementation:

1. **Run on same dataset split as paper**
   ```bash
   # Prune4Web paper uses Mind2Web test split
   python -m Prune4Web.evaluate --dataset mind2web --split test
   ```

2. **Check Recall@20 calculation**
   ```python
   # In your evaluation
gt_uid = ground_truth_element['uid']
   recall_hit = any(c.uid == gt_uid for c in candidates[:20])
   ```

3. **Verify element scoring**
   ```python
   # Compare your scoring with paper's Equation 1
   # Paper: S(e) = Σ_k w_base(k) · Σ_(a,β) · Σ_m [α_m · β_a · 1{match_m(k,a)}]
   # Your: Check if implementation matches
   ```

4. **Compare candidate pool sizes**
   - Prune4Web: 25-50x reduction
   - Your DeBERTa: What reduction factor?

---

## 7. Summary of Gaps

| Gap | Severity | Your Status | Action Needed |
|-----|----------|-------------|---------------|
| Candidate filtering method | HIGH | DeBERTa instead of keyword weights | Implement Prune4Web filter OR compare both |
| Evaluation on same data | HIGH | SeeAct eval exists, need Prune4Web eval | Add Prune4Web evaluation module |
| DOM Delta Processing | HIGH | Not implemented | Create dom_delta_processor.py |
| Privacy-aware LLM | HIGH | Only PolicyHub (action risks) | Add PII masking, DP, anonymization |
| Two-turn protocol | MEDIUM | Different architecture | Document difference, justify approach |
| Recall@N metrics | MEDIUM | Not in SeeAct eval | Add to evaluation |

---

## 8. Recommended Priority Order

**For 20-day submission:**

1. **Days 1-3:** Implement DOM Delta Processing
2. **Days 4-6:** Implement Privacy Filters + DP
3. **Days 7-8:** Integrate with Prune4Web pipeline
4. **Days 9-10:** Run comparative experiments (SeeAct vs Prune4Web vs Yours)
5. **Days 11-20:** Writing (paper, thesis, presentation)

**Key Insight:**
Your Prune4Web implementation in `run_prune4web.py` looks correct! The main issue is likely that you're not evaluating it on the same benchmark (Mind2Web) with the same metrics (Recall@20, Element Accuracy) as the paper.

---

*Analysis completed. Priority: Fix evaluation pipeline, then add DOM Delta + Privacy features.*
