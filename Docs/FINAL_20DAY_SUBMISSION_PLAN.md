# 20-Day Mtech Submission Plan
## "Improving Web Automation with DOM Delta Processing and Privacy-Aware LLM Agents"

**Start Date:** April 9, 2026  
**Submission Deadline:** April 29, 2026 (20 days)  
**Current Status:** Prune4Web baseline implemented, DOM Delta & Privacy components pending

---

## Executive Summary

This plan divides the remaining 20 days into **4 phases**:
1. **Implementation Phase (Days 1-8):** DOM Delta Processing + Privacy-Aware LLM + Working Prototype
2. **Evaluation Phase (Days 9-11):** Comparative experiments and results generation
3. **Writing Phase (Days 12-16):** Research paper + Thesis compilation
4. **Presentation Phase (Days 17-20):** Final presentation + Practice + Submission prep

**Daily Work Hours Recommended:** 10-12 hours/day (High intensity required)

---

## Phase 1: Implementation (Days 1-8) - Critical Path

### Day 1: April 9 (Wednesday) - Foundation
**Focus:** DOM Delta Processing Architecture Design

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-12:00 | Design DOM Delta Processing architecture | Architecture diagram |
| 12:00-13:00 | Lunch break | - |
| 13:00-17:00 | Create DOM state capture module | `src/dom_state.py` |
| 17:00-18:00 | Document approach in research notes | Notes file |
| 18:00-21:00 | Implement DOM serialization (hash-based) | DOM snapshot working |

**Day 1 Deliverables:**
- [ ] Architecture diagram for DOM Delta Processing
- [ ] `src/dom_state.py` - DOM snapshot capture
- [ ] Research notes on delta algorithm design

---

### Day 2: April 10 (Thursday) - Core Algorithm
**Focus:** DOM Diff Algorithm Implementation

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-13:00 | Implement tree diff algorithm (Myers or custom) | `src/dom_diff.py` |
| 13:00-14:00 | Lunch break | - |
| 14:00-17:00 | Handle edge cases: text changes, attribute updates, node moves | Unit tests passing |
| 17:00-19:00 | Integrate with Prune4Web pipeline | Delta processing integrated |
| 19:00-21:00 | Test on 3 sample websites | Results log |

**Day 2 Deliverables:**
- [ ] `src/dom_diff.py` - Tree comparison algorithm
- [ ] Unit tests for diff algorithm
- [ ] Integration with Prune4Web pipeline

---

### Day 3: April 11 (Friday) - Privacy Framework
**Focus:** Privacy-Aware LLM Implementation

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-12:00 | Design privacy filtering system (PII detection) | `src/privacy_filters.py` skeleton |
| 12:00-13:00 | Lunch break | - |
| 13:00-17:00 | Implement element anonymization (text hashing, ID obfuscation) | Privacy filters working |
| 17:00-18:00 | Implement differential privacy for attributes | DP module ready |
| 18:00-21:00 | Create privacy-preserving prompt builder | Privacy pipeline working |

**Day 3 Deliverables:**
- [ ] `src/privacy_filters.py` - PII detection and masking
- [ ] `src/privacy_aware_llm.py` - Privacy wrapper for LLM calls
- [ ] Privacy configuration system

---

### Day 4: April 12 (Saturday) - Integration
**Focus:** Combine DOM Delta + Privacy + Prune4Web

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-13:00 | Integrate delta processor with Prune4Web stages | `src/prune4web_delta.py` |
| 13:00-14:00 | Lunch break | - |
| 14:00-17:00 | Add privacy layer to all LLM calls | Privacy integration complete |
| 17:00-19:00 | Implement terminal-based automation runner | CLI tool working |
| 19:00-21:00 | Debug integration issues | All modules working together |

**Day 4 Deliverables:**
- [ ] `src/prune4web_delta.py` - Integrated system
- [ ] Terminal-based automation interface
- [ ] End-to-end pipeline working

---

### Day 5: April 13 (Sunday) - Browser Showcase
**Focus:** Selenium/Playwright Showcase Implementation

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-12:00 | Create showcase script with live browser | `showcase/web_automation_demo.py` |
| 12:00-13:00 | Lunch break | - |
| 13:00-17:00 | Add visual annotations (highlight elements before action) | Visual feedback working |
| 17:00-19:00 | Create demo scenarios (flight booking, form filling) | 3 demo scenarios |
| 19:00-21:00 | Record demo videos/screenshots | Demo assets ready |

**Day 5 Deliverables:**
- [ ] Live browser automation showcase
- [ ] Visual feedback system (element highlighting)
- [ ] 3 working demo scenarios
- [ ] Screenshot/video assets for presentation

---

### Day 6: April 14 (Monday) - Testing & Refinement
**Focus:** Testing and Bug Fixes

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-12:00 | Test on 10 real websites | Test results spreadsheet |
| 12:00-13:00 | Lunch break | - |
| 13:00-16:00 | Fix bugs and edge cases | Stable implementation |
| 16:00-18:00 | Performance optimization | Optimized code |
| 18:00-21:00 | Documentation of implementation | Technical documentation |

**Day 6 Deliverables:**
- [ ] Test results from 10 websites
- [ ] Bug fixes complete
- [ ] Performance benchmarks
- [ ] Implementation documentation

---

### Day 7: April 15 (Tuesday) - Comparative Baselines
**Focus:** Implement Comparison Approaches

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-12:00 | Implement baseline: Standard LLM (no delta, no privacy) | `baselines/standard_llm.py` |
| 12:00-13:00 | Lunch break | - |
| 13:00-16:00 | Implement baseline: Prune4Web only (no delta/privacy) | `baselines/prune4web_only.py` |
| 16:00-18:00 | Research 2-3 related papers for comparison | Paper comparison notes |
| 18:00-21:00 | Set up evaluation framework | `experiments/evaluate.py` |

**Day 7 Deliverables:**
- [ ] Baseline implementations
- [ ] Paper comparison matrix
- [ ] Evaluation framework

---

### Day 8: April 16 (Wednesday) - Buffer/Completion
**Focus:** Complete any pending implementation

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-12:00 | Complete any pending implementation tasks | All code complete |
| 12:00-13:00 | Lunch break | - |
| 13:00-17:00 | Code review and refactoring | Clean codebase |
| 17:00-19:00 | Final integration testing | All tests passing |
| 19:00-21:00 | Prepare for evaluation phase | Evaluation dataset ready |

**Day 8 Deliverables:**
- [ ] Complete implementation
- [ ] Code reviewed and cleaned
- [ ] Evaluation dataset prepared

---

## Phase 2: Evaluation (Days 9-11)

### Day 9: April 17 (Thursday) - Experiment Execution
**Focus:** Run All Comparative Experiments

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-13:00 | Run experiments: Standard LLM baseline | Results saved |
| 13:00-14:00 | Lunch break | - |
| 14:00-18:00 | Run experiments: Prune4Web baseline | Results saved |
| 18:00-21:00 | Run experiments: Our approach (Delta + Privacy) | Results saved |

**Day 9 Deliverables:**
- [ ] All baseline results
- [ ] Our approach results
- [ ] Raw data organized

---

### Day 10: April 18 (Friday) - Analysis
**Focus:** Results Analysis and Visualization

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-12:00 | Compute all metrics (accuracy, tokens, latency, privacy score) | Metrics computed |
| 12:00-13:00 | Lunch break | - |
| 13:00-17:00 | Create comparison tables and charts | Figures ready |
| 17:00-19:00 | Statistical significance testing | p-values computed |
| 19:00-21:00 | Document results section | Results notes |

**Metrics to Track:**
- Element Accuracy (EA)
- Operation F1
- Step Success Rate (SSR)
- Token Usage (input/output)
- Latency per step
- DOM Reduction Factor
- Privacy Score (elements masked / total sensitive elements)

**Day 10 Deliverables:**
- [ ] Metrics computed for all approaches
- [ ] Comparison tables
- [ ] Charts/figures for paper
- [ ] Statistical analysis

---

### Day 11: April 19 (Saturday) - Results Finalization
**Focus:** Finalize Results and Key Findings

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-12:00 | Write results section draft | Results section written |
| 12:00-13:00 | Lunch break | - |
| 13:00-16:00 | Create ablation studies (with/without delta, with/without privacy) | Ablation results |
| 16:00-18:00 | Identify key findings and contributions | Key findings document |
| 18:00-21:00 | Review all results for consistency | Finalized results |

**Day 11 Deliverables:**
- [ ] Results section draft
- [ ] Ablation study results
- [ ] Key findings documented

---

## Phase 3: Writing (Days 12-16)

### Day 12: April 20 (Sunday) - Paper Structure
**Focus:** Research Paper Draft - Core Sections

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-12:00 | Write Abstract and Introduction | Sections drafted |
| 12:00-13:00 | Lunch break | - |
| 13:00-17:00 | Write Related Work | Related work section |
| 17:00-19:00 | Write Methodology (DOM Delta + Privacy) | Methodology section |
| 19:00-21:00 | Review and refine sections | Sections reviewed |

**Paper Structure (8-10 pages):**
1. Abstract (150-250 words)
2. Introduction (1-1.5 pages)
3. Related Work (1-1.5 pages)
4. Methodology (2-3 pages)
   - DOM Delta Processing
   - Privacy-Aware LLM
5. Experiments & Results (2 pages)
6. Discussion & Limitations (0.5 page)
7. Conclusion (0.5 page)

**Day 12 Deliverables:**
- [ ] Abstract drafted
- [ ] Introduction section
- [ ] Related Work section
- [ ] Methodology section

---

### Day 13: April 21 (Monday) - Paper Completion
**Focus:** Complete Research Paper

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-12:00 | Write Experiments & Results section (from Day 11 notes) | Results section |
| 12:00-13:00 | Lunch break | - |
| 13:00-15:00 | Write Discussion and Conclusion | Closing sections |
| 15:00-17:00 | Compile references | Reference list complete |
| 17:00-21:00 | Full paper review and formatting | Complete draft |

**Day 13 Deliverables:**
- [ ] Complete research paper draft
- [ ] All sections written
- [ ] References compiled

---

### Day 14: April 22 (Tuesday) - Paper Polish
**Focus:** Paper Refinement and Figures

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-12:00 | Create all figures (architecture, results charts) | Figures complete |
| 12:00-13:00 | Lunch break | - |
| 13:00-16:00 | Polish writing, improve clarity | Improved draft |
| 16:00-18:00 | Format for target venue (ACL/EMNLP/AAAI style) | Properly formatted |
| 18:00-21:00 | Final review, read-through | Submission-ready draft |

**Day 14 Deliverables:**
- [ ] All figures created
- [ ] Paper polished and formatted
- [ ] Submission-ready draft

---

### Day 15: April 23 (Wednesday) - Thesis Start
**Focus:** Thesis Document Structure

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-12:00 | Set up LaTeX thesis template | Thesis template ready |
| 12:00-13:00 | Lunch break | - |
| 13:00-17:00 | Write Chapter 1: Introduction | Ch. 1 complete |
| 17:00-19:00 | Write Chapter 2: Literature Review | Ch. 2 outline |
| 19:00-21:00 | Review thesis guidelines | Guidelines checklist |

**Thesis Structure:**
1. Introduction (15-20 pages)
2. Literature Review (20-25 pages)
3. Methodology (25-30 pages)
4. Implementation (15-20 pages)
5. Results & Discussion (15-20 pages)
6. Conclusion (5-10 pages)

**Day 15 Deliverables:**
- [ ] LaTeX template set up
- [ ] Chapter 1 complete
- [ ] Chapter 2 outline

---

### Day 16: April 24 (Thursday) - Thesis Completion
**Focus:** Complete Thesis Writing

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-13:00 | Write Chapters 2-3 (Literature Review + Methodology) | Ch. 2-3 complete |
| 13:00-14:00 | Lunch break | - |
| 14:00-17:00 | Write Chapters 4-5 (Implementation + Results) | Ch. 4-5 complete |
| 17:00-18:00 | Write Chapter 6 (Conclusion) | Ch. 6 complete |
| 18:00-21:00 | Add front matter, references, appendices | Complete thesis |

**Day 16 Deliverables:**
- [ ] All thesis chapters complete
- [ ] Front matter added
- [ ] References formatted
- [ ] Complete thesis draft

---

## Phase 4: Presentation & Submission (Days 17-20)

### Day 17: April 25 (Friday) - Presentation Creation
**Focus:** Create Final Presentation

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-12:00 | Create presentation slides (based on 03_Evaluation.pdf + new results) | Slide deck |
| 12:00-13:00 | Lunch break | - |
| 13:00-16:00 | Add demo videos, animations, visuals | Enhanced slides |
| 16:00-18:00 | Create speaker notes | Speaker notes |
| 18:00-21:00 | First practice run | Initial feedback |

**Presentation Structure (20-25 minutes):**
1. Title slide (1 min)
2. Problem & Motivation (3 min)
3. Related Work (2 min)
4. Proposed Approach (5 min)
5. Implementation Demo (3 min)
6. Results (4 min)
7. Conclusion & Q&A (2 min)

**Day 17 Deliverables:**
- [ ] Complete slide deck
- [ ] Demo videos embedded
- [ ] Speaker notes
- [ ] First practice completed

---

### Day 18: April 26 (Saturday) - Practice & Refinement
**Focus:** Practice and Polish

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-12:00 | Second practice run, timing check | Timing verified |
| 12:00-13:00 | Lunch break | - |
| 13:00-15:00 | Refine slides based on practice | Improved slides |
| 15:00-18:00 | Third practice run | Smooth delivery |
| 18:00-21:00 | Prepare backup slides (technical details) | Backup slides |

**Day 18 Deliverables:**
- [ ] Three practice runs completed
- [ ] Timing optimized
- [ ] Slides refined
- [ ] Backup slides ready

---

### Day 19: April 27 (Sunday) - Final Review
**Focus:** Complete All Deliverables

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-12:00 | Final paper review and proofreading | Paper finalized |
| 12:00-13:00 | Lunch break | - |
| 13:00-16:00 | Final thesis review and formatting | Thesis finalized |
| 16:00-18:00 | Final presentation practice | Presentation ready |
| 18:00-21:00 | Package all deliverables | Submission package |

**Submission Package:**
- [ ] Research paper (PDF + source)
- [ ] Thesis document (PDF + source)
- [ ] Presentation (PPT/PDF)
- [ ] Source code (zip + repository)
- [ ] Demo videos
- [ ] README with instructions

**Day 19 Deliverables:**
- [ ] All documents finalized
- [ ] Submission package prepared
- [ ] Final practice completed

---

### Day 20: April 28 (Monday) - Submission Day
**Focus:** Submit Everything

| Time | Task | Deliverable |
|------|------|-------------|
| 9:00-11:00 | Final proofreading of all documents | Final check complete |
| 11:00-13:00 | Generate final PDFs, check formatting | Final PDFs ready |
| 13:00-14:00 | Lunch break | - |
| 14:00-16:00 | Submit research paper (arXiv/conference portal) | Paper submitted |
| 16:00-18:00 | Submit thesis to university portal | Thesis submitted |
| 18:00-20:00 | Submit presentation (if required) | Presentation submitted |
| 20:00-21:00 | Confirmation of all submissions | Submission confirmed |

**Day 20 Deliverables:**
- [ ] Research paper submitted
- [ ] Thesis submitted
- [ ] All confirmations received
- [ ] Backup copies saved

---

## Key Milestones Summary

| Milestone | Date | Success Criteria |
|-----------|------|------------------|
| DOM Delta Implementation Complete | Day 2 (Apr 10) | Diff algorithm working, tested |
| Privacy-Aware LLM Complete | Day 3 (Apr 11) | PII masking, DP working |
| Working Prototype | Day 5 (Apr 13) | Terminal + showcase working |
| All Experiments Complete | Day 9 (Apr 17) | All baselines run |
| Results Finalized | Day 11 (Apr 19) | Tables, figures, analysis ready |
| Paper Complete | Day 14 (Apr 22) | Submission-ready draft |
| Thesis Complete | Day 16 (Apr 24) | All chapters written |
| Presentation Ready | Day 18 (Apr 26) | 3+ practice runs done |
| Submission Complete | Day 20 (Apr 28) | All documents submitted |

---

## Risk Mitigation

### High-Risk Items

1. **DOM Delta Algorithm Complexity**
   - *Mitigation:* Use simple hash-based diff if tree diff is too complex
   - *Fallback:* Day 2 - switch to attribute-level comparison only

2. **Privacy Implementation Complexity**
   - *Mitigation:* Start with simple regex-based PII detection
   - *Fallback:* Day 3 - basic masking without differential privacy

3. **Experimental Results Don't Show Improvement**
   - *Mitigation:* Focus on token reduction + privacy as main contributions
   - *Alternative:* Show accuracy maintained with reduced token cost

4. **Paper Writing Takes Longer Than Expected**
   - *Mitigation:* Reuse content from thesis where possible
   - *Backup:* Day 14 - submit to arXiv for quick publication

5. **Technical Issues with LLM API**
   - *Mitigation:* Cache responses, use local models if needed
   - *Backup:* Pre-compute results for demo scenarios

---

## Daily Checklist Template

```
Date: _________

□ Morning: [Planned Task]
□ Afternoon: [Planned Task]
□ Evening: [Planned Task]

Deliverables Completed:
- [ ] 
- [ ] 

Blockers:
- 

Notes:
-
```

---

## Resources Needed

### Technical Resources
- OpenAI API key (for experiments)
- Local LLM (Qwen2-VL-2B) for privacy-sensitive testing
- Playwright/Selenium for browser automation
- LaTeX installation for thesis

### Reference Papers for Comparison
1. **Prune4Web** (Zhang et al., 2025) - DOM tree pruning
2. **SeeAct** (Zhou et al., 2024) - GPT-4V web agent
3. **WebArena** (Zhou et al., 2024) - Web agent benchmark
4. **Mind2Web** (Deng et al., 2023) - Web agent dataset
5. **AgentBench** (Liu et al., 2023) - LLM agent evaluation

### Datasets for Evaluation
- Mind2Web (already available)
- Custom test cases (20-30 tasks)
- Real websites (10-15 diverse sites)

---

## Success Metrics

### Implementation Success
- [ ] DOM Delta Processing reduces token usage by 30%+
- [ ] Privacy filters mask 90%+ of PII
- [ ] System works on 80%+ of test websites
- [ ] Terminal demo runs without errors
- [ ] Browser showcase completes 3 scenarios

### Writing Success
- [ ] Paper: 8-10 pages, all sections complete
- [ ] Paper: Submitted to arXiv or conference
- [ ] Thesis: 100+ pages, all chapters complete
- [ ] Thesis: Follows university guidelines

### Presentation Success
- [ ] 20-25 minute presentation ready
- [ ] Demo video included
- [ ] 3+ practice sessions completed
- [ ] Can present without notes

---

## Final Checklist (Day 20)

Before submission, verify:

- [ ] All code is committed to git
- [ ] Research paper is polished and formatted
- [ ] Thesis follows university template
- [ ] Presentation is tested and timed
- [ ] Demo videos are recorded and embedded
- [ ] All deliverables are backed up
- [ ] Submission portals are accessible
- [ ] Supervisor has reviewed (if possible)

---

**Good luck! This is achievable with focused effort.**

*Created: April 8, 2026*
*Last Updated: April 8, 2026*
