# Mermaid Diagram Reference

All diagrams are delivered as Mermaid source only. Render them at
<https://mermaid.live> or with `mmdc -i <file>.mmd -o <file>.png -b white`
and drop the resulting image into the report via `\includegraphics`.

## Figure 4.1 — Proposed system overview
- **Source:** `diagrams/fig_4_1_system_overview.mmd`
- **Placement:** Chapter 4, §4.1 (`04ProposedWork.tex`, Fig \ref{fig:ProposeArchitecture})
- **Caption:** "Proposed system overview. Prune4Web Planner → Filter → Grounder backbone, augmented with DOM Delta Processing at the filter stage and a Privacy Hub + risk-based HITL gate around the grounder."
- **Sanity check on render:** should show the three novel blocks (DOM Delta, Privacy-Aware LLM, HITL Gate) highlighted with the `novel` class (light blue); the `next step` arrow loops from the browser driver back to the DOM extractor.

## Figure 4.2 — DOM Delta Processing pipeline
- **Source:** `diagrams/fig_4_2_dom_delta.mmd`
- **Placement:** Chapter 4, §4.2 (`04ProposedWork.tex`, Fig \ref{fig:DomDelta})
- **Caption:** "DOM Delta Processing: structural-UID tracking plus hybrid keyword + MiniLM scoring on the delta subset."
- **Sanity check on render:** three input nodes (Task, Current DOM, Previous Snapshot) should fan into the structural UID + classification path; the late-fusion box should have both the keyword scorer and the MiniLM cosine as inputs, with the `0.6 / 0.4` split visible.

## Figure 4.3 — HITL Gate + Privacy Hub
- **Source:** `diagrams/fig_4_3_privacy_hitl.mmd`
- **Placement:** Chapter 4, §4.3 (`04ProposedWork.tex`, Fig \ref{fig:HitlPrivacy})
- **Caption:** "HITL gate and Privacy Hub. PolicyHub injects policy rules and risk keywords into the Planner prompt; the Privacy-Aware LLM wrapper scrubs PII from element summaries before the Grounder call."
- **Sanity check on render:** PolicyHub on the top-left feeds two arrows — one into the Planner Prompt, one into the HITL risk-keyword check (bottom-right). The HITL gate has three decision diamonds in sequence (`policy_risk`, keyword scan, confidence); any `yes` branch goes to `Block → Human Review`, all-`no` path proceeds to `Pass → Execute`.

## How to render and drop into the report
1. `mmdc -i diagrams/fig_4_1_system_overview.mmd -o diagrams/fig_4_1_system_overview.png -b white -w 1600`
2. Replace the current `\includegraphics{Implementation Architecture.png}` / `SeeAct Diagram (white-BG).png` references in `04ProposedWork.tex` and `03BasePaperSeeAct.tex` with the rendered PNGs.
3. Rebuild the report with `pdflatex && bibtex && pdflatex && pdflatex`.
