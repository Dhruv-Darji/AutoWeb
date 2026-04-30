# LaTeX Beamer Presentation

## Folder Structure

```
latex-ppt/
├── main.tex                  # Main document (compile this)
├── mybib.bib                 # Bibliography file with all references
├── README.md                 # This file
├── diagrams/                 # Place all images/diagrams here
│   └── (upload your images)  # e.g., architecture_diagram.png, dom_delta_diagram.png
└── slides/                   # Individual slide files
    ├── slide01_title.tex
    ├── slide02_contents.tex
    ├── slide03_introduction.tex
    ├── slide04_problem.tex
    ├── slide05_litreview1.tex
    ├── slide06_litreview2.tex
    ├── slide07_litreview3.tex
    ├── slide08_litreview4.tex
    ├── slide09_litreview5.tex
    ├── slide10_litreview6.tex
    ├── slide11_gaps.tex
    ├── slide12_objectives.tex
    ├── slide13_existing.tex
    ├── slide14_dataset.tex
    ├── slide15_novelty1.tex
    ├── slide16_novelty2.tex
    ├── slide17_architecture.tex
    ├── slide18_improvement.tex
    ├── slide19_algorithm.tex
    ├── slide20_implnotes.tex
    ├── slide21_filter_recall.tex
    ├── slide22_grounder_ea.tex
    ├── slide23_hitl_results.tex
    ├── slide24_future.tex
    ├── slide25_conclusion.tex
    ├── slide26_references.tex
    └── slide29_thanks.tex
```

## How to Compile

```bash
# Compile with bibliography support
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

Or use `latexmk`:
```bash
latexmk -pdf main.tex
```

## Adding Images

1. Place all your diagram/image files (PNG, JPG, PDF) in the `diagrams/` folder.
2. In the relevant slide `.tex` files, uncomment the `\includegraphics` lines and update the filename.
3. The `\graphicspath{{diagrams/}}` in `main.tex` means you only need the filename without the path.

## Key Packages Used

- **beamer** — Presentation framework (Madrid theme)
- **biblatex + biber** — Bibliography management from `mybib.bib`
- **booktabs** — Professional tables
- **tabularx** — Flexible-width tables for literature review
- **amsmath** — Mathematical formulas (improvement/algorithm slides)
- **tikz** — Graphics (available for custom diagrams)
- **hyperref** — Clickable links
- **listings** — Code formatting

## Notes

- All content from the original 29-slide PPTX is preserved.
- References use `\cite{}` commands mapped to keys in `mybib.bib`.
- The `[allowframebreaks]` option on the References slide automatically splits across multiple frames.
- The design uses a Navy + Peach color scheme similar to the original PPT.
