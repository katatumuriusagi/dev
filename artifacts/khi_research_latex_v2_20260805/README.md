# KHI PatchCore Research LaTeX Documents v2

This directory contains six independent Japanese LuaLaTeX documents:

1. `01_research_plan.tex`
2. `02_experiment_specification.tex`
3. `03_beginner_knowledge_guide.tex`
4. `04_related_work_review.tex`
5. `05_normal_mode_definition_spec.tex`
6. `06_experiment_matrix.tex`

All documents use `khi-common.sty`.

## Local build

Run from this directory with TeX Live 2025 or later:

```powershell
latexmk -lualatex -interaction=nonstopmode -halt-on-error .\01_research_plan.tex
latexmk -lualatex -interaction=nonstopmode -halt-on-error .\02_experiment_specification.tex
latexmk -lualatex -interaction=nonstopmode -halt-on-error .\03_beginner_knowledge_guide.tex
latexmk -lualatex -interaction=nonstopmode -halt-on-error .\04_related_work_review.tex
latexmk -lualatex -interaction=nonstopmode -halt-on-error .\05_normal_mode_definition_spec.tex
latexmk -lualatex -interaction=nonstopmode -halt-on-error .\06_experiment_matrix.tex
```

Required packages include LuaTeX-ja, TikZ, booktabs, longtable, tabularx, pdflscape, fancyhdr, titlesec and hyperref.

## Notes

- Japanese prose uses the `である` style.
- Figures are primarily redrawn with TikZ and do not depend on external image files.
- The workflow `.github/workflows/build-khi-research-latex-v2.yml` compiles all documents and uploads PDFs plus sources as one artifact.
