# `submission/` — the Overleaf project

A 1:1 mirror of the Overleaf project. **Single LaTeX file**, no `\input`, no
subfiles.

```
submission/
├── manuscript.tex   ← the whole paper: preamble + frontmatter + §1–§6
│                       + Appendix A (blind re-check) + Appendix B
│                       (Supplementary Material, S1–S6) + internal bibliography
└── figures/
    ├── Figure1_study_area.pdf        (fig:studyarea)
    ├── Figure2_workflow.pdf          (fig:flowchart)
    ├── Figure3_independence.png      (fig:independence)
    ├── Figure4_maps_maintext.pdf     (fig:map_maintext)
    ├── FigureS1–S4_map_*.png         (fig:map_s1–fig:map_s4, Appendix B)
    ├── FigureS5_c2_chips.{pdf,png}   (fig:S5chips, Appendix B)
    └── FIGURES_TODO.md               per-figure status/build notes
```

**Rule (2026-09-11): everything lives in this one `manuscript.tex`, main text
and supplementary alike — never split supplementary (or anything else) into
its own `.tex` file.** `../docs/manuscript_supplementary.md` remains the Markdown
source of record for wording (update it first, then port changes into
Appendix B here), but the actual submission content, including every
supplementary table and figure, is folded into `manuscript.tex` as Appendix B.

Section boundaries inside `manuscript.tex` are marked with
`%% ===== N  TITLE =====` banners; each section keeps its original `%%` header
block (source, fixes, cite keys, labels defined/expected).

## Getting it onto Overleaf

The Overleaf project already exists and already has the `Definitions/` bundle
(`mdpi.cls`, `logo-mdpi`, `mdpi.bst`). Do **not** make a new project.

1. Replace the project's main `.tex` with this `manuscript.tex` (it already
   carries `remotesensing` in the class line and the real frontmatter). Leave
   `Definitions/` untouched.
   - If merging by hand instead: change `journal` → `remotesensing`; drop the
     template's `\Title/\Author/\abstract/\keyword` placeholders, the
     `\setcounter{section}{-1}` line and the "How to Use this Template" section;
     paste this file's `\begin{document}`-onward over the template body; and
     put this file's `\bibitem`s inside the template's `\isAPAandChicago{}{…}`
     wrapper (don't end up with two `thebibliography` blocks).
2. Upload the `figures/` folder.
3. Overleaf menu → **Main document** → `manuscript.tex`. Compile.

After that just edit `manuscript.tex`.

## Figures

4 main-text figures (`fig:studyarea`, `fig:flowchart`, `fig:independence`,
`fig:map_maintext`) plus 5 supplementary figures in Appendix B
(`fig:map_s1`–`fig:map_s4`, `fig:S5chips`), all `\includegraphics`'d. Details
in `figures/FIGURES_TODO.md`.

## Relationship to the rest of `docs/`

| File | Role |
|---|---|
| `../docs/manuscript.md` | Full prose master — untrimmed, all 9 figures / 11 tables. Source of truth for wording. |
| `submission/manuscript.tex` | The actual submission: main text §1–6, Appendix A, Appendix B (= Supplementary Material, S1–S6), bibliography — all one file. |
| `../docs/manuscript_supplementary.md` | Markdown source of record for the Supplementary Material's wording (S1–S6); mirrored into Appendix B here, not submitted itself. |
| `../archive/manuscript_untrimmed_sections/manuscript_sec{3,4}_untrimmed.tex` | Old standalone LaTeX for §3/§4. Superseded. Delete once the §3/§4 review pass is merged into `manuscript.md`. |

When numbers change, update all three of: `manuscript.md`, `submission/manuscript.tex`
(both main text and Appendix B), `manuscript_supplementary.md`.

## Regenerating the split (if ever needed)

The per-section files were removed in favour of the single file. To go back to a
split, cut `manuscript.tex` at the `%% =====` banners — or rebuild from
`../docs/manuscript.md` via the section-fragment workflow.
