# submission figures — status

**This folder (`submission/figures/`) is now the single location for all
paper figures**, main text and supplementary alike (consolidated 2026-09-11;
`docs/manuscript_figures/` — the old parallel copy used by the full-draft
`manuscript.md` — was deleted, and its supplementary maps/chips moved here).
Regenerating any figure should save (or be copied) straight into this folder.

The trimmed submission uses **4 main-text figures**, plus 5 supplementary
figures (S1–S5) shared with `../../docs/manuscript_supplementary.md`.

| In text | `\label` | File here | Built by | Status |
|---|---|---|---|---|
| §2.1 | `fig:studyarea` | `Figure1_study_area.pdf` | `scripts/make_study_area_map.py` | **redrawn 2026-09-11** — see below; caption in `manuscript.tex` (~line 376) rewritten to match (same day) |
| §3 | `fig:flowchart` | `Figure2_workflow.pdf` | `scripts/make_workflow_figure.py` | **fixed 2026-09-12**: the diagram's own figure/table cross-references were stale (pointed at a since-cut "Fig 3–9" numbering from an earlier draft where T1–T4 each had a companion figure and the maps were "Fig 8–9"). Corrected to T1–T4 (no figure), "Tables S2–S3 · Fig 3" for panel (e), and "Fig 4 (main) · Fig S1–S4" for the LOCO maps — matching the actual 4-main-figure submission; regenerated. |
| §4.5 | `fig:independence` | `Figure3_independence.png` | **author-edited 2026-09-10** (started from the diagnostic figure `scripts/make_diag_figures.py` produces) | **edited by the author — do NOT overwrite by re-running the script.** Values match §4.5 + Table S2 (0.87/0.62; +0.13/+0.14; Q 0.97/0.08/0.14). Caption fixed 2026-09-12: it claimed the image shows both Yule's $Q$ and Cohen's $\kappa$, but the right-hand panel only plots $Q$ — caption now says so and points to Table S2 for $\kappa$. Image itself untouched (still locked). |
| §4.6 | `fig:map_maintext` | `Figure4_maps_maintext.pdf` | composite of the Nanjing + Changsha maps (2 rows) via `scripts/make_maintext_map_figure.py` | **rebuilt 2026-09-12**: briefly trimmed to Nanjing-only for page budget, reverted the same day — the two-city figure is more convincing, and the caption/prose now states explicitly why Nanjing and Changsha are the two shown (they bracket the operator's performance range). Counts match §4.6 / S6: Nanjing 28 corrected / 102 expert≠WC, Changsha 63 / 74. |
| Supp. §S4 | `fig:map_s1`–`fig:map_s4` | `FigureS1_map_Wuhan.png` … `FigureS4_map_Hangzhou.png` | `scripts/make_correction_maps.py`, then copied here by hand | before/after maps for the four cities not in the main-text composite |
| Supp. §S6 | `fig:S5chips` | `FigureS5_c2_chips.{pdf,png}` | `scripts/c2_chip_{select,fetch,assemble}.py` | boundary-error chip gallery |

**`Figure3_independence.png` carries the author's 2026-09-10 edits** — only
replace it with a newer author version, never with the raw script output.
`Figure4_maps_maintext.pdf` is a composite — rebuild it (re-run
`scripts/make_maintext_map_figure.py`) after the underlying city maps change.

**Regenerating the six before/after city maps:** `scripts/make_correction_maps.py`
writes its raw output to `../../data/analysis_outputs/correction_maps/Fig_diag_map_{city}.png`
(the working analysis-outputs folder, not this one) for all six cities.
`scripts/make_maintext_map_figure.py` reads Nanjing + Changsha straight from
there to rebuild `Figure4_maps_maintext.pdf` in this folder automatically. The
other four (`Fig_diag_map_Wuhan/Hefei/Nanchang/Hangzhou.png`) still need a
manual copy + rename into `FigureS1–S4_map_*.png` here — no script does that
step yet.

### Figure 1 redesign (2026-09-11)

`Figure1_study_area` was rebuilt end to end (see `memory/figure1-study-area-redesign.md`
if reading from an assistant session with access to it): dropped the Yangtze
River Economic Belt outline (panel used to show all 11 YREB provinces, mostly
empty); now shows only the six study cities' own provinces, named, with the
Yangtze main stem drawn in. Each city's AOI is now its real FAO GAUL 2015 ADM2
(prefecture) polygon — the old version drew a bounding-box rectangle, which
didn't match §2.2's AOI definition. The China locator is no longer a separate
lettered "(b)" panel; it is an unlabelled inset floated over the East China Sea
inside panel (a) (panel (a) widened, sea extended east, sea tinted so land/sea
read clearly), sized/positioned with an equal gap from the top and right map
edges. The locator now draws every country in the region from NE 10m
admin_0_countries (`data/boundaries/ne_countries_eastasia.geojson`), all in
one uniform style — China is not singled out, its border is just one among its
neighbours'. The caption in `manuscript.tex` has been rewritten to match (no
more lettered panels or a boxed AOI); `manuscript.md` §2.1's prose already only
cited the label, so it needed no change.

### Consistency check (2026-09-10)

All six before/after maps were regenerated from the current two-stage data
(`scripts/make_correction_maps.py`); operator-change / expert≠WC counts now:
Wuhan 35/88, Hefei 91/74, Nanchang 38/81, Nanjing 28/102, Changsha 63/74,
Hangzhou 30/61 — identical to the S6 block in `manuscript.tex`. The pre-refresh
maps (Nanjing 94, Changsha 181) were stale.

## Not in the submission (full draft only, not regenerated here)

- `Figure3_fourway`, `Figure4_confidence`, `Figure5_error_direction`,
  `Figure6_scene_localisation` (old `manuscript_figures/` naming) — **cut**:
  each duplicated a table (T1–T4), so they're not kept as files anywhere.
  Re-run `scripts/make_diag_figures.py` if they're needed again.

`FigureS1–S4` (before-after maps) and `FigureS5` (chip gallery) are **in the
submission**, `\includegraphics`'d directly in `../manuscript.tex` Appendix B
(§S4/§S6, added 2026-09-11 — supplementary content is folded into the single
`manuscript.tex`, not a separate file, see `../README.md`), and mirrored in
`../../docs/manuscript_supplementary.md` §S4/§S6 as the Markdown wording source —
see the table above, not a cut list.

## PNG vs PDF

- Fig1 / Fig2 are line art → **vector PDF** (sharp at any zoom, MDPI's preference).
- Fig3 (independence) and the Fig4 maps contain raster map imagery → PNG at
  ≥300 dpi is fine for MDPI; a PDF wrapper would not add real resolution.
- The two scripts that emit PDF: `make_study_area_map.py`, `make_workflow_figure.py`.
  `make_diag_figures.py` / `make_correction_maps.py` currently save PNG only —
  add a `fig.savefig(..., '.pdf')` line if a vector copy is wanted.
