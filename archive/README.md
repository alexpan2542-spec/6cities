# Archive

Superseded work from before the final "diagnostic-first + LOCO correction operator"
methodology was locked in (see `docs/planning/paper_thesis_and_contributions.txt`,
2026-09-09 skeleton). Nothing here is read by any script in `scripts/` or cited by
the manuscript in `submission/`. Kept for provenance/audit trail, not for reuse.

Full decision history: `docs/planning/CLEANUP_LOG.md`.

- **`pe_era_scripts/`** — the abandoned "Prototype Expansion" (PE) approach: 38
  per-city one-off experiment scripts (`*_pe.py`, `bams_prototype_expansion.py`,
  `scheme_a_train_hard_correction.py`, etc.), originally deleted 2026-09-06, restored
  here from the sibling `gee-project2` project as a historical reference copy.
- **`pe_era_notebooks/`** — the 4 exploratory baseline notebooks (`01_baseline.ipynb`
  … `04_hefei.ipynb`) from the same PE-era, plus their CSV/PNG outputs.
- **`pe_era_paper_draft/`** — `paper_report_BAMS_Confidence_PE.md` (2026-08-08), the
  technical report written around the PE method before the diagnostic-first pivot.
- **`manuscript_untrimmed_sections/`** — pre-trim `.tex` drafts of §3/§4, superseded
  by the current `submission/manuscript.tex`.
- **`results_snapshot_pre_hangzhou_standardisation/`** — the full frozen-numbers
  snapshot as it stood before Hangzhou was re-sampled onto the two-stage BAMS150
  rule (2026-09-10). Superseded by `docs/results_snapshot/`.
- **`pe_era_scripts_related/`** — `Nanjing_phase2_audit70.*` (shapefile) and
  `Nanjing_phase2_inspect.js`, moved out of `data/gee_shp/`. Undocumented
  anywhere in `docs/planning/`, presumed leftover from the same abandoned
  `nanjing_phase_b/c/d` investigation (see `pe_era_scripts/`).
