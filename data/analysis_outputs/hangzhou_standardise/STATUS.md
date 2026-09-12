# Hangzhou standardisation — status (2026-09-10) — COMPLETE

Goal: drop Hangzhou's weighted-blend BAMS150 rule; put all six cities on the
frozen two-stage rule so the paper has no special case to explain.

**DONE.** 104 points re-labelled, DW/ESRI re-sampled, all analyses re-run,
snapshot refreshed (blend version at `docs/results_snapshot.blend_bak/`),
manuscript fully updated. See `RESULTS_before_after.md` for the full
before/after and the manuscript-edit list.

## Done

| step | state |
|---|---|
| two-stage BAMS150 recomputed for Hangzhou | ✅ 46/150 overlap with the old blend set |
| 104 new points expert-labelled (GEE) | ✅ `labelled_104.csv` (11 built / 61 non-built / 32 water; conf 37h/54m/13l) |
| new `Hangzhou_BAMS150_Manual.csv` assembled (46 kept + 104 new) | ✅ 150 rows; old saved as `*.blend.bak.csv` |
| `SELECTION_RULE["Hangzhou"]` → `two_stage` | ✅ `scripts/reproduce_bams_selection.py` |
| `03_top150/Hangzhou_BAMS150.csv` + `Top150.csv` → two-stage | ✅ blend versions saved as `*.blend.bak.csv` |
| `all_cities_BAMS150_second_ref.csv` rolled back to 5 cities | ✅ 6-city blend version saved as `*.6city_blend.bak.csv` |

Sneak peek (needs no DW/ESRI): new Hangzhou expert ≠ WC = 0.407 all / 0.547
high-only (old blend set: 0.467 / 0.580). Still the mildest of the six cities.

## BLOCKED ON: one GEE export

Hangzhou's DW / ESRI values are only sampled at the *old* 150 blend points. The
104 new points have none.

1. Run `figures/gee_shp/Hangzhou_second_ref_DW_ESRI_v2_twostage.js` in the GEE
   Code Editor (150 points inlined — no asset upload needed).
2. `Export.table.toDrive` → download `Hangzhou_BAMS150_dwesri_raw.csv`.
3. Overwrite
   `data2/shared_reference/cross_city_second_ref_dw_esri/Hangzhou_BAMS150_dwesri_raw.csv`
   with it.
4. Tell me — I run the rest.

## Then (mechanical, I do it)

```
python scripts/add_hangzhou_second_ref.py          # rebuild second_ref, 5 → 6 cities, n=900
python scripts/diagnostic_analysis.py              # T1–T5
python scripts/independence_diagnostic.py          # S2/S3, Yule Q
python scripts/loco_correction_operator.py         # Phase 1–3
```
then refresh `docs/results_snapshot/`, diff every headline number against the
frozen (blend) snapshot, and show you the before/after so you can decide
keep-Hangzhou vs drop-Hangzhou.

## Manuscript edits pending that decision

- delete Eq. `hzblend`, the "five of the six cities … Hangzhou was sampled with
  an earlier weighted-blend variant" paragraph, and Supplementary S4 / Table S5
  (all currently in `manuscript_sec3.tex`, `manuscript.md`,
  `manuscript_supplementary.md` as the Option-1 fallback)
- "five of six" → "each of the six cities"; drop the per-city Hangzhou caveats
- if the 46 carry-over scene notes get recoded to the `be/br/fe/veg/bl/re/ct/
  we/wr/pf/tw` vocab, Hangzhou can also join the T4 scene stratification
  (five-city → six-city)

## Rollback

Everything replaced is saved next to it as `*.blend.bak.csv` /
`*.6city_blend.bak.csv`. To revert: restore those, set
`SELECTION_RULE["Hangzhou"]="blend"`, re-run the three analysis scripts.
