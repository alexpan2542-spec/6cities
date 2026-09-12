# BAMS selection — reproduction report

- generated: 2026-09-10T07:26:57+00:00
- scikit-learn: 1.9.0 (frozen artefacts: 1.9.0)
- RF: {'n_estimators': 300, 'random_state': 42, 'n_jobs': -1}
- margin features (9): B2, B3, B4, B8, B11, B12, NDVI, NDBI, MNDWI
- BoundaryScore: sum of 3x3 stdDev over B2_stdDev, B3_stdDev, B4_stdDev, B8_stdDev
- BAMS150: two-stage (margin<=rank 1000 -> top 150 BoundaryScore); Hangzhou uses the 0.7/0.3 min-max blend

| City | rule | margin max|Δ| | BAMS150 match | Top150 match |
|------|------|--------------|---------------|--------------|
| Hangzhou | two_stage | 1.11e-16 | 46/150 | 150/150 |

## Mismatches (first 10 Original_IDs each)
- **Hangzhou**
  - bams150: only_new=[61, 172, 224, 400, 404, 442, 467, 489, 530, 583] only_old=[57, 85, 175, 220, 355, 438, 460, 726, 825, 861]

## Notes
- `margin` is recomputed bit-for-bit (Δ ~1e-16) from the frozen pools, so downstream code that reads only `Original_ID`/`margin` is unaffected.
- The canonical `*_MarginScores.csv` standardises all six cities to per-class `prob_1/prob_2/prob_3` + `RF_Pred`; the older Wuhan file stored only the top-two sorted posteriors.
- `*_Top150.csv` is a plain min-margin reference set and is not consumed downstream; Wuhan's on-disk `Top150.csv` came from a superseded 70/30-split run (`scripts_bk/wuhan_pipeline_step1_2.py`) and will not match.
- Hangzhou's BAMS150 uses the weighted-blend variant (`scripts_bk/hangzhou_init_pipeline.py`); pass `--standardise-hangzhou` to also emit the two-stage set for comparison.
- Hangzhou's on-disk `BAMS150.csv` stored a degenerate (all-zero) `BoundaryScore` column from an in-place min-max bug; the reproduction writes the real raw sum. Point selection, order and `margin` are unaffected (row order identical, Δmargin = 0).
