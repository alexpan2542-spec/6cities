# Phase C verdict

## Bottom line
**H2 strong form rejected: BAMS locations are not uniquely toxic.**

With **WC labels** on every sampler, Nanjing PE gains stay in a narrow band (~**+0.45–0.56 pp vs RF**). Random/Margin are only ~0.05 pp above BAMS_WC. The city remains far below the +1–1.8 pp success-city regime.

**Human labels on BAMS are the weakest boundary option** (+0.33), so conflict still amplifies failure — but sampling strategy is not the main元凶.

## Ranking (ΔOA vs RF)

| Method | n_seed | ΔOA vs RF | ΔOA vs MLP | BE | n_exp |
|--------|-------:|----------:|-----------:|---:|------:|
| Baseline_RF | 0 | +0.00 | -0.63 | 304.2 | nan |
| Baseline_MLP | 0 | +0.63 | +0.00 | 283.0 | nan |
| PE_Margin150_WC | 150 | +0.56 | -0.07 | 284.6 | 41.0 |
| PE_Random150_WC | 150 | +0.52 | -0.11 | 281.4 | 14.8 |
| PE_BAMS_WC_high | 73 | +0.50 | -0.13 | 283.4 | 21.4 |
| PE_BAMS_WC_all | 150 | +0.45 | -0.18 | 287.4 | 23.4 |
| PE_BAMS_Human_high | 73 | +0.33 | -0.30 | 289.8 | 16.6 |
| PE_Easy150_WC | 150 | +0.25 | -0.38 | 289.8 | 0.2 |


## Contrasts
| Contrast | Δ (pp, vs RF deltas) |
|----------|---------------------:|
| Random_WC − BAMS_WC_high | +0.02 |
| Margin_WC − BAMS_WC_high | +0.06 |
| BAMS_WC_high − BAMS_Human_high | +0.17 |
| Easy_WC − BAMS_WC_high | -0.25 |

## Hypothesis update
| Hyp | Status after Phase C |
|-----|----------------------|
| H2 strong: “BAMS geometry uniquely kills WC-OA” | **Rejected** (Random/Margin ≈ BAMS_WC) |
| H2 weak: “hard/boundary seeds slightly better than easy” | **Supported** (Easy worst; Margin best; Easy n_exp≈0) |
| City-level PE ceiling on Nanjing | **Supported** — all WC samplers ~+0.5 pp vs RF |
| H1 weak (human worse than WC on same points) | **Reconfirmed** (Human_high +0.33 < WC_high +0.50) |

## Combined A+B+C story
1. Phase A: passport locked — PE_high ≈ +0.33 vs RF.
2. Phase B: switching seed labels to WC on BAMS does **not** unlock +1 pp.
3. Phase C: switching **sampling** (Random/Margin/Easy) with WC labels also stays ~+0.5 pp.

**元凶 (current best statement):**
Nanjing sits under a **city-level PE / boundary-hardness ceiling** under WC-referenced evaluation. Human–WC conflict and BAMS boundary focus are **secondary amplifiers**, not the root switch that would restore success-city gains.

## Next (optional)
- Phase D: PE ablation (no expand / tight K / WC-agree-only write) — is expansion wasted effort?
- Phase E: scene leave-one-group — which BAMS scenes hurt most?
- Or stop and write the root-cause section for the paper with A–C evidence.
