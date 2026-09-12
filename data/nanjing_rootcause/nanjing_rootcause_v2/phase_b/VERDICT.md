# Phase B verdict

## Bottom line
**H1 (label–exam conflict) is only a partial / secondary factor — not the sole 元凶.**

On the **same BAMS positions**:
- Human seeds are somewhat **worse** for WC-OA than WC seeds (~0.15–0.25 pp).
- But **WC seeds still do not deliver success-city gains** (still ≈0 / slightly negative vs MLP baseline).
- Agree-only vs Disagree-only both fail to unlock ~+1 pp WC-OA.

So Nanjing is not fixed by “just use WC-consistent labels.” The failure sits deeper in **boundary regime / sampled locations / PE transfer** (H2–H3), with human–WC conflict as an amplifier.

## Design reminder
- WC-track baseline here is **MLP** (OA≈0.9095), matched to PE backbone.
- Phase A passport baseline was **RF** (OA≈0.9032). Absolute PE_Human_high OA **matches passport** (0.9065).
- Dual-track K is still picked by WC (BE, −OA), so Human-OA is diagnostic, not the training objective.

## WC-track (mean over 5 seeds)

| Method | n_seed | WC_OA | Δ vs MLP (pp) | Δ vs Phase-A RF (pp) | BE |
|--------|-------:|------:|-------------:|---------------------:|---:|
| Baseline | 0 | 0.9095 | +0.00 | +0.63 | 283.0 |
| PE_Human_all | 150 | 0.9053 | -0.42 | +0.21 | 294.4 |
| PE_Human_high | 73 | 0.9065 | -0.30 | +0.33 | 289.8 |
| PE_WC_all | 150 | 0.9077 | -0.18 | +0.45 | 287.4 |
| PE_WC_high | 73 | 0.9082 | -0.13 | +0.50 | 283.4 |
| PE_Agree_all | 48 | 0.9068 | -0.26 | +0.37 | 286.6 |
| PE_Disagree_all | 102 | 0.9059 | -0.36 | +0.28 | 291.8 |
| PE_Agree_high | 28 | 0.9066 | -0.28 | +0.35 | 291.4 |
| PE_Disagree_high | 45 | 0.9072 | -0.23 | +0.40 | 288.8 |


### Key contrasts (ΔWC pp, relative to MLP baseline deltas)
| Contrast | Value |
|----------|------:|
| PE_Human_high − PE_WC_high | -0.17 pp |
| PE_Human_all − PE_WC_all | -0.24 pp |
| PE_Disagree_all − PE_Agree_all | -0.09 pp |
| PE_Disagree_high − PE_Agree_high | +0.06 pp |

## Dual-track (human hold-out; K picked by WC)

| Method | ΔWC pp | Human OA | ΔH pp |
|--------|-------:|---------:|------:|
| Baseline_dual | +0.00 | 0.665 | +0.00 |
| PE_Human_holdout | -0.08 | 0.609 | -5.58 |
| PE_WC_holdout | -0.13 | 0.586 | -7.91 |


Both PE_Human_holdout and PE_WC_holdout **hurt** Human-OA when K is chosen for WC. This does **not** reproduce the earlier “Human↑ / WC flat” story under Human-oriented model selection; under WC-oriented selection, neither seed source helps the human hold-out.

## Hypothesis update
| Hyp | Status after Phase B |
|-----|----------------------|
| H1 strong: “WC seeds on same points restore +1 pp” | **Rejected** |
| H1 weak: “Human seeds worse than WC on same points” | **Supported** (~0.2 pp) |
| H2 sampling / hard boundary locations | **Elevated** — even WC@BAMS fails |
| H3 expansion amplifier | Still open — needs Phase D |
| Pure “exam mismatch” as sole story | **Not persuasive enough** (confirmed by this run) |

## Next
Phase C (Random/Margin vs BAMS with WC seeds) and/or Phase E (scene leave-one-group) to test whether **where** BAMS samples is the main remaining元凶.
