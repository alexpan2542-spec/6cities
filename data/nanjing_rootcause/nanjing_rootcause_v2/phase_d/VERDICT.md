# Phase D verdict

## Bottom line
**Expansion is not wasted under the standard PE protocol — and is not the main元凶.**

- **SeedsOnly ≈ 0** for Human_high (+0.03 pp): without expansion, human hard seeds barely move WC-OA.
- **PE_std helps**: Human_high +0.33; WC_high +0.50.
- **Ungated expansion amplifies noise**: `PE_nolimit` drops to ~+0.13–0.17.
- **Tight margin helps Human seeds** (`PE_tight` +0.44 > `PE_std` +0.33).
- Filtering expansions to WC-agree does **not** unlock +1 pp.

So H3 “expansion is pure waste” is **rejected**; H3 “ungated expansion is noisy” is **supported**. The city ceiling remains.

## ΔOA vs RF (mean over 5 seeds)

### BAMS_Human_high (61.6% ≠ WC)
| Method | ΔOA pp | n_exp | note |
|--------|-------:|------:|------|
| SeedsOnly | +0.03 | 0 | almost dead |
| PE_std | +0.33 | 16.6 | passport |
| PE_tight | +0.44 | 6.2 | **best human** |
| PE_nolimit | +0.13 | 275 | noise |
| PE_expAgreeWC | +0.35 | 9.6 | ≈ std |

### BAMS_WC_high
| Method | ΔOA pp | n_exp | note |
|--------|-------:|------:|------|
| SeedsOnly | +0.20 | 0 | weak |
| PE_std | +0.50 | 21.4 | **best WC** |
| PE_tight | +0.45 | 6.6 | |
| PE_nolimit | +0.17 | 237 | noise |
| PE_expAgreeWC | +0.36 | 8.4 | worse than std |

## Contrasts
| Contrast | pp |
|----------|---:|
| Human: PE_std − SeedsOnly | +0.30 |
| WC: PE_std − SeedsOnly | +0.31 |
| Human: PE_std − PE_nolimit | +0.20 |
| WC: PE_std − PE_nolimit | +0.33 |

## Hypothesis update
| Hyp | Status |
|-----|--------|
| H3 waste: SeedsOnly ≥ PE | **Rejected** (PE clearly better) |
| H3 amplify under std PE | **Rejected** (net helpful) |
| H3 amplify if no margin gate | **Supported** (nolimit hurts) |
| Expansion can break Nanjing ceiling | **Rejected** (still ≤ +0.5 pp) |

## Implication for root cause
Expansion is a **necessary small booster** on Nanjing (~+0.3 pp over seeds-only), not the bug and not the cure. Margin gating matters; removing it wastes the booster. The ceiling from Phases B–C still stands.
