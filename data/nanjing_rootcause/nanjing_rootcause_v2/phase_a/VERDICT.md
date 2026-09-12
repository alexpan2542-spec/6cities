# Phase A verdict: **PASS**

Reproduced with `scripts/nanjing_relabel_bams_pe.py --seeds 0 1 2 3 4 --epochs 80`.

Acceptance: key-method `|Δ dOA| ≤ 0.20 pp`; Baseline `|Δ OA| ≤ 0.002`.

| Method | passport dOA | repro dOA | Δ pp | status |
|--------|-------------:|----------:|-----:|--------|
| Baseline | +0.000 | +0.000 | +0.000 | PASS |
| BAMS150_new | -0.213 | -0.213 | +0.000 | PASS |
| BAMS150_new_high | -0.076 | -0.076 | +0.000 | PASS |
| PE_new | +0.213 | +0.213 | +0.000 | PASS |
| PE_new_high | +0.333 | +0.333 | +0.000 | PASS |
| PE_new_hm | +0.169 | +0.169 | +0.000 | PASS |
| PE_old | +0.356 | +0.356 | +0.000 | PASS |

## Locked Nanjing passport (for Phase B+)
- Baseline OA = **0.9032**, BE = **304.2**
- PE_new = **+0.21 pp**, PE_new_high = **+0.33 pp**, PE_new_hm = **+0.17 pp**
- BAMS150_new = **−0.21 pp**
- Confidence: high=73, med=60, low=17; Human changed vs bk = 29

## Next
Phase B: Human vs WC seed labels × WC/Human dual eval.
