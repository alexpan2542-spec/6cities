# Nanjing baseline autopsy — what lowers OA (before AL/PE)

**Scope:** RF/MLP trained only on WC labels (no human/BAMS/PE).  
**Goal:** locate the baseline error mass that keeps Nanjing OA lowest.

## Headline

| | Nanjing RF | Other-4 mean RF | Δ |
|--|--:|--:|--:|
| OA | **0.903** | 0.921 | **−1.8 pp** |
| BE (C1↔C2) | **304** | 244 | **+60** |
| C2 recall | **0.813** | 0.854 | **−4.1 pp** |
| C1 recall | 0.925 | 0.942 | −1.7 pp |
| C3 recall | 0.971 | 0.968 | ≈0 |
| low-margin frac | **4.5%** | 3.6% | +0.9 pp |

MLP same story: NJ 0.909 vs others ~0.93–0.95; BE still worst (283).

## Where the OA points go (test ≈ 1500/class)

Nanjing RF mean confusion (counts):

```
          pred C1   C2    C3
true C1   1388    105     7
true C2    199   1220    81
true C3     12     31  1457
```

vs Hefei (higher baseline):

```
true C1   1415     83     2
true C2    157   1308    35
true C3      9     18  1473
```

**Error budget (approx contribution to OA gap):**

1. **C2→C1** (199 vs Hefei 157): ~+42 → **~0.9 pp OA**  
2. **C2→C3** (81 vs 35): ~+46 → **~1.0 pp OA**  
3. **C1→C2** (105 vs 83): ~+22 → **~0.5 pp OA**  
4. Water class itself is fine (C3 recall high)

→ Baseline OA is dragged down almost entirely by **non-built (C2) being hard**: confused into built **and** into water. Boundary Error is the visible tip; **C2→water is an extra Nanjing-specific slice**.

## What this is / isn’t

- **Is:** feature–WC mapping for C2 is messier in Nanjing under the same 3×3 S2 features (lowest C2 recall, highest BE, largest low-margin pool).  
- **Isn’t:** class prior (15k is balanced); water recall; “model can’t learn” (MLP still helps but stays last).  
- **Implication for raising OA:** any fix must recover **C2** (cut C2→C1 and C2→C3). Pure built-seed PE that doesn’t stabilize C2 won’t close the baseline gap.

## Actionable levers (baseline-centric)

1. Target **C2 false negatives** (predicted C1/C3): scene audit / features for veg–bare–water edge.  
2. Reduce **C1↔C2** confusion (main BE).  
3. Don’t expect OA to jump by reinforcing C3 or easy C1 interiors.

Outputs: `data2/nanjing_baseline_autopsy/`
