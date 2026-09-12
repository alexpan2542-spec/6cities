# Second-reference VERDICT (Human × WC × DW × ESRI)

**Setup:** BAMS150 × 5 cities; DW/ESRI year=2021; remap to built/non-built/water.  
QA: local WC vs EE WC ≈ 1.00 (Nanchang 0.987).

## Pre-registered question

**Is Nanjing highest on human≈DW≠WC?**

| City | human≈DW≠WC | human≈ESRI≠WC | human=WC | human=DW | WC=DW |
|------|------------:|--------------:|---------:|---------:|------:|
| **Nanjing** | **30.0%** | **30.0%** | **32.0%** | 42.7% | 46.7% |
| Wuhan | 29.3% | 29.3% | 41.3% | **50.7%** | 46.7% |
| Nanchang | 21.3% | 16.7% | 55.3% | 40.0% | 36.0% |
| Changsha | 19.3% | 16.0% | 50.7% | 43.3% | 49.3% |
| Hefei | 19.3% | 17.3% | 50.7% | 39.3% | 48.7% |

→ **Yes, Nanjing is #1**, but only **+0.7 pp over Wuhan** — not a large gap.

## What this does / does not prove

**Supports (soft):**
- Nanjing has the **lowest human=WC** (32%) and jointly highest **human≈second-product≠WC** (DW & ESRI both 30%).
- Among human≠WC, Nanjing still agrees with DW/ESRI **44%** of the time → many conflicts are not “random human noise.”

**Does not prove “Nanjing WC uniquely broken”:**
- Wuhan is almost identical on human≈DW≠WC (29.3%).
- WC=DW is **not** worst in Nanjing (46.7%; Nanchang 36% is worse).
- Among human≠WC, Wuhan/Nanchang match DW as often or more often than Nanjing.

## Paper-safe line

> On BAMS150, Nanjing ranks highest on the rate of human–Dynamic World (and ESRI) agreement against WC (~30%), with the lowest human–WC agreement (~32%). This is consistent with a **reference–objective tension**, but Wuhan is nearly as high on human≈DW≠WC, so the result supports **elevated mismatch**, not a unique WC product failure confined to Nanjing.

Outputs: `data2/cross_city_second_ref_dw_esri/`
