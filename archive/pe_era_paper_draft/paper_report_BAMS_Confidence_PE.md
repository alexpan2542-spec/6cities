# Boundary-Aware Active Sampling with Confidence-Filtered Prototype Expansion for Urban Land-Cover Correction

**Technical Report (Paper-Ready Experimental Summary)**  
**Task:** 3-class urban land cover (built-up / non-built / water), corrected against ESA WorldCover  
**Cities:** Wuhan, Changsha, Nanchang, Hefei, Nanjing (≈15,000 samples each)  
**Human budget:** BAMS150 per city  
**Date:** 2026-08-08

---

## Abstract

We study fixed-budget active learning for correcting urban WorldCover (WC) labels at built/non-built boundaries. **BAMS150** selects boundary-informative points for human annotation. Directly injecting all 150 labels into training yields little or no OA gain (**supervision dilution**). **Prototype Expansion (PE)** with human seeds improves OA in most cities. We further introduce **Confidence filtering**: only high-confidence labels are used as PE hard seeds. Across five cities, PE with confidence-aware seeds delivers about **+1.0 to +1.8 pp OA** in four cities, while **Nanjing remains a failure case (~+0.3 pp)** under WC-referenced evaluation, consistent with the strongest human–WC disagreement and the hardest baseline. The contribution is a complete pipeline—**select (BAMS) → filter (Confidence) → expand (PE)**—plus an applicability boundary.

---

## 1. Problem and Method

### 1.1 Problem

Product labels (WorldCover) are often reliable in homogeneous interiors but unreliable at urban **Class1–Class2 boundaries**. Under a fixed labeling budget, the questions are: which points to label, which labels are safe as hard supervision, and how to amplify scarce clean seeds.

### 1.2 Pipeline

| Stage                        | Role                                                                                                                             |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **BAMS150**                  | Boundary-aware active sampling of 150 points per city                                                                            |
| **Human label + Confidence** | Visual class; confidence ∈ {high, med, low} (or high/med by rule; see §2.2)                                                      |
| **PE**                       | Train teacher on WC train split + hard seeds; expand prototypes in feature space; soft-label low-margin neighbors; train student |

**Evaluation:** pointwise OA / Boundary Error (BE = C1↔C2 confusions) vs original WC `Class`; mean over **5 random seeds**.

**Notation**

- `Baseline`: RF/MLP on WC train labels only (tables use the RF baseline of each city’s confidence-PE run unless noted).
- `BAMS150_*`: overwrite WC labels at BAMS locations (RF), no PE.
- `PE_new`: PE using all human labels as seeds.
- `PE_new_high`: PE using **Confidence=high** seeds only.
- `PE_new_hm`: PE using high+med seeds.

---

## 2. Experimental Setup

### 2.1 Data

Five Chinese cities; balanced ~5k samples per class; 3×3 spectral/index features for the classifier; 70/30 stratified train/test (`random_state=42`). PE sweeps `K ∈ {5,10,20,30}`, margin threshold 0.15, soft weight 0.3; best K by (BE, −OA).

### 2.2 Confidence assignment

| City                          | Protocol                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------ |
| **Hefei, Nanjing**            | Visual Confidence (`h`/`m`/`l`) during re-annotation                           |
| **Wuhan, Changsha, Nanchang** | Rule: **C1↔C2 human–WC swap → high**; all other BAMS points → **med** (no low) |

### 2.3 BAMS150 Confidence counts and human–WC disagreement

| City        | n   | high | med | low | Human≠WC        | C1↔C2 swaps    | high-subset Human≠WC |
| ----------- | --- | ---- | --- | --- | --------------- | -------------- | -------------------- |
| Wuhan       | 150 | 73   | 77  | 0   | 58.0% (87)      | 48.7% (73)     | 100%\*               |
| Changsha    | 150 | 51   | 99  | 0   | 52.7% (79)      | 34.0% (51)     | 100%\*               |
| Nanchang    | 150 | 42   | 108 | 0   | 44.7% (67)      | 28.0% (42)     | 100%\*               |
| Hefei       | 150 | 81   | 64  | 5   | 49.3% (74)      | 44.0% (66)     | 53.1%                |
| **Nanjing** | 150 | 73   | 60  | 17  | **68.0% (102)** | **51.3% (77)** | **61.6%**            |

For Wuhan/Changsha/Nanchang, high is defined as C1↔C2 conflict, so high-subset disagreement is 100% by construction.

**Nanjing has the highest human–WC disagreement among the five cities under current labels.**

---

## 3. Main Results

### 3.1 Overall accuracy and boundary error (mean ± std over 5 seeds)

#### Wuhan

| Method           | OA                  | ΔOA (pp)  | BE        | n_seed |
| ---------------- | ------------------- | --------- | --------- | ------ |
| Baseline         | 0.9100 ± 0.0012     | —         | 296.8     | —      |
| BAMS150_new      | 0.9103 ± 0.0015     | +0.03     | 294.6     | —      |
| BAMS150_new_high | 0.9104 ± 0.0009     | +0.04     | 293.8     | —      |
| PE_new (all)     | **0.9283 ± 0.0023** | **+1.82** | **217.6** | 150    |
| PE_new_high      | 0.9265 ± 0.0017     | +1.65     | 225.0     | 73     |
| PE_new_hm        | 0.9283 ± 0.0023     | +1.82     | 217.6     | 150    |

#### Changsha

| Method           | OA                  | ΔOA (pp)  | BE        | n_seed |
| ---------------- | ------------------- | --------- | --------- | ------ |
| Baseline         | 0.9295 ± 0.0007     | —         | 234.2     | —      |
| BAMS150_new      | 0.9306 ± 0.0004     | +0.11     | 230.0     | —      |
| BAMS150_new_high | 0.9295 ± 0.0003     | 0.00      | 232.2     | —      |
| PE_new (all)     | **0.9447 ± 0.0024** | **+1.52** | **176.4** | 150    |
| PE_new_high      | 0.9441 ± 0.0013     | +1.46     | 179.6     | 51     |
| PE_new_hm        | 0.9447 ± 0.0024     | +1.52     | 176.4     | 150    |

#### Nanchang

| Method           | OA                  | ΔOA (pp)  | BE        | n_seed |
| ---------------- | ------------------- | --------- | --------- | ------ |
| Baseline         | 0.9125 ± 0.0011     | —         | 206.0     | —      |
| BAMS150_new      | 0.9132 ± 0.0008     | +0.07     | 201.8     | —      |
| BAMS150_new_high | 0.9126 ± 0.0011     | +0.01     | 203.6     | —      |
| PE_new (all)     | 0.9260 ± 0.0016     | +1.35     | 164.6     | 150    |
| PE_new_high      | **0.9271 ± 0.0017** | **+1.46** | **159.6** | 42     |
| PE_new_hm        | 0.9260 ± 0.0016     | +1.35     | 164.6     | 150    |

#### Hefei（visual Confidence；相对旧标 PE_old = +0.87）

| Method           | OA                  | ΔOA (pp)  | BE        | n_seed |
| ---------------- | ------------------- | --------- | --------- | ------ |
| Baseline         | 0.9325 ± 0.0005     | —         | 239.6     | —      |
| BAMS150_old      | 0.9319 ± 0.0014     | −0.06     | 243.6     | —      |
| BAMS150_new      | 0.9312 ± 0.0014     | −0.13     | 245.0     | —      |
| BAMS150_new_high | 0.9322 ± 0.0009     | −0.03     | 241.6     | —      |
| PE_old           | 0.9412 ± 0.0028     | +0.87     | 196.6     | 150    |
| PE_new (all)     | **0.9440 ± 0.0014** | **+1.15** | **186.4** | 150    |
| PE_new_high      | 0.9432 ± 0.0016     | +1.08     | 192.0     | 81     |
| PE_new_hm        | 0.9428 ± 0.0013     | +1.04     | 186.0     | 145    |

#### Nanjing（visual Confidence）

| Method           | OA                  | ΔOA (pp)  | BE        | n_seed |
| ---------------- | ------------------- | --------- | --------- | ------ |
| Baseline         | 0.9032 ± 0.0019     | —         | 304.2     | —      |
| BAMS150_old      | 0.9004 ± 0.0015     | −0.27     | 315.4     | —      |
| BAMS150_new      | 0.9010 ± 0.0004     | −0.21     | 314.8     | —      |
| BAMS150_new_high | 0.9024 ± 0.0005     | −0.08     | 309.0     | —      |
| PE_old           | 0.9067 ± 0.0022     | +0.36     | 286.2     | 150    |
| PE_new (all)     | 0.9053 ± 0.0016     | +0.21     | 294.4     | 150    |
| **PE_new_high**  | **0.9065 ± 0.0007** | **+0.33** | **289.8** | 73     |
| PE_new_hm        | 0.9048 ± 0.0029     | +0.17     | 294.4     | 133    |

### 3.2 Cross-city summary (ΔOA in percentage points)

| City                     | Baseline OA | PE_all    | PE_high   | PE_hm     | PE_high − PE_all |
| ------------------------ | ----------- | --------- | --------- | --------- | ---------------- |
| Wuhan                    | 0.910       | **+1.82** | +1.65     | +1.82     | −0.17            |
| Changsha                 | 0.930       | **+1.52** | +1.46     | +1.52     | −0.06            |
| Nanchang                 | 0.912       | +1.35     | **+1.46** | +1.35     | **+0.11**        |
| Hefei                    | 0.932       | **+1.15** | +1.08     | +1.04     | −0.07            |
| **Nanjing**              | **0.903**   | +0.21     | **+0.33** | +0.17     | **+0.12**        |
| **Mean (5 cities)**      | —           | **+1.21** | **+1.19** | **+1.18** | —                |
| **Mean (excl. Nanjing)** | —           | **+1.46** | **+1.41** | **+1.43** | —                |

### 3.3 Core findings

1. **BAMS alone does not raise OA.** RF overwrite with BAMS150 is ≈0 ΔOA (often slightly negative in Nanjing/Hefei).
2. **PE is required** for large gains in successful cities (+1.1–1.8 pp).
3. **Confidence filtering:**

- Successful cities: `PE_high` ≈ `PE_all` (within ~0.2 pp), i.e. high seeds already carry most PE benefit under a smaller effective budget.
- **Nanjing:** `PE_high` (**+0.33**) **>** `PE_all` (**+0.21**) **>** `PE_hm` (**+0.17**) → including uncertain labels dilutes PE (**supervision dilution**).

4. **Nanjing is a failure case** under WC-OA: gain remains ~+0.3 pp despite high-only PE.

---

## 4. Why Nanjing Fails (WC-referenced evaluation)

### 4.1 Difficulty fingerprint (Phase-1 diagnostics)

| City        | Base OA   | Base BE   | low-margin frac (MLP) | BAMS ΔOA     | PE ΔOA (Phase-1 / pre-confidence table) |
| ----------- | --------- | --------- | --------------------- | ------------ | --------------------------------------- |
| Wuhan       | 0.910     | 296.8     | 0.0157                | +0.03 pp     | +1.82                                   |
| Hefei       | 0.932     | 239.6     | 0.0148                | −0.06 pp     | +0.87†                                  |
| Nanchang    | 0.912     | 206.0     | 0.0195                | +0.07 pp     | +1.35                                   |
| Changsha    | 0.930     | 234.2     | 0.0127                | +0.11 pp     | +1.52                                   |
| **Nanjing** | **0.903** | **304.2** | **0.0214**            | **−0.27 pp** | **+0.36**                               |

†Hefei after visual Confidence re-annotation: PE_all = **+1.15** (§3).

Nanjing ranks worst on baseline OA/BE, BAMS ΔOA, and low-margin pool size.

### 4.2 Help / Hurt on the WC test set (Phase-1, mean over seeds)

| City        | BAMS help/hurt | PE help/hurt | BAMS net  |
| ----------- | -------------- | ------------ | --------- |
| Wuhan       | 1.14           | 2.06         | +1.2      |
| Changsha    | 1.34           | 2.12         | +5.0      |
| Nanchang    | 1.18           | 1.64         | +3.0      |
| Hefei       | 0.92           | 1.47         | −2.6      |
| **Nanjing** | **0.66**       | **1.14**     | **−12.2** |

Only Nanjing shows strong **hurt ≫ help** after BAMS under WC scoring.

### 4.3 Interpretation (for the paper)

Five cities share the same WC evaluation protocol. Nanjing combines:

1. **Highest human–WC disagreement on BAMS150 (68%);**
2. **Hardest baseline** (lowest OA, highest BE, largest low-margin pool);
3. **BAMS net-negative under WC-OA.**

Therefore WC-OA gains stay small even with Confidence-filtered PE. This is a **reference–objective mismatch at boundaries**, not evidence that PE is broken. Hefei shows that moderate disagreement (~49%) still allows ~+1.1 pp PE gains—disagreement alone is not sufficient for failure.

---

## 5. Ablations

### 5.1 Hefei: remove greenhouse-tagged seeds

Nine BAMS points tagged 大棚/温室 in Scene/Notes (protocol: greenhouse → Class2).

| Method                     | n_seed | ΔOA (pp) | BE    |
| -------------------------- | ------ | -------- | ----- |
| PE_all                     | 150    | +1.15    | 186.4 |
| PE_all without greenhouse  | 141    | +1.01    | 187.0 |
| PE_high                    | 81     | +1.08    | 192.0 |
| PE_high without greenhouse | 74     | +0.99    | 190.0 |

Removing greenhouse seeds changes OA by ≈ **−0.1 pp** (negligible).

### 5.2 Direct label injection vs PE

In all cities, `BAMS150_*` RF overwrite ≈ 0 ΔOA, while PE yields large gains where applicable → **expansion**, not mere label rewrite, drives improvement.

---

## 6. Conclusions

1. **BAMS150** effectively targets boundary errors but does not by itself improve WC-OA.
2. **PE** converts scarce human seeds into consistent OA/BE gains in four cities (**~+1.1–1.8 pp**).
3. **Confidence** operationalizes “not all sampled points should be hard seeds”:

- In successful cities, high-only PE nearly matches full-label PE with fewer seeds;
- In Nanjing, high-only is necessary to avoid dilution, yet gains remain limited.

4. **Nanjing** is a documented **failure case** under WC-referenced metrics: maximum human–WC conflict × hardest baseline × BAMS hurt≫help. The method’s applicability boundary is part of the contribution.

**One-sentence claim:**  
_Under a fixed budget, boundary active sampling must be paired with confidence filtering and prototype expansion; the same protocol succeeds when human corrections align sufficiently with the WC evaluation target, and fails when boundary reference conflict and scene difficulty couple—as in Nanjing._

---

## 7. Reproducibility paths

| Result                                    | Path                                                                                                           |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Wuhan / Changsha / Nanchang Confidence PE | `data2/{wuhan,changsha,nanchang}_confidence_bams_pe/`                                                          |
| Hefei visual Confidence PE                | `data2/hefei_relabel_bams_pe/`                                                                                 |
| Nanjing visual Confidence PE              | `data2/nanjing_relabel_bams_pe/`                                                                               |
| Cross-city PE summary                     | `data2/cross_city_confidence_pe/`                                                                              |
| Phase-1 diagnostics (help/hurt, ranks)    | `data2/nanjing_phase1_diagnostics/`                                                                            |
| Hefei greenhouse ablation                 | `data2/hefei_greenhouse_ablation/`                                                                             |
| Manuals                                   | `data2/{City}/04_manual/{City}_BAMS150_Manual.csv`                                                             |
| Runner                                    | `scripts/city_confidence_bams_pe.py`, `scripts/hefei_relabel_bams_pe.py`, `scripts/nanjing_relabel_bams_pe.py` |

---

## Appendix A — Full method list per city run

Each confidence-PE experiment reports:  
`Baseline`, `BAMS150_old`, `BAMS150_new`, `BAMS150_new_high`, `BAMS150_new_hm`, `PE_old`, `PE_new`, `PE_new_high`, `PE_new_hm`.

For Wuhan/Changsha/Nanchang, human labels are unchanged vs backup ⇒ `*_old` ≡ `*_new` for label content; differences appear only when Confidence filters seeds (`PE_new_high`).

## Appendix B — Mean BE reduction under PE_new_high

| City     | Baseline BE | PE_high BE | ΔBE   |
| -------- | ----------- | ---------- | ----- |
| Wuhan    | 296.8       | 225.0      | −71.8 |
| Changsha | 234.2       | 179.6      | −54.6 |
| Nanchang | 206.0       | 159.6      | −46.4 |
| Hefei    | 239.6       | 192.0      | −47.6 |
| Nanjing  | 304.2       | 289.8      | −14.4 |

---

_End of report._
