# Hangzhou standardisation — results, blend vs two-stage

All six cities now on the frozen two-stage BAMS150 rule (Hangzhou re-labelled:
46 carry-over + 104 new expert labels). Snapshot refreshed; the blend snapshot
is preserved at `docs/results_snapshot.blend_bak/`.

Only Hangzhou's 150 points changed. The other five cities' numbers are
byte-identical (verified: T3a rows, per-city T1/T2/T3b unchanged).

## Core result — INTACT

| statistic (pooled n=900) | blend | two-stage | Δ |
|---|---|---|---|
| **DW, Esri error coupling — Yule's Q** | 0.971 | **0.972** | +0.001 |
| DW, Esri error κ | 0.777 | 0.781 | +0.004 |
| DW, Esri co-error ratio (obs / indep) | 1.57 | 1.60 | +0.03 |
| DW = Esri agreement | 0.870 | 0.874 | +0.004 |
| DW = Esri excess over cond. independence | 0.255 | 0.258 | +0.003 |
| DW = Esri excess, LOCO range | [0.24, 0.27] | [0.24, 0.27] | — |
| S2c: DW = Esri consensus lift vs prior | +0.162 | +0.181 | +0.019 |
| S3a: two-class gap (DW=Esri − expert=DW) | 0.356 | 0.340 | −0.016 |
| S3c by-scene coupling | identical (five-city table, Hangzhou not in it) | | |

## Confidence story (T2) — INTACT

| pooled | blend | two-stage | Δ |
|---|---|---|---|
| expert ≠ WC, all labels | 0.543 | 0.533 | −0.010 |
| expert ≠ WC, high + medium | 0.563 | 0.546 | −0.017 |
| expert ≠ WC, **high confidence only** | 0.672 | 0.671 | −0.001 |

## Shifted — all Hangzhou-driven, all in the WC-pair quantities

| pooled | blend | two-stage | Δ |
|---|---|---|---|
| WC = DW agreement | 0.438 | 0.484 | **+0.046** |
| WC = Esri agreement | 0.452 | 0.497 | **+0.045** |
| **WC, DW error Q** | −0.091 | **+0.080** | CI [−0.05, 0.21] |
| **WC, Esri error Q** | −0.036 | **+0.142** | CI [0.01, 0.27] — now excludes 0 |
| WC, DW error κ | −0.045 | +0.040 | |
| WC, Esri error κ | −0.018 | +0.070 | |
| four-way agreement | 0.156 | 0.184 | +0.028 |
| net built over-call Δ_built | 0.020 | 0.050 | +0.030 |
| S2a WC = DW excess | 0.108 | 0.126 | +0.018 |
| S2a WC = Esri excess | 0.124 | 0.143 | +0.019 |

Driver: Hangzhou per-city WC = DW went 0.353 → **0.633**, WC = Esri 0.333 →
**0.600**. The two-stage rule selects Hangzhou points where all three products
mutually agree far more often (while the expert still dissents — Hangzhou
expert = WC only 0.593). This injects some three-way product co-agreement, and
hence some WC-pair co-error, that the blend set did not have.

Consequence for the write-up: the clean line "WorldCover pairs show no error
coupling (Q ≈ 0)" must become "WorldCover pairs Q ≤ 0.14, versus 0.97 for
DW/Esri" — still a 7× gap, κ 0.04–0.07 vs 0.78, co-error ratio ~1.05 vs 1.60.
The differential that carries the argument is unchanged; the "exactly zero"
phrasing is not available any more.

## LOCO (supporting) — conclusions unchanged

| Phase 2 overall (six-city mean) | blend | two-stage |
|---|---|---|
| B0 baseline downstream Human-OA | 0.397 | 0.375 |
| B2 proposed (gated) Human-OA | 0.421 | 0.389 |
| between-seed std | ~0.07 | ~0.07 |

Still within between-seed noise → "a few dozen boundary corrections do not move
a random forest trained on ~10,000 points" holds.

| Phase 3 referee (six-city mean) | blend | two-stage |
|---|---|---|
| WC_raw OA vs expert | 0.457 | 0.467 |
| LOCO_corrected OA vs expert | 0.522 | 0.536 |
| random_flip OA vs expert | 0.442 | 0.456 |
| LOCO changed-toward-expert frac | 0.828 | 0.825 |

LOCO still beats the random-flip control and moves the label map toward the
expert. Phase 1 operator accuracy 0.67–0.75 per city (Hangzhou 0.687 → 0.753,
fire precision 1.00 → 0.909 on 11 fires).

## Verdict

The paper's central claim (DW/Esri errors move in lockstep, Q ≈ 0.97; product
consensus does not track truth at the urban boundary) survives the rule change
untouched. Unifying the sampling rule removes the special case with **no cost to
the headline result** — the only edit forced is softening "WC pairs Q ≈ 0" to
"WC pairs ≤ 0.14 vs 0.97". Keeping Hangzhou (unified) is the clean choice;
dropping it is no longer necessary.

---

## Manuscript updated (2026-09-10)

Snapshot refreshed; pre-change snapshot at `docs/results_snapshot.blend_bak/`.
All manuscript representations brought into line with the two-stage numbers:

- **`manuscript_sec3.tex` / `body/methods.tex`** — deleted Eq. `hzblend` and the
  "five of the six cities … weighted-blend variant" paragraph; "For five of the
  six cities" → "For all six cities"; per-city median BAMS150 margin restated
  0.41–0.47; full-pool comparison figure corrected to the actual median
  ≈ 0.95–1.00 (was "0.77–0.91"); the Hangzhou "later pass" / "12 of 150 WC
  version drift" asides trimmed to one sentence.
- **`manuscript_sec4.tex` / `body/results.tex`** — every T1/T2/T3/S2a/S2b/S2c
  cell and every sentence that quotes them; LOCO Table T5, Phase 2/2b prose,
  Table phase3, Phase 3 prose, the maps table and captions. "WC pairs Q ≈ 0"
  → "WC pairs Q ≤ 0.14 vs 0.97" in S2b, S3(b), the summary paragraph and the
  Fig. 7 caption.
- **`manuscript_supplementary.md`** — S4 section and Table S5 (the Option-1
  robustness table) deleted; S1 (LOCO gate sweep) and S3-1 (ontology) tables
  and prose refreshed; S3-2 / Table S4 unchanged (five-city).
- **`body/frontmatter.tex` + `manuscript.md` abstract** — 0.47 / 0.45 / 0.42;
  four-way 18%; consensus 0.44 vs 0.45 base; "WorldCover errs independently of
  both (Q ≈ 0)" → "at most weakly coupled to either (Q ≤ 0.14)".
- **`body/discussion.tex`, `body/conclusions.tex`, `manuscript.md` §1 gap /
  §5 / §6** — same Q-phrasing change and the handful of quoted numbers
  (baseline 0.32–0.59, random-flip 33%, per-city WC range 0.32–0.59,
  high-conf rise 0.53→0.67).

Figures regenerated: `figures/Fig_diag_*.png` (fourway, confidence, error
direction, independence, all six correction maps). The Overleaf figure files
(`Figure3…Figure9`, `Figure S1…S4`) are the author's renamed copies — re-export
those from the refreshed `Fig_diag_*.png` before compiling.

---

## Scene stratification now six-city (2026-09-10)

Follow-up so no "Hangzhou is special" call-out remains. Hangzhou's 150 scene
notes were mapped to `Scene_group` (`scripts/hangzhou_scene_rows.py`); its 43
vegetation-edge points were split urban-green vs forest-edge from imagery
(`scripts/gee_hangzhou_veg_subclass.js` -> `veg_subclass.csv`: ug 10, fe 24,
ot 9). The 9 `ot` (veg-edge code did not survive review, no replacement type)
are dropped, so the pooled scene set is **n = 891** (750 + 141 Hangzhou).

**T4 five-city -> six-city:**

| Scene group | 5-city n / % | 6-city n / % |
|---|---|---|
| Built-up edge / urban fabric | 158 / 69.0 | 174 / 66.1 |
| Urban green space | 42 / 69.0 | 52 / 67.3 |
| Road / road edge | 106 / 63.2 | 117 / 62.4 |
| Water / water edge | 162 / 58.6 | 211 / 50.2 |
| Vegetation / forest edge | 106 / 55.7 | 130 / 52.3 |
| Bare land / construction | 65 / 38.5 | 88 / 43.2 |
| Rural settlement / surfaces | 40 / 40.0 | 41 / 41.5 |
| Paddy / cropland / field | 18 / 38.9 | 19 / 36.8 |
| Cropland / paddy transition | 11 / (below thr.) | 16 / 31.2 (**new row**) |
| Mixed / complex | 24 / 29.2 | 25 / 28.0 |

**Headline change:** "built-up edge and urban green space tie at 69.0%" ->
"urban green space 67.3% is the single highest, built-up edge 66.1% second".
Still the two highest, still well above the low groups; the "urban green space
disagrees as often as the built-up edge" finding holds and is arguably cleaner
(urban green space now #1 outright). Water/water-edge dropped from 58.6% to
50.2% because Hangzhou's 49 water points are relatively mild (22% disagree).

**S3c** (DW=ESRI by scene): rate 0.83-0.98 (was "0.81-0.98, five-city"),
still highest in urban green space (0.98) and mixed/complex (0.96). Unchanged
story.

**Manuscript:** every "five-city (n=750)" / "Hangzhou has no scene-code table"
/ "short codes in Wuhan… Chinese free text in Hefei…" call-out removed from
sec3/sec4/body/manuscript.md/supplementary. T4 and Table S4 are six-city; the
S3c method line drops "(five-city)". Scene grouping now framed as "the 891
boundary points with a resolvable scene note".
