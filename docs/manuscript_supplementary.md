# Supplementary Material

> Companion to `docs/manuscript.md` (MDPI *Remote Sensing*). Single source of
> record for framing and frozen numbers: `docs/Paper_Writing_Plan.md` §4;
> result snapshots in `docs/results_snapshot/`.
>
> Created 2026-09-09 to hold material moved out of the main text: the LOCO
> operator gate-sensitivity sweep (Phase 2b, S1) and reduced-feature variant
> (`transfer15`, S2), the full confound-check tables for the independence
> diagnostic (ontology + acquisition date, S3), and the Prototype-Expansion
> leakage evidence (S5). Sections 4.5 and 4.6 of the main text carry the
> one-line conclusions and point here for the tables. All tables regenerated
> 2026-09-10 from the refreshed `docs/results_snapshot/` (post the Hangzhou
> two-stage standardisation). S4 (Tables S5–S6, Figures S1–S4: LOCO operator
> per-city detail and the four remaining cities' before/after maps) added
> 2026-09-11 — these were cited from the main text but missing here. S6
> (Figure S5: example boundary-error chip gallery, C1/C2) added 2026-09-11.
> 2026-09-12: briefly trimmed main-text Figure 4 from Nanjing+Changsha to
> Nanjing only for page budget (which would have moved Changsha's map here
> as Figure S5 and bumped the chip gallery to Figure S6), then reverted the
> same day — the two-city figure is more convincing and the main text now
> states explicitly why Nanjing and Changsha are the two shown. S4/S6
> numbering below is therefore unchanged from 2026-09-11.

---

## S1. LOCO operator: gate-sensitivity sweep (Phase 2b)

Referenced from main text Section 4.6 (Phase 2). The proposed downstream
correction (transferred operator, boundary- and confidence-gated) moves the
six-city mean overall accuracy against the 150 expert points from 0.375 to only
0.389, inside the between-seed standard deviation (0.07). The sweep below varies
the confidence threshold τ and the within-city margin quantile *q* that gates
which candidates are eligible for correction, and reports the six-city mean
downstream overall accuracy against the expert labels (Human-OA) and against
WorldCover (WC-OA), together with the mean number of the ≈ 15,000 city labels
rewritten.

| τ | *q* | Points changed (of ≈ 15,000) | Human-OA | WC-OA |
|---|---|---|---|---|
| baseline | — | 0 | 0.375 | 0.929 |
| 0.90 | 0.05 | 1 | 0.375 | 0.929 |
| 0.90 | 0.10 | 5 | 0.379 | 0.929 |
| 0.90 | 0.20 | 8 | 0.382 | 0.929 |
| 0.85 | 0.05 | 9 | 0.379 | 0.929 |
| 0.85 | 0.10 | 23 | 0.389 | 0.929 |
| 0.85 | 0.20 | 41 | 0.405 | 0.928 |
| 0.80 | 0.10 | 60 | 0.408 | 0.928 |
| 0.80 | 0.20 | 112 | 0.449 | 0.925 |
| 0.70 | 0.10 | 186 | 0.454 | 0.923 |
| 0.70 | 0.20 | 345 | 0.516 | 0.913 |

*Table S1. Gate sensitivity, six-city mean. Source:
`docs/results_snapshot/loco_correction_operator/phase2b_sweep_summary.csv`.*

Reaching +14 points of downstream Human-OA (0.375 → ≈ 0.516) requires loosening
to τ = 0.70, *q* = 0.20 and rewriting ≈ 345 of 15,000 labels, at a cost of
≈ 1.5 points of WC-OA (0.929 → 0.913). At the safe operating thresholds used in
the main text (τ ≥ 0.85) the downstream effect is within between-seed noise. A
few dozen boundary-label corrections do not move a random forest trained on
≈ 10,000 points; this is why the label-map referee (main text Section 4.6,
Phase 3) evaluates the corrected map directly rather than through a retrained
classifier.

---

## S2. LOCO operator: reduced feature subset (`transfer15`)

Referenced from main text Section 4.6 (Phase 1). The main-text operator uses a
27-feature vector (nine point-level spectral features plus their nine 3 × 3
neighbourhood means and nine 3 × 3 neighbourhood standard deviations). The
`transfer15` variant drops the six raw reflectance bands and their neighbourhood
means — which carry cross-city radiometric offsets — keeping the three spectral
indices and all nine neighbourhood standard-deviation terms.

`transfer15` is slightly worse than the full 27-feature operator in every city
(operator-vs-expert agreement lower by roughly 0.02–0.05), so the raw
reflectance bands remain useful for the transfer despite the radiometric
offsets. The main text reports only the 27-feature operator.

*Source: `docs/results_snapshot/loco_correction_operator/phase1_corrector_transfer.csv`
(`feature_set` column: `all27` vs `transfer15`).*

---

## S3. Independence diagnostic: confound-check tables

Referenced from main text Section 4.5 (S3). The main text states the
conclusions; the full tables are below. The negative-control comparison (S3b in
the main text — WorldCover shares the Sentinel-2 input with Dynamic World and
ESRI but not the deep-segmentation model family, and its excess agreement is
about half that of DW = ESRI with error *Q* ≤ 0.14) uses quantities already
tabulated in main text Tables S2a/S2b and is not repeated here.

### S3-1. Ontology (main text S3a)

DW = ESRI agreement and the DW = ESRI − Human = DW gap under three class
schemes, six-city pool (n = 900; water points dropped for the third scheme,
n = 623).

| Scheme | DW = ESRI | Human = DW | Human = ESRI | Gap |
|---|---|---|---|---|
| Three-class (built / non-built / water) | 0.874 | 0.446 | 0.422 | 0.429 |
| Two-class (built vs rest) | 0.908 | 0.568 | 0.538 | 0.340 |
| Built vs vegetation (water points dropped) | 0.917 | 0.440 | 0.417 | 0.477 |

*Table S3. Collapsing the class ontology leaves DW = ESRI at 0.87–0.92 and the
gap at 0.34–0.48: the coupling is not an artefact of the three-class remap.
Source: `docs/results_snapshot/independence_diagnostic/S3a_ontology_confound.csv`.*

### S3-2. Acquisition date, by scene type (main text S3c)

DW = ESRI agreement resolved by physical scene group, pooled over the 891
boundary points with a resolvable scene note, groups with n ≥ 15.

| Scene group | n | DW = ESRI | Human = DW | Gap |
|---|---|---|---|---|
| Urban green space | 52 | 0.981 | 0.135 | 0.846 |
| Mixed / complex | 25 | 0.960 | 0.280 | 0.680 |
| Built-up edge / urban fabric | 174 | 0.943 | 0.828 | 0.115 |
| Cropland / paddy transition | 16 | 0.938 | 0.125 | 0.812 |
| Rural settlement / surfaces | 41 | 0.878 | 0.244 | 0.634 |
| Bare land / construction | 88 | 0.852 | 0.205 | 0.648 |
| Paddy / cropland / field | 19 | 0.842 | 0.316 | 0.526 |
| Vegetation / forest edge | 130 | 0.831 | 0.215 | 0.615 |
| Road / road edge | 117 | 0.829 | 0.427 | 0.402 |
| Water / water edge | 211 | 0.829 | 0.578 | 0.251 |

*Table S4. The DW = ESRI rate stays in 0.83–0.98 across every scene group and is
*highest* in the hardest transition zones (urban green space 0.98, mixed /
complex 0.96), rather than concentrated in the spectrally stable classes. If the
coupling came from the two products reading the same single acquisition it
should peak on the easy, stable classes and break down in the transition zones;
the observed pattern is the reverse. Source:
`docs/results_snapshot/independence_diagnostic/S3c_coupling_by_scene.csv`.*

---

## S4. LOCO correction operator: per-city detail

Referenced from main text Section 4.6 (LOCO Correction Operator, including the
before/after correction maps). Table S5 gives the Phase 1 operator-transfer
numbers behind the "0.67 to 0.75 against a baseline of 0.32 to 0.59" range
quoted in the main text; Table S6 gives the per-city operator-change and
expert $\neq$ WC counts behind main text Figure 4 (Nanjing, Changsha) and
Figures S1–S4 below (the other four cities).

| City | WC $\leftrightarrow$ expert | Majority-class baseline | Operator $\leftrightarrow$ expert | Confident firings ($\tau=0.85$) | Firing precision vs expert |
|---|---|---|---|---|---|
| Wuhan | 0.41 | 0.53 | 0.71 | 18 | 0.78 |
| Hefei | 0.51 | 0.63 | 0.67 | 28 | 0.68 |
| Nanchang | 0.46 | 0.65 | 0.73 | 11 | 0.82 |
| Nanjing | 0.32 | 0.55 | 0.74 | 15 | 0.93 |
| Changsha | 0.51 | 0.55 | 0.75 | 18 | 0.83 |
| Hangzhou | 0.59 | 0.53 | 0.75 | 11 | 0.91 |

*Table S5. Phase 1 operator transfer, per city. "WC $\leftrightarrow$ expert" is
the raw do-nothing baseline agreement (150 boundary points per city).
"Majority-class baseline" always predicts non-built, learned from the other
five cities' human labels (same leak-proof, leave-one-city-out pattern as the
operator itself; added 2026-09-11 per Remote_Sensing_投稿评估.md, to
contextualise the operator's gain against class imbalance alone, since
non-built is the majority class pooled across cities). "Operator
$\leftrightarrow$ expert" is the transferred operator's agreement with the
expert label. Confident firings are predictions with $p_{\max} \ge 0.85$ and
predicted class $\neq$ WC; firing precision is the fraction of those firings
that match the expert label. Source:
`data/analysis_outputs/loco_correction_operator/phase1_corrector_transfer.csv`.*

| City | Operator changes (of $\approx$15,000) | Expert $\neq$ WC (of 150) |
|---|---|---|
| Nanjing | 28 | 102 |
| Changsha | 63 | 74 |
| Wuhan | 35 | 88 |
| Hefei | 91 | 74 |
| Nanchang | 38 | 81 |
| Hangzhou | 30 | 61 |

*Table S6. Per-city operator-change and expert $\neq$ WC counts underlying the
before/after maps (main text Figure 4; Figures S1–S4 below). "Operator changes"
is the number of the $\approx$15,000 stratified city points the gated operator
($\tau=0.85$, lowest-decile within-city margin) overwrites relative to raw
WorldCover. Source: `docs/results_snapshot/loco_correction_operator/`.*

### Figures S1–S4. Before/after correction maps, remaining four cities

Companion to main text Figure 4 (Nanjing, Changsha; Section 4.6), which shows
those two cities because they bracket the range of operator performance —
Nanjing has the lowest raw WC-versus-expert agreement of the six and the
largest transfer gain, Changsha a mid-range city with high firing precision.
Each figure below: (**A**) raw WorldCover class map over the $\approx$15,000
stratified city points; (**B**) LOCO-corrected map ($\tau=0.85$,
lowest-decile within-city margin), the operator's changed points circled;
(**C**) the 150 expert boundary points, expert $=$ WC (green) vs
expert $\neq$ WC (red), over a faded backdrop of the full point set.

![Figure S1. Wuhan before/after correction maps.](submission/figures/FigureS1_map_Wuhan.png)

*Figure S1. Wuhan. (A) raw WorldCover; (B) LOCO-corrected, 35 points changed;
(C) 150 expert boundary points, expert $=$ WC (green, $n=62$) vs expert
$\neq$ WC (red, $n=88$).*

![Figure S2. Hefei before/after correction maps.](submission/figures/FigureS2_map_Hefei.png)

*Figure S2. Hefei. (A) raw WorldCover; (B) LOCO-corrected, 91 points changed;
(C) 150 expert boundary points, expert $=$ WC (green, $n=76$) vs expert
$\neq$ WC (red, $n=74$).*

![Figure S3. Nanchang before/after correction maps.](submission/figures/FigureS3_map_Nanchang.png)

*Figure S3. Nanchang. (A) raw WorldCover; (B) LOCO-corrected, 38 points
changed; (C) 150 expert boundary points, expert $=$ WC (green, $n=69$) vs
expert $\neq$ WC (red, $n=81$).*

![Figure S4. Hangzhou before/after correction maps.](submission/figures/FigureS4_map_Hangzhou.png)

*Figure S4. Hangzhou. (A) raw WorldCover; (B) LOCO-corrected, 30 points
changed; (C) 150 expert boundary points, expert $=$ WC (green, $n=89$) vs
expert $\neq$ WC (red, $n=61$).*

---

## S5. Within-city label propagation leaks (Prototype Expansion)

Referenced from main text Section 3.3. Before adopting the leave-one-city-out
design, we tested within-city propagation of the boundary labels to their
spectral neighbours (our variant of the prototype / pseudo-label
rectification-and-expansion family; "Prototype Expansion").

- **Leakage diagnostic.** Across five cities and neighbourhood sizes
  *K* ∈ {5, 10, 20, 30}, the fraction of expanded points whose spectral
  neighbours crossed the evaluation set was `leak_fraction = 1.0` in every
  configuration: without a spatial partition, every propagated label draws on a
  point that is also being scored.
- **Leak-proof re-run.** With the train/test split drawn by spatial block and
  the evaluation points explicitly removed from the neighbour search, the
  procedure produced **`n_expanded = 0`** — no valid expansions survive.
- **Apparent gain was the classifier swap, not propagation.** In the
  non-partitioned runs the "PE_Boundary" scheme reported overall accuracy 0.924
  / Boundary Error 227 against 0.914 / 276 for the raw-WC baseline, but with
  `n_expanded = 0` under the leak-proof control that ≈ +1 point is attributable
  to the RF→MLP classifier change and to treating the 150 hand labels as
  anchors, not to neighbour propagation.

*Source: `docs/results_snapshot/leak_diagnostic/` (`leak_diagnostic.csv`,
`spatial_block_results.csv`, `spatial_block_summary.csv`); generation script
`../gee-project2/scripts/spatial_block_validation.py`.*

---

## S6. Illustrative example chips: WorldCover boundary error (C1/C2)

Referenced from main text Section 4.6 (Direction of the WorldCover Error).
Table 3 there establishes the quantitative claim: once WorldCover calls a
boundary point "built-up", the expert reads non-built about seven times in
ten ($P(\text{expert}=\text{non-built} \mid \text{WC}=\text{built}) = 0.71$).
That number says nothing about what the error looks like on the ground.
Figure S5 puts real Sentinel-2 imagery behind it — 24 boundary points across
the six cities, chosen by a stated rule rather than eyeballed — so a reader
can see the error, not just read its rate. The gallery is illustration only;
it adds no statistic the main text does not already report, and no claim in
this paper rests on it.

**What the three panels show.** Panel **A** — consensus error (supports
C1 and C2), 12 points, 2 per city — is the figure's core case: WorldCover,
Dynamic World and Esri all call the point built-up, and the expert reads
non-built. This is not a rare corner case: it is the common outcome in the
WC$=$built $\wedge$ expert$=$non-built cell of the reference table (159 of
191 such points, 83%). It illustrates C2 directly (WorldCover's directional
error) and C1 by extension, since all three products land on the same wrong
answer at once. Panel **B** — coupled-pair error (supports C1), 6 points,
1 per city — isolates the C1 argument on its own: Dynamic World and Esri, the
two products whose errors are most correlated with each other (Section 4.5),
both call the point built-up and both are wrong; WorldCover — the product
least coupled to the other two — dissents and is correct. Read together,
panels A and B show what "agreement is not independent corroboration" means
in practice. In A, three products agree and are still wrong. In B, the two
most-correlated products agree and are wrong, while WorldCover — the least
correlated of the three — dissents and is right. Panel **C** — counter-examples,
6 points, 1 per city — is WorldCover calling built-up and the expert
agreeing. Without it a reader could reasonably suspect the gallery was built
by cherry-picking WorldCover's failures; Panel C shows the same product
getting the same kind of boundary call right, at the same rank-selection
severity as the other two panels.

**How the points were chosen.** Every point in the underlying reference table
(the same BAMS150 boundary-candidate pool used throughout the paper; see
Section 2's Boundary score / Selection paragraphs) carries a classifier
margin — the gap between the top two class posteriors of the random forest
used to propose boundary-ambiguous points in the first place. A small margin
means the point's spectral signature sits between classes; because the whole
BAMS150 pool is already selected for small margins relative to the full
15,000-point candidate set, ranking *within* that pool by margin
**descending** picks out the points the classifier was, relatively speaking,
most committed to — the clearest cases of each pattern, not the most
ambiguous ones. Points are ranked this way per city per group, ties broken by
`Original_ID` ascending, and the top `N_PER_CITY` taken (2 for A, 1 each for
B and C) — a rule fixed before any image was looked at
(`scripts/c2_chip_select.py`). A confidence filter is layered on top: Panels
A and B keep points at expert confidence `high` or `med`; Panel C keeps
`high` only (falling back to `med` solely where a city has no high-confidence
built$\leftrightarrow$built point — Nanchang), so the counter-examples sit
unambiguously inside built fabric and cannot be dismissed as boundary noise
themselves.

**Visual review and exclusions.** The rule above is deterministic, but a
margin ranking cannot see the imagery, so two rounds of visual review sit on
top of it. The first pass dropped points that were a poor illustration of the
case they were meant to represent once the actual chip was pulled: a scene
too ambiguous to read at a glance, a "built" counter-example that was not an
obvious urban core, or a point whose visible structure read as an unambiguous
building complex rather than vegetation. A second, wider pass compared the
top six to twelve ranked candidates per city and group side by side and
hand-picked the clearest of that set, rather than only accepting or rejecting
the single next-ranked fallback. Combined, 34 points were dropped across the
two passes and each replaced by the next surviving candidate for that city,
with the selection rule itself left untouched: ids 378, 558, 853, 927, 951,
1061, 1537, 1719, 1804, 1831, 1874, 1919, 2183, 2217, 2863, 2961, 2972, 3030,
3173, 3273, 3382, 3557, 3806, 3900, 3996, 4053, 4251, 4271, 4317, 4440, 4886,
4904, 4930, 9555. This is a visual-quality filter on the illustration, not a
re-selection of which cases count as errors — the counts behind Table 3 and
the 159/191 figure above are unaffected.

**Imagery.** `COPERNICUS/S2_SR_HARMONIZED`, 2021,
`CLOUDY_PIXEL_PERCENTAGE < 20`, median — the same recipe as the sampling
pipeline that produced the 15,000-point candidate pool and the WC/DW/Esri
class calls being compared. Using any sharper or more recent image source
here would let a reader attribute the error to poor image legibility rather
than to the product classification itself, which is exactly the confound the
§5 "product error vs. S2 legibility" discussion is written to rule out — so
the chips deliberately show the same imagery the products themselves were
scored against, nothing better.

![Figure S5. Example boundary chips illustrating the WorldCover directional error.](submission/figures/FigureS5_c2_chips.png)

*Figure S5. Twenty-four Sentinel-2 boundary chips (750 m window, true colour
with a false-colour NIR inset — vegetation red, water black), grouped as
(A) consensus error, (B) coupled-pair error, (C) counter-examples, as
described above. Each chip gives the WorldCover / Dynamic World / Esri /
expert calls, expert confidence, city and point id; a product's call is
printed in green where it matches the expert and red where it differs, so the
per-product pattern in each panel is readable at a glance without checking
the labels one by one. Selection rule, confidence filter and exclusions
detailed in the text above; illustration only, the quantitative claim rests
on Table 3 (§4.6).*

*[TODO: selection/fetch/assembly code for this figure will be available at
the repository DOI given in the Data Availability statement upon
publication.]*
