# Cross-product agreement does not arbitrate reference labels at the urban built-up/vegetation boundary: a six-city diagnostic

> Working manuscript draft for MDPI *Remote Sensing* (research article).
> Single source of record for framing, decisions and frozen numbers:
> `docs/planning/Paper_Writing_Plan.md`.
> Section 3 drafted 2026-09-07; revised 2026-09-07 after the BAMS-selection
> reproduction (`scripts/reproduce_bams_selection.py`) and the addition of
> Hangzhou's Dynamic World / ESRI layers (multi-reference subset now n = 900);
> §3.1 corrected 2026-09-08 — area of interest is the FAO GAUL 2015 ADM2
> (prefecture) polygon, not a built-up mask; the labelling protocol and the
> shared-Sentinel-2 defence (plan §7.4) are now written in.
>
> **2026-09-13.** Contribution list collapsed from five to four: former C4
> (the leave-one-city-out check) folded into C2 as a robustness clause —
> "the pattern is not a single-city artifact, since a leave-one-city-out
> check reproduces the same signal ... with a bounded downstream footprint"
> — because as a standalone numbered item it read as a method/process
> ("a leave-one-city-out correction operator showing...") rather than a
> finding, even after the earlier "supporting evidence, not a correction
> framework" hedge. Former C5 renumbered to C4; the back-reference "the C5
> practice of building consensus products..." in §5.3 updated to "C4". No
> other section (abstract, §5.4, §5.5, the (i)–(iv) scope list) uses the
> letter labels, so nothing else changed. See [[paper-pivot-loco]].
>
> **2026-09-13 (follow-up).** Trimmed the trailing ", with a bounded
> downstream footprint" off that same C2 clause: it describes the LOCO
> operator's limited effect on a downstream classifier, not evidence that
> the directional-error pattern generalises across cities, so grafted onto
> "not a single-city artifact, since ..." it read as a non sequitur. The
> point is preserved intact in §5.3 ("the right scale for auditing a label
> map, not for retraining one").
>
> **2026-09-13 (sync fix).** The abstract here (and its `manuscript_zh.txt`
> translation) still carried the LOCO sentence ("A leave-one-city-out
> operator shows the disagreement carries a signal that transfers across
> cities, but with a bounded downstream effect") that the 2026-09-11
> word-count trim removed from `submission/manuscript.tex` and
> `manuscript_for_ai_review.txt`. Dropped it here and in the Chinese
> translation to restore sync across the four-file group (see
> [[section-delivery-as-tex-fragments]]); no other wording changed.
>
> **Status.** All sections now drafted. Section 1 (Introduction + 1.1 Related
> work) drafted 2026-09-09; closes with an explicit C1–C4 contribution
> statement (final paragraph, collapsed from C1–C5 2026-09-13). Section 2 (Study Area & Data) drafted 2026-09-09
> — 2.1 study area + site-selection rationale (monitoring demand, policy
> citations per `study-area-citations`), 2.2 AOI + Sentinel-2 basis, 2.3 the
> three products, 2.4 annotators (author-supplied); two placeholders remain
> (`[N]+` years in §2.4; the `fig:studyarea` locator map). Sections 3 (Methods)
> and 4 (Results) drafted 2026-09-07 against the frozen six-city snapshot
> (`docs/results_snapshot/`, n = 900). Section 5 (Discussion) drafted
> 2026-09-09. Section 6 (Conclusions) drafted 2026-09-09 (three paragraphs;
> spine order, existence register, C4 caution [was C5, renumbered 2026-09-13]). Abstract drafted 2026-09-09
> (leads with the DW = ESRI coupling; agreement rates last). §4.1 opening
> reordered to lead with DW = ESRI 0.87 vs Human = DW 0.44; the "Q ≈ 0.97 does
> not use the expert at all" wording softened to "robust to an imperfect
> expert" in §3.1, §4.5 and §5.5. Trim done (2026-09-09): LOCO Phase 2b sweep
> and `transfer15` moved to `docs/manuscript_supplementary.md` (S1, S2);
> confound tables S3a/S3c moved there (Tables S3, S4) with the one-line
> conclusions kept in §4.5. Citations are `(Author, year)` placeholders; a
> numbered list is substituted at typesetting.
>
> **Phase 3 (2026-09-09) — done.** All `Section N.x` cross-reference
> placeholders resolved to real subsection numbers; `\ref{app:pe}` removed (its
> content is now Supplementary S5); `Section 3.2d/e/b` normalised. All three
> analysis scripts re-run 2026-09-09 (gee env) — deterministic, reproduce the
> current data. `docs/results_snapshot/` fully refreshed from that re-run
> (SNAPSHOT_INFO note appended); it now matches the manuscript. Manuscript
> caught up to the current data where the 2026-09-08 Nanchang scene-26 relabel
> had been missed: §4.3 (P(expert = non-built | WC = built) 0.69 → 0.71;
> net_built_over_call 0.017 → 0.02; T3 confusion row 70/181/12 → 64/187/12),
> §5.3 (0.69 → 0.71), §4.6 Phase 2 (0.400 → 0.397 raw, 0.414 → 0.421 gated,
> +1.4 → +2.4 pt, sd 0.06 → 0.07). §4.4 was already current (n = 158).
> `manuscript_supplementary.md` S1 (phase2b) and S3/S4 (ontology + by-scene)
> regenerated from the fresh CSVs.
> §3.1 inter-annotator count corrected 100 → 120 (20/city; κ values were
> already the 120-point ones; Fleiss κ on the 116 points all three completed).
> §2.3 legend-crosswalk table added (Table `tab:legend_map`) from the locked
> remap code; wetland/flooded-veg → water and Esri Clouds → no-data confirmed
> by the authors 2026-09-09. §2.4 annotator paragraph finalised 2026-09-09
> (first author's role + research; three colleagues, 10+ yr each) — no
> placeholders left.
> `make_correction_maps.py` re-run 2026-09-09: §4.7 table counts unchanged
> (already current); §4.7 prose fixed (Hangzhou 21 → 30, 3 → 4 firings); the
> six `Fig_diag_map_*` PNGs regenerated and re-snapshotted.
> **Open — needs the authors:** the `fig:studyarea`, `fig:flowchart` and
> `fig:map_*` figures (the `Fig_diag_*` analysis figures and `Fig_diag_map_*`
> maps already exist and are current; `fig:map_*` just needs the two chosen for
> the main text wired to the real filenames at typesetting).

---

## Abstract

Global 10 m land-cover products are increasingly used together, on the
assumption that where independent products agree the agreed label is close to
correct — so that consulting a second or third product can arbitrate a disputed
pixel. Rather than ask which product is best, this study tests whether agreement
among ESA WorldCover, Google/WRI Dynamic World and Esri/Impact Observatory Land
Cover is independent evidence of correctness at the urban built-up/vegetation
boundary, the setting where the products are weakest. Using 900
expert-interpreted boundary points — 150 in each of six middle- and lower-Yangtze cities
(China), placed on the most spectrally ambiguous transitions by a boundary-aware
sampling step and labelled from a 2021 Sentinel-2 composite — we find that the
only agreement rate that is high everywhere is between Dynamic World and Esri
(0.87), while each product matches the expert on fewer than half the points
(Human = WorldCover 0.47, Human = Dynamic World 0.45, Human = Esri 0.42). That
mutual agreement is not independent corroboration: the two products' errors move
almost in lockstep (Yule's *Q* ≈ 0.97 between their error indicators), whereas
WorldCover's errors are at most weakly coupled to either (*Q* ≤ 0.14). Product
consensus predicts the expert label no better than a single product (0.44
against a 0.45 base rate), and all four sources agree on only 18% of points. The
WorldCover error is directional — a "built-up" call is wrong toward non-built
about seven times in ten — so an agreement-filtered label set systematically
under-represents urban
vegetation. These
results are an existence characterisation for six cities and one sensor, not a
global prevalence estimate; within that scope, cross-product agreement should
not be used on its own as a pseudo-reference or a high-confidence label at the
urban boundary.

**Keywords:** global land-cover products; ESA WorldCover; Dynamic World; Esri
Land Cover; map agreement; correlated error; urban vegetation boundary; accuracy
assessment; weak supervision; Sentinel-2

---

## 1. Introduction

> Drafted 2026-09-09. Framing per plan §1.2 and §6.4: diagnostic-first;
> "agreement-as-label" assumption named concretely here (Option D, 2026-09-09)
> but with the modest register kept — an existence result in a bounded domain,
> not a global claim. Citations are `(Author, year)` placeholders; note the
> two distinct 2024 first-author-Wang papers, tagged 2024a / 2024b in
> *References to insert (Section 1)*.

Global 10 m land-cover products — ESA WorldCover, Google/WRI Dynamic World, and
Esri/Impact Observatory Land Cover — are now routine inputs to urban and
environmental analysis, distributed ready-to-use on cloud platforms and cited
with headline overall accuracies in the 70–90% range (Venter et al., 2022;
Xu et al., 2024). Those headline figures are whole-map averages, and they hide
a well-documented weak spot: the transition between built-up land and
vegetation, where mixed pixels, spectral similarity and differing class
definitions push the products apart from one another and from field observation
(Xu et al., 2024; Yuan et al., 2026). Intra-urban green space — parks, street
trees, residential and campus greenery embedded in the built matrix — is
absorbed into the built class by several products, a bias large enough to
matter for the urban-ecology and planning applications that consume these maps
(Yuan et al., 2026).

A common response to this uncertainty, where independent reference data are
scarce or costly, is to lean on several products at once. Pixels on which
multiple independent products agree are treated as reliable: merged into
accuracy-weighted consensus layers (Tuanmu and Jetz, 2014), used as
pseudo-reference for accuracy assessment, or harvested as high-confidence
training and validation samples for new classifications (Zhang and Roy, 2017;
Wang et al., 2024a). The step is convenient and intuitive, and it rests on an
assumption that is seldom stated, let alone tested: that where independent
products agree, the agreed label is close to correct — and, by extension, that
consulting a second or third product can arbitrate a disputed pixel. Whether
that assumption holds at the urban built-up/vegetation boundary, the very place
the products are weakest, has not been examined. The product-comparison
literature reports how far the maps sit from reference data on average, but not
whether their mutual agreement is independent corroboration or shared error,
and not specifically on boundary pixels.

Testing it requires two things the standard comparisons do not provide. First,
reference labels placed on the contested pixels rather than spread across a
globally representative sample, so that the hard cases are not diluted to a
handful of points. Second, a test that separates agreement-as-corroboration
from agreement-as-correlated-error: two products that agree because both are
right, and two that agree because they share a model family, a training corpus
or a preprocessing chain, look identical in a pairwise agreement table but
carry opposite implications for arbitration.

This paper supplies both, for six cities in the middle and lower reaches of the Yangtze River in China
(Wuhan, Hefei, Nanchang, Nanjing, Changsha, Hangzhou). A boundary-aware
sampling step spends a deliberately small photo-interpretation budget — 900
points, 150 per city — on the most spectrally ambiguous built-up/vegetation
transitions, each labelled by expert interpretation of a 2021 Sentinel-2
composite. On these points we (i) quantify how far ESA WorldCover, Dynamic
World and Esri Land Cover depart from the expert labels and from one another;
(ii) test directly whether a second and third product arbitrate the
disagreement, formalising the arbitration assumption as a violation of
conditional independence and reporting the joint error structure of the three
products; (iii) characterise the direction of the WorldCover error and the
physical scenes in which the disagreement concentrates; and (iv), as a
supporting analysis, ask with a leave-one-city-out correction operator whether
the disagreement carries a signal that transfers to a city holding out all of
its own labels.

The scope is stated plainly. The six cities are one country, one climate, one
broad urban-build morphology and one sensor; every rate reported here is
conditional on a boundary-candidate population and is not a map-wide accuracy
of any product. The result is an existence characterisation within that domain,
not an estimate of how often the behaviour occurs elsewhere. The middle and
lower Yangtze is not put forward as an easy case — its paddy, aquaculture and wetland mosaics
make it one of the harder land-cover settings — but as a region where the
monitoring demand is concrete.

This paper makes four contributions. **C1**, the central one, is a direct test
of an assumption that is widely relied on but seldom examined — that agreement
among global products is evidence of correctness, so that a second or third
product can arbitrate a disputed pixel — showing that at the urban
built-up/vegetation boundary, in these six cities, it can fail. The failure is
both operational and mechanistic: product consensus predicts the expert label no
better than a single product (arbitration PPV 0.44 against a 0.45 base rate),
and Dynamic World and Esri err together (Yule's *Q* ≈ 0.97 between their error
indicators) while WorldCover's errors are at most weakly coupled to either
(*Q* ≤ 0.14), so agreement between the two coupled products is correlated error
rather than independent
corroboration. **C2**: the WorldCover boundary error is directional, not
symmetric noise — once WorldCover calls "built-up", the expert reads non-built
about seven times in ten — so an agreement-filtered label set systematically
under-represents urban vegetation; the pattern is not a single-city artifact,
since a leave-one-city-out check reproduces the same signal in a city holding
out all of its own labels. **C3**: the
boundary-aware sampling step, offered as a practical and reproducible way to
spend a small photo-interpretation budget on the hard pixels, not as a new
active-learning method. **C4**: the resulting methodological caution —
cross-product agreement should not be used on its own as a pseudo-reference, a
high-confidence label source, or a quality filter, because the spectral and
temporal safeguards in current weak-supervision pipelines are not designed to
catch error that is correlated across the products being combined.

**Product inter-comparison.** Several studies benchmark the 10 m global
products against independent ground data. Venter et al. (2022), in this
journal, compare Dynamic World, WorldCover and Esri Land Cover at a global
reference sample (overall accuracy on their scheme: Esri ≈ 75%, Dynamic
World ≈ 72%, WorldCover ≈ 65%) and document systematic per-class biases; they
also record the definitional split that matters here — Dynamic World folds
urban vegetation into "Built Area", whereas WorldCover's "Built-up" nominally
excludes it. Xu et al. (2024) run a comparative independent validation of the
same three maps, with overall accuracies of 73–83% depending on how
reference-data uncertainty is handled. Wang et al. (2024b) evaluate six
high-resolution products over China and find inter-product consistency running
below per-product accuracy. Chakraborty et al. (2024) show large disagreements
between products in estimates of urban land *area*, driven by scale and
differing urban definitions. Together these establish that the products
disagree; none focuses on boundary pixels, places reference labels on the hard
cases by design, or tests whether product agreement functions as arbitration.

**Labels derived from existing maps.** A parallel line of work treats existing
maps as a label source. Zhang and Roy (2017) use the 500 m MODIS land-cover
product to derive training labels for a 30 m Landsat classification;
Wang et al. (2024a) fuse several 10 m products by weighted majority vote into a
training-sample pool, with a downstream spectral-outlier filter;
Tong et al. (2025) train a global high-categorical-resolution map by weak
supervision from existing products. These pipelines operationalise the
agreement-as-reliable-label assumption that Section 4.5 tests. The same paper's
within-region prototype / pseudo-label rectification-and-expansion approach
("PRE") propagates sparse labels to their feature neighbours; we adapted it to
a within-city setting during this project and found that it leaks under spatial
autocorrelation once evaluation points are properly held out (Section 3.3),
which is why the correction operator here is trained across cities with the
target city entirely excluded.

**The gap.** No existing study combines (i) a boundary-pixel focus rather than
a whole-map average, (ii) active sampling that places a small manual budget
precisely on built-up/vegetation transitions, and (iii) an explicit test of the
arbitration assumption, formalised as a testable violation of conditional
independence and supported by the joint error structure of the three products
(Section 4.5), with class ontology, shared imagery and acquisition date ruled
out as cheap explanations. *Why* the two
products are coupled is left as a hypothesis for the Discussion, not asserted as
a result. Sections 3–6 present the methods, the diagnostic and correction
results, their interpretation, and the bounded conclusions.

---

## 2. Study Area and Data

> Drafted 2026-09-09. Site-selection framing per plan §1.2 (2026-09-05 /
> 2026-09-06 lock): regional monitoring demand, not author convenience or
> regional expertise. Citations resolved per `study-area-citations`
> (2026-09-09). Two placeholders remain for the authors: `[N]+` years in
> §2.4, and the Figure \ref{fig:studyarea} locator map, not yet made.

### 2.1. Study area

The study covers six provincial capitals in the middle and lower reaches of the
Yangtze River in China — Wuhan (Hubei), Hefei (Anhui), Nanchang (Jiangxi),
Nanjing (Jiangsu), Changsha (Hunan) and Hangzhou (Zhejiang)
(Figure \ref{fig:studyarea}). The region has urbanised rapidly over the past
three decades, with built-up land expanding largely at the expense of cropland
and water (Yi et al., 2021; Liao et al., 2025), and it falls under several
overlapping land-management regimes — ecological protection red lines, permanent
basic farmland protection, and shoreline and flood-detention-area controls —
that all rely on accurate and repeatable land-cover monitoring (Yu et al., 2023;
Tong et al., 2024). It is not an easy classification setting: paddy, aquaculture
ponds and wetland interleave finely with the built and vegetated classes, mixed
pixels are common at 10 m, and the built-up/vegetation transition — the focus of
this study — is among the boundaries global products place least reliably
(Lu et al., 2026; Zhao et al., 2026). The six cities were selected because this
monitoring demand is concrete; they are not put forward as a globally
representative sample (Section 5.5).

### 2.2. Area of interest and Sentinel-2 basis

For each city the area of interest is the whole second-level administrative unit
(FAO GAUL 2015, ADM2 — the prefecture-level municipality; FAO, 2015), taken in
full with no built-up or urban mask, so that the candidate pool spans the
built-up core together with the surrounding rural land. All spectral inputs come
from a single 2021 Sentinel-2 surface-reflectance composite per city
(`COPERNICUS/S2_SR_HARMONIZED`, full calendar year, scenes below 20% cloud,
per-pixel median): six bands (B2, B3, B4, B8, B11, B12) and three indices
(NDVI, NDBI, MNDWI), each also summarised over a 3 × 3 pixel neighbourhood. The
same composite is the primary reference imagery for the expert labelling. The
compositing recipe, the stratified candidate pool and the boundary-aware
selection are given in Section 3.1. All image processing was carried out in
Google Earth Engine (Gorelick et al., 2017).

### 2.3. Global land-cover products

Three global 10 m land-cover products are compared, all for the 2021 epoch.

- **ESA WorldCover v200** (Zanaga et al., 2022) — an annual gradient-boosted-tree
  classification of Sentinel-1 and Sentinel-2 with a rule-based post-process.
  WorldCover has a dual role here: its remapped classes drive the stratified
  candidate pool (Section 3.1), and it is one of the three references in the
  diagnostic.
- **Dynamic World** (Brown et al., 2022) — a near-real-time deep
  semantic-segmentation model on Sentinel-2; we take the 2021 annual mode of the
  class band (`GOOGLE/DYNAMICWORLD/V1`).
- **Esri / Impact Observatory Land Cover** (Karra et al., 2021) — an annual deep
  semantic-segmentation model on Sentinel-2; we take the 2021 layer of the
  time-series product (`ESRI_Global-LULC_10m_TS`).

Each product's native legend is collapsed to the same three classes used for the
expert labels — built-up, non-built, water — as set out in
Table \ref{tab:legend_map}.

| Product (native classes) | → built-up | → water | → non-built |
|---|---|---|---|
| **ESA WorldCover v200** (11) | 50 Built-up | 80 Permanent water, 90 Herbaceous wetland, 95 Mangroves | 10 Tree cover, 20 Shrubland, 30 Grassland, 40 Cropland, 60 Bare/sparse vegetation, 70 Snow and ice, 100 Moss and lichen |
| **Dynamic World** (9) | 6 Built | 0 Water, 3 Flooded vegetation | 1 Trees, 2 Grass, 4 Crops, 5 Shrub and scrub, 7 Bare, 8 Snow and ice |
| **Esri / Impact Observatory 2021** (9) | 7 Built Area | 1 Water, 4 Flooded vegetation | 2 Trees, 5 Crops, 8 Bare ground, 9 Snow/Ice, 11 Rangeland |

*Table \ref{tab:legend_map}. Legend crosswalk from each product's native classes
to the three-class scheme. Wetland and flooded/inundated vegetation are assigned
to **water** for all three products, a deliberate and consistent ontology choice
for this wetland- and aquaculture-rich region. Esri's "Clouds" class (code 10)
is treated as no-data and excluded. The same three-class WorldCover remap defines
the stratified candidate pool (Section 3.1).*

The schemas also differ at the boundary in a way that matters for this study:
Dynamic World folds urban trees and grass into its "Built Area" class, whereas
WorldCover's "Built-up" nominally excludes vegetation (Venter et al., 2022); this
split is taken up again in Section 5.3. Which analysis draws on which product is
stated in Sections 3.2–3.3 — in outline, WorldCover enters every analysis, while
Dynamic World and Esri enter only the four-way agreement and the independence
diagnostic (Section 3.2), and the correction-operator referee (Section 3.3).

### 2.4. Annotators

All 900 reference points were interpreted by the first author, a remote-sensing
analyst at the Land Satellite Remote Sensing Application Center, Ministry of
Natural Resources of China, where she is responsible for satellite-imagery
coordination and provision across the ministry's natural-resources system and
carries out applied research on feature-based land-cover classification from
high-resolution optical imagery, including multi-feature cloud and snow
detection for accurate automated recognition and extraction of target features.
The reliability exercise of Section 3.1 was carried out by three of her
colleagues at the same centre (denoted A, B and C; not authors of this study),
each with more than ten years of operational satellite-image interpretation
experience and none of whom took part in the original labelling; the blind
re-check of Appendix A was done by one of the three. This operational and
research experience is what made a single consistent reference feasible on the
boundary candidates; it is an enabling condition for the reference, not a
rationale for the choice of study area (Section 2.1), and the resulting
single-annotator reference is treated as a limitation in Section 5.5.

---

## 3. Methods

The analysis has three stages, summarised in Figure \ref{fig:flowchart}
(`submission/figures/Figure2_workflow.pdf`). First, a boundary-aware sampling step
(Section 3.1) spends a small photo-interpretation budget on the pixels where
global products are most likely to depart from expert judgement, producing 900
hand-labelled boundary points across six cities. Second, a multi-reference
agreement diagnostic (Section 3.2) quantifies *how much*, *where*, and *in
which direction* ESA WorldCover, Dynamic World and ESRI Land Cover disagree
with the expert labels, and tests directly whether consulting a second and a
third product resolves the disagreement. Third, as a supporting analysis, a
leave-one-city-out (LOCO) correction operator (Section 3.3) asks whether the
expert corrections learned on five cities transfer to a sixth city that
contributes no labels of its own, together with the evaluation protocol and
leakage controls specific to that test.

Every agreement and disagreement rate in this paper is computed on the
boundary-candidate points of Section 3.1. Because those points are chosen to
be the hardest and most error-prone pixels, the rates are **conditional on
that population and are not map-wide accuracies** of any product; the
qualifier is repeated wherever the numbers appear.

### 3.1. Boundary-aware candidate sampling

**Candidate pool.** The area of interest for each city is its second-level
administrative polygon (FAO GAUL 2015, ADM2 — the prefecture-level
municipality; FAO, 2015), taken whole, with no built-up or urban mask. ESA
WorldCover 2021, v200, at 10 m (Zanaga et al., 2022) is remapped from its 11
classes to the three target classes used throughout: *built-up* (WorldCover
class 50), *water* (80 permanent water, 90 herbaceous wetland, 95 mangrove),
and *non-built* (all remaining vegetated and bare classes: 10, 20, 30, 40, 60,
70, 100). Within each polygon we draw a stratified random sample of 5,000
points per remapped class — 15,000 points per city (Google Earth Engine
`stratifiedSample`, 10 m scale, fixed seed) — so the three classes are
balanced before any boundary-aware selection. Because the pool spans the whole
prefecture rather than a built-up envelope, it includes rural land; this is
why the scene stratification of Section 3.2(d) resolves paddy/cropland,
forest-edge and rural-settlement boundary types alongside the urban ones. It
is the boundary-aware step below, not a spatial mask, that concentrates the
900 annotated points onto cover transitions.

**Inputs to the sampler.** Feature values come from a single 2021 Sentinel-2
composite (`COPERNICUS/S2_SR_HARMONIZED`, full calendar year, scenes with
`CLOUDY_PIXEL_PERCENTAGE` < 20, per-pixel median, clipped to the administrative
polygon): six surface-reflectance bands (B2, B3, B4, B8, B11, B12) and three
spectral indices (NDVI, NDBI, MNDWI), each as a point value and as a 3 × 3
pixel (30 m) neighbourhood mean and standard deviation. This is the same
composite used for the expert labelling described below.

**Boundary score.** Boundary-Aware Margin Sampling (BAMS) combines a
classifier-uncertainty signal with a local-heterogeneity signal; the exact
procedure is reproduced deterministically in
`scripts/reproduce_bams_selection.py`
(`data/analysis_outputs/bams_reproduction/`), which regenerates the on-disk
point sets to which the 900 hand labels are anchored (150/150 match, all six
cities).

1. *Classifier margin.* A random forest (300 trees, fixed seed) is fitted on
   all 15,000 candidates of a city with the remapped WorldCover label as
   target and the **nine point-level** spectral features as predictors (no
   neighbourhood terms enter this model). Class posteriors are read back
   in-sample; the margin of a point is
   *m* = *p*<sub>(1)</sub> − *p*<sub>(2)</sub>, the gap between its two largest
   class posteriors. Small *m* means the spectral signature lies between two
   classes rather than inside one.
2. *Local spectral heterogeneity.* BoundaryScore is the raw, unnormalised sum
   of the 3 × 3 neighbourhood standard deviations of the four visible and
   near-infrared bands, σ<sub>3×3</sub>(B2) + σ<sub>3×3</sub>(B3) +
   σ<sub>3×3</sub>(B4) + σ<sub>3×3</sub>(B8). Large values indicate the 30 m
   window straddles a cover transition.

We emphasise that this random forest serves strictly as a proposal mechanism
to isolate spectrally ambiguous zones for annotation; it plays no part in
assigning class labels. Because all 900 boundary candidates were labelled
independently by expert photo-interpretation, bias in the initial WorldCover
training targets governs only *which* pixels are examined and does not enter
the reference against which the products are subsequently scored. The
consequent conditioning of the reported rates is addressed below
("Sampling-induced conditioning").

**Selection.** For all six cities, BAMS150 is a two-stage cut: keep the 1,000
lowest-margin points, then, of those, keep the 150 with the largest
BoundaryScore (ascending `Original_ID` as the deterministic final tie-break). A
plain 150-smallest-margin set ("Top150") is generated for reference only and is
not used anywhere downstream. This places every city's sample deep in the
ambiguous regime: per-city median BAMS150 margin 0.41–0.47, against a full-pool
median of ≈ 0.95–1.00. Each diagnostic below is additionally reported per city.
Selection yields 150 points per city, 900 in total.

**Expert labelling.** The 150 points per city were interpreted one by one in
the Google Earth Engine Code Editor at roughly 1:2 000 zoom, under a single
protocol common to all six cities. The primary reference was the 2021
Sentinel-2 composite itself, viewed both as a true-colour rendering (B4/B3/B2,
stretched 0–3 000) and as a false-colour infrared rendering (B8/B4/B3,
stretched 0–4 000); a Google high-resolution basemap was consulted only as
ancillary spatial context — for form and edge localisation, when the
Sentinel-2 view was insufficient. Because that basemap carries no single
acquisition date, the 2021 Sentinel-2 composite is the temporal anchor of the
reference. Each point was assigned the dominant cover within its 10 m pixel —
{built-up, non-built, water} — with a three-level confidence tag
(high / medium / low) and a free-text scene note. This produces 900 expert
boundary labels, the reference anchor for every analysis below. (The
site-selection rationale is given in Section 2.1 and the annotator background
in Section 2.4.)

Reliability of these single-annotator labels was assessed with an independent
blind re-labelling exercise: three colleagues at the same centre (A, B and C;
Section 2.4), with operational remote-sensing image-interpretation experience,
each re-labelled a stratified subsample of 120 points (20 per city, proportional
to the confidence-tier distribution) without access to the original labels, the
product outputs, or one another's judgements. Agreement with the reference was
almost perfect for all three (Cohen's κ = 0.95, 0.91, 0.94; Landis and Koch,
1977). Mean pairwise exact agreement across the six interpreter pairs was
92.67%, and multi-rater agreement on the 116 points all three completed was
almost perfect (Fleiss' κ = 0.89).

**Sampling-induced conditioning.** BAMS is deliberately non-representative: it
concentrates the labelling budget on the pixels that are hardest to classify,
so every agreement or disagreement rate in Section 4 is conditional on this
boundary-candidate population and is not a map-wide accuracy of WorldCover,
Dynamic World or ESRI Land Cover (see also Sections 4.2 and 5.x).

A second point of conditioning deserves statement here. BAMS selects points
from Sentinel-2 spectra (a WorldCover-trained random forest's margin) and
Sentinel-2 local heterogeneity, while the expert reference is read chiefly
from a 2021 Sentinel-2 composite, so selection and reference share a common
Sentinel-2 information base. This is examined in Section 5.5.

### 3.2. Multi-reference agreement diagnostic

**Reference layers.** Three global 10 m products are compared with the expert
labels: ESA WorldCover 2021 v200 (WC), a gradient-boosted-tree classifier with
a rule-based post-process on Sentinel-1/2 (Zanaga et al., 2022); Dynamic World
(DW), a fully-convolutional semantic-segmentation model producing
near-instantaneous per-scene Sentinel-2 predictions, here taken as the 2021
annual label mode (Brown et al., 2022); and ESRI Land Cover
(ESRI / Impact Observatory), a deep segmentation model on annual Sentinel-2
composites (Karra et al., 2021). Each product is sampled at the 900 boundary
points and remapped to the same three classes. WC, DW and ESRI are now
available for all six cities (multi-reference subset *n* = 900); Hangzhou's DW
and ESRI were sampled in a later pass (`scripts/add_hangzhou_second_ref.py`).
For Hangzhou, the freshly sampled WorldCover class differs from the stored
value at 12 of 150 points (WorldCover version drift; ≤ 0.3% for the other five
cities); no reported quantity uses the re-sampled WC layer. Binomial
proportions are reported with Wilson 95% confidence intervals (Wilson, 1927).

The diagnostic has five parts.

**(a) Four-way agreement (Table \ref{tab:T1}, Figure \ref{fig:fourway}).**
Pairwise agreement rates among {expert, WC, DW, ESRI}, plus the tri-agreement
fraction (expert = DW = ESRI) and the four-way agreement fraction, pooled
(*n* = 900) and per city.

**(b) Confidence stratification (Table \ref{tab:T2},
Figure \ref{fig:confidence}).** The expert ≠ WC disagreement rate is computed
over all labels, over the high + medium subset, and over the
high-confidence-only subset, per city and pooled. The purpose is to test
whether the disagreement is merely annotator noise on ambiguous pixels — if so
it should shrink on the high-confidence subset. We note the built-in bias of
this test (an annotator may assign "high" precisely when a WC error is
obvious) and read the result accordingly (Sections 4.2, 5.x).

**(c) Direction of the WC error (Table \ref{tab:T3},
Figure \ref{fig:errordir}).** From the WC → expert confusion matrix on the
boundary points we report the conditional error rates
P(expert = non-built | WC = built) and P(expert = built | WC = non-built), and
the net over-/under-call of built-up,
[*n*(expert = non-built | WC = built) − *n*(expert = built | WC = non-built)] / *n*.

**(d) Physical scene stratification (Table \ref{tab:T4},
Figure \ref{fig:scene}).** Each boundary point carries a free-text scene note
recorded during annotation. Notes were grouped into physical boundary types
(built-up edge / urban fabric; urban green space; road / road edge; water /
water edge; vegetation / forest edge; paddy / cropland; rural settlement;
bare land / construction) using the annotator-supplied scene-code table, and
the expert ≠ WC disagreement rate with Wilson intervals is reported for groups
with *n* ≥ 15. This is a keyword-based localisation aid, not a formal
stratified sample (Section 5.5); the scene notes are free-text, so scene-type
contrasts are drawn only in the pooled set. The grouping covers the 891
boundary points whose note resolves to a single physical boundary type; the
remaining nine carried no resolvable type and are omitted. The 26 Nanchang
points initially recorded without a scene note — their expert class
placeholder-filled from WorldCover during annotation — were re-annotated on
sub-metre imagery by two annotators and assigned an expert class and a scene
code; 13 of the 26 disagree with WorldCover.

**(e) Independence and arbitration test (Section 4.5; Tables \ref{tab:S2}–\ref{tab:S3},
Figure \ref{fig:independence}).** Multi-reference arbitration implicitly
assumes the references approach the truth *independently*, so that agreement
between two of them corroborates a label. We test that assumption on the
*n* = 900 subset; the central contrast — S2a and S2b, below — does not require
the expert label to be ground truth. S2c is the exception, since it asks how
well the expert label is predicted, and is accordingly capped by the expert's
own reliability (Section 5.5).

- *S2a — excess agreement.* Observed pairwise agreement is compared with the
  agreement expected if the two products were conditionally independent given
  the true class,
  Σ<sub>j</sub> P(Y = j) Σ<sub>c</sub> P(A = c | Y = j) P(B = c | Y = j). The
  conditional class distributions are estimated in-sample, so they partly
  absorb the true dependence they are meant to null out; this biases the
  expectation *toward* the observed agreement, so the reported excess is
  expected to understate the true value. Uncertainty is a 4,000-sample point bootstrap; a
  leave-one-city-out range is also reported.
- *S2b — error dependence.* With e<sub>A</sub> = 1{A ≠ expert}, we compute
  Yule's *Q* and Cohen's κ between each pair of product error indicators, and
  the ratio of the observed co-error rate to its marginal-independent value.
  Whether DW errs when ESRI errs is a statement about the joint error
  structure of the three products alone.
- *S2c — arbitration value.* We evaluate P(expert = ℓ | DW = ESRI = ℓ) —
  whether product consensus predicts the expert label better than a single
  product — against the single-product baselines P(expert = DW),
  P(expert = ESRI) and against the consensus of the non-deep-learning pair
  (WC = DW).
- *S3a–c — confound checks.* (S3a) The ontology is collapsed to two classes
  (built vs rest) and to built-vs-vegetation (water points dropped), and the
  test repeated. (S3b) WorldCover is a negative control: it shares the
  Sentinel-2 input with DW and ESRI but not the deep-segmentation model
  family. (S3c) The DW = ESRI agreement rate is resolved by scene type to check
  whether the coupling is concentrated in the transition zones where a shared
  image date would matter, or is roughly uniform. The
  ontology and by-scene tables are in the Supplementary Material
  (Tables S3, S4); S3b uses quantities already in Tables S2a/S2b.

### 3.3. LOCO correction operator (supporting analysis)

We first tried propagating the boundary labels to their spectral neighbours
*within the same city* — our variant of the prototype / pseudo-label
rectification-and-expansion family, of which PRE (Tong et al., 2024,
arXiv:2406.00891; 2025) is a representative instance (*Prototype Expansion*).
It leaks under spatial autocorrelation: once evaluation points are excluded
from the neighbour search, the procedure produces no valid expansions, and the
apparent gain it showed without that control traces to a classifier change,
not to propagation (full leakage diagnostic in Supplementary S5). We instead
define a correction operator trained on the labelled points of *N* − 1 cities
and transferred to a held-out city that contributes no labels of its own —
leave-one-city-out (LOCO) — which removes this dependence by construction, and
report it as a supporting result, not the paper's main claim.

**Operator.** For a held-out city *c*, a random forest (300 trees) is trained
on the pooled 150-point boundary sets of the other five cities, with a
27-feature vector as input (the nine point-level spectral features plus their
nine 3 × 3 neighbourhood means and nine 3 × 3 neighbourhood standard
deviations), the expert class as target, and sample weights by confidence tier
(high 1.0, medium 0.6, low 0.3). Applied to city *c*'s 150 boundary points it
returns, per point, a predicted class and its posterior *p*<sub>max</sub>. The
operator *fires* — overwrites the WorldCover label — only where it is both
confident and dissenting: *p*<sub>max</sub> ≥ τ (τ = 0.85) and predicted class
≠ WC class. For the downstream-map experiment (Phase 2) firing is further
restricted by a boundary gate: only candidates in the lowest decile of
within-city margin are eligible. A reduced-feature variant is reported in
Supplementary S2.

**Phase 1 — operator transfer.** The LOCO operator is applied to the held-out
city's 150 boundary points and its predicted class scored against the expert
label, relative to the "do nothing" baseline (the WC = expert rate). We report
operator accuracy, the number of confident firings, and their precision
against the expert label.

**Phase 2 — downstream map correction.** Every baseline and corrected variant
in this phase is scored with the same downstream classifier — a random
forest, 300 trees, scikit-learn defaults otherwise — so no change of
classifier architecture can be mistaken for a method effect; this is a
deliberate departure from an earlier version of the pipeline, in which part of
the apparent gain traced to a random-forest → MLP swap rather than to the
correction step. For each held-out city and each of five spatial-block
partition seeds, this classifier is trained on the city's ≈ 10,000
non-boundary training points; splits are made by whole spatial cells, not by
random points (each city gridded into 6 × 6 cells on latitude/longitude
quantiles, whole cells assigned to the test set until ≈ 30% of points are held
out), so a training point and a test point are never near-duplicates across
the split. Five labelling schemes are compared: (B0) raw WC labels; (B2) WC
labels corrected by the transferred LOCO operator on boundary-gated,
confidence-gated points — *the proposed method*; (B1) WC labels corrected by a
*within-city oracle* operator that does see city *c*'s expert labels, an
optimistic ceiling; (B3) the transferred operator applied without the
boundary gate (ablation); and (B4) the same number of labels flipped to a
random other class (control). The correction operator never sees the held-out
city, and the downstream classifier never sees the held-out city's expert
labels — its 150 boundary points are evaluation-only. Each trained classifier
is evaluated for overall accuracy (OA) and Boundary Error (BE), defined as the
number of built ↔ non-built confusions, BE = *cm*[built, non-built] +
*cm*[non-built, built], on the held-out spatial-block test set (against WC)
and on the 150 expert points (against the expert). A τ × margin-quantile
sensitivity sweep is reported in Supplementary Table S1.

**Phase 3 — multi-reference referee.** The corrected label map is evaluated
directly, with no downstream classifier, on the 150 held-out points of each
city (all six now carry DW and ESRI), against (i) the expert label and
(ii) the strict tri-consensus subset expert = DW = ESRI. Of
the points the operator changed, we report the fraction moved toward the
expert label and the fraction moved toward the DW/ESRI consensus, with the
random-flip control and the within-city oracle ceiling for reference.

**Reproducibility.** All inputs, analysis scripts and frozen result snapshots
are listed in the Data Availability statement; BAMS point selection is
regenerated by `scripts/reproduce_bams_selection.py`. The pipeline runs in a
fixed conda environment with a scikit-learn RandomForest throughout (no GPU or
neural-network components).

---

## 4. Results

All rates below are computed on the 900 boundary-candidate points of
Section 3.1. Because BAMS deliberately concentrates the labelling budget on the
most spectrally ambiguous pixels of each city, these are **conditional rates on
that population, not map-wide accuracies** of ESA WorldCover, Dynamic World or
ESRI Land Cover; the qualifier is not repeated at every number but applies
throughout. Sections 4.1–4.5 report the multi-reference diagnostic: how much
the three products depart from the expert labels (4.1), whether that departure
is annotator noise (4.2), which direction the WorldCover error runs (4.3),
which physical boundary types it concentrates in (4.4), and whether a second
and third product supply independent corroboration (4.5). Section 4.6 reports
the supporting LOCO correction operator, including the before/after maps.
Class codes are 1 = built-up, 2 = non-built, 3 = water.

### 4.1. Multi-reference agreement at the boundary

Table \ref{tab:T1} and Figure \ref{fig:fourway} give the pairwise agreement
rates among {expert, WC, DW, ESRI} on the boundary candidates, pooled
(n = 900) and per city.

| City | n | H=WC | H=DW | H=ESRI | WC=DW | WC=ESRI | DW=ESRI | H=DW=ESRI | all four |
|---|---|---|---|---|---|---|---|---|---|
| Wuhan | 150 | 0.41 | 0.51 | 0.50 | 0.47 | 0.45 | 0.91 | 0.47 | 0.19 |
| Hefei | 150 | 0.51 | 0.39 | 0.41 | 0.49 | 0.53 | 0.83 | 0.33 | 0.17 |
| Nanchang | 150 | 0.46 | 0.39 | 0.36 | 0.36 | 0.45 | 0.81 | 0.29 | 0.11 |
| Nanjing | 150 | 0.32 | 0.43 | 0.43 | 0.47 | 0.47 | 0.93 | 0.40 | 0.11 |
| Changsha | 150 | 0.51 | 0.43 | 0.35 | 0.49 | 0.47 | 0.87 | 0.33 | 0.18 |
| Hangzhou | 150 | 0.59 | 0.53 | 0.48 | 0.63 | 0.60 | 0.90 | 0.46 | 0.34 |
| **Pooled** | **900** | **0.47** | **0.45** | **0.42** | **0.48** | **0.50** | **0.87** | **0.38** | **0.18** |

*Table \ref{tab:T1}. Four-way agreement on the boundary candidates. H = expert.
Pooled Human = WC 0.47, Wilson 95% CI [0.43, 0.50]. Per-city n = 150.*

The one rate that is high everywhere is DW = ESRI: 0.81 to 0.93 across all six
cities, pooled 0.87 — against a Human = DW of only 0.45. The two products that
agree closely with each other do not agree with the expert, and neither does the
third: pooled, Human = WC is 0.47 [0.43, 0.50], Human = DW 0.45,
Human = ESRI 0.42. Substituting one product for another does not help — the
cross-product rates WC = DW (0.48) and WC = ESRI (0.50) sit barely above the
product-versus-expert rates and nowhere near a level that would resolve the
boundary. The four sources coincide unanimously on only 18%
of boundary candidates; on the remaining ≈ 82% at least one dissents, and the
largest internally consistent subset — expert, DW and ESRI all agreeing —
covers just 38%. Consulting a second and a third global product therefore does
not arbitrate the urban boundary: outside that 38% there is no reference
majority to appeal to.

The per-city Human = WC rate ranges from 0.32 in Nanjing (the lowest of the
six) to 0.59 in Hangzhou, with Wuhan at 0.41, Nanchang at 0.46, and Hefei and
Changsha at 0.51. Whether the DW = ESRI agreement is independent corroboration —
and therefore usable for arbitration — is tested directly in Section 4.5.

### 4.2. Confidence stratification

If the expert–WC disagreement were annotator noise on inherently ambiguous
pixels, it should shrink when the analysis is restricted to the labels the
annotator marked "high" confidence. Table \ref{tab:T2} and
Figure \ref{fig:confidence} show the opposite.

| Stratum | Pooled | Wuhan | Hefei | Nanchang | Nanjing | Changsha | Hangzhou |
|---|---|---|---|---|---|---|---|
| All labels | 0.53 (n = 900) | 0.59 | 0.49 | 0.54 | 0.68 | 0.49 | 0.41 |
| High + medium | 0.55 (n = 813) | 0.58 | 0.49 | 0.54 | 0.68 | 0.57 | 0.42 |
| High only | **0.67 (n = 419)** | 0.88 | 0.53 | 1.00 (n = 42) | 0.62 | 0.58 | 0.55 |

*Table \ref{tab:T2}. Expert ≠ WC disagreement rate by expert confidence tier.
Pooled high-only rate 0.67, Wilson 95% CI [0.62, 0.71].*

The pooled expert ≠ WC rate rises from 0.53 over all labels to 0.67
[0.62, 0.71] on the high-confidence subset. The rise holds in five of six
cities and is steep in two: Wuhan 0.59 → 0.88, and Nanchang, where all 42
high-confidence boundary labels disagree with WorldCover. Nanjing is the sole
exception, easing from 0.68 to 0.62, and it starts from the highest all-label
rate of the six. Where the expert is most certain, WorldCover is not closer to
the expert but further from it.

This test carries a known upward bias (Section 3.2(b)): an annotator may reserve
the "high" tag precisely for cases where a WC error is visually unambiguous, so
the high-only rate is better read as an upper bound on how systematic the
disagreement is than as an unbiased estimate. Even so, two observations do not
follow from that selection effect alone. First, the result still refutes the
"noise on ambiguous pixels" reading — noise would not concentrate on the
confident calls. Second, the magnitude in Nanchang (42/42) and Wuhan (0.88) is
larger than a labelling-confidence artefact would plausibly produce. The
sampling-induced conditioning of this subset is examined further in
Section 5.5.

A targeted check addresses the same concern from the other side. From the
high-confidence expert ≠ WorldCover subset (281 points; the count before the
Hangzhou re-labelling was 293), we drew 30 — five per city, in decreasing BAMS
classifier margin — and had them
re-classified by an analyst who had taken no part in the original labelling,
working from sub-metre satellite imagery and the 2021 Sentinel-2 composite,
blind to both the expert label and the WorldCover label
(Appendix \ref{app:recheck}). The re-classification reproduced the original
expert label on 29 of the 30 points and matched the WorldCover label on none;
the one exception was a mixed water/vegetation edge pixel the second analyst
could not resolve. This removes the same-analyst circularity in reading these
high-confidence disagreements as WorldCover error rather than annotator error —
with the caveat that the second analyst shared the imagery sources, so this is
an independent judgement, not a fully independent measurement.

### 4.3. Direction of the WorldCover error

Table \ref{tab:T3} is the pooled WC → expert confusion matrix on the 900
boundary points; Figure \ref{fig:errordir} shows it per city.

| WC label \ expert | built | non-built | water | row n |
|---|---|---|---|---|
| WC = built | 72 | 197 | 10 | 279 |
| WC = non-built | 152 | 262 | 57 | 471 |
| WC = water | 7 | 57 | 86 | 150 |

*Table \ref{tab:T3}. WC → expert confusion on the boundary candidates (counts),
six cities pooled.*

Derived from the matrix: P(expert = non-built | WC = built) = 0.71,
P(expert = built | WC = non-built) = 0.32, and the net over-call of built-up,
[n(expert = non-built | WC = built) − n(expert = built | WC = non-built)] / n
= 45/900 = 0.05. In aggregate the boundary error is close to symmetric —
WorldCover is not simply over- or under-painting built-up along the edge — but
it is conditionally asymmetric: once WorldCover commits to "built-up" at a
boundary candidate, roughly seven times in ten the expert reads non-built. The
non-built | built rate is between 0.60 and 0.79 in every one of the six cities.
This one-directional signal — a WC "built-up" call at the boundary is
untrustworthy in a specific, predictable way — is what the correction operator
of Section 4.6 exploits. Figure S5 in the Supplementary Material shows example
Sentinel-2 chips illustrating this directional pattern; the quantitative claim
rests on Table \ref{tab:T3} above, not on the gallery.

### 4.4. Physical scene stratification

Each boundary point carries a free-text scene note recorded during annotation
(Section 3.2(d)). Notes were grouped into physical boundary types using the
annotator's scene-code table, and the expert ≠ WC rate computed per group for
groups with n ≥ 15. The grouping covers the 891 boundary points whose note
resolves to a single boundary type (all 900 less nine), and includes the
26 Nanchang points that were re-annotated from imagery after an annotation
placeholder was found (Section 3.2(d)). Results are in Table \ref{tab:T4} and
Figure \ref{fig:scene}.

| Scene group | n | Expert ≠ WC | Wilson 95% CI |
|---|---|---|---|
| Urban green space | 52 | 67.3% | [54, 78] |
| Built-up edge / urban fabric | 174 | 66.1% | [59, 73] |
| Road / road edge | 117 | 62.4% | [53, 71] |
| Vegetation / forest edge | 130 | 52.3% | [44, 61] |
| Water / water edge | 211 | 50.2% | [44, 57] |
| Bare land / construction | 88 | 43.2% | [33, 54] |
| Rural settlement / surfaces | 41 | 41.5% | [28, 57] |
| Paddy / cropland / field | 19 | 36.8% | [19, 59] |
| Cropland / paddy transition | 16 | 31.2% | [14, 56] |
| Mixed / complex | 25 | 28.0% | [14, 48] |

*Table \ref{tab:T4}. Expert ≠ WC disagreement by physical boundary type, pooled
over the 891 boundary points with a resolvable scene note, groups with n ≥ 15.*

Disagreement is highest exactly where built-up meets vegetation. Urban green
space (67.3%) and the built-up edge itself (66.1%) are the two highest groups,
both well above bare land / construction (43.2%) and rural settlement (41.5%),
where the cover classes are more spectrally separable. That urban green
space — parks, street trees, campus and residential greenery embedded in the
built matrix — disagrees as often as the built-up edge is consistent with the
documented tendency of global products to absorb intra-urban vegetation into the
built class (Section 5.3; Yuan et al., 2026), and is reported here as a finding
in its own right, not only a category in the table.

Two caveats from Section 3.2(d) apply. The grouping is a keyword-based
localisation aid, not a formal stratified sample, so the rates locate the
disagreement rather than estimate a population quantity. And the scene notes
are free-text — a mix of short codes and Chinese, mapped post hoc to a common
code table — so scene-type contrasts are drawn only in the pooled set, never
city against city (Section 5.5).

### 4.5. Independence and arbitration

Multi-reference arbitration assumes the references approach the truth
*independently*, so that agreement between two of them corroborates a label.
Section 4.1 showed DW and ESRI agreeing with each other far more (0.87) than
either agrees with the expert (≈ 0.43). This section tests whether that mutual
agreement is independent corroboration. All statistics are on the n = 900
subset unless noted; the central contrast (S2a–S2b) does not require the
expert label to be ground truth. The arbitration-value check (S2c) is the
exception (Section 3.2(e)). Figure \ref{fig:independence} summarises.

**S2a — excess agreement (Table \ref{tab:S2}a).** Observed pairwise agreement
is compared with the agreement expected if the two products were conditionally
independent given the true class.

| Pair | Observed | Expected if cond. indep. | Excess | Bootstrap 95% CI | LOCO range |
|---|---|---|---|---|---|
| DW = ESRI | 0.87 | 0.62 | **+0.26** | [0.23, 0.29] | [0.24, 0.27] |
| WC = DW | 0.48 | 0.36 | +0.13 | [0.10, 0.15] | [0.11, 0.14] |
| WC = ESRI | 0.50 | 0.35 | +0.14 | [0.12, 0.17] | [0.13, 0.15] |

DW and ESRI agree 0.26 more often than conditional independence allows — about
twice the excess of either WorldCover pair (+0.13, +0.14). Because the
conditional class distributions are estimated in-sample, the expected value is
biased *toward* independence, so +0.26 is expected to understate the true
value (Section 3.2(e)).

**S2b — error dependence (Table \ref{tab:S2}b).** With
e<sub>A</sub> = 1{A ≠ expert}, we compute the co-error rate, Yule's *Q* and
Cohen's κ between each pair of product error indicators.

| Error pair | Error rate A / B | Co-error obs / indep (ratio) | Yule's *Q* [95% CI] | κ(errors) | Same wrong class \| both wrong |
|---|---|---|---|---|---|
| DW, ESRI | 0.55 / 0.58 | 0.51 / 0.32 (× 1.60) | **0.97** [0.96, 0.98] | 0.78 | 0.97 |
| WC, DW | 0.53 / 0.55 | 0.31 / 0.30 (× 1.03) | **+0.08** [−0.05, 0.21] | 0.04 | 0.87 |
| WC, ESRI | 0.53 / 0.58 | 0.33 / 0.31 (× 1.06) | **+0.14** [0.01, 0.27] | 0.07 | 0.87 |

This is the sharpest result of the diagnostic. DW and ESRI err on 55% and 58%
of boundary candidates and co-err on 51%, against 32% expected if their errors
were independent; Yule's *Q* between their error indicators is 0.97
[0.96, 0.98], and when both err they select the same wrong class 97% of the
time. For the WorldCover pairs *Q* is an order of magnitude smaller — +0.08
[−0.05, 0.21] for WC, DW and +0.14 [0.01, 0.27] for WC, ESRI, against 0.97,
with κ(errors) of 0.04 and 0.07 against 0.78. The WC, DW interval still spans
zero; WC, ESRI is marginally positive but nowhere near the DW/ESRI coupling.
WorldCover's errors are at most weakly coupled to the two deep-segmentation
products; those two err together. This is a
differential between the product pairs: a noisy or biased expert reference would
inflate the apparent error correlation of all three pairs alike, not just one,
so the contrast is robust to an imperfect expert rather than merely independent
of it. (The "same wrong class | both wrong"
column is high for the WorldCover pairs too, ≈ 0.87, because with three classes
two products that both err on a boundary point usually land in the same
built ↔ non-built confusion; the pairs are separated by *Q* and κ, not by this
column.)

**S2c — arbitration value (Table \ref{tab:S2}c).** Does product consensus
predict the expert label better than a single product?

| Condition | n | P(expert = that label) | Note |
|---|---|---|---|
| Baseline P(expert = DW) | 900 | 0.45 | — |
| Baseline P(expert = ESRI) | 900 | 0.42 | — |
| DW = ESRI → | 787 | 0.44 [0.40, 0.47] | 87% of points |
| DW = ESRI = WC → | 391 | 0.43 [0.38, 0.47] | adding WC does not help |
| WC = DW → | 436 | 0.45 [0.40, 0.50] | — |

On the 787 boundary points (87% of the subset) where DW = ESRI, the expert
agrees with that shared label 0.44 of the time — indistinguishable from the
0.45 base rate of simply following DW. Requiring WorldCover to join the
consensus does not raise it. Product consensus carries essentially no
additional information about the expert label; the nominal "+0.18 lift" over
the class prior is just the 0.25 prior probability of the agreed-on class.

**S3 — confound checks.** *(a) Ontology (Supplementary Table S3).* Collapsing the
three classes to built-vs-rest, or to built-vs-vegetation with water points
dropped, leaves DW = ESRI at 0.90–0.92 and the DW = ESRI − Human = DW gap at
0.34–0.48; the coupling is not an artefact of the three-class remap.
*(b) Shared imagery.* WorldCover ingests the same
Sentinel-2 input as DW and ESRI but is architecturally distinct — a
gradient-boosted-tree classifier against their deep-segmentation models; its
excess agreement is about half that of
DW = ESRI and its error *Q* stays an order of magnitude smaller (≤ 0.14 vs
0.97), so shared imagery alone does not generate the coupling. *(c) Acquisition date (Supplementary Table S4).* The DW = ESRI rate is
high in every scene type (0.83–0.98) and is *highest* in the hardest
transition zones — urban green space ≈ 0.98, mixed / complex ≈ 0.96 — rather than
concentrated in the spectrally stable classes, which is the reverse of the
pattern a shared single-scene acquisition would produce.

Taken together: the independence assumption behind multi-reference arbitration
fails on urban boundary pixels. DW and ESRI do not supply a second and third
independent opinion; their errors move together (Yule's *Q* ≈ 0.97) while
WorldCover's error is at most weakly coupled to either (*Q* ≤ 0.14), so their
agreement is correlated error, and it adds nothing to a single product's ability
to predict the expert label. Ontology, shared Sentinel-2 input and acquisition
timing are ruled out as cheap explanations.
*Why* the two are coupled — a shared deep-segmentation model family is the least
speculative reading, with a shared training-label lineage a further, less
testable possibility — is taken up in Section 5.2; the statistical result here
does not depend on it.

### 4.6. LOCO correction operator (supporting analysis)

The operator is trained on five cities' boundary points and applied to the
held-out sixth, which contributes no labels of its own (Section 3.3).

**Phase 1 — operator transfer (Table \ref{tab:T5}).**

| Held-out city | WC ↔ expert | Majority-class baseline | Operator ↔ expert | Confident firings (τ = 0.85) | Firing precision vs expert |
|---|---|---|---|---|---|
| Wuhan | 0.41 | 0.53 | 0.71 | 18 | 0.78 |
| Hefei | 0.51 | 0.63 | 0.67 | 28 | 0.68 |
| Nanchang | 0.46 | 0.65 | 0.73 | 11 | 0.82 |
| Nanjing | 0.32 | 0.55 | 0.74 | 15 | 0.93 |
| Changsha | 0.51 | 0.55 | 0.75 | 18 | 0.83 |
| Hangzhou | 0.59 | 0.53 | 0.75 | 11 | 0.91 |

*Table \ref{tab:T5}. LOCO operator transfer (27-feature model). Majority-class
baseline always predicts non-built, learned from the other five cities
(leak-proof, same pattern as the operator).*

The transferred operator predicts the held-out city's expert labels at 0.67 to
0.75, against a "do nothing" WC-versus-expert baseline of 0.32 to 0.59 and a
majority-class (always predict non-built) baseline of 0.53 to 0.65 — a gain
over both baselines in every city, though the margin over the majority-class
baseline narrows to 0.05 in Hefei. Confident firings (p<sub>max</sub> ≥ 0.85 and predicted class
≠ WC) number from 11 in Nanchang and Hangzhou to 28 in Hefei; their precision
against the expert ranges from 0.68 (Hefei) to 0.93 (Nanjing), with Nanchang at
0.82. Nanjing shows the largest transfer
gain (0.32 → 0.74) and high firing precision (14/15); it is at once the
city with the lowest raw WC-versus-expert agreement and the one with the most
headroom for the operator to recover — both stated as diagnostic facts, not as
a corrected failure case.

**Phase 2 — downstream map correction.** Feeding the corrected labels into the
fixed land-cover random forest (Section 3.3) moves the six-city mean overall
accuracy against the 150 expert points from 0.375 (raw WC) to only 0.389
(proposed gated operator), against 0.422 for the within-city oracle ceiling and
0.378 for the random-flip control — a +1.5-point shift, inside the between-seed
standard deviation (0.07). A gate-sensitivity sweep (Supplementary Table S1)
shows why: a materially larger downstream gain is available only by loosening
the confidence threshold and rewriting far more labels, at a real cost to
WC accuracy, while at the safe threshold used here the effect stays within
noise. A few dozen boundary-label corrections do not move a random forest
trained on ≈ 10,000 points — the downstream classifier is the wrong instrument
for this signal, which is why Phase 3 evaluates the label map directly.

**Phase 3 — multi-reference referee (Table \ref{tab:phase3}).** The corrected
label map is scored directly on the 150 held-out points of each city, with no
downstream classifier, against the expert label and against the strict
tri-consensus subset (expert = DW = ESRI).

| Label map | OA vs expert | OA vs tri-consensus | BE vs expert | Changes → expert | Changes → DW = ESRI | Changes / city |
|---|---|---|---|---|---|---|
| WC raw | 0.47 | 0.48 | 58 | — | — | 0 |
| LOCO corrected (τ = 0.85) | **0.54** | 0.51 | 49 | **0.83** | 0.30 | 17 |
| Random-flip control | 0.46 | 0.47 | 53 | 0.33 | 0.20 | 17 |
| Within-city oracle (ceiling) | 0.81 | 0.78 | 16 | 1.00 | 0.36 | 51 |

*Table \ref{tab:phase3}. Label-map referee, six-city mean.*

Of the ≈ 17 points per city the operator changes, 83% move toward the expert
label, against 33% for the same number of random flips — the signal is real,
not noise. In aggregate the effect is nonetheless modest: +7 points of OA
against the expert, ≈ 3 points against the tri-consensus subset, and BE against
the expert falls from 58 to 49. That recovers about one-fifth of the
within-city oracle's OA gain (≈ +7 points out of ≈ +34, on a
150-point base). Per city the pattern tracks Phase 1: Nanjing 0.32 → 0.41
(14 of 15 changes toward the expert), Changsha 0.51 → 0.59, and Hangzhou
0.59 → 0.65, where 11 points change — matching its 11 confident firings.

Figure \ref{fig:map_nanjing} and Figure \ref{fig:map_changsha} show this
pattern spatially for two cities chosen to bracket the range of operator
performance: Nanjing, which has the lowest raw WC-versus-expert agreement of
the six and the largest transfer gain, and Changsha, a mid-range city with
high firing precision. (A) the raw WorldCover class map over the 15,000 city
points; (B) the LOCO-corrected map with the operator's changed points
circled; (C) the 150 expert boundary points coloured by expert = WC /
expert ≠ WC. The remaining four cities are in Supplementary Figures S1–S4
(per-city counts in Supplementary Table S6). In every city the operator's
changes and the expert ≠ WC points cluster along the built-up edge and the
river and water margins rather than scattering across the map — qualitative
evidence that BAMS sampling placed the labelling budget on genuine boundary
zones (Section 5.5), though it does not by itself establish that the
disagreement rate there exceeds what random sampling would give.

---

## 5. Discussion

> Drafted 2026-09-09. Register is deliberately modest: a bounded regional
> characterisation, not a claim about global product behaviour. Citations are
> `(Author, year)` placeholders; the bucket-B "agreement-as-label" citations in
> 5.1 await an authors' decision (see *References to insert (Section 5)*).

### 5.1. Multi-reference agreement does not arbitrate the urban boundary

The three global products examined here reproduce the expert boundary label on
fewer than half of the candidate points (Human = WC 0.47, Human = DW 0.45,
Human = ESRI 0.42), and — the central result — consulting a second and a third
product does not recover the shortfall. All four sources coincide on only 18%
of boundary candidates; outside the 38% covered by the largest self-consistent
subset (expert, Dynamic World and ESRI), there is no reference majority to
appeal to. Where Dynamic World and ESRI do agree with each other (0.87 of
points), that agreement is not independent corroboration: their error
indicators move almost in lockstep (Yule's *Q* = 0.97, against ≤ 0.14 for either
WorldCover pair), they exceed the agreement expected under conditional
independence by twice the margin of the WorldCover pairs (+0.26 against
+0.13–0.14), and their consensus predicts the expert label no better than a
single product does (0.44 against a 0.45 base rate). The practical implication
is narrow but concrete: at the urban built-up/vegetation boundary, treating
cross-product agreement as a high-confidence signal — the C4 practice of
building consensus products (Tuanmu and Jetz, 2014) or generating training and
reference samples from the pixels where existing maps agree (Zhang and Roy,
2017; Wang et al., 2024a) — selects a subset no closer to expert judgement than
any single product, and discards disagreements that are themselves informative
(Section 5.3). The safeguards built into these pipelines are spectral or
temporal (spectral-outlier rejection, multi-date consistency) and are not
designed to catch error that is correlated across the products being combined.

The scope of this claim is stated carefully. It is an existence result within a
bounded domain: in six middle- and lower-Yangtze cities, on Sentinel-2-era 10 m products,
at deliberately hard pixels, the assumption that product agreement implies a
reliable label fails. We do not claim the failure is pervasive across regions,
sensors or land-cover regimes, and the design — six cities, one sensor — is not
a prevalence estimate and should not be read as one. But the assumption it
tests is normally stated without qualification, and a single well-characterised
counterexample is enough to show it cannot be applied unconditionally.

### 5.2. Why Dynamic World and ESRI err together: a hypothesis

The coupling between Dynamic World and ESRI is strong, systematic and — from
the public record — plausibly structural rather than incidental to this study
area. We list candidate explanations, none of them established, because deciding
between them needs product-internal information that is not published; only the
first can be argued from published documentation.

- *Model family.* WorldCover is a gradient-boosted-tree classifier with a
  rule-based post-process (Zanaga et al., 2022); Dynamic World and ESRI are
  both deep semantic-segmentation models on Sentinel-2 (Brown et al., 2022;
  Karra et al., 2021). Two members of one modelling family failing on the same
  pixels in the same direction is the least speculative reading, and it fits
  WorldCover — the architectural odd-one-out — whose errors are at most weakly
  coupled to either (*Q* ≤ 0.14). A further, less testable possibility is that
  the two share overlapping training-label provenance — deep land-cover models
  are trained on large human-annotated point sets whose lineage may overlap
  across products — but the training corpora are not fully public and we
  cannot document the overlap.
- *Shared inputs and preprocessing.* All three ingest Sentinel-2 surface
  reflectance; Dynamic World and ESRI additionally share an annual-composite
  Sentinel-2 basis. The negative control argues against this being sufficient
  on its own — WorldCover has the same Sentinel-2 input but does not join the
  coupling — yet shared inputs may still contribute.

What can be said without product internals is the statistical fact: the two
deep-segmentation products are not two independent opinions. Establishing which
mechanism dominates requires access to the training data and model internals
and is left to future work.

### 5.3. Where the disagreement falls: the built–vegetation edge and intra-urban green space

The WorldCover error at the boundary is not symmetric noise. Once WorldCover
assigns "built-up" to a boundary candidate, the expert reads non-built roughly
seven times in ten (P(expert = non-built | WC = built) = 0.71), a rate between
0.60 and 0.79 in every one of the six cities. The disagreement is largest
exactly where built-up meets vegetation: intra-urban green space (67.3%) and the
built-up edge (66.1%) are the two highest groups, well above spectrally more
separable settings such as bare land or construction ground (43.2%). The
urban-green-space result echoes a documented tendency of global products to
absorb parks, street trees and residential greenery into the built class
(Yuan et al., 2026; Xu et al., 2024), and it has a definitional component:
Dynamic World's schema folds urban vegetation into "Built Area", whereas
WorldCover's "Built-up" nominally excludes it (Venter et al., 2022) — yet
WorldCover still misses this greenery at the rate observed here. For downstream
users the consequence is directional: an agreement-filtered label set at the
urban fringe will systematically under-represent vegetated cover, in the same
direction across all six cities.

### 5.4. What the correction operator shows

The correction operator answers a question the diagnostic raises: is the
disagreement structured enough to be corrected automatically, without new
reference data in the target city? The answer is a qualified yes: transferred
to a held-out city that contributes no labels of its own, the operator beats
both the do-nothing and majority-class baselines in every city (Section 4.6)
and, as a direct label-map referee, moves 83% of the points it changes toward
the expert against 33% for random flips. The effect is nonetheless modest — it
recovers about one-fifth of the accuracy gap a within-city oracle closes, and
is within noise for a downstream classifier at safe operating thresholds. This
is a diagnostic result in its own right — the disagreement carries a
transferable signal, but its downstream footprint is bounded — not a
general-purpose correction method. A few dozen boundary corrections are the
right scale for auditing a label map, not for retraining one.

### 5.5. Scope and limitations

Several features of the design bound what can be concluded.

*Shared Sentinel-2 basis.* Boundary candidates are selected from Sentinel-2
spectra and Sentinel-2 local heterogeneity, and the expert reference is read
chiefly from a 2021 Sentinel-2 composite, so selection and reference share an
information base; one may question whether Section 4 measures product error or
Sentinel-2 legibility. Four considerations bound this: the
interpreter integrates spatial configuration, texture, context and a
false-colour infrared view rather than acting as a per-pixel spectral
classifier, and falls back to a sub-metre basemap; three independent
interpreters converge on the same labels (Fleiss' κ = 0.89); the disagreement
concentrates in high-confidence labels and interpretable physical scenes, which
is not the shape spectral noise would take; and the sharpest result — the
Dynamic World/ESRI error coupling — is a differential between product pairs
(*Q* ≈ 0.97 against ≤ 0.14 for the WorldCover pairs) that is robust to an
imperfect expert reference, which would raise all pairs alike. An
independent blind re-classification of 30 high-confidence expert ≠ WorldCover
points reproduced the expert label on 29 and the WorldCover label on none
(Appendix \ref{app:recheck}), which further argues the disagreements are not an
artefact of the original annotator.

*Single-annotator reference.* The reference is one expert's interpretation,
supported but not replaced by the blind re-labelling exercises. Where a result
depends on the expert being correct — the confidence stratification, the
error-direction analysis, the arbitration PPV — its strength is capped by that
reliability. The independence contrast (Section 4.5) is the exception: it rests
only on the relationship among the three products' errors.

*Confidence-stratum bias.* The rise in the expert ≠ WorldCover rate on
high-confidence labels (0.53 to 0.67) is subject to a known upward bias — an
annotator may reserve "high" precisely when a product error is visually
obvious — so that figure is better read as an upper bound than an unbiased
estimate. It still refutes the "noise on ambiguous pixels" reading, and the
magnitude in Nanchang (42/42) and Wuhan (0.88) is larger than a
labelling-confidence artefact alone would produce.

*Scene grouping.* The physical-scene rates locate the disagreement; they are
not a formal stratified sample, and the scene notes are free-text, mapped post
hoc to a common code table, so scene-type contrasts are drawn only in the
pooled set.

*Geographic and sensor scope.* All six cities lie in the middle and lower
reaches of the Yangtze River in China — one country, one climate (subtropical
monsoon), one broad urban-build morphology — and the analysis uses a single
sensor. No result should be read as a statement about global product behaviour;
each is framed as holding "in these six cities". This is a bounded
characterisation, not a prevalence estimate, and the region is not offered as
an easy or representative case: the middle and lower Yangtze, with its paddy,
aquaculture and wetland mosaics, is
among the more difficult land-cover settings. Two points nonetheless travel
beyond the specific numbers. First, the six cities are internally
heterogeneous — per-city expert = WorldCover agreement ranges from 0.32
(Nanjing) to 0.53 (Hangzhou), and the high-confidence disagreement reaches
42/42 in Nanchang — yet the direction of the effect is the same in all six, so
it is not a single-city artefact. Second, the Dynamic World/ESRI coupling
follows from those models being trained once and applied globally; the specific
coupling strength here (0.87, *Q* = 0.97) will not carry over unchanged, but
the qualitative mechanism plausibly does. Replicating the boundary-aware
sampling and the four-way diagnostic in other climates and urban morphologies
is the natural next step.

---

## 6. Conclusions

Global 10 m land-cover products are increasingly consulted in combination, on
the assumption that where they agree the agreed label can be trusted — and, by
extension, that a second or third product can arbitrate a disputed pixel. This
study tested that assumption directly at the urban built-up/vegetation boundary,
using 900 expert-interpreted boundary candidates across six cities of the middle
and lower Yangtze.

Within that domain the assumption does not hold. The only agreement that is high
across all six cities is between Dynamic World and Esri (0.87), yet each of the
three products matches the expert on fewer than half of the boundary
candidates, and the Dynamic World–Esri agreement is not independent
corroboration: the two products' errors move almost in lockstep (Yule's
*Q* ≈ 0.97), whereas WorldCover's errors are at most weakly coupled to either
(*Q* ≤ 0.14). Product consensus predicts the expert label no better than a
single product (0.44 against a 0.45 base rate), and all four sources agree on
only 18% of points, so outside a small self-consistent core there is no
reference majority to appeal
to. The WorldCover error is directional rather than random: once WorldCover
calls "built-up" the expert reads non-built about seven times in ten, so an
agreement-filtered label set at the urban fringe systematically
under-represents vegetated cover, in the same direction in every city. A
leave-one-city-out correction operator shows the disagreement is structured
enough to carry a signal that transfers to a city contributing no labels of its
own, but with a bounded downstream footprint — the signal is real, not a
general-purpose correction.

These results are an existence characterisation for six cities and one sensor,
not an estimate of how often the behaviour occurs elsewhere; whether the
specific coupling strengths carry over is open, although the mechanism behind
the Dynamic World–Esri coupling — two models trained once and applied globally —
plausibly does. The practical implication is narrow and concrete: at the urban
built-up/vegetation boundary, cross-product agreement should not be used on its
own as a pseudo-reference, a high-confidence training label, or a quality
filter, because the spectral and temporal safeguards in current weak-supervision
pipelines are not built to catch error that is correlated across the products
being combined. Replicating the boundary-aware sampling and the multi-reference
diagnostic in other climates and urban morphologies is the natural next step.

---

## Appendix A. Blind re-check of high-confidence disagreements {#app:recheck}

> Drafted 2026-09-09 from `data/interannotator_recheck/top30_disagreement_recheck.csv`
> and `scripts/gee_top30_recheck.js`. Separate from, and additional to, the
> 120-point three-colleague reliability exercise in Section 3.1.

**Purpose.** To check whether the high-confidence expert ≠ WorldCover
disagreements of Section 4.2 could be an artefact of the original single
annotator, rather than WorldCover error.

**Point selection.** From the multi-reference table
(`all_cities_BAMS150_second_ref.csv`), points with a high-confidence expert
label and `expert ≠ WorldCover` were retained. Thirty were drawn for the check
— five per city, in decreasing BAMS classifier margin (5 × 6 cities). The
subset numbers 281 points under the final reference labelling (293 when the
check was run, before the Hangzhou re-labelling); all 30 checked points remain
high-confidence `expert ≠ WorldCover` points.

**Protocol.** A second analyst — with remote-sensing land-cover interpretation
experience, and who had not taken part in the original reference labelling —
re-classified all 30 points into {built-up, non-built, water} from sub-metre
Google satellite imagery together with the 2021 Sentinel-2 composite. The
stepper script was modified to hide the `expert` and `wc` fields, so the
analyst was blind to both the original expert label and the WorldCover label;
the returned classes were cross-tabulated against the original expert label
afterwards.

**Result.** The second analyst's blind classification matched the original
expert label on **29 of the 30 points** and the WorldCover label on **none**.
The single unresolved point (Changsha, `Original_ID` 10031; expert = non-built,
WorldCover = water) is an edge/mixed pixel the analyst could not assign a
confident class to from the available imagery. Of the 30 points, 20 are cases
where the expert reads non-built (WorldCover called 19 of them built-up and one
water), 6 where the expert reads water (WorldCover called five non-built and
one built-up), and 4 where the expert reads built-up (WorldCover called all
four non-built) — the same directional pattern as the pooled WorldCover →
expert confusion matrix of Section 4.3.

**What it does and does not establish.** It removes the same-analyst
circularity in attributing the high-confidence disagreements to WorldCover. It
is not a fully independent measurement: the original labelling protocol already
consulted the same Google high-resolution basemap as ancillary context, so the
second analyst is an independent *judgement* sharing the same imagery sources.
Fully cross-independent annotation at scale is the separate reliability
exercise reported in Section 3.1.

---

### References to insert (Section 1)

- **Wang et al., 2024a** = Y. Wang, Y. Sun, et al., "Automatic training sample
  collection utilizing multi-source land cover products and time-series
  Sentinel-2 images," *GIScience & Remote Sensing* 61(1), 2352957. DOI
  10.1080/15481603.2024.2352957. (weighted-majority-vote fusion → training
  samples; used in §1 para 2, §1.1 "labels from existing maps", §5.1)
- **Wang et al., 2024b** = Y. Wang, Y. Xu, X. Xu, et al., "Evaluation of six
  global high-resolution land cover products over China," *Int. J. Digital
  Earth* 17(1), 2301673. DOI 10.1080/17538947.2023.2301673. ("consistency
  lower than accuracy"; used in §1.1 "product inter-comparison")
- Tuanmu and Jetz, 2014 — *Global Ecology and Biogeography* 23(9), 1031–1045.
  DOI 10.1111/geb.12182. EarthEnv consensus land cover.
- Zhang and Roy, 2017 — *Remote Sensing of Environment* 197, 15–34. MODIS
  500 m land cover as a 30 m training-label source.
- Tong et al., 2025 — *ISPRS J. Photogramm. Remote Sens.* 220, 535–549. DOI
  10.1016/j.isprsjprs.2024.12.017 (arXiv:2406.00891). Weak supervision from
  existing maps; method "PRE". (also cited in Section 3.3)
- Venter et al., 2022 — *Remote Sensing* 14(16), 4101. DOI 10.3390/rs14164101.
- Xu et al., 2024 — *Remote Sensing of Environment* 311, 114316. DOI
  10.1016/j.rse.2024.114316.
- Chakraborty et al., 2024 — *Nature Communications* 15, 9165. DOI
  10.1038/s41467-024-52241-5.
- Yuan et al., 2026 — *Urban Forestry & Urban Greening*, 129299. DOI
  10.1016/j.ufug.2026.129299.
- Zanaga et al., 2022; Brown et al., 2022; Karra et al., 2021 — the three
  products (also in Section 3).
- Herold et al., 2008 (*RSE* 112, 2538–2556, DOI 10.1016/j.rse.2007.11.013) —
  optional, prior 1 km agreement/accuracy assessment, if a longer-history
  sentence is wanted.

---

### References to insert (Section 5)

Framing decision (2026-09-09): Option D — general framing in the abstract and
introduction, specific "agreement-as-label" citations here and in Related Work.

- 5.1 "agreement-as-label" practice — **Tuanmu and Jetz, 2014** (*Global Ecology
  and Biogeography* 23(9), EarthEnv consensus land cover; accuracy-weighted
  merge of multiple products); **Zhang and Roy, 2017** (*RSE* 197, MODIS 500 m
  land cover as a training-label source for a 30 m Landsat classification);
  **Wang et al., 2024** (*GIScience & Remote Sensing* 61(1),
  weighted-majority-vote fusion of multiple 10 m products → training samples,
  with a spectral-outlier filter). Cui et al., 2023 (*Ecological Indicators*
  154, single-product temporal self-consistency) dropped from the main list —
  off-target unless the framing is widened to "map self-consistency across
  products *or* time".
- Yuan et al., 2026 — systematic underestimation of urban green space by global
  land-cover products (*Urban Forestry & Urban Greening*), lead cite for 5.3.
- Xu et al., 2024 — comparative validation of 10 m global land-cover maps
  (*RSE* 311, 114316), secondary for 5.3.
- Venter et al., 2022 — Dynamic World / WorldCover / ESRI comparison
  (*Remote Sensing* 14(16), 4101); Built-Area vs Built-up definitional
  contrast, 5.3.
- Brown et al., 2022; Karra et al., 2021; Zanaga et al., 2022 — the three
  products (also cited in Section 3), for the model-family contrast in 5.2.

---

### References to insert (Section 4)

- Author, year — global land-cover products absorbing intra-urban vegetation
  into the built class (for Section 4.4 / 4.5 urban-green-space finding).
- Yule, 1900 — Yule's *Q* (also cited in Section 3).
- Wilson, 1927 — binomial confidence interval (also cited in Section 3).
- Landis and Koch, 1977 — κ interpretation benchmarks (also cited in
  Section 3).

---

### References to insert (Section 3)

- FAO, 2015 — Global Administrative Unit Layers (GAUL) 2015, level 2. *(exact
  citation form to be supplied by the authors.)*
- Zanaga et al., 2022 — ESA WorldCover 10 m 2021 v200 (product / ATBD).
- Brown et al., 2022 — Dynamic World, near-real-time global land cover
  (*Scientific Data*).
- Karra et al., 2021 — ESRI / Impact Observatory 10 m global land cover
  (IGARSS).
- Tong et al., 2024/2025 — PRE: prototype-based pseudo-label rectification and
  expansion (arXiv:2406.00891 / *ISPRS J. Photogramm. Remote Sens.*).
- Wilson, 1927 — binomial confidence interval.
- Landis and Koch, 1977 — κ interpretation benchmarks.
- Yule, 1900 — association of attributes (Yule's Q), if Q stays in the main
  text.
