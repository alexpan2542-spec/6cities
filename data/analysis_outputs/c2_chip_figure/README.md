# C2 / C1 example-chip gallery — Supplementary Figure S5

Illustrates **C2** (WorldCover directional error) and **C1** (DW/Esri coupled
error) with real Sentinel-2 boundary chips. Illustration only — the quantitative
claims stay in Table 3 / §4.6.

## Pipeline (3 steps)

| Step | Script | Needs | Output |
|---|---|---|---|
| 1 select | `scripts/c2_chip_select.py` | pandas only | `chip_manifest.csv`, `points_for_gee.js` |
| 2 fetch  | `scripts/c2_chip_fetch.py --project ee-XXXX` | `gee` env + EE cloud project | `chips/*.png` |
| 2 alt    | `scripts/gee_c2_chip_export.js` | GEE Code Editor | GeoTIFFs → drop in `chips/` |
| 3 build  | `scripts/c2_chip_assemble.py` | matplotlib | `submission/figures/FigureS5_c2_chips.{pdf,png}` |

Step 3 runs any time and draws `chip pending` placeholders for chips not yet in
`chips/`, so the layout is reviewable before the imagery is fetched.

## Selection rule (goes in the caption)

Three groups:
- **A** — C1+C2, *consensus error*: `WC=DW=Esri=built ∧ expert=non-built`,
  2 per city (12). All three products call built-up; the expert reads vegetation.
  In the WC=built∧expert=veg cell this is the common case (159 / 191 = 83 %).
- **B** — C1, *coupled-pair error*: `DW=Esri=built ∧ WC=non-built ∧ expert=non-built`,
  1 per city (6). The pair that agrees most errs together; WorldCover — least
  coupled to either — dissents and is right here.
- **C** — counter-examples: `WC=built ∧ expert=built`, 1 per city (6).

All groups ranked by BAMS selection margin **descending** (the points the
consensus was *most* committed to), ties broken by `Original_ID`. Fully
deterministic. Confidence tiers (`CONF_TIERS`): A and B take `Conf ∈ {high, med}`;
**C takes high-conf only**, falling back to med only where a city has no
high-confidence built–built point (Nanchang). `EXCLUDE_IDS` drops points
rejected on visual review — the next-ranked candidate for that city fills the
slot.

Two review passes are folded into the current `EXCLUDE_IDS` (34 ids):
- **First pass** (2026-09-11) — dropped for being a poor illustration once the
  chip was pulled: 558 (ambiguous scene); 378, 927, 2183, 3557 (built, but not
  an obvious urban core); 4251, 1831 (reads as a built structure —
  greenhouse film / metal roof — not clear vegetation); 3173, 4886
  (weaker/patchier built counter-example); 4271, 1537, 2217, 4053 (Changsha
  group-A slot — 4271 itself swapped out for an unclear chip, 1537/2217 show
  an unambiguous built complex undercutting the "expert reads vegetation"
  narrative, 4053 a weaker vegetation read); 3996, 2961, 442, 4440 (Wuhan
  group-C slot, same pattern).
- **Second pass** (2026-09-12) — user reviewed 6–12 ranked candidates per
  city/group side by side (rendered comparison grids, one PNG per city, cyan
  crosshair marking the exact sample point) and hand-picked the clearest,
  rather than only accepting/rejecting the next-ranked fallback. This
  re-opened 442 (now the Wuhan C pick) and dropped: 1804, 1061 (group A,
  Wuhan/Hefei rank-2 not picked); 2863, 4930, 3382, 1919, 3030, 951 (group A
  Nanchang ranks 1–6, all passed over for ranks 7–8 — margin dropped from
  ~0.53→0.39 but the chips read more clearly as built); 4904, 3900, 1719,
  4317 (group A Nanjing ranks 1,3,4,5); 853, 3273, 3806 (group A Changsha
  ranks 2,3,4 — 853 had been the first-pass pick, replaced again); 2972, 1874
  (group A Hangzhou ranks 2,3); 9555 (group B Wuhan rank 1, replaced by
  rank 2).

Change `N_PER_CITY` / `CONF_TIERS` / `EXCLUDE_IDS` in `c2_chip_select.py` and
regenerate (steps 1→2→3; step 2 skips chips already on disk, so only the
newly-promoted candidates get fetched).

## Imagery

`COPERNICUS/S2_SR_HARMONIZED`, 2021, `CLOUDY_PIXEL_PERCENTAGE < 20`, median —
the same recipe as the sampling pipeline. Each chip: 750 m window, true colour
(B4 B3 B2) with a false-colour (B8 B4 B3) inset so vegetation (red) is
unambiguous.
