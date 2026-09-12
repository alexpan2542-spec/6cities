# gee-project

Remote Sensing (MDPI) submission: a multi-reference diagnostic study of six
middle- and lower-Yangtze cities, examining whether agreement among global
land-cover products (WorldCover, Dynamic World, Esri) can arbitrate the correct
label at the urban built-up/vegetation boundary — it cannot, due to a measurable
error coupling between Dynamic World and Esri — with a LOCO (leave-one-city-out)
correction operator as a supporting result.

- **Authoritative narrative/decision source:** `docs/planning/Paper_Writing_Plan.md`
- **File/data cleanup history:** `docs/planning/CLEANUP_LOG.md`
- **The actual manuscript:** `submission/manuscript.tex`

## Layout

```
scripts/    Live analysis pipeline (22 files: Python + Earth Engine .js).
            Entry points include reproduce_bams_selection.py, diagnostic_analysis.py,
            independence_diagnostic.py, loco_correction_operator.py.
data/       6-city sample data + diagnostic outputs.
              cities/               per-city pipeline stages (01_original … 08_logs)
              shared_reference/     cross-city DW/ESRI + scene-grouping reference data
              analysis_outputs/     outputs of the current scripts/ pipeline, incl.
                                    correction_maps/ (raw per-city before/after maps,
                                    all 6 cities — curated subsets are copied/composited
                                    into submission/figures/)
              nanjing_rootcause/    Nanjing root-cause investigation, cited in
                                    manuscript Discussion §5.3
              interannotator_recheck/  blind re-check of expert labels
              boundaries/           basemap shapefiles/geojson for Figure 1 (Natural Earth + GAUL)
              gee_shp/              AOI shapefiles + their Earth Engine export/inspect scripts
docs/       Process/working files.
  manuscript.md                Full prose master (untrimmed) — source of truth for wording.
  manuscript_supplementary.md  Wording source for the Supplementary Material (S1-S6).
  planning/                    Decision log & single source of truth (see above).
  results_snapshot/            Frozen numbers backing every figure quoted in the manuscript.
  supervisor_reports/          Dated progress reports.
submission/ The final output: single-file manuscript.tex + figures/ — a 1:1
            mirror of the Overleaf project. See submission/README.md.
archive/    Superseded pre-final-methodology work (the abandoned "Prototype
            Expansion" approach and earlier drafts/snapshots). Not read by
            anything live — see archive/README.md.
```

## Reproducing a result

Each script's docstring names its inputs/outputs under `data/`. Run with the
`gee` conda env, e.g.:

```
/opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 scripts/diagnostic_analysis.py
```

Or `pip install -r requirements.txt` into your own environment.

## License

Code and data: MIT (see `LICENSE`). The manuscript text itself is not covered
by this license.

