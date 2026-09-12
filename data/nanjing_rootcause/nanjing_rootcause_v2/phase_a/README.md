# Phase A — Nanjing baseline reproduction

## Goal
Re-run the locked Nanjing Confidence+PE protocol and verify OA within **±0.2 pp** of the passport.

## Passport (do not overwrite)
- Source: `data2/nanjing_relabel_bams_pe/`
- Copied: `passport_locked_overall_summary.csv`, `passport_locked_config.json`

## Protocol
- Script: `scripts/nanjing_relabel_bams_pe.py`
- Seeds: 0 1 2 3 4
- Epochs: 80
- Manual: `data2/Nanjing/04_manual/Nanjing_BAMS150_Manual.csv` (+ `_bk.csv`)
- Eval: pointwise vs WorldCover `Class`

## Acceptance
For key methods (`Baseline`, `PE_new`, `PE_new_high`, `BAMS150_new`):
`|dOA_repro - dOA_passport| ≤ 0.20 pp` (and Baseline OA within ~0.002 absolute).
