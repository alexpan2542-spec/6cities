#!/usr/bin/env python3
"""
Robustness check for Phase 2's Human-OA evaluation set (loco_correction_operator.py).

Phase 2 trains its downstream classifier on non-boundary WorldCover points
from a spatial-block train split, and evaluates WC-OA on the matching
spatial-block test split -- but evaluates Human-OA on ALL 150 expert
boundary points regardless of which spatial block they geographically fall
into. The 150 points are always excluded from training (is_bams points are
never in tr_idx), but some of them may sit inside the same spatial cells as
training points, so Human-OA is not spatially isolated the same way WC-OA is
(flagged 2026-09-15 review of loco_correction_operator.py:270-285 vs.
manuscript text describing "disjoint spatial blocks").

This script re-runs the B0/B2 downstream classifiers exactly as Phase 2 does,
but additionally scores Human-OA restricted to only the expert points that
fall in the test block for that (city, seed), to check whether the B2-over-B0
gain is an artefact of the non-isolated evaluation set.

Not part of the main pipeline (does not modify loco_correction_operator.py or
its outputs); run standalone with the `gee` conda env:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 \
      scripts/loco_human_oa_testblock_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import loco_correction_operator as L  # noqa: E402

OUT_DIR = ROOT / "data" / "analysis_outputs" / "loco_correction_operator"


def main():
    city_df = {c: L.load_city(c) for c in L.CITIES}
    feats = L.FEATURE_SETS[L.PHASE2_FEATURE_SET]

    rows = []
    for held in L.CITIES:
        op = L.fit_operator(city_df, held, feats, L.CITIES)
        for seed in L.PARTITION_SEEDS:
            df = city_df[held]
            X = df[feats].to_numpy(float)
            y_wc = df["wc"].to_numpy(int)
            margin = df["margin"].to_numpy(float)
            is_bams = df["is_bams"].to_numpy(bool)
            human = df["human"].to_numpy()
            tr_mask, te_mask = L.spatial_block_split(
                df["lat"].to_numpy(), df["lon"].to_numpy(), seed)
            tr_idx = np.flatnonzero(tr_mask & ~is_bams)
            bams_idx_full = np.flatnonzero(is_bams)
            in_test_block = te_mask[bams_idx_full]

            op_proba = op.predict_proba(X[tr_idx])
            op_pred = op.classes_[op_proba.argmax(1)]
            op_pmax = op_proba.max(1)
            gated = margin[tr_idx] <= np.quantile(margin[tr_idx], L.BOUNDARY_GATE_Q)
            y_wc_tr = y_wc[tr_idx]

            def corrected(apply_mask):
                y = y_wc_tr.copy()
                fire = apply_mask & (op_pmax >= L.CORR_CONF_TAU) & (op_pred != y)
                y[fire] = op_pred[fire]
                return y

            for method, y_tr in [("B0_baseline", y_wc_tr.copy()),
                                  ("B2_loco_gated", corrected(gated))]:
                clf = RandomForestClassifier(**{**L.RF_PARAMS, "random_state": seed})
                clf.fit(X[tr_idx], y_tr)
                pred_bams = clf.predict(X[bams_idx_full])
                y_true_bams = human[bams_idx_full].astype(int)

                oa_full = float((pred_bams == y_true_bams).mean())
                oa_test = (float((pred_bams[in_test_block] == y_true_bams[in_test_block]).mean())
                           if in_test_block.any() else np.nan)
                rows.append({
                    "held_out": held, "seed": seed, "method": method,
                    "Human_OA_full150": oa_full, "n_full150": len(bams_idx_full),
                    "Human_OA_testblock_only": oa_test, "n_testblock": int(in_test_block.sum()),
                })
        print(f"{held} done")

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT_DIR / "phase2_human_oa_testblock_check.csv", index=False)

    b0 = raw[raw.method == "B0_baseline"].set_index(["held_out", "seed"])
    b2 = raw[raw.method == "B2_loco_gated"].set_index(["held_out", "seed"])
    gain_full = b2["Human_OA_full150"] - b0["Human_OA_full150"]
    gain_test = b2["Human_OA_testblock_only"] - b0["Human_OA_testblock_only"]

    summary = pd.DataFrame([
        {"quantity": "Human_OA B0 (mean)", "full150": b0["Human_OA_full150"].mean(),
         "testblock_only": b0["Human_OA_testblock_only"].mean()},
        {"quantity": "Human_OA B2 (mean)", "full150": b2["Human_OA_full150"].mean(),
         "testblock_only": b2["Human_OA_testblock_only"].mean()},
        {"quantity": "B2-B0 paired gain (mean)", "full150": gain_full.mean(),
         "testblock_only": gain_test.mean()},
        {"quantity": "B2-B0 paired gain (std)", "full150": gain_full.std(),
         "testblock_only": gain_test.std()},
        {"quantity": "n expert points per city-seed (mean)",
         "full150": raw["n_full150"].mean(), "testblock_only": raw["n_testblock"].mean()},
    ])
    summary.to_csv(OUT_DIR / "phase2_human_oa_testblock_check_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nwritten -> {OUT_DIR / 'phase2_human_oa_testblock_check.csv'}")
    print(f"written -> {OUT_DIR / 'phase2_human_oa_testblock_check_summary.csv'}")


if __name__ == "__main__":
    main()
