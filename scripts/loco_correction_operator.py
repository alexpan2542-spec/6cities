#!/usr/bin/env python3
"""
LOCO Correction Operator (Method A, redesign for MDPI Remote Sensing).

Idea
----
Do NOT propagate seed *labels* in feature space (that was Prototype Expansion,
which leaked and, once the leak was plugged, contributed 0 -- see
docs/Paper_Writing_Plan.md sec 1.1). Instead, learn a *correction operator*
from the hand-labelled boundary points of N-1 cities and transfer it to a
held-out city that contributes NO manual labels of its own.

Leak-proof by construction:
  * The correction operator never sees the target city.
  * The land-cover classifier never sees the target city's human labels
    (its 150 BAMS points are held out purely for evaluation).
  * WC evaluation uses spatial-block train/test partitions (whole lat/lon
    cells assigned to train or test), not point-level random splits.
  * RandomForest is used for BOTH the baseline and every corrected variant,
    so no classifier-architecture effect can masquerade as a method gain.

Phase 1 -- Corrector transfer (standalone)
  For each held-out city: train operator on the other cities' BAMS150
  points, predict the held-out city's 150 boundary points, score against
  Human_Class. Compare with the "do nothing" reference (WC == Human rate).

Phase 2 -- Downstream map correction (TIGHTENED gate)
  For each held-out city x spatial-partition seed:
    B0 Baseline        RF on raw WC labels
    B1 LocalOracle     RF on WC labels + a within-city operator's corrections
                       (reference ceiling; uses within-city human labels)
    B2 LOCO-Corr       RF on WC labels corrected by the transferred operator
                       on boundary-gated + HIGH-confidence-gated points only
                       (NO within-city human labels -- the proposed method)
    B3 LOCO-Corr-ungated   operator applied to every training point (ablation)
    B4 RandomFlip      same #labels flipped to a random other class (control)
  Evaluate WC-OA / WC-BE on the spatial-block test set and Human-OA /
  Human-BE on the 150 held-out BAMS points.

Phase 2b -- gate sensitivity sweep (tau x margin-quantile).

Phase 3 -- multi-reference referee on the 150 held-out boundary points
  Independent references: Dynamic World (DW_C3) and ESRI (ESRI_C3), from
  data/shared_reference/cross_city_second_ref_dw_esri/ (6 cities; Hangzhou added 2026-09-07). Evaluates
  the *corrected label map* directly (no downstream classifier) against:
    - Human_Class
    - the strict tri-consensus subset  Human == DW == ESRI
  plus a correction-direction analysis: of the points the operator changed,
  how many moved toward Human, and toward the DW/ESRI consensus.

Outputs under data/analysis_outputs/loco_correction_operator/.

Run with the `gee` conda env:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 \
      scripts/loco_correction_operator.py
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA2 = ROOT / "data"
CITIES_DIR = DATA2 / "cities"
OUT_DIR = DATA2 / "analysis_outputs" / "loco_correction_operator"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SECOND_REF = DATA2 / "shared_reference" / "cross_city_second_ref_dw_esri" / "all_cities_BAMS150_second_ref.csv"

CITIES = ["Wuhan", "Hefei", "Nanchang", "Nanjing", "Changsha", "Hangzhou"]
# Hangzhou's DW/ESRI were added 2026-09-07 (all_cities_BAMS150_second_ref.csv
# 750 -> 900 rows); Phase 3 is now leave-one-city-out over all six cities.
DWESRI_CITIES = ["Wuhan", "Hefei", "Nanchang", "Nanjing", "Changsha", "Hangzhou"]

BASE_FEATURES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDBI", "MNDWI"]
NB_FEATURES = [
    "B2_mean", "B2_stdDev", "B3_mean", "B3_stdDev", "B4_mean", "B4_stdDev",
    "B8_mean", "B8_stdDev", "B11_mean", "B11_stdDev", "B12_mean", "B12_stdDev",
    "NDVI_mean", "NDVI_stdDev", "NDBI_mean", "NDBI_stdDev", "MNDWI_mean", "MNDWI_stdDev",
]
TRANSFER_FEATURES = [
    "NDVI", "NDBI", "MNDWI",
    "NDVI_mean", "NDVI_stdDev", "NDBI_mean", "NDBI_stdDev", "MNDWI_mean", "MNDWI_stdDev",
    "B2_stdDev", "B3_stdDev", "B4_stdDev", "B8_stdDev", "B11_stdDev", "B12_stdDev",
]
FEATURE_SETS = {"all27": BASE_FEATURES + NB_FEATURES, "transfer15": TRANSFER_FEATURES}
CONF_WEIGHT = {"h": 1.0, "m": 0.6, "l": 0.3}

# Phase 2 knobs -- TIGHTENED
PARTITION_SEEDS = [0, 1, 2, 3, 4]
GRID_BINS = 6
TEST_FRAC = 0.30
RF_PARAMS = dict(n_estimators=300, random_state=0, n_jobs=-1)
BOUNDARY_GATE_Q = 0.10   # was 0.20 -- correct only the lowest-margin 10%
CORR_CONF_TAU = 0.85     # was 0.60 -- operator must be highly confident
PHASE2_FEATURE_SET = "all27"

# Phase 2b sweep
SWEEP_TAU = [0.70, 0.80, 0.85, 0.90]
SWEEP_Q = [0.05, 0.10, 0.20]

# Phase 3
PHASE3_TAU = 0.85


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _norm_conf(x: object) -> str:
    s = str(x).strip().lower()
    return s if s in CONF_WEIGHT else "m"


def load_second_ref() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(SECOND_REF)
    df["Original_ID"] = df["Original_ID"].astype(int)
    for col in ("DW_C3", "ESRI_C3", "WC_C3"):
        df[col] = df[col].round().astype(int)
    out = {}
    for c in DWESRI_CITIES:
        sub = df[df["City"] == c][["Original_ID", "DW_C3", "ESRI_C3"]].copy()
        sub["dw"] = sub["DW_C3"] - 1
        sub["esri"] = sub["ESRI_C3"] - 1
        out[c] = sub.set_index("Original_ID")[["dw", "esri"]]
    return out


def load_city(city: str) -> pd.DataFrame:
    croot = CITIES_DIR / city
    feat3 = pd.read_csv(croot / "06_3x3" / f"{city}_3x3_Features.csv")
    orig = pd.read_csv(croot / "01_original" / f"{city}_WC_Samples_15000.csv")
    margin = pd.read_csv(croot / "02_margin" / f"{city}_MarginScores.csv")
    manual = pd.read_csv(croot / "04_manual" / f"{city}_BAMS150_Manual.csv")

    for df in (feat3, orig, margin):
        df["Original_ID"] = df["Original_ID"].astype(int)

    base = orig[["Original_ID", "lat", "lon", "Class"] + BASE_FEATURES].drop_duplicates("Original_ID")
    nb = feat3[["Original_ID"] + NB_FEATURES].drop_duplicates("Original_ID")
    mg = margin[["Original_ID", "margin"]].drop_duplicates("Original_ID")

    df = base.merge(nb, on="Original_ID", validate="one_to_one")
    df = df.merge(mg, on="Original_ID", validate="one_to_one")
    df = df.rename(columns={"Class": "wc"})
    df["wc"] = df["wc"].astype(int) - 1
    if not df["wc"].between(0, 2).all():
        raise ValueError(f"{city}: WC class outside 1..3")

    manual["Original_ID"] = manual["Original_ID"].astype(int)
    manual["Human_Class"] = pd.to_numeric(manual["Human_Class"], errors="coerce")
    manual = manual[manual["Human_Class"].notna()].copy()
    manual["human"] = manual["Human_Class"].astype(int) - 1
    if not manual["human"].between(0, 2).all():
        raise ValueError(f"{city}: Human_Class outside 1..3")
    manual["conf"] = manual["Confidence"].map(_norm_conf)
    hmap = manual.set_index("Original_ID")["human"]
    cmap = manual.set_index("Original_ID")["conf"]

    df["is_bams"] = df["Original_ID"].isin(hmap.index)
    df["human"] = df["Original_ID"].map(hmap)
    df["conf"] = df["Original_ID"].map(cmap)
    df["city"] = city
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.nan_to_num(np.diag(cm) / cm.sum(axis=1))
    return {
        "OA": float(accuracy_score(y_true, y_pred)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2])),
        "BE": int(cm[0, 1] + cm[1, 0]),
        "rec_C1": float(rec[0]), "rec_C2": float(rec[1]), "rec_C3": float(rec[2]),
    }


def spatial_block_split(lat, lon, seed, n_bins=GRID_BINS, test_frac=TEST_FRAC):
    lat_b = pd.qcut(lat, q=n_bins, labels=False, duplicates="drop").astype(int)
    lon_b = pd.qcut(lon, q=n_bins, labels=False, duplicates="drop").astype(int)
    cell = lat_b * 1000 + lon_b
    rng = np.random.RandomState(seed)
    counts = pd.Series(cell).value_counts()
    cids = counts.index.to_numpy().copy()
    rng.shuffle(cids)
    target = int(round(test_frac * len(lat)))
    test_cells, run = set(), 0
    for cid in cids:
        if run >= target:
            break
        test_cells.add(int(cid))
        run += int(counts.loc[cid])
    is_test = np.isin(cell, list(test_cells))
    return ~is_test, is_test


# ---------------------------------------------------------------------------
# operator
# ---------------------------------------------------------------------------
def fit_operator(city_df, held, feats, cities):
    tr = pd.concat([city_df[c][city_df[c]["is_bams"]] for c in cities if c != held])
    clf = RandomForestClassifier(**{**RF_PARAMS, "random_state": 0})
    clf.fit(tr[feats].to_numpy(float), tr["human"].to_numpy(int),
            sample_weight=tr["conf"].map(CONF_WEIGHT).to_numpy(float))
    return clf


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------
def phase1(city_df, log) -> pd.DataFrame:
    rows = []
    for fs_name, feats in FEATURE_SETS.items():
        for held in CITIES:
            te = city_df[held][city_df[held]["is_bams"]].copy()
            clf = fit_operator(city_df, held, feats, CITIES)
            proba = clf.predict_proba(te[feats].to_numpy(float))
            pred = clf.classes_[proba.argmax(1)]
            pmax = proba.max(1)
            y_h = te["human"].to_numpy(int)
            y_wc = te["wc"].to_numpy(int)
            hyb = y_wc.copy()
            fire = (pmax >= CORR_CONF_TAU) & (pred != y_wc)
            hyb[fire] = pred[fire]
            bmask = np.isin(y_h, [0, 1]) & np.isin(y_wc, [0, 1])

            # Trivial majority-class baseline: predict the modal human label of
            # the OTHER five cities (same leak-proof pattern as the operator
            # itself) for every held-out point. Added 2026-09-11 per
            # Remote_Sensing_投稿评估.md, to contextualise the 0.67-0.75
            # operator accuracy against class-imbalance alone (non-built is the
            # majority class, 516/900 pooled).
            train_human = pd.concat(
                [city_df[c][city_df[c]["is_bams"]] for c in CITIES if c != held]
            )["human"].to_numpy(int)
            majority_class = int(Counter(train_human).most_common(1)[0][0])
            majority_baseline_acc = float((y_h == majority_class).mean())

            rows.append({
                "feature_set": fs_name, "held_out": held, "n_test": len(te),
                "wc_vs_human_acc": float((y_wc == y_h).mean()),
                "majority_baseline_acc": majority_baseline_acc,
                "majority_baseline_class": majority_class,
                "operator_acc": float((pred == y_h).mean()),
                "hybrid_acc": float((hyb == y_h).mean()),
                "operator_acc_boundary": float((pred[bmask] == y_h[bmask]).mean()) if bmask.any() else np.nan,
                "n_operator_fires": int(fire.sum()),
                "fire_precision_vs_human": float((pred[fire] == y_h[fire]).mean()) if fire.any() else np.nan,
            })
            log(f"[P1|{fs_name}] held={held:9s} WC={rows[-1]['wc_vs_human_acc']:.3f} "
                f"MAJ={rows[-1]['majority_baseline_acc']:.3f} "
                f"OP={rows[-1]['operator_acc']:.3f} HYB={rows[-1]['hybrid_acc']:.3f} "
                f"fires={rows[-1]['n_operator_fires']:3d} fire_prec={rows[-1]['fire_precision_vs_human']:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 2 + 2b
# ---------------------------------------------------------------------------
def _prep_city_seed(city_df, held, seed, feats):
    df = city_df[held]
    feat_all = FEATURE_SETS[PHASE2_FEATURE_SET]
    X = df[feat_all].to_numpy(float)
    Xop = df[feats].to_numpy(float)
    y_wc = df["wc"].to_numpy(int)
    margin = df["margin"].to_numpy(float)
    is_bams = df["is_bams"].to_numpy(bool)
    human = df["human"].to_numpy()
    tr_mask, te_mask = spatial_block_split(df["lat"].to_numpy(), df["lon"].to_numpy(), seed)
    tr_idx = np.flatnonzero(tr_mask & ~is_bams)
    te_idx = np.flatnonzero(te_mask & ~is_bams)
    bams_idx = np.flatnonzero(is_bams)
    return dict(df=df, X=X, Xop=Xop, y_wc=y_wc, margin=margin,
                tr_idx=tr_idx, te_idx=te_idx, bams_idx=bams_idx,
                y_te=y_wc[te_idx], y_bams_true=human[bams_idx].astype(int))


def phase2_city_seed(city_df, held, seed, operator, feats, log,
                     tau=CORR_CONF_TAU, q=BOUNDARY_GATE_Q, full=True):
    P = _prep_city_seed(city_df, held, seed, feats)
    tr_idx = P["tr_idx"]
    op_proba = operator.predict_proba(P["Xop"][tr_idx])
    op_pred = operator.classes_[op_proba.argmax(1)]
    op_pmax = op_proba.max(1)
    m_tr = P["margin"][tr_idx]
    gated = m_tr <= np.quantile(m_tr, q)
    y_wc_tr = P["y_wc"][tr_idx]

    def corrected(apply_mask):
        y = y_wc_tr.copy()
        fire = apply_mask & (op_pmax >= tau) & (op_pred != y)
        y[fire] = op_pred[fire]
        return y, int(fire.sum())

    y_b2, n_b2 = corrected(gated)

    if not full:  # sweep mode: only need B2 Human-OA / n_corrected
        clf = RandomForestClassifier(**{**RF_PARAMS, "random_state": seed})
        clf.fit(P["X"][tr_idx], y_b2)
        m_hu = metrics(P["y_bams_true"], clf.predict(P["X"][P["bams_idx"]]))
        m_wc = metrics(P["y_te"], clf.predict(P["X"][P["te_idx"]]))
        return {"held_out": held, "seed": seed, "tau": tau, "q": q,
                "n_corrected": n_b2, "Human_OA": m_hu["OA"], "Human_BE": m_hu["BE"],
                "WC_OA": m_wc["OA"], "WC_BE": m_wc["BE"]}

    # full mode: all 5 variants
    variants = {"B0_baseline": (y_wc_tr.copy(), 0)}
    own = city_df[held][city_df[held]["is_bams"]]
    loc_op = RandomForestClassifier(**{**RF_PARAMS, "random_state": 0})
    loc_op.fit(own[feats].to_numpy(float), own["human"].to_numpy(int),
               sample_weight=own["conf"].map(CONF_WEIGHT).to_numpy(float))
    lp = loc_op.predict_proba(P["Xop"][tr_idx])
    lpred = loc_op.classes_[lp.argmax(1)]
    lpmax = lp.max(1)
    y_oracle = y_wc_tr.copy()
    lfire = gated & (lpmax >= tau) & (lpred != y_oracle)
    y_oracle[lfire] = lpred[lfire]
    variants["B1_local_oracle"] = (y_oracle, int(lfire.sum()))
    variants["B2_loco_gated"] = (y_b2, n_b2)
    variants["B3_loco_ungated"] = corrected(np.ones_like(gated, bool))
    rng = np.random.RandomState(1000 + seed)
    cand = np.flatnonzero(gated)
    pick = rng.choice(cand, size=min(n_b2, len(cand)), replace=False)
    y_rand = y_wc_tr.copy()
    for i in pick:
        y_rand[i] = rng.choice([c for c in (0, 1, 2) if c != y_rand[i]])
    variants["B4_random_flip"] = (y_rand, len(pick))

    out = []
    for name, (y_tr, n_corr) in variants.items():
        clf = RandomForestClassifier(**{**RF_PARAMS, "random_state": seed})
        clf.fit(P["X"][tr_idx], y_tr)
        m_wc = metrics(P["y_te"], clf.predict(P["X"][P["te_idx"]]))
        m_hu = metrics(P["y_bams_true"], clf.predict(P["X"][P["bams_idx"]]))
        out.append({
            "held_out": held, "seed": seed, "method": name,
            "n_train": len(tr_idx), "n_corrected": n_corr,
            "WC_OA": m_wc["OA"], "WC_Kappa": m_wc["Kappa"], "WC_BE": m_wc["BE"],
            "Human_OA": m_hu["OA"], "Human_BE": m_hu["BE"],
        })
        log(f"[P2|{held:9s}|s{seed}] {name:17s} n_corr={n_corr:4d} "
            f"WC_OA={m_wc['OA']:.4f} WC_BE={m_wc['BE']:3d} "
            f"Hum_OA={m_hu['OA']:.4f} Hum_BE={m_hu['BE']:3d}")
    return out


# ---------------------------------------------------------------------------
# Phase 3 -- multi-reference referee on the 150 held-out boundary points
# ---------------------------------------------------------------------------
def phase3(city_df, second_ref, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = FEATURE_SETS[PHASE2_FEATURE_SET]
    div_rows, eval_rows = [], []
    for held in DWESRI_CITIES:
        te = city_df[held][city_df[held]["is_bams"]].copy()
        sr = second_ref[held]
        te = te[te["Original_ID"].isin(sr.index)].copy()
        dw = sr.loc[te["Original_ID"], "dw"].to_numpy(int)
        es = sr.loc[te["Original_ID"], "esri"].to_numpy(int)
        y_h = te["human"].to_numpy(int)
        y_wc = te["wc"].to_numpy(int)

        # reference divergence (standalone table)
        strict = (y_h == dw) & (dw == es)
        div_rows.append({
            "city": held, "n": len(te),
            "WC=Human": float((y_wc == y_h).mean()),
            "DW=Human": float((dw == y_h).mean()),
            "ESRI=Human": float((es == y_h).mean()),
            "DW=ESRI": float((dw == es).mean()),
            "tri_consensus_frac": float(strict.mean()),
            "WC=tri_consensus": float((y_wc[strict] == y_h[strict]).mean()) if strict.any() else np.nan,
        })

        # operator (trained on the OTHER cities) predicts these points
        op = fit_operator(city_df, held, feats, DWESRI_CITIES)
        proba = op.predict_proba(te[feats].to_numpy(float))
        pred = op.classes_[proba.argmax(1)]
        pmax = proba.max(1)
        fire = (pmax >= PHASE3_TAU) & (pred != y_wc)

        loco_map = y_wc.copy()
        loco_map[fire] = pred[fire]

        # within-city oracle operator (5-fold-ish: just LOO within the 150)
        # -> use a simple within-city RF; corrections where it is confident
        own = city_df[held][city_df[held]["is_bams"]]
        loc = RandomForestClassifier(**{**RF_PARAMS, "random_state": 0})
        loc.fit(own[feats].to_numpy(float), own["human"].to_numpy(int),
                sample_weight=own["conf"].map(CONF_WEIGHT).to_numpy(float))
        # (this is optimistic -- trained incl. these points -- labelled as ceiling)
        lp = loc.predict_proba(te[feats].to_numpy(float))
        lpred = loc.classes_[lp.argmax(1)]
        lfire = (lp.max(1) >= PHASE3_TAU) & (lpred != y_wc)
        oracle_map = y_wc.copy()
        oracle_map[lfire] = lpred[lfire]

        rng = np.random.RandomState(7)
        rand_map = y_wc.copy()
        ridx = rng.choice(np.arange(len(te)), size=int(fire.sum()), replace=False)
        for i in ridx:
            rand_map[i] = rng.choice([c for c in (0, 1, 2) if c != rand_map[i]])

        dwes_consensus = dw == es  # points where the two products agree
        for name, mp, nf in [
            ("WC_raw", y_wc, 0),
            ("LOCO_corrected", loco_map, int(fire.sum())),
            ("local_oracle(ceiling)", oracle_map, int(lfire.sum())),
            ("random_flip", rand_map, len(ridx)),
        ]:
            m_h = metrics(y_h, mp)
            row = {
                "city": held, "method": name, "n_changed": nf,
                "OA_vs_Human": m_h["OA"], "BE_vs_Human": m_h["BE"],
                "OA_vs_tri_consensus": float((mp[strict] == y_h[strict]).mean()) if strict.any() else np.nan,
                "n_tri_consensus": int(strict.sum()),
            }
            if name in ("LOCO_corrected", "local_oracle(ceiling)", "random_flip"):
                fmask = (mp != y_wc)
                if fmask.any():
                    row["changed_toward_Human_frac"] = float((mp[fmask] == y_h[fmask]).mean())
                    dcm = fmask & dwes_consensus
                    row["changed_toward_DWESRI_frac"] = float((mp[dcm] == dw[dcm]).mean()) if dcm.any() else np.nan
            eval_rows.append(row)
        log(f"[P3|{held:9s}] LOCO changed {int(fire.sum()):2d}/{len(te)}  "
            f"OA_vs_Human {metrics(y_h,y_wc)['OA']:.3f}->{metrics(y_h,loco_map)['OA']:.3f}  "
            f"tri-consensus n={int(strict.sum())}")
    return pd.DataFrame(div_rows), pd.DataFrame(eval_rows)


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    logf = (OUT_DIR / "run.log").open("w", encoding="utf-8")

    def log(msg: str):
        line = f"[{time.time()-t0:6.0f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    log("loading cities ...")
    city_df = {c: load_city(c) for c in CITIES}
    second_ref = load_second_ref()
    for c in CITIES:
        d = city_df[c]
        dis = 1 - (d.loc[d["is_bams"], "wc"] == d.loc[d["is_bams"], "human"]).mean()
        log(f"  {c:9s} n={len(d)} n_bams={int(d['is_bams'].sum())} "
            f"conf={dict(Counter(d.loc[d['is_bams'],'conf']))} wc_vs_human_disagree={dis:.3f}")

    log("=" * 72); log("PHASE 1  corrector transfer (leave-one-city-out)"); log("=" * 72)
    p1 = phase1(city_df, log)
    p1.to_csv(OUT_DIR / "phase1_corrector_transfer.csv", index=False)
    (OUT_DIR / "phase1_corrector_transfer.txt").write_text(p1.to_string(index=False) + "\n", "utf-8")

    log("=" * 72); log(f"PHASE 2  downstream (TIGHTENED: tau={CORR_CONF_TAU}, q={BOUNDARY_GATE_Q})"); log("=" * 72)
    feats = FEATURE_SETS[PHASE2_FEATURE_SET]
    rows = []
    for held in CITIES:
        op = fit_operator(city_df, held, feats, CITIES)
        for seed in PARTITION_SEEDS:
            rows.extend(phase2_city_seed(city_df, held, seed, op, feats, log))
    p2 = pd.DataFrame(rows)
    p2.to_csv(OUT_DIR / "phase2_downstream.csv", index=False)
    agg = p2.groupby(["held_out", "method"]).agg(
        WC_OA=("WC_OA", "mean"), WC_BE=("WC_BE", "mean"),
        Human_OA=("Human_OA", "mean"), Human_BE=("Human_BE", "mean"),
        n_corrected=("n_corrected", "mean")).reset_index()
    agg.to_csv(OUT_DIR / "phase2_by_city.csv", index=False)
    overall = p2.groupby("method").agg(
        WC_OA_mean=("WC_OA", "mean"), WC_OA_std=("WC_OA", "std"), WC_BE_mean=("WC_BE", "mean"),
        Human_OA_mean=("Human_OA", "mean"), Human_OA_std=("Human_OA", "std"),
        Human_BE_mean=("Human_BE", "mean"), n_corrected_mean=("n_corrected", "mean")).reindex(
        ["B0_baseline", "B1_local_oracle", "B2_loco_gated", "B3_loco_ungated", "B4_random_flip"])
    overall.to_csv(OUT_DIR / "phase2_summary.csv")

    # NOTE: overall.*_std above pools across both cities and seeds in one
    # std() call, so it is dominated by between-city spread, not seed noise.
    # This decomposition separates the two and adds the paired
    # same-city-same-seed gain, which is what "is the correction within
    # noise" actually needs (flagged by external review, 2026-09-15).
    var_rows = []
    for method in ["B0_baseline", "B2_loco_gated"]:
        sub = p2[p2["method"] == method]
        seed_means = sub.groupby("seed")["Human_OA"].mean()
        var_rows.append({
            "method": method,
            "pooled_city_seed_std": sub["Human_OA"].std(),
            "seed_only_std_of_city_mean": seed_means.std(),
        })
    b0 = p2[p2["method"] == "B0_baseline"].set_index(["held_out", "seed"])["Human_OA"]
    b2 = p2[p2["method"] == "B2_loco_gated"].set_index(["held_out", "seed"])["Human_OA"]
    paired_gain = (b2 - b0).dropna()
    var_rows.append({
        "method": "B2_minus_B0_paired_same_city_seed",
        "pooled_city_seed_std": paired_gain.mean(),
        "seed_only_std_of_city_mean": paired_gain.std(),
    })
    # City-level unit: average the 5 seeds within each city first (seeds
    # reuse overlapping evaluation points within a city, so they are not
    # independent draws -- the 6 cities are the closest thing to an
    # independent unit here). n=6 is small but this is the right level at
    # which to ask "is the gain consistent," not the pooled 30 city-seed
    # rows (external review, 2026-09-15).
    city_gain = paired_gain.groupby(level="held_out").mean()
    from scipy.stats import ttest_1samp
    t_res = ttest_1samp(city_gain.to_numpy(), 0.0)
    var_rows.append({
        "method": "B2_minus_B0_city_level_mean_of_5_seeds",
        "pooled_city_seed_std": city_gain.mean(),
        "seed_only_std_of_city_mean": city_gain.std(),
    })
    var_rows.append({
        "method": (f"city_level_paired_t (n=6, one-sided-positive count="
                   f"{int((city_gain > 0).sum())}/6, t-test p={t_res.pvalue:.3f})"),
        "pooled_city_seed_std": np.nan,
        "seed_only_std_of_city_mean": np.nan,
    })
    pd.DataFrame(var_rows).rename(columns={
        "pooled_city_seed_std": "value_or_mean",
        "seed_only_std_of_city_mean": "seed_std_or_gain_std",
    }).to_csv(OUT_DIR / "phase2_variance_decomposition.csv", index=False)

    log("=" * 72); log("PHASE 2b  gate sensitivity sweep"); log("=" * 72)
    sweep = []
    for held in CITIES:
        op = fit_operator(city_df, held, feats, CITIES)
        for tau in SWEEP_TAU:
            for q in SWEEP_Q:
                for seed in PARTITION_SEEDS:
                    sweep.append(phase2_city_seed(city_df, held, seed, op, feats,
                                                  lambda *_: None, tau=tau, q=q, full=False))
        log(f"  {held} swept")
    sw = pd.DataFrame(sweep)
    sw.to_csv(OUT_DIR / "phase2b_sweep_raw.csv", index=False)
    sw_agg = sw.groupby(["tau", "q"]).agg(
        n_corrected=("n_corrected", "mean"),
        Human_OA=("Human_OA", "mean"), Human_BE=("Human_BE", "mean"),
        WC_OA=("WC_OA", "mean")).reset_index()
    sw_agg.to_csv(OUT_DIR / "phase2b_sweep_summary.csv", index=False)

    log("=" * 72); log("PHASE 3  multi-reference referee (DW + ESRI, 6 cities)"); log("=" * 72)
    div, ev = phase3(city_df, second_ref, log)
    div.to_csv(OUT_DIR / "phase3_reference_divergence.csv", index=False)
    ev.to_csv(OUT_DIR / "phase3_referee_eval.csv", index=False)
    ev_agg = ev.groupby("method").agg(
        OA_vs_Human=("OA_vs_Human", "mean"), BE_vs_Human=("BE_vs_Human", "mean"),
        OA_vs_tri_consensus=("OA_vs_tri_consensus", "mean"),
        changed_toward_Human_frac=("changed_toward_Human_frac", "mean"),
        changed_toward_DWESRI_frac=("changed_toward_DWESRI_frac", "mean"),
        n_changed=("n_changed", "mean")).reindex(
        ["WC_raw", "LOCO_corrected", "local_oracle(ceiling)", "random_flip"])
    ev_agg.to_csv(OUT_DIR / "phase3_referee_summary.csv")

    log("\nPHASE 1\n" + p1.to_string(index=False))
    log("\nPHASE 2 by city\n" + agg.to_string(index=False))
    log("\nPHASE 2 overall\n" + overall.to_string())
    log("\nPHASE 2b sweep\n" + sw_agg.to_string(index=False))
    log("\nPHASE 3 reference divergence\n" + div.to_string(index=False))
    log("\nPHASE 3 referee eval (by city)\n" + ev.to_string(index=False))
    log("\nPHASE 3 referee summary\n" + ev_agg.to_string())
    log(f"\ndone in {time.time()-t0:.0f}s -> {OUT_DIR}")
    logf.close()


if __name__ == "__main__":
    main()
