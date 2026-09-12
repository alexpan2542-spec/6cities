#!/usr/bin/env python3
"""
Nanjing high-confidence × 3×3 WC neighborhood experiment

Uses:
  BAMS150 Manual (Confidence) + Nanjing_BAMS150_3x3nb_wide.csv

Diagnostics (high only):
  - Human vs WC / vs suggest_C_if_uncertain / vs has_C1_nb8 consistency

PE methods (seeds = high only, then filtered by nb rules):
  PE_high                 : all Confidence=high
  PE_high_nb_agree        : high & Human == suggest_C_if_uncertain
  PE_high_nb_disagree_drop: high & Human != suggest  → drop
  PE_high_c1gate          : Human=1 requires has_C1_nb8=1; else keep
  PE_high_iso2            : Human=1 & has_C1_nb8=0 → relabel seed as Class2

Eval vs original WC Class.
Outputs -> data2/nanjing_high_3x3nb/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".python_pkgs"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA2 = ROOT / "data2"
OUT_DIR = DATA2 / "nanjing_high_3x3nb"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITY = "Nanjing"
FEATURES_3X3 = [
    "B2_mean", "B2_stdDev", "B3_mean", "B3_stdDev", "B4_mean", "B4_stdDev",
    "B8_mean", "B8_stdDev", "B11_mean", "B11_stdDev", "B12_mean", "B12_stdDev",
    "NDVI_mean", "NDVI_stdDev", "NDBI_mean", "NDBI_stdDev",
    "MNDWI_mean", "MNDWI_stdDev",
]
SPLIT = dict(test_size=0.3, random_state=42)
SEEDS = [0, 1, 2, 3, 4]
K_LIST = [5, 10, 20, 30]
MARGIN_THRESH = 0.15
SOFT_WEIGHT = 0.3
REAL_WEIGHT = 1.0
EPOCHS = 80
LR = 1e-3
BATCH_SIZE = 256

METHODS = [
    "Baseline",
    "RF_high",
    "RF_high_nb_agree",
    "PE_high",
    "PE_high_nb_agree",
    "PE_high_c1gate",
    "PE_high_iso2",
]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class WeightedDataset(Dataset):
    def __init__(self, X, y, w):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.w = torch.from_numpy(w.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i]


def norm_conf(x):
    if pd.isna(x) or str(x).strip() == "":
        return None
    s = str(x).strip().lower()
    if s in ("h", "high"):
        return "high"
    if s in ("m", "med", "medium"):
        return "med"
    if s in ("l", "low"):
        return "low"
    return None


def train_mlp(X, y, sample_weight=None, device="cpu", seed: int = 0, epochs: int = EPOCHS):
    set_all_seeds(seed)
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=np.float32)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        WeightedDataset(X, y, sample_weight),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
    )
    model = MLP(X.shape[1], 3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(reduction="none")
    model.train()
    for _ in range(epochs):
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            loss = (crit(model(xb), yb) * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_proba_mlp(model, X, device="cpu"):
    model.eval()
    probs = torch.softmax(
        model(torch.from_numpy(X.astype(np.float32)).to(device)), dim=1
    ).cpu().numpy()
    order = np.argsort(probs, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    conf = probs[np.arange(len(probs)), top1]
    margin = conf - probs[np.arange(len(probs)), top2]
    return top1.astype(np.int64), margin


def majority_label(votes: list[int]) -> int:
    counts = Counter(votes)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def expand_prototypes(feature, pos, lab, k: int):
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn_model.fit(feature)
    neigh_idx = nn_model.kneighbors(feature[pos], return_distance=False)
    votes: dict[int, list[int]] = defaultdict(list)
    for proto_i, neighbors in enumerate(neigh_idx):
        proto_pos = int(pos[proto_i])
        proto_lab = int(lab[proto_i])
        for nidx in neighbors.tolist():
            nidx = int(nidx)
            if nidx == proto_pos:
                continue
            votes[nidx].append(proto_lab)
    indices = np.asarray(sorted(votes.keys()), dtype=int)
    labels = np.asarray([majority_label(votes[int(i)]) for i in indices], dtype=np.int64)
    return indices, labels


def full_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.nan_to_num(np.diag(cm) / cm.sum(axis=1), nan=0.0)
    return {
        "OA": float(accuracy_score(y_true, y_pred)),
        "AA": float(per.mean()),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "Boundary Error": int(cm[0, 1] + cm[1, 0]),
        "C1_to_C2": int(cm[0, 1]),
        "C2_to_C1": int(cm[1, 0]),
    }


def pick_best(cands: list[dict]) -> dict:
    return sorted(cands, key=lambda c: (c["m"]["Boundary Error"], -c["m"]["OA"]))[0]


def build_diagnostics(merged: pd.DataFrame) -> dict:
    h = merged[merged["Conf"] == "high"].copy()
    h["agree_suggest"] = h["Human_Class"] == h["suggest_C_if_uncertain"]
    h["flip_wc"] = h["Human_Class"] != h["Class"]
    h["human1"] = h["Human_Class"] == 1
    h["human2"] = h["Human_Class"] == 2
    # consistency checks
    h1 = h[h["human1"]]
    h2 = h[h["human2"]]
    diag = {
        "n_high": int(len(h)),
        "high_flip_wc": float(h["flip_wc"].mean()),
        "high_agree_suggest": float(h["agree_suggest"].mean()),
        "high_agree_suggest_n": int(h["agree_suggest"].sum()),
        "human1_n": int(len(h1)),
        "human1_has_C1_nb8": float(h1["has_C1_nb8"].mean()) if len(h1) else None,
        "human1_no_C1_nb8_n": int(((h1["has_C1_nb8"] == 0)).sum()) if len(h1) else 0,
        "human2_n": int(len(h2)),
        "human2_has_C1_nb8": float(h2["has_C1_nb8"].mean()) if len(h2) else None,
        "suggest_vs_human_crosstab": pd.crosstab(
            h["Human_Class"], h["suggest_C_if_uncertain"]
        ).to_dict(),
        "seed_counts": {},
    }
    # seed set sizes for methods
    diag["seed_counts"]["PE_high"] = int(len(h))
    diag["seed_counts"]["PE_high_nb_agree"] = int(h["agree_suggest"].sum())
    diag["seed_counts"]["PE_high_c1gate"] = int(
        ((h["Human_Class"] != 1) | (h["has_C1_nb8"] == 1)).sum()
    )
    diag["seed_counts"]["PE_high_iso2"] = int(len(h))  # same n, some labels flipped
    return diag, h


def make_pack(id_to_pos, y_all, oids, labs):
    pos, lab = [], []
    for oid, hc in zip(oids, labs):
        oid = int(oid)
        if oid not in id_to_pos:
            continue
        pos.append(id_to_pos[oid])
        lab.append(int(hc) - 1)
    pos = np.asarray(pos, dtype=int)
    lab = np.asarray(lab, dtype=np.int64)
    y_full = y_all.copy()
    if len(pos):
        y_full[pos] = lab
    return {"pos_seed": pos, "lab_seed": lab, "y_full": y_full, "n_seed": int(len(pos))}


def load_city():
    city_root = DATA2 / CITY
    feat3 = pd.read_csv(city_root / "06_3x3" / f"{CITY}_3x3_Features.csv")
    man = pd.read_csv(city_root / "04_manual" / f"{CITY}_BAMS150_Manual.csv")
    nb = pd.read_csv(city_root / "04_manual" / f"{CITY}_BAMS150_3x3nb_wide.csv")

    man["Human_Class"] = pd.to_numeric(man["Human_Class"], errors="coerce")
    man = man[man["Human_Class"].notna()].copy()
    man["Human_Class"] = man["Human_Class"].astype(int)
    man["Original_ID"] = man["Original_ID"].astype(int)
    man["Conf"] = man["Confidence"].map(norm_conf)

    nb_cols = [
        "Original_ID",
        "Class",
        "has_C1_nb8",
        "n_C1_nb8",
        "n_C2_nb8",
        "n_C3_nb8",
        "suggest_C_if_uncertain",
        "margin",
    ]
    merged = man.merge(nb[nb_cols], on="Original_ID", how="left")
    if merged["has_C1_nb8"].isna().any():
        raise RuntimeError("missing 3x3nb for some BAMS points")

    diag, high = build_diagnostics(merged)
    high.to_csv(OUT_DIR / "high_points_with_nb8.csv", index=False)

    X_all = feat3[FEATURES_3X3].to_numpy(dtype=np.float32)
    ids = feat3["Original_ID"].astype(int).to_numpy()
    y_all = feat3["Class"].astype(int).to_numpy() - 1
    id_to_pos = {int(oid): i for i, oid in enumerate(ids)}

    idx = np.arange(len(X_all))
    idx_train, idx_test = train_test_split(
        idx, test_size=SPLIT["test_size"], random_state=SPLIT["random_state"], stratify=y_all
    )
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all).astype(np.float32)

    # packs
    h = high
    agree = h[h["agree_suggest"]]
    c1gate = h[(h["Human_Class"] != 1) | (h["has_C1_nb8"] == 1)]
    # iso2: flip isolated human1 to 2
    iso_labs = []
    for _, r in h.iterrows():
        hc = int(r.Human_Class)
        if hc == 1 and int(r.has_C1_nb8) == 0:
            iso_labs.append(2)
        else:
            iso_labs.append(hc)

    packs = {
        "high": make_pack(id_to_pos, y_all, h.Original_ID, h.Human_Class),
        "high_nb_agree": make_pack(id_to_pos, y_all, agree.Original_ID, agree.Human_Class),
        "high_c1gate": make_pack(id_to_pos, y_all, c1gate.Original_ID, c1gate.Human_Class),
        "high_iso2": make_pack(id_to_pos, y_all, h.Original_ID, iso_labs),
    }
    diag["seed_counts"] = {k: int(v["n_seed"]) for k, v in packs.items()}
    print("Diagnostics high×3x3:", json.dumps({k: v for k, v in diag.items() if k != "suggest_vs_human_crosstab"}, ensure_ascii=False))
    print("suggest crosstab Human×suggest:", diag["suggest_vs_human_crosstab"])
    print("seed counts", diag["seed_counts"])

    return {
        "X_all": X_all,
        "X_all_s": X_all_s,
        "y_all": y_all,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "packs": packs,
        "diag": diag,
    }


def run_pe(X_all, X_all_s, y_all, idx_train, idx_test, pack, device, seed, epochs, tag):
    if pack["n_seed"] == 0:
        raise ValueError(f"empty seeds: {tag}")
    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[pack["pos_seed"]] = True
    X_train_s = X_all_s[idx_train]
    X_test_s = X_all_s[idx_test]
    X_seed_s = X_all_s[pack["pos_seed"]]
    y_true = y_all[idx_test]

    teacher = train_mlp(
        np.concatenate([X_train_s, X_seed_s], axis=0),
        np.concatenate([y_all[idx_train], pack["lab_seed"]], axis=0),
        device=device,
        seed=seed,
        epochs=epochs,
    )
    _, margin_pool = predict_proba_mlp(teacher, X_all_s, device=device)

    cands = []
    for k in K_LIST:
        exp_idx, exp_lab = expand_prototypes(X_all, pack["pos_seed"], pack["lab_seed"], k)
        keep = (~used[exp_idx]) & (margin_pool[exp_idx] < MARGIN_THRESH)
        exp_idx, exp_lab = exp_idx[keep], exp_lab[keep]
        X_pe = np.concatenate([X_train_s, X_seed_s, X_all_s[exp_idx]], axis=0)
        y_pe = np.concatenate([y_all[idx_train], pack["lab_seed"], exp_lab], axis=0)
        w_pe = np.concatenate(
            [
                np.full(len(idx_train) + len(pack["lab_seed"]), REAL_WEIGHT, np.float32),
                np.full(len(exp_lab), SOFT_WEIGHT, np.float32),
            ]
        )
        st = train_mlp(
            X_pe, y_pe, sample_weight=w_pe, device=device, seed=seed + 100 + k, epochs=epochs
        )
        pred, _ = predict_proba_mlp(st, X_test_s, device=device)
        m = full_metrics(y_true, pred)
        cands.append({"k": k, "n_exp": int(len(exp_idx)), "m": m})
        print(
            f"  [seed={seed}] {tag} K={k:<2d} exp={len(exp_idx):<3d} "
            f"OA={m['OA']:.4f} BE={m['Boundary Error']}"
        )
    return pick_best(cands)


def run(device: str, seeds: list[int], epochs: int) -> pd.DataFrame:
    data = load_city()
    X_all, X_all_s, y_all = data["X_all"], data["X_all_s"], data["y_all"]
    idx_train, idx_test = data["idx_train"], data["idx_test"]
    packs = data["packs"]

    rows = []
    for seed in seeds:
        set_all_seeds(seed)
        X_train, X_test = X_all[idx_train], X_all[idx_test]
        y_true = y_all[idx_test]

        rf_specs = [
            ("Baseline", y_all),
            ("RF_high", packs["high"]["y_full"]),
            ("RF_high_nb_agree", packs["high_nb_agree"]["y_full"]),
        ]
        for name, y_src in rf_specs:
            rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
            rf.fit(X_train, y_src[idx_train])
            m = full_metrics(y_true, rf.predict(X_test))
            rows.append({"City": CITY, "Method": name, "Seed": seed, "K": None, "n_exp": None, "n_seed": None, **m})
            print(f"  [seed={seed}] {name:22s} OA={m['OA']:.4f} BE={m['Boundary Error']}")

        pe_specs = [
            ("PE_high", packs["high"]),
            ("PE_high_nb_agree", packs["high_nb_agree"]),
            ("PE_high_c1gate", packs["high_c1gate"]),
            ("PE_high_iso2", packs["high_iso2"]),
        ]
        for name, pack in pe_specs:
            best = run_pe(
                X_all, X_all_s, y_all, idx_train, idx_test, pack, device, seed, epochs, name
            )
            rows.append(
                {
                    "City": CITY,
                    "Method": name,
                    "Seed": seed,
                    "K": best["k"],
                    "n_exp": best["n_exp"],
                    "n_seed": pack["n_seed"],
                    **best["m"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_seed_metrics.csv", index=False)

    base_oa = df[df.Method == "Baseline"].OA.mean()
    base_be = df[df.Method == "Baseline"]["Boundary Error"].mean()
    summary_rows = []
    for method in METHODS:
        sub = df[df.Method == method]
        if sub.empty:
            continue
        summary_rows.append(
            {
                "Method": method,
                "OA_mean": sub.OA.mean(),
                "OA_std": sub.OA.std(ddof=0),
                "dOA_pp": (sub.OA.mean() - base_oa) * 100,
                "BE_mean": sub["Boundary Error"].mean(),
                "dBE": sub["Boundary Error"].mean() - base_be,
                "n_seed": sub["n_seed"].dropna().mean() if sub["n_seed"].notna().any() else None,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("dOA_pp", ascending=False)
    summary.to_csv(OUT_DIR / "summary_vs_baseline.csv", index=False)

    diag = data["diag"]
    # json-safe
    meta = {
        "n_high": diag["n_high"],
        "high_flip_wc": diag["high_flip_wc"],
        "high_agree_suggest": diag["high_agree_suggest"],
        "high_agree_suggest_n": diag["high_agree_suggest_n"],
        "human1_n": diag["human1_n"],
        "human1_has_C1_nb8": diag["human1_has_C1_nb8"],
        "human1_no_C1_nb8_n": diag["human1_no_C1_nb8_n"],
        "human2_n": diag["human2_n"],
        "human2_has_C1_nb8": diag["human2_has_C1_nb8"],
        "seed_counts": diag["seed_counts"],
        "suggest_vs_human_crosstab": {str(k): {str(k2): int(v2) for k2, v2 in v.items()} for k, v in diag["suggest_vs_human_crosstab"].items()},
    }
    (OUT_DIR / "diagnostics.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    md = [
        "# Nanjing high-confidence × 3×3 neighborhood",
        "",
        f"- High points: {meta['n_high']}; flip vs WC: {meta['high_flip_wc']:.1%}",
        f"- Agree with WC-neighbor suggest: {meta['high_agree_suggest']:.1%} ({meta['high_agree_suggest_n']}/{meta['n_high']})",
        f"- Human=1 with has_C1_nb8=0 (isolated built): {meta['human1_no_C1_nb8_n']}/{meta['human1_n']}",
        "",
        "## Summary vs Baseline",
        "",
        summary.to_string(index=False),
        "",
        "Note: Round2 high points are not included (no 3x3nb export yet).",
    ]
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(md))

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    run(args.device, args.seeds, args.epochs)


if __name__ == "__main__":
    main()
