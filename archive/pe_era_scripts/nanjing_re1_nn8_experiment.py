#!/usr/bin/env python3
"""
Nanjing protocol update experiment
----------------------------------
1) Force selected Scene human labels -> Class 1:
     re / road / urban road  (road edge)
     fe / ub                 (building-adjacent / urban-boundary scenes
                              that were labeled Class2)
2) Evaluate OA / Boundary Error against original WorldCover Class (pointwise).
   NOTE: discrete-sample 8NN "neighborhood OA" is NOT used (not true 3x3 pixels).

Methods: Baseline, BAMS150 (original human), BAMS150_proto1, PE_Boundary_proto1
Seeds: 0..4

Outputs -> data2/nanjing_proto1_eval/
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
OUT_DIR = DATA2 / "nanjing_proto1_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITY = "Nanjing"
FEATURES_3X3 = [
    "B2_mean",
    "B2_stdDev",
    "B3_mean",
    "B3_stdDev",
    "B4_mean",
    "B4_stdDev",
    "B8_mean",
    "B8_stdDev",
    "B11_mean",
    "B11_stdDev",
    "B12_mean",
    "B12_stdDev",
    "NDVI_mean",
    "NDVI_stdDev",
    "NDBI_mean",
    "NDBI_stdDev",
    "MNDWI_mean",
    "MNDWI_stdDev",
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
# Scenes forced to Human_Class = 1 under the revised protocol
PROTO1_SCENES = {"re", "road", "urban road", "fe", "ub"}


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


def get_labeled_manual(path: Path) -> pd.DataFrame:
    manual = pd.read_csv(path)
    manual["Human_Class"] = pd.to_numeric(manual["Human_Class"], errors="coerce")
    manual = manual[manual["Human_Class"].notna()].copy()
    manual["Human_Class"] = manual["Human_Class"].astype(int)
    manual["Original_ID"] = manual["Original_ID"].astype(int)
    if "Scene" not in manual.columns:
        manual["Scene"] = ""
    manual["Scene"] = manual["Scene"].astype(str)
    return manual


def apply_re_to_class1(manual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = manual.copy()
    scene = out["Scene"].str.lower().str.strip()
    mask = scene.isin(ROAD_SCENES)
    changed = out.loc[mask].copy()
    out.loc[mask, "Human_Class"] = 1
    changed["Human_Class_new"] = 1
    return out, changed


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
    xt = torch.from_numpy(X.astype(np.float32)).to(device)
    probs = torch.softmax(model(xt), dim=1).cpu().numpy()
    order = np.argsort(probs, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    conf = probs[np.arange(len(probs)), top1]
    margin = conf - probs[np.arange(len(probs)), top2]
    return top1.astype(np.int64), conf, margin


def majority_label(votes: list[int]) -> int:
    counts = Counter(votes)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def expand_prototypes(feature, bams_pos, bams_label, k: int):
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn_model.fit(feature)
    neigh_idx = nn_model.kneighbors(feature[bams_pos], return_distance=False)
    votes: dict[int, list[int]] = defaultdict(list)
    for proto_i, neighbors in enumerate(neigh_idx):
        proto_pos = int(bams_pos[proto_i])
        proto_lab = int(bams_label[proto_i])
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


def spatial_8nn_majority(lon: np.ndarray, lat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    For each sample, take 8 spatially nearest other samples (approx. 3x3 neighbors
    on a sparse point set) and assign majority Class as neighborhood GT.
    """
    coords = np.column_stack([np.radians(lat), np.radians(lon)])
    nn = NearestNeighbors(n_neighbors=9, metric="haversine")
    nn.fit(coords)
    idx = nn.kneighbors(coords, return_distance=False)
    y_maj = np.empty(len(y), dtype=np.int64)
    for i in range(len(y)):
        neigh = [int(j) for j in idx[i].tolist() if int(j) != i][:8]
        y_maj[i] = majority_label([int(y[j]) for j in neigh])
    return y_maj


def load_nanjing() -> dict:
    city_root = DATA2 / CITY
    feat3 = pd.read_csv(city_root / "06_3x3" / f"{CITY}_3x3_Features.csv")
    orig = pd.read_csv(city_root / "01_original" / f"{CITY}_WC_Samples_15000.csv")
    manual = get_labeled_manual(city_root / "04_manual" / f"{CITY}_BAMS150_Manual.csv")
    manual_re1, changed = apply_re_to_class1(manual)

    # persist protocol labels
    manual_re1.to_csv(OUT_DIR / "Nanjing_BAMS150_Manual_re1.csv", index=False)
    changed.to_csv(OUT_DIR / "Nanjing_re_to_class1_changes.csv", index=False)

    X_all = feat3[FEATURES_3X3].to_numpy(dtype=np.float32)
    ids = feat3["Original_ID"].astype(int).to_numpy()
    y_all = feat3["Class"].astype(int).to_numpy() - 1

    id_to_pos = {int(oid): i for i, oid in enumerate(ids)}
    lon_map = dict(zip(orig["Original_ID"].astype(int), orig["lon"].astype(float)))
    lat_map = dict(zip(orig["Original_ID"].astype(int), orig["lat"].astype(float)))
    lon_all = np.asarray([lon_map[int(i)] for i in ids], dtype=float)
    lat_all = np.asarray([lat_map[int(i)] for i in ids], dtype=float)

    y_nn8 = spatial_8nn_majority(lon_all, lat_all, y_all)

    idx = np.arange(len(X_all))
    idx_train, idx_test = train_test_split(
        idx, test_size=SPLIT["test_size"], random_state=SPLIT["random_state"], stratify=y_all
    )
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all).astype(np.float32)

    def labels_from_manual(man: pd.DataFrame):
        pos = np.asarray([id_to_pos[int(r.Original_ID)] for _, r in man.iterrows()], dtype=int)
        lab = np.asarray([int(r.Human_Class) - 1 for _, r in man.iterrows()], dtype=np.int64)
        y_full = y_all.copy()
        y_full[pos] = lab
        return pos, lab, y_full

    bams_pos, bams_label, y_bams = labels_from_manual(manual)
    bams_pos_re1, bams_label_re1, y_bams_re1 = labels_from_manual(manual_re1)

    # agreement stats
    agree_point = float((y_nn8 == y_all).mean())
    n_re_changed = int((manual["Human_Class"].to_numpy() != manual_re1["Human_Class"].to_numpy()).sum())

    return {
        "X_all": X_all,
        "X_all_s": X_all_s,
        "y_all": y_all,
        "y_nn8": y_nn8,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "bams_pos": bams_pos,
        "bams_label": bams_label,
        "y_bams": y_bams,
        "bams_pos_re1": bams_pos_re1,
        "bams_label_re1": bams_label_re1,
        "y_bams_re1": y_bams_re1,
        "n_re_changed": n_re_changed,
        "nn8_agree_with_point_class": agree_point,
        "ids": ids,
        "lon_all": lon_all,
        "lat_all": lat_all,
    }


def pick_best(cands: list[dict]) -> dict:
    return sorted(cands, key=lambda c: (c["m_point"]["Boundary Error"], -c["m_point"]["OA"]))[0]


def eval_both(y_point, y_nn8, pred) -> dict:
    return {
        "m_point": full_metrics(y_point, pred),
        "m_nn8": full_metrics(y_nn8, pred),
    }


def run(device: str, seeds: list[int], epochs: int) -> pd.DataFrame:
    data = load_nanjing()
    print(
        f"Nanjing re->1 changed={data['n_re_changed']}; "
        f"8NN majority agree with point Class={data['nn8_agree_with_point_class']:.4f}"
    )

    X_all = data["X_all"]
    X_all_s = data["X_all_s"]
    y_all = data["y_all"]
    y_nn8 = data["y_nn8"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    bams_pos = data["bams_pos"]
    bams_label = data["bams_label"]
    y_bams = data["y_bams"]
    bams_pos_re1 = data["bams_pos_re1"]
    bams_label_re1 = data["bams_label_re1"]
    y_bams_re1 = data["y_bams_re1"]

    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[bams_pos_re1] = True

    rows = []
    for seed in seeds:
        set_all_seeds(seed)
        X_train, X_test = X_all[idx_train], X_all[idx_test]
        X_train_s, X_test_s = X_all_s[idx_train], X_all_s[idx_test]
        X_bams_s = X_all_s[bams_pos]
        X_bams_re1_s = X_all_s[bams_pos_re1]
        y_true_point = y_all[idx_test]
        y_true_nn8 = y_nn8[idx_test]

        # Baseline
        rf_base = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf_base.fit(X_train, y_all[idx_train])
        pred_base = rf_base.predict(X_test).astype(np.int64)
        ev = eval_both(y_true_point, y_true_nn8, pred_base)
        print(
            f"  [seed={seed}] Baseline     "
            f"OA_pt={ev['m_point']['OA']:.4f} BE_pt={ev['m_point']['Boundary Error']} | "
            f"OA_nn8={ev['m_nn8']['OA']:.4f} BE_nn8={ev['m_nn8']['Boundary Error']}"
        )
        rows.append({"Seed": seed, "Method": "Baseline", **_flat(ev)})

        # BAMS original human
        rf_bams = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf_bams.fit(X_train, y_bams[idx_train])
        pred_bams = rf_bams.predict(X_test).astype(np.int64)
        ev = eval_both(y_true_point, y_true_nn8, pred_bams)
        print(
            f"  [seed={seed}] BAMS150      "
            f"OA_pt={ev['m_point']['OA']:.4f} BE_pt={ev['m_point']['Boundary Error']} | "
            f"OA_nn8={ev['m_nn8']['OA']:.4f} BE_nn8={ev['m_nn8']['Boundary Error']}"
        )
        rows.append({"Seed": seed, "Method": "BAMS150", **_flat(ev)})

        # BAMS with re->1
        rf_re1 = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf_re1.fit(X_train, y_bams_re1[idx_train])
        pred_re1 = rf_re1.predict(X_test).astype(np.int64)
        ev = eval_both(y_true_point, y_true_nn8, pred_re1)
        print(
            f"  [seed={seed}] BAMS150_re1  "
            f"OA_pt={ev['m_point']['OA']:.4f} BE_pt={ev['m_point']['Boundary Error']} | "
            f"OA_nn8={ev['m_nn8']['OA']:.4f} BE_nn8={ev['m_nn8']['Boundary Error']}"
        )
        rows.append({"Seed": seed, "Method": "BAMS150_re1", **_flat(ev)})

        # PE with re1 human labels
        teacher = train_mlp(
            np.concatenate([X_train_s, X_bams_re1_s], axis=0),
            np.concatenate([y_all[idx_train], bams_label_re1], axis=0),
            device=device,
            seed=seed,
            epochs=epochs,
        )
        _, _, margin_pool = predict_proba_mlp(teacher, X_all_s, device=device)
        pe_cands = []
        for k in K_LIST:
            exp_idx_all, exp_lab_all = expand_prototypes(
                X_all, bams_pos_re1, bams_label_re1, k
            )
            keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
            exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]
            X_pe = np.concatenate([X_train_s, X_bams_re1_s, X_all_s[exp_idx]], axis=0)
            y_pe = np.concatenate([y_all[idx_train], bams_label_re1, exp_lab], axis=0)
            w_pe = np.concatenate(
                [
                    np.full(len(idx_train) + len(bams_label_re1), REAL_WEIGHT, np.float32),
                    np.full(len(exp_lab), SOFT_WEIGHT, np.float32),
                ]
            )
            st = train_mlp(
                X_pe, y_pe, sample_weight=w_pe, device=device, seed=seed + 100 + k, epochs=epochs
            )
            pred_pe, _, _ = predict_proba_mlp(st, X_test_s, device=device)
            ev = eval_both(y_true_point, y_true_nn8, pred_pe)
            pe_cands.append(
                {
                    "k": k,
                    "n_exp": int(len(exp_idx)),
                    "m_point": ev["m_point"],
                    "m_nn8": ev["m_nn8"],
                    "pred": pred_pe,
                }
            )
            print(
                f"  [seed={seed}] PE_re1 K={k:<2d} exp={len(exp_idx):<3d} "
                f"OA_pt={ev['m_point']['OA']:.4f} BE_pt={ev['m_point']['Boundary Error']} | "
                f"OA_nn8={ev['m_nn8']['OA']:.4f} BE_nn8={ev['m_nn8']['Boundary Error']}"
            )
        best = pick_best(pe_cands)
        rows.append(
            {
                "Seed": seed,
                "Method": "PE_Boundary_re1",
                "BestK": best["k"],
                "n_expanded": best["n_exp"],
                **_flat({"m_point": best["m_point"], "m_nn8": best["m_nn8"]}),
            }
        )
        print(
            f"  [seed={seed}] PE_Boundary_re1 bestK={best['k']} "
            f"OA_pt={best['m_point']['OA']:.4f} OA_nn8={best['m_nn8']['OA']:.4f}"
        )

    return pd.DataFrame(rows)


def _flat(ev: dict) -> dict:
    mp, mn = ev["m_point"], ev["m_nn8"]
    return {
        "OA_point": mp["OA"],
        "BE_point": mp["Boundary Error"],
        "OA_nn8": mn["OA"],
        "BE_nn8": mn["Boundary Error"],
        "AA_point": mp["AA"],
        "Kappa_point": mp["Kappa"],
        "AA_nn8": mn["AA"],
        "Kappa_nn8": mn["Kappa"],
    }


def summarize(df: pd.DataFrame, meta: dict) -> str:
    methods = ["Baseline", "BAMS150", "BAMS150_re1", "PE_Boundary_re1"]
    overall = (
        df.groupby("Method", sort=False)
        .agg(
            Mean_OA_point=("OA_point", "mean"),
            Mean_BE_point=("BE_point", "mean"),
            Mean_OA_nn8=("OA_nn8", "mean"),
            Mean_BE_nn8=("BE_nn8", "mean"),
        )
        .reindex(methods)
        .reset_index()
    )
    lines = []
    lines.append("# Nanjing re→1 + 8NN-majority evaluation\n")
    lines.append(
        f"- Road-edge human labels forced to Class1: n_changed={meta['n_re_changed']}\n"
        f"- 8NN majority GT agrees with point Class: {meta['nn8_agree_with_point_class']:.4f}\n"
    )
    lines.append("## Mean over seeds\n")
    lines.append(overall.to_string(index=False))
    lines.append(
        "\n\n## Reading guide\n"
        "- OA_point / BE_point: vs original WorldCover Class at the sample point\n"
        "- OA_nn8 / BE_nn8: vs majority Class of 8 spatially nearest other samples\n"
        "- BAMS150_re1 uses human labels with Scene=re/road/urban road set to Class1\n"
        "- PE_Boundary_re1 expands from those re1 human prototypes\n"
    )
    return "\n".join(lines), overall


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [0] if args.quick else args.seeds
    epochs = min(args.epochs, 40) if args.quick else args.epochs
    print("device:", device, "seeds:", seeds, "epochs:", epochs)

    # preload meta via load inside run; also expose agreement
    # lightweight preload for summary meta
    from copy import deepcopy  # noqa: F401

    # run experiment
    # first load once for meta printed in run()
    pre = load_nanjing()
    meta = {
        "n_re_changed": pre["n_re_changed"],
        "nn8_agree_with_point_class": pre["nn8_agree_with_point_class"],
    }
    # avoid double heavy work: patch run to accept optional preloaded? simpler: just run()
    # Actually run() loads again - fine for clarity/correctness.

    df = run(device=device, seeds=seeds, epochs=epochs)
    # reload meta cheaply from saved change file
    changed = pd.read_csv(OUT_DIR / "Nanjing_re_to_class1_changes.csv")
    meta = {
        "n_re_changed": int(len(changed)),
        "nn8_agree_with_point_class": float(
            (pre["y_nn8"] == pre["y_all"]).mean()
        ),
    }
    md, overall = summarize(df, meta)
    df.to_csv(OUT_DIR / "seed_results.csv", index=False)
    overall.to_csv(OUT_DIR / "overall_summary.csv", index=False)
    (OUT_DIR / "SUMMARY.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "city": CITY,
                "seeds": seeds,
                "epochs": epochs,
                "road_scenes": sorted(ROAD_SCENES),
                "k_list": K_LIST,
                "margin_thresh": MARGIN_THRESH,
                "n_re_changed": meta["n_re_changed"],
                "nn8_agree_with_point_class": meta["nn8_agree_with_point_class"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + md)
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
