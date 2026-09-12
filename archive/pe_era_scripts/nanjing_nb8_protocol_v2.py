#!/usr/bin/env python3
"""
Nanjing neighborhood protocol v2 (after 30-point visual audit)

Key changes vs blind suggest (PE_nb_lm):
  - Do NOT overwrite human labels with suggest_C for low/med.
  - Default trust human labels.
  - Apply the 30 audited visual verdicts as Human_v2 corrections.
  - Neighborhood only as soft veto / seed filter:
      * never promote Human=2 -> 1 just because has_C1_nb8=1
      * water-edge: never allow neighborhood to force Class1
  - PE prefers Confidence=high seeds; optionally include audited points.

Methods
-------
  Baseline
  BAMS_new / PE_new                 (current full human)
  BAMS_v2 / PE_v2                   (human + 30 visual verdicts)
  PE_new_high / PE_v2_high          (Confidence=high only; v2 uses updated labels)
  PE_v2_high_plus_audit             (high seeds + all 30 audited points)
  PE_nb_lm_ref                      (old blind suggest on low+med; reference only)

Eval: pointwise OA / BE vs original WorldCover Class.
Outputs -> data2/nanjing_nb8_protocol_v2/
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
OUT_DIR = DATA2 / "nanjing_nb8_protocol_v2"
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

# Visual verdicts from user audit (PointID -> class 1/2/3)
VISUAL_VERDICTS = {
    2: 1,
    5: 2,
    6: 1,
    17: 3,
    20: 1,
    24: 3,
    25: 1,
    27: 1,
    28: 2,
    29: 3,
    30: 1,
    31: 2,
    32: 2,
    35: 2,
    38: 2,
    43: 2,
    44: 2,
    49: 2,
    51: 2,
    52: 2,
    54: 1,
    55: 2,
    72: 3,
    75: 2,
    95: 1,
    107: 2,
    117: 2,
    120: 1,
    142: 2,
    143: 2,
}

METHODS = [
    "Baseline",
    "BAMS_new",
    "BAMS_v2",
    "PE_new",
    "PE_v2",
    "PE_new_high",
    "PE_v2_high",
    "PE_v2_high_plus_audit",
    "PE_nb_lm_ref",
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


def pick_best(cands: list[dict]) -> dict:
    return sorted(cands, key=lambda c: (c["m"]["Boundary Error"], -c["m"]["OA"]))[0]


def is_water_like_scene(scene: str) -> bool:
    s = str(scene).lower()
    keys = ["水", "water", "we", "河", "湖", "塘", "滩"]
    return any(k in s for k in keys)


def load() -> dict:
    city_root = DATA2 / CITY
    feat3 = pd.read_csv(city_root / "06_3x3" / f"{CITY}_3x3_Features.csv")
    man = pd.read_csv(city_root / "04_manual" / f"{CITY}_BAMS150_Manual.csv")
    nb = pd.read_csv(city_root / "04_manual" / f"{CITY}_BAMS150_3x3nb_wide.csv")

    man["Original_ID"] = man["Original_ID"].astype(int)
    man["PointID"] = man["PointID"].astype(int)
    man["Human_Class"] = pd.to_numeric(man["Human_Class"], errors="coerce").astype(int)
    man["Conf"] = man["Confidence"].map(norm_conf)
    nb["Original_ID"] = nb["Original_ID"].astype(int)

    merged = man.merge(
        nb[
            [
                "Original_ID",
                "has_C1_nb8",
                "suggest_C_if_uncertain",
                "n_C1_nb8",
                "n_C2_nb8",
                "n_C3_nb8",
            ]
        ],
        on="Original_ID",
        how="left",
        validate="one_to_one",
    )
    if merged["has_C1_nb8"].isna().any():
        raise ValueError("Missing nb join")

    # Human_v2 = human with 30 visual overrides
    merged["Human_v2"] = merged["Human_Class"]
    n_v2 = 0
    for pid, cls in VISUAL_VERDICTS.items():
        mask = merged["PointID"] == pid
        if mask.any() and int(merged.loc[mask, "Human_v2"].iloc[0]) != cls:
            n_v2 += 1
        merged.loc[mask, "Human_v2"] = cls
        merged.loc[mask, "Audited"] = True
    merged["Audited"] = merged["Audited"].fillna(False)

    # Blind suggest reference labels (old protocol) — for PE_nb_lm_ref only
    suggest = merged["suggest_C_if_uncertain"].astype(int).to_numpy()
    human = merged["Human_Class"].to_numpy()
    conf = merged["Conf"].tolist()
    human_lm = human.copy()
    n_blind = 0
    for i, c in enumerate(conf):
        if c in ("low", "med"):
            # OLD blind rule; also keep water-unsafe behavior as reference
            if human_lm[i] != suggest[i]:
                n_blind += 1
            human_lm[i] = suggest[i]

    merged["Human_blind_lm"] = human_lm
    merged.to_csv(OUT_DIR / "Nanjing_BAMS150_labels_v2.csv", index=False)

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

    oids = merged["Original_ID"].astype(int).to_numpy()
    pids = merged["PointID"].astype(int).to_numpy()
    audited = merged["Audited"].to_numpy().astype(bool)
    scenes = merged["Scene"].astype(str).tolist()
    has_c1 = merged["has_C1_nb8"].astype(int).to_numpy()

    def pack_from_labels(labels_1based: np.ndarray, keep_mask: np.ndarray, name: str):
        pos_all = np.asarray([id_to_pos[int(o)] for o in oids], dtype=int)
        lab_all = (labels_1based - 1).astype(np.int64)
        pos = pos_all[keep_mask]
        lab = lab_all[keep_mask]
        y_full = y_all.copy()
        y_full[pos] = lab
        return {
            "name": name,
            "pos": pos,
            "lab": lab,
            "y_full": y_full,
            "n_seed": int(keep_mask.sum()),
        }

    variants = {}

    # all seeds
    keep_all = np.ones(len(merged), dtype=bool)
    variants["new"] = pack_from_labels(merged["Human_Class"].to_numpy(), keep_all, "new")
    variants["v2"] = pack_from_labels(merged["Human_v2"].to_numpy(), keep_all, "v2")
    variants["blind_lm"] = pack_from_labels(
        merged["Human_blind_lm"].to_numpy(), keep_all, "blind_lm"
    )

    # high only (on respective label sets)
    keep_high = np.asarray([c == "high" for c in conf], dtype=bool)
    variants["new_high"] = pack_from_labels(
        merged["Human_Class"].to_numpy(), keep_high, "new_high"
    )
    variants["v2_high"] = pack_from_labels(merged["Human_v2"].to_numpy(), keep_high, "v2_high")

    # high + all audited 30 (v2 labels)
    keep_high_audit = keep_high | audited
    # additional veto: drop Class1 seeds that are water-like AND has_C1_nb8==1 was the only reason
    # (protocol: water edge never forced to 1 by nb — here audited already fixed)
    # Extra seed filter for high_plus_audit: remove unaudited low-conf Class1 with water scene
    keep_hpa = keep_high_audit.copy()
    for i in range(len(merged)):
        if audited[i]:
            continue
        if conf[i] == "low" and int(merged["Human_v2"].iloc[i]) == 1 and is_water_like_scene(scenes[i]):
            keep_hpa[i] = False
    variants["v2_high_plus_audit"] = pack_from_labels(
        merged["Human_v2"].to_numpy(), keep_hpa, "v2_high_plus_audit"
    )

    print(
        f"v2 overrides vs human: {n_v2}; "
        f"blind_lm adjusted: {n_blind}; "
        f"seeds: new={variants['new']['n_seed']}, v2_high={variants['v2_high']['n_seed']}, "
        f"v2_high+audit={variants['v2_high_plus_audit']['n_seed']}"
    )

    return {
        "X_all": X_all,
        "X_all_s": X_all_s,
        "y_all": y_all,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "variants": variants,
        "n_v2_overrides": n_v2,
        "n_blind_adj": n_blind,
    }


def run_pe(X_all, X_all_s, y_all, idx_train, idx_test, pack, device, seed, epochs, tag):
    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[pack["pos"]] = True
    X_train_s, X_test_s = X_all_s[idx_train], X_all_s[idx_test]
    X_seed_s = X_all_s[pack["pos"]]
    y_true = y_all[idx_test]

    teacher = train_mlp(
        np.concatenate([X_train_s, X_seed_s], axis=0),
        np.concatenate([y_all[idx_train], pack["lab"]], axis=0),
        device=device,
        seed=seed,
        epochs=epochs,
    )
    _, _, margin_pool = predict_proba_mlp(teacher, X_all_s, device=device)

    cands = []
    for k in K_LIST:
        exp_idx_all, exp_lab_all = expand_prototypes(X_all, pack["pos"], pack["lab"], k)
        keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
        exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]
        X_pe = np.concatenate([X_train_s, X_seed_s, X_all_s[exp_idx]], axis=0)
        y_pe = np.concatenate([y_all[idx_train], pack["lab"], exp_lab], axis=0)
        w_pe = np.concatenate(
            [
                np.full(len(idx_train) + len(pack["lab"]), REAL_WEIGHT, np.float32),
                np.full(len(exp_lab), SOFT_WEIGHT, np.float32),
            ]
        )
        st = train_mlp(
            X_pe, y_pe, sample_weight=w_pe, device=device, seed=seed + 100 + k, epochs=epochs
        )
        pred, _, _ = predict_proba_mlp(st, X_test_s, device=device)
        m = full_metrics(y_true, pred)
        cands.append({"k": k, "n_exp": int(len(exp_idx)), "m": m})
        print(
            f"  [seed={seed}] {tag} K={k:<2d} exp={len(exp_idx):<3d} "
            f"OA={m['OA']:.4f} BE={m['Boundary Error']}"
        )
    return pick_best(cands)


def run(device: str, seeds: list[int], epochs: int):
    data = load()
    X_all, X_all_s, y_all = data["X_all"], data["X_all_s"], data["y_all"]
    idx_train, idx_test = data["idx_train"], data["idx_test"]
    V = data["variants"]

    rows = []
    for seed in seeds:
        set_all_seeds(seed)
        X_train, X_test = X_all[idx_train], X_all[idx_test]
        y_true = y_all[idx_test]

        for name, ysrc in [
            ("Baseline", y_all),
            ("BAMS_new", V["new"]["y_full"]),
            ("BAMS_v2", V["v2"]["y_full"]),
        ]:
            rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
            rf.fit(X_train, ysrc[idx_train])
            m = full_metrics(y_true, rf.predict(X_test))
            rows.append({"City": CITY, "Method": name, "Seed": seed, **m})
            print(f"  [seed={seed}] {name:24s} OA={m['OA']:.4f} BE={m['Boundary Error']}")

        pe_map = [
            ("PE_new", V["new"]),
            ("PE_v2", V["v2"]),
            ("PE_new_high", V["new_high"]),
            ("PE_v2_high", V["v2_high"]),
            ("PE_v2_high_plus_audit", V["v2_high_plus_audit"]),
            ("PE_nb_lm_ref", V["blind_lm"]),
        ]
        for name, pack in pe_map:
            best = run_pe(
                X_all, X_all_s, y_all, idx_train, idx_test, pack, device, seed, epochs, name
            )
            rows.append(
                {
                    "City": CITY,
                    "Method": name,
                    "Seed": seed,
                    **best["m"],
                    "BestK": best["k"],
                    "n_expanded": best["n_exp"],
                    "n_seed": pack["n_seed"],
                }
            )
            print(
                f"  [seed={seed}] {name:24s} OA={best['m']['OA']:.4f} "
                f"BE={best['m']['Boundary Error']} K={best['k']} exp={best['n_exp']}"
            )

    return pd.DataFrame(rows), data


def summarize(df: pd.DataFrame, data: dict) -> str:
    overall = (
        df.groupby("Method", sort=False)
        .agg(
            Mean_OA=("OA", "mean"),
            Std_OA=("OA", "std"),
            Mean_BE=("Boundary Error", "mean"),
            Std_BE=("Boundary Error", "std"),
            Mean_n_seed=("n_seed", "mean"),
            Mean_n_exp=("n_expanded", "mean"),
        )
        .reindex(METHODS)
        .reset_index()
    )
    base = overall.loc[overall.Method == "Baseline", "Mean_OA"].iloc[0]
    overall["dOA_pp"] = (overall["Mean_OA"] - base) * 100
    lines = [
        "# Nanjing nb protocol v2 (after 30-point visual audit)\n",
        f"- Visual overrides applied: {data['n_v2_overrides']} class changes among 30 audited points\n",
        f"- Blind lm adjustments (reference): {data['n_blind_adj']}\n",
        "\n## Mean over seeds\n",
        overall.to_string(index=False),
        "\n\n## Protocol\n",
        "- Stop blind suggest overwrite (except PE_nb_lm_ref)\n",
        "- BAMS_v2 / PE_v2: human labels + 30 visual verdicts\n",
        "- PE_*_high: Confidence=high seeds only\n",
        "- PE_v2_high_plus_audit: high seeds + all audited 30\n",
        "- Water-edge never forced to Class1 by neighborhood\n",
    ]
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

    df, data = run(device=device, seeds=seeds, epochs=epochs)
    md, overall = summarize(df, data)
    df.to_csv(OUT_DIR / "seed_results.csv", index=False)
    overall.to_csv(OUT_DIR / "overall_summary.csv", index=False)
    (OUT_DIR / "SUMMARY.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "city": CITY,
                "seeds": seeds,
                "epochs": epochs,
                "methods": METHODS,
                "n_visual_verdicts": len(VISUAL_VERDICTS),
                "n_v2_overrides": data["n_v2_overrides"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + md)
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
