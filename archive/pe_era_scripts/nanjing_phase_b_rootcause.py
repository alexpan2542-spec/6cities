#!/usr/bin/env python3
"""
Nanjing Phase B — root-cause contrast for H1 (label–exam conflict)

Same BAMS geometry; vary only seed labels; evaluate WC test + Human hold-out.

WC-track (full BAMS seeds; K picked by WC BE, −OA — Phase A protocol):
  Baseline
  PE_Human_all / PE_Human_high
  PE_WC_all / PE_WC_high          (WC class on the same point positions)
  PE_Agree_all / PE_Disagree_all  (Human==WC vs Human!=WC among BAMS150)
  PE_Agree_high / PE_Disagree_high

Dual-track (BAMS high ∪ Round2 high; 60% seeds / 40% human test per RNG seed):
  Baseline_dual
  PE_Human_holdout
  PE_WC_holdout

Outputs -> data2/nanjing_rootcause_v2/phase_b/
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
HUMAN_TEST_SIZE = 0.4
SEEDS = [0, 1, 2, 3, 4]
K_LIST = [5, 10, 20, 30]
MARGIN_THRESH = 0.15
SOFT_WEIGHT = 0.3
REAL_WEIGHT = 1.0
EPOCHS = 80
LR = 1e-3
BATCH_SIZE = 256


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


def load_manual(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path)
    m["Human_Class"] = pd.to_numeric(m["Human_Class"], errors="coerce")
    m = m[m["Human_Class"].notna()].copy()
    m["Human_Class"] = m["Human_Class"].astype(int)
    m["Original_ID"] = m["Original_ID"].astype(int)
    if "Confidence" in m.columns:
        m["Conf"] = m["Confidence"].map(norm_conf)
    else:
        m["Conf"] = "high"
    return m


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
def predict_mlp(model, X, device="cpu"):
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
    return {
        "OA": float(accuracy_score(y_true, y_pred)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "BE": int(cm[0, 1] + cm[1, 0]),
        "n": int(len(y_true)),
    }


def make_pack(pos: np.ndarray, lab: np.ndarray) -> dict:
    pos = np.asarray(pos, dtype=int)
    lab = np.asarray(lab, dtype=np.int64)
    return {"pos_seed": pos, "lab_seed": lab, "n_seed": int(len(pos))}


def run_pe_wc_pick(X_all, X_all_s, y_wc, idx_train, idx_test, pack, device, seed, epochs, tag: str):
    if pack["n_seed"] == 0:
        print(f"  [seed={seed}] {tag}: empty seed set, skip")
        return None
    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[pack["pos_seed"]] = True

    teacher = train_mlp(
        np.concatenate([X_all_s[idx_train], X_all_s[pack["pos_seed"]]], axis=0),
        np.concatenate([y_wc[idx_train], pack["lab_seed"]], axis=0),
        device=device,
        seed=seed,
        epochs=epochs,
    )
    _, margin_pool = predict_mlp(teacher, X_all_s, device=device)

    cands = []
    for k in K_LIST:
        exp_idx_all, exp_lab_all = expand_prototypes(
            X_all, pack["pos_seed"], pack["lab_seed"], k
        )
        keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
        exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]
        X_pe = np.concatenate(
            [X_all_s[idx_train], X_all_s[pack["pos_seed"]], X_all_s[exp_idx]], axis=0
        )
        y_pe = np.concatenate([y_wc[idx_train], pack["lab_seed"], exp_lab], axis=0)
        w_pe = np.concatenate(
            [
                np.full(len(idx_train) + len(pack["lab_seed"]), REAL_WEIGHT, np.float32),
                np.full(len(exp_lab), SOFT_WEIGHT, np.float32),
            ]
        )
        st = train_mlp(
            X_pe, y_pe, sample_weight=w_pe, device=device, seed=seed + 100 + k, epochs=epochs
        )
        pred, _ = predict_mlp(st, X_all_s, device=device)
        mw = full_metrics(y_wc[idx_test], pred[idx_test])
        cands.append({"k": k, "n_exp": int(len(exp_idx)), "pred": pred, "m_w": mw})
        print(
            f"  [seed={seed}] {tag:22s} K={k:<2d} n_seed={pack['n_seed']:<3d} "
            f"exp={len(exp_idx):<3d} WC_OA={mw['OA']:.4f} BE={mw['BE']}"
        )
    return sorted(cands, key=lambda c: (c["m_w"]["BE"], -c["m_w"]["OA"]))[0]


def load_city():
    city_root = DATA2 / CITY
    feat3 = pd.read_csv(city_root / "06_3x3" / f"{CITY}_3x3_Features.csv")
    bams = load_manual(city_root / "04_manual" / f"{CITY}_BAMS150_Manual.csv")
    r2_path = city_root / "04_manual" / "Nanjing_Round2_C12Boundary50.csv"
    r2 = load_manual(r2_path) if r2_path.exists() else bams.iloc[0:0].copy()

    X_all = feat3[FEATURES_3X3].to_numpy(dtype=np.float32)
    ids = feat3["Original_ID"].astype(int).to_numpy()
    y_wc = feat3["Class"].astype(int).to_numpy() - 1
    id_to_pos = {int(oid): i for i, oid in enumerate(ids)}

    bams = bams.copy()
    bams["pos"] = bams["Original_ID"].map(id_to_pos)
    bams = bams[bams["pos"].notna()].copy()
    bams["pos"] = bams["pos"].astype(int)
    bams["WC0"] = y_wc[bams["pos"].to_numpy()]
    bams["H0"] = bams["Human_Class"].astype(int) - 1
    bams["agree"] = bams["H0"] == bams["WC0"]

    hum = pd.concat([bams.assign(src="bams"), r2.assign(src="r2")], ignore_index=True)
    hum = hum[hum["Conf"] == "high"].copy()
    hum = hum.sort_values("src").drop_duplicates("Original_ID", keep="last")
    hum["pos"] = hum["Original_ID"].map(id_to_pos)
    hum = hum[hum["pos"].notna()].copy()
    hum["pos"] = hum["pos"].astype(int)
    hum["WC0"] = y_wc[hum["pos"].to_numpy()]
    hum["H0"] = hum["Human_Class"].astype(int) - 1
    hum["agree"] = hum["H0"] == hum["WC0"]

    idx = np.arange(len(X_all))
    idx_train, idx_test = train_test_split(
        idx, test_size=SPLIT["test_size"], random_state=SPLIT["random_state"], stratify=y_wc
    )
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all).astype(np.float32)

    meta = {
        "n_bams": int(len(bams)),
        "n_bams_high": int((bams["Conf"] == "high").sum()),
        "n_agree_all": int(bams["agree"].sum()),
        "n_disagree_all": int((~bams["agree"]).sum()),
        "n_agree_high": int(((bams["Conf"] == "high") & bams["agree"]).sum()),
        "n_disagree_high": int(((bams["Conf"] == "high") & ~bams["agree"]).sum()),
        "n_dual_high_pool": int(len(hum)),
        "dual_flip_rate": float((~hum["agree"]).mean()) if len(hum) else None,
        "human_test_frac": HUMAN_TEST_SIZE,
    }
    print(
        f"Phase B Nanjing: BAMS n={meta['n_bams']} high={meta['n_bams_high']} "
        f"agree/disagree={meta['n_agree_all']}/{meta['n_disagree_all']} "
        f"(high {meta['n_agree_high']}/{meta['n_disagree_high']}); "
        f"dual_high_pool={meta['n_dual_high_pool']} flip={meta['dual_flip_rate']:.1%}"
    )
    return {
        "X_all": X_all,
        "X_all_s": X_all_s,
        "y_wc": y_wc,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "bams": bams,
        "hum": hum,
        "meta": meta,
    }


def build_wc_track_packs(bams: pd.DataFrame) -> dict[str, dict]:
    pos_all = bams["pos"].to_numpy(dtype=int)
    h_all = bams["H0"].to_numpy(dtype=np.int64)
    w_all = bams["WC0"].to_numpy(dtype=np.int64)
    high = (bams["Conf"] == "high").to_numpy()
    agree = bams["agree"].to_numpy()

    return {
        "PE_Human_all": make_pack(pos_all, h_all),
        "PE_Human_high": make_pack(pos_all[high], h_all[high]),
        "PE_WC_all": make_pack(pos_all, w_all),
        "PE_WC_high": make_pack(pos_all[high], w_all[high]),
        "PE_Agree_all": make_pack(pos_all[agree], h_all[agree]),
        "PE_Disagree_all": make_pack(pos_all[~agree], h_all[~agree]),
        "PE_Agree_high": make_pack(pos_all[high & agree], h_all[high & agree]),
        "PE_Disagree_high": make_pack(pos_all[high & ~agree], h_all[high & ~agree]),
    }


def run(device: str, seeds: list[int], epochs: int) -> tuple[pd.DataFrame, dict]:
    data = load_city()
    X_all = data["X_all"]
    X_all_s = data["X_all_s"]
    y_wc = data["y_wc"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    bams = data["bams"]
    hum = data["hum"]
    packs = build_wc_track_packs(bams)

    rows = []
    for seed in seeds:
        set_all_seeds(seed)

        # ----- WC-track baseline -----
        base = train_mlp(
            X_all_s[idx_train], y_wc[idx_train], device=device, seed=seed, epochs=epochs
        )
        pred_b, _ = predict_mlp(base, X_all_s, device=device)
        mw_b = full_metrics(y_wc[idx_test], pred_b[idx_test])
        rows.append(
            {
                "Track": "WC",
                "Seed": seed,
                "Method": "Baseline",
                "n_seed": 0,
                "K": None,
                "n_exp": None,
                "WC_OA": mw_b["OA"],
                "WC_BE": mw_b["BE"],
                "WC_Kappa": mw_b["Kappa"],
                "H_OA": np.nan,
                "H_BE": np.nan,
                "H_n": np.nan,
                "H_agree_WC": np.nan,
            }
        )
        print(f"  [seed={seed}] Baseline WC_OA={mw_b['OA']:.4f} BE={mw_b['BE']}")

        for name, pack in packs.items():
            best = run_pe_wc_pick(
                X_all, X_all_s, y_wc, idx_train, idx_test, pack, device, seed, epochs, name
            )
            if best is None:
                continue
            rows.append(
                {
                    "Track": "WC",
                    "Seed": seed,
                    "Method": name,
                    "n_seed": pack["n_seed"],
                    "K": best["k"],
                    "n_exp": best["n_exp"],
                    "WC_OA": best["m_w"]["OA"],
                    "WC_BE": best["m_w"]["BE"],
                    "WC_Kappa": best["m_w"]["Kappa"],
                    "H_OA": np.nan,
                    "H_BE": np.nan,
                    "H_n": np.nan,
                    "H_agree_WC": np.nan,
                }
            )

        # ----- Dual-track: human hold-out -----
        h_idx = np.arange(len(hum))
        hy = hum["H0"].to_numpy()
        try:
            seed_h, test_h = train_test_split(
                h_idx, test_size=HUMAN_TEST_SIZE, random_state=seed, stratify=hy
            )
        except ValueError:
            seed_h, test_h = train_test_split(
                h_idx, test_size=HUMAN_TEST_SIZE, random_state=seed
            )
        seed_df = hum.iloc[seed_h]
        test_df = hum.iloc[test_h]
        pos_seed = seed_df["pos"].to_numpy(dtype=int)
        lab_h = seed_df["H0"].to_numpy(dtype=np.int64)
        lab_w = seed_df["WC0"].to_numpy(dtype=np.int64)
        pos_htest = test_df["pos"].to_numpy(dtype=int)
        y_htest = test_df["H0"].to_numpy(dtype=np.int64)
        h_agree = float((y_htest == y_wc[pos_htest]).mean())

        # baseline dual metrics
        mh_b = full_metrics(y_htest, pred_b[pos_htest])
        rows.append(
            {
                "Track": "Dual",
                "Seed": seed,
                "Method": "Baseline_dual",
                "n_seed": 0,
                "K": None,
                "n_exp": None,
                "WC_OA": mw_b["OA"],
                "WC_BE": mw_b["BE"],
                "WC_Kappa": mw_b["Kappa"],
                "H_OA": mh_b["OA"],
                "H_BE": mh_b["BE"],
                "H_n": mh_b["n"],
                "H_agree_WC": h_agree,
            }
        )

        for name, lab in [("PE_Human_holdout", lab_h), ("PE_WC_holdout", lab_w)]:
            pack = make_pack(pos_seed, lab)
            # generate candidates; pick by WC (primary), also store human metrics
            used = np.zeros(len(X_all), dtype=bool)
            used[idx_train] = True
            used[pos_seed] = True
            teacher = train_mlp(
                np.concatenate([X_all_s[idx_train], X_all_s[pos_seed]], axis=0),
                np.concatenate([y_wc[idx_train], lab], axis=0),
                device=device,
                seed=seed,
                epochs=epochs,
            )
            _, margin_pool = predict_mlp(teacher, X_all_s, device=device)
            cands = []
            for k in K_LIST:
                exp_idx_all, exp_lab_all = expand_prototypes(X_all, pos_seed, lab, k)
                keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
                exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]
                X_pe = np.concatenate(
                    [X_all_s[idx_train], X_all_s[pos_seed], X_all_s[exp_idx]], axis=0
                )
                y_pe = np.concatenate([y_wc[idx_train], lab, exp_lab], axis=0)
                w_pe = np.concatenate(
                    [
                        np.full(len(idx_train) + len(lab), REAL_WEIGHT, np.float32),
                        np.full(len(exp_lab), SOFT_WEIGHT, np.float32),
                    ]
                )
                st = train_mlp(
                    X_pe, y_pe, w_pe, device=device, seed=seed + 200 + k, epochs=epochs
                )
                pred, _ = predict_mlp(st, X_all_s, device=device)
                mw = full_metrics(y_wc[idx_test], pred[idx_test])
                mh = full_metrics(y_htest, pred[pos_htest])
                cands.append(
                    {"k": k, "n_exp": int(len(exp_idx)), "m_w": mw, "m_h": mh}
                )
                print(
                    f"  [seed={seed}] {name:22s} K={k:<2d} n_seed={len(pos_seed):<3d} "
                    f"exp={len(exp_idx):<3d} WC_OA={mw['OA']:.4f} H_OA={mh['OA']:.4f}"
                )
            best = sorted(cands, key=lambda c: (c["m_w"]["BE"], -c["m_w"]["OA"]))[0]
            rows.append(
                {
                    "Track": "Dual",
                    "Seed": seed,
                    "Method": name,
                    "n_seed": int(len(pos_seed)),
                    "K": best["k"],
                    "n_exp": best["n_exp"],
                    "WC_OA": best["m_w"]["OA"],
                    "WC_BE": best["m_w"]["BE"],
                    "WC_Kappa": best["m_w"]["Kappa"],
                    "H_OA": best["m_h"]["OA"],
                    "H_BE": best["m_h"]["BE"],
                    "H_n": best["m_h"]["n"],
                    "H_agree_WC": h_agree,
                }
            )

    return pd.DataFrame(rows), data["meta"]


def summarize(df: pd.DataFrame, meta: dict) -> tuple[str, pd.DataFrame]:
    rows = []
    for (track, method), g in df.groupby(["Track", "Method"], sort=False):
        base = df[(df.Track == track) & (df.Method.str.startswith("Baseline"))]
        # pair baseline by track
        if track == "WC":
            b = df[(df.Track == "WC") & (df.Method == "Baseline")]
        else:
            b = df[(df.Track == "Dual") & (df.Method == "Baseline_dual")]
        base_wc = float(b.WC_OA.mean())
        base_h = float(b.H_OA.mean()) if b.H_OA.notna().any() else np.nan
        rows.append(
            {
                "Track": track,
                "Method": method,
                "n_seed_mean": float(g.n_seed.mean()),
                "WC_OA_mean": float(g.WC_OA.mean()),
                "WC_OA_std": float(g.WC_OA.std(ddof=0)),
                "dWC_pp": (float(g.WC_OA.mean()) - base_wc) * 100,
                "WC_BE_mean": float(g.WC_BE.mean()),
                "H_OA_mean": float(g.H_OA.mean()) if g.H_OA.notna().any() else np.nan,
                "dH_pp": (
                    (float(g.H_OA.mean()) - base_h) * 100
                    if g.H_OA.notna().any() and np.isfinite(base_h)
                    else np.nan
                ),
                "H_BE_mean": float(g.H_BE.mean()) if g.H_BE.notna().any() else np.nan,
                "n_exp_mean": float(g.n_exp.dropna().mean()) if g.n_exp.notna().any() else np.nan,
            }
        )
    overall = pd.DataFrame(rows)

    def fmt(v, nd=2):
        if v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v))):
            return "—"
        return f"{v:+.{nd}f}" if nd == 2 else f"{v:.{nd}f}"

    lines = [
        "# Nanjing Phase B — Human vs WC seed contrast\n",
        f"- BAMS: n={meta['n_bams']}, high={meta['n_bams_high']}, "
        f"agree/disagree={meta['n_agree_all']}/{meta['n_disagree_all']}\n",
        f"- Dual high pool: n={meta['n_dual_high_pool']}, "
        f"flip_vs_WC={meta['dual_flip_rate']:.1%}, holdout={meta['human_test_frac']}\n",
        "\n## Mean over seeds\n\n",
        overall.to_string(index=False),
        "\n\n## Reading guide\n",
        "- PE_WC_*: same BAMS positions, seed label = WorldCover (method can lift WC-OA?).\n",
        "- PE_Human_*: seed label = human (passport-style).\n",
        "- PE_Agree_* vs PE_Disagree_*: only WC-consistent / conflicting human seeds.\n",
        "- Dual track: 40% high points held out for Human-OA; never used as seeds.\n",
        "- H1 supported if PE_WC lifts WC-OA while PE_Human does not, "
        "and/or PE_Human lifts Human-OA while WC stays flat; "
        "and Disagree hurts WC while Agree helps.\n",
    ]
    return "\n".join(lines), overall


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DATA2 / "nanjing_rootcause_v2" / "phase_b",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [0] if args.quick else args.seeds
    epochs = min(args.epochs, 40) if args.quick else args.epochs
    print("device:", device, "seeds:", seeds, "epochs:", epochs, "out:", out)

    df, meta = run(device=device, seeds=seeds, epochs=epochs)
    md, overall = summarize(df, meta)

    df.to_csv(out / "seed_results.csv", index=False)
    overall.to_csv(out / "overall_summary.csv", index=False)
    (out / "SUMMARY.md").write_text(md, encoding="utf-8")
    (out / "config.json").write_text(
        json.dumps(
            {
                "city": CITY,
                "seeds": seeds,
                "epochs": epochs,
                "out_dir": str(out),
                "k_list": K_LIST,
                "margin_thresh": MARGIN_THRESH,
                **meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + md)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
