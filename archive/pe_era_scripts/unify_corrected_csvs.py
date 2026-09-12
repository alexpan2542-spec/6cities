#!/usr/bin/env python3
"""Unify *_top100_corrected.csv files to columns: lon, lat, Class, margin."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

FILES = [
    "Nanchang_top100_corrected.csv",
    "Wuhan_top100_corrected.csv",
    "Changsha_top100_corrected.csv",
    "Hefei_top100_corrected.csv",
]

KEEP_COLS = ["lon", "lat", "Class", "margin"]


def unify(df: pd.DataFrame, name: str) -> pd.DataFrame:
    print(f"\n=== {name} ===")
    print("原始列名:", df.columns.tolist())

    if "Human_Class" in df.columns:
        df = df.copy()
        df["Class"] = df["Human_Class"]
        print("已从 Human_Class 创建 Class")

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要列: {missing}")

    out = df[KEEP_COLS].copy()

    n_missing = int(out["Class"].isna().sum())
    print("shape:", out.shape)
    print("Class 缺失值:", n_missing)
    print("Class value_counts():")
    print(out["Class"].value_counts(dropna=False).sort_index())

    if n_missing > 0:
        raise ValueError(f"{name} 存在 Class 缺失值: {n_missing}")

    return out


def main() -> None:
    for name in FILES:
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        out = unify(df, name)
        out.to_csv(path, index=False)
        print(f"已覆盖保存: {path}")

    print("\n========== 最终检查 ==========")
    for name in FILES:
        path = DATA_DIR / name
        df = pd.read_csv(path)
        print(f"\n文件名: {name}")
        print("columns:", df.columns.tolist())
        print("shape:", df.shape)
        print("Class缺失值数量:", int(df["Class"].isna().sum()))


if __name__ == "__main__":
    main()
