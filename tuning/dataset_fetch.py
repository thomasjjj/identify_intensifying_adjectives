#!/usr/bin/env python3
"""Download, clean, and split the AI vs Human Kaggle dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Tuple
import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

from metrics_utils import safe_makedirs, set_seed


RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")

DATASET_SLUG = "shamimhasan8/ai-vs-human-text-dataset"
CSV_NAME = "ai_vs_human_text.csv"
LABEL_MAP = {"human": 0, "ai": 1}
LABEL_NORMALIZATION = {
    "human-written": "human",
    "human written": "human",
    "human_written": "human",
    "ai-generated": "ai",
    "ai generated": "ai",
    "ai_generated": "ai",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    return parser.parse_args()


def ensure_dirs() -> None:
    for directory in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR):
        safe_makedirs(directory)


def download_csv() -> Path:
    """Download the dataset using kagglehub and copy it into data/raw."""

    download_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    csv_candidates = list(download_path.rglob(CSV_NAME))
    if not csv_candidates:
        raise FileNotFoundError(f"Could not locate {CSV_NAME} inside downloaded archive.")

    target_path = RAW_DIR / CSV_NAME
    shutil.copyfile(csv_candidates[0], target_path)
    return target_path


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean columns, normalize labels, and drop invalid rows."""

    available_cols = [col for col in ["id", "text", "label", "prompt", "model", "date"] if col in df.columns]
    df = df[available_cols].copy()

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    df["text"] = df["text"].astype(str).fillna("").str.strip()
    df = df[df["text"].astype(bool)]

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].replace(LABEL_NORMALIZATION)
    df = df[df["label"].isin(LABEL_MAP)]
    df["y"] = df["label"].map(LABEL_MAP)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df.reset_index(drop=True)


def stratified_splits(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/val/test using stratified sampling."""

    train_val, test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["y"],
        random_state=seed,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.2,
        stratify=train_val["y"],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs()

    print(f"Downloading dataset '{DATASET_SLUG}'...")
    csv_path = download_csv()
    print(f"Dataset copied to {csv_path}")

    print("Loading and cleaning data...")
    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
    df = normalize_dataframe(df)

    class_counts = df["y"].value_counts().to_dict()
    total = len(df)
    formatted_counts = {cls: f"{count} ({count / total:.1%})" for cls, count in class_counts.items()}
    print(f"Total rows: {total}")
    print("Class distribution:", formatted_counts)

    safe_makedirs(INTERIM_DIR)
    interim_path = INTERIM_DIR / "cleaned.csv"
    df.to_csv(interim_path, index=False)
    print(f"Saved cleaned data to {interim_path}")

    train_df, val_df, test_df = stratified_splits(df, args.seed)
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = PROCESSED_DIR / f"{name}.csv"
        split.to_csv(out_path, index=False)
        pos = split["y"].mean()
        print(f"{name.title()} split -> {len(split)} rows | AI ratio: {pos:.2%} (saved to {out_path})")


if __name__ == "__main__":
    main()
