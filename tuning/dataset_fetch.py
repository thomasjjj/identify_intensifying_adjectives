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

PRIMARY_DATASET_SLUG = "shamimhasan8/ai-vs-human-text-dataset"
PRIMARY_CSV_NAME = "ai_vs_human_text.csv"
SECONDARY_DATASET_SLUG = "shanegerami/ai-vs-human-text"
SECONDARY_CSV_NAME = "AI_Human.csv"
PRIMARY_SOURCE = "ai_vs_human_text"
SECONDARY_SOURCE = "ai_vs_human_text_large"
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
    parser.add_argument(
        "--max-secondary-rows",
        type=int,
        default=50000,
        help="Maximum rows to sample from the large AI_vs_Human dataset (0 = use all rows).",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    for directory in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR):
        safe_makedirs(directory)


def download_csv(dataset_slug: str, csv_name: str) -> Path:
    """Download a dataset using kagglehub and copy it into data/raw."""

    download_path = Path(kagglehub.dataset_download(dataset_slug))
    csv_candidates = list(download_path.rglob(csv_name))
    if not csv_candidates:
        raise FileNotFoundError(f"Could not locate {csv_name} inside downloaded archive.")

    target_path = RAW_DIR / csv_name
    shutil.copyfile(csv_candidates[0], target_path)
    return target_path


def normalize_primary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean columns, normalize labels, and drop invalid rows."""

    available_cols = [
        col for col in ["id", "text", "label", "prompt", "model", "date", "data_source"] if col in df.columns
    ]
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

    df["data_source"] = PRIMARY_SOURCE

    return df.reset_index(drop=True)


def load_secondary_dataframe(csv_path: Path, max_rows: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
    if "text" not in df.columns or "generated" not in df.columns:
        raise ValueError(f"Secondary dataset at {csv_path} must include 'text' and 'generated' columns.")

    df["text"] = df["text"].astype(str).fillna("").str.strip()
    df = df[df["text"].astype(bool)]
    df["generated"] = pd.to_numeric(df["generated"], errors="coerce")
    df = df[df["generated"].isin([0, 1])]
    df["label"] = df["generated"].map({0: "human", 1: "ai"})
    df["y"] = df["generated"].astype(int)
    df["prompt"] = ""
    df["model"] = ""
    df["date"] = pd.NaT
    df["data_source"] = SECONDARY_SOURCE
    df["id"] = df.get("id", pd.Series(dtype=object))
    df["id"] = df["id"].where(df["id"].notna(), df.index.map(lambda idx: f"{SECONDARY_SOURCE}_{idx}"))
    df = df.drop(columns=["generated"])

    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

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

    print(f"Downloading dataset '{PRIMARY_DATASET_SLUG}'...")
    primary_path = download_csv(PRIMARY_DATASET_SLUG, PRIMARY_CSV_NAME)
    print(f"Primary dataset copied to {primary_path}")

    print(f"Downloading dataset '{SECONDARY_DATASET_SLUG}'...")
    secondary_path = download_csv(SECONDARY_DATASET_SLUG, SECONDARY_CSV_NAME)
    print(f"Secondary dataset copied to {secondary_path}")

    print("Loading and cleaning data...")
    primary_df = pd.read_csv(primary_path, encoding="utf-8", encoding_errors="replace")
    primary_df = normalize_primary_dataframe(primary_df)

    secondary_df = load_secondary_dataframe(secondary_path, args.max_secondary_rows, args.seed)

    df = pd.concat([primary_df, secondary_df], ignore_index=True, sort=False)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    deduped = before_dedup - len(df)
    if deduped:
        print(f"Removed {deduped} duplicate texts after merging datasets.")
    source_counts = df.groupby("data_source")["y"].agg(["count", "mean"]).to_dict("index")

    class_counts = df["y"].value_counts().to_dict()
    total = len(df)
    formatted_counts = {cls: f"{count} ({count / total:.1%})" for cls, count in class_counts.items()}
    print(f"Total rows: {total}")
    print("Class distribution:", formatted_counts)
    for source, stats in source_counts.items():
        ratio = stats["mean"]
        print(f"  - {source}: {int(stats['count'])} rows | AI ratio {ratio:.2%}")

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
