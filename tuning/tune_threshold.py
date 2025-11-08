#!/usr/bin/env python3
"""Interactive and automated probability threshold tuner."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from compare_texts import TextIntensificationComparator
from metrics_utils import safe_makedirs


PROCESSED_DIR = Path("data/processed")
SWEEP_PATH = Path("artifacts/metrics/threshold_sweep_val.csv")
CHOSEN_PATH = Path("artifacts/metrics/chosen_threshold.json")
DEFAULT_AUTO_METRIC = "f1"
METRIC_ALIASES = {
    "f1": ["f1", "f1_macro", "f1_weighted"],
    "youden_j": ["youden_j"],
    "mcc": ["mcc"],
}


def ensure_y_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "y" in df.columns:
        return df
    mapping = {"human": 0, "ai": 1}
    if "label" not in df.columns:
        df["label"] = ""
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["y"] = df["label"].map(mapping)
    df = df[df["y"].isin([0, 1])]
    return df


def probability_from_analysis(analysis: Dict[str, float]) -> float:
    noun_rate = analysis.get("noun_intensification_rate", 0.0)
    adj_rate = analysis.get("adj_intensification_rate", 0.0)
    combined_score = noun_rate * 2 + adj_rate
    return float(np.clip(combined_score / 100.0, 0.0, 1.0))


def compute_probabilities(df: pd.DataFrame, comparator: TextIntensificationComparator, desc: str) -> np.ndarray:
    probs: List[float] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        text = str(row.get("text", ""))
        label = str(row.get("label", ""))
        analysis = comparator.analyze_text(text, label)
        probs.append(probability_from_analysis(analysis))
    return np.array(probs, dtype=float)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.logical_and(y_pred == 1, y_true == 1).sum())
    tn = int(np.logical_and(y_pred == 0, y_true == 0).sum())
    fp = int(np.logical_and(y_pred == 1, y_true == 0).sum())
    fn = int(np.logical_and(y_pred == 0, y_true == 1).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    numerator = (tp * tn) - (fp * fn)
    denominator = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = numerator / denominator if denominator else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2
    youden_j = recall - (fp / (fp + tn) if (fp + tn) else 0.0)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "youden_j": youden_j,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def confusion_matrix_raw(metrics: Dict[str, float]) -> np.ndarray:
    return np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])


def confusion_matrix_normalized(cm: np.ndarray) -> np.ndarray:
    norm = cm.astype(float)
    for i in range(norm.shape[0]):
        total = norm[i].sum()
        norm[i] = norm[i] / total if total else 0
    return norm


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split at {path}. Run tuning.dataset_fetch first.")
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    return ensure_y_column(df)


def load_sweep() -> Optional[pd.DataFrame]:
    if not SWEEP_PATH.exists():
        return None
    return pd.read_csv(SWEEP_PATH)


def show_grid(df: pd.DataFrame, metric: str, label: str) -> None:
    subset = df.sort_values(by=metric, ascending=False).head(10)
    print(f"Top {len(subset)} thresholds by {label}:")
    for _, row in subset.iterrows():
        print(f"  t={row['threshold']:.2f} -> {row[metric]:.4f}")


def maybe_compute_sweep(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    existing = load_sweep()
    if existing is not None:
        return existing

    print("No threshold sweep found; computing quickly for reference...")
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 3)
    records = []
    y_true = df["y"].to_numpy()
    for t in thresholds:
        records.append(classification_metrics(y_true, probs, float(t)))
    sweep_df = pd.DataFrame(records)
    safe_makedirs(SWEEP_PATH.parent)
    sweep_df.to_csv(SWEEP_PATH, index=False)
    return sweep_df


def print_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(f"{name} metrics @ threshold {metrics['threshold']:.3f}:")
    print(
        f"  Accuracy={metrics['accuracy']:.3f} | Precision={metrics['precision']:.3f} | "
        f"Recall={metrics['recall']:.3f} | F1={metrics['f1']:.3f}"
    )
    print(
        f"  Balanced Acc={metrics['balanced_accuracy']:.3f} | MCC={metrics['mcc']:.3f} | Youden's J={metrics['youden_j']:.3f}"
    )
    cm = confusion_matrix_raw(metrics)
    cm_norm = confusion_matrix_normalized(cm)
    print("  Confusion matrix (raw):")
    print(f"    [[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]")
    print("  Confusion matrix (normalized by true label):")
    print(f"    [[{cm_norm[0,0]:.3f}, {cm_norm[0,1]:.3f}], [{cm_norm[1,0]:.3f}, {cm_norm[1,1]:.3f}]]")


def save_threshold(threshold: float) -> None:
    payload = {
        "threshold": threshold,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "source": "auto" if getattr(save_threshold, "_auto_mode", False) else "analyst",
    }
    safe_makedirs(CHOSEN_PATH.parent)
    with CHOSEN_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved threshold to {CHOSEN_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto", action="store_true", help="Run in non-interactive auto-tuning mode.")
    parser.add_argument(
        "--metric",
        choices=list(METRIC_ALIASES.keys()),
        default=DEFAULT_AUTO_METRIC,
        help="Metric to maximize when --auto is provided.",
    )
    parser.add_argument(
        "--apply-test",
        action="store_true",
        help="When --auto, also score the test split at the chosen threshold.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist the auto-selected threshold to artifacts/metrics.",
    )
    return parser.parse_args()


def resolve_metric_column(df: pd.DataFrame, metric: str) -> str:
    for name in METRIC_ALIASES.get(metric, [metric]):
        if name in df.columns:
            return name
    raise ValueError(f"Metric '{metric}' unavailable in sweep results.")


def auto_mode(metric: str, apply_test: bool, save_choice: bool) -> None:
    comparator = TextIntensificationComparator()
    val_df = load_split("val")
    val_probs = compute_probabilities(val_df, comparator, "Scoring val")
    sweep_df = maybe_compute_sweep(val_df, val_probs)

    metric_column = resolve_metric_column(sweep_df, metric)

    best_row = sweep_df.sort_values(by=metric_column, ascending=False).head(1)
    if best_row.empty:
        raise ValueError("Threshold sweep produced no rows to evaluate.")
    threshold = float(best_row["threshold"].iloc[0])
    metric_value = float(best_row[metric_column].iloc[0])

    print(f"[AUTO] Selecting threshold {threshold:.3f} maximizing {metric_column}={metric_value:.4f}")
    val_metrics = classification_metrics(val_df["y"].to_numpy(), val_probs, threshold)
    print_metrics("VAL", val_metrics)

    test_metrics = None
    if apply_test:
        test_df = load_split("test")
        test_probs = compute_probabilities(test_df, comparator, "Scoring test")
        test_metrics = classification_metrics(test_df["y"].to_numpy(), test_probs, threshold)
        print_metrics("TEST", test_metrics)

    if save_choice:
        setattr(save_threshold, "_auto_mode", True)
        save_threshold(threshold)
        setattr(save_threshold, "_auto_mode", False)


def interactive_mode() -> None:
    comparator = TextIntensificationComparator()
    val_df = load_split("val")
    test_df = load_split("test")

    val_probs = compute_probabilities(val_df, comparator, "Scoring val")
    test_probs: Optional[np.ndarray] = None

    sweep_df = maybe_compute_sweep(val_df, val_probs)

    print("\nInteractive threshold tuner ready. Press Ctrl+C to exit any time.")

    while True:
        try:
            user_input = input("Enter a threshold 0-1 (or 'grid' to show top suggestions, or 'quit'): ").strip()
        except KeyboardInterrupt:
            print("\nExiting tuner.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "grid":
            show_grid(sweep_df, "f1_macro" if "f1_macro" in sweep_df.columns else "f1", "F1")
            show_grid(sweep_df, "youden_j", "Youden's J")
            continue

        try:
            threshold = float(user_input)
        except ValueError:
            print("Please enter a numeric threshold, 'grid', or 'quit'.")
            continue

        if not 0 <= threshold <= 1:
            print("Threshold must be between 0 and 1.")
            continue

        val_metrics = classification_metrics(val_df["y"].to_numpy(), val_probs, threshold)
        print_metrics("VAL", val_metrics)

        try:
            apply_test = input("Apply to TEST? (y/n): ").strip().lower()
        except KeyboardInterrupt:
            print("\nExiting tuner.")
            break
        if apply_test in {"y", "yes"}:
            if test_probs is None:
                test_probs = compute_probabilities(test_df, comparator, "Scoring test")
            test_metrics = classification_metrics(test_df["y"].to_numpy(), test_probs, threshold)
            print_metrics("TEST", test_metrics)

        try:
            save_choice = input("Save chosen threshold? (y/n): ").strip().lower()
        except KeyboardInterrupt:
            print("\nExiting tuner.")
            break
        if save_choice in {"y", "yes"}:
            save_threshold(threshold)


if __name__ == "__main__":
    args = parse_args()
    if args.auto:
        auto_mode(args.metric, args.apply_test, args.save)
    else:
        interactive_mode()
