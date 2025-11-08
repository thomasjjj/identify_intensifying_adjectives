#!/usr/bin/env python3
"""Evaluate TextIntensificationComparator on the Kaggle AI vs Human dataset."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from compare_texts import TextIntensificationComparator
from metrics_utils import plot_cm, plot_pr, plot_roc, safe_makedirs, save_json, set_seed


PROCESSED_DIR = Path("data/processed")
ARTIFACTS_PLOTS = Path("artifacts/plots")
ARTIFACTS_METRICS = Path("artifacts/metrics")
THRESHOLD_SWEEP_PATH = ARTIFACTS_METRICS / "threshold_sweep_val.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--subset",
        type=str,
        default="",
        help="Optional substring to filter rows by prompt or model.",
    )
    return parser.parse_args()


def ensure_y_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "y" in df.columns:
        return df

    if "label" not in df.columns:
        raise ValueError("Expected 'label' column to infer numeric targets.")

    labels = df["label"].astype(str).str.lower().str.strip()
    mapping = {"human": 0, "ai": 1}
    df["y"] = labels.map(mapping)
    df = df[df["y"].isin([0, 1])]
    return df


def apply_subset(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if not subset:
        return df

    subset_lower = subset.lower()
    mask = pd.Series(False, index=df.index)
    for column in ("prompt", "model"):
        if column in df.columns:
            mask |= df[column].astype(str).str.lower().str.contains(subset_lower, na=False)

    filtered = df[mask]
    if filtered.empty:
        print(f"[WARN] Subset '{subset}' produced no rows; falling back to full split.")
        return df
    print(f"Applying subset '{subset}' -> {len(filtered)} rows (from {len(df)}).")
    return filtered


def probability_from_analysis(analysis: Dict[str, float]) -> float:
    noun_rate = analysis.get("noun_intensification_rate", 0.0)
    adj_rate = analysis.get("adj_intensification_rate", 0.0)
    combined_score = noun_rate * 2 + adj_rate
    return float(np.clip(combined_score / 100.0, 0.0, 1.0))


def compute_probabilities(
    df: pd.DataFrame,
    comparator: TextIntensificationComparator,
    desc: str,
) -> np.ndarray:
    probs: List[float] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        text = str(row.get("text", ""))
        label = str(row.get("label", ""))
        try:
            analysis = comparator.analyze_text(text, label)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Analysis failed ({exc}); defaulting score to 0.")
            analysis = {"noun_intensification_rate": 0.0, "adj_intensification_rate": 0.0}
        probs.append(probability_from_analysis(analysis))
    return np.array(probs, dtype=float)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    tp = np.logical_and(y_pred == 1, y_true == 1).sum()
    fn = np.logical_and(y_pred == 0, y_true == 1).sum()
    fp = np.logical_and(y_pred == 1, y_true == 0).sum()
    tn = np.logical_and(y_pred == 0, y_true == 0).sum()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    metrics["youden_j"] = tpr - fpr
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["tn"] = int(tn)
    metrics["fn"] = int(fn)
    return metrics


def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 3)
    records = []
    for threshold in thresholds:
        metrics = classification_metrics(y_true, y_prob, float(threshold))
        records.append(metrics)
    df = pd.DataFrame(records)
    safe_makedirs(THRESHOLD_SWEEP_PATH.parent)
    df.to_csv(THRESHOLD_SWEEP_PATH, index=False)
    return df


def best_thresholds(sweep_df: pd.DataFrame, key: str, top_k: int = 3) -> List[Tuple[float, float]]:
    ordered = sweep_df.sort_values(by=key, ascending=False)
    return list(zip(ordered["threshold"].head(top_k), ordered[key].head(top_k)))


def load_split(name: str, subset: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected split at {path}. Run tuning.dataset_fetch first.")
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    df = ensure_y_column(df)
    df = apply_subset(df, subset)
    return df


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    comparator = TextIntensificationComparator()

    splits = {name: load_split(name, args.subset) for name in ("train", "val", "test")}
    probabilities = {
        name: compute_probabilities(df, comparator, f"Scoring {name}")
        for name, df in splits.items()
    }

    y_val = splits["val"]["y"].to_numpy()
    val_probs = probabilities["val"]
    y_test = splits["test"]["y"].to_numpy()
    test_probs = probabilities["test"]

    sweep_df = sweep_thresholds(y_val, val_probs)
    print(f"Threshold sweep saved to {THRESHOLD_SWEEP_PATH}")

    criteria = {
        "F1_max": ("f1_macro", best_thresholds(sweep_df, "f1_macro")),
        "YoudenJ_max": ("youden_j", best_thresholds(sweep_df, "youden_j")),
        "MCC_max": ("mcc", best_thresholds(sweep_df, "mcc")),
    }
    for label, (_, values) in criteria.items():
        print(f"{label} top thresholds:")
        for thr, score in values:
            print(f"  t={thr:.2f} -> {score:.4f}")

    roc_val = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else float("nan")
    roc_test = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else float("nan")
    pr_val = average_precision_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else float("nan")
    pr_test = average_precision_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else float("nan")

    plot_roc(y_val, val_probs, ARTIFACTS_PLOTS / "roc_val.png", "ROC Curve (Validation)")
    plot_roc(y_test, test_probs, ARTIFACTS_PLOTS / "roc_test.png", "ROC Curve (Test)")
    plot_pr(y_val, val_probs, ARTIFACTS_PLOTS / "pr_val.png", "PR Curve (Validation)")
    plot_pr(y_test, test_probs, ARTIFACTS_PLOTS / "pr_test.png", "PR Curve (Test)")

    threshold_summary = {
        label: {
            "threshold": float(values[0][0]),
            "metric": metric_key,
            "score": float(values[0][1]),
        }
        for label, (metric_key, values) in criteria.items()
        if values
    }

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "subset": args.subset,
        "splits": {
            name: {
                "rows": int(len(df)),
                "ai_ratio": float(df["y"].mean()) if len(df) else None,
            }
            for name, df in splits.items()
        },
        "auc": {
            "val": {"roc": roc_val, "pr": pr_val},
            "test": {"roc": roc_test, "pr": pr_test},
        },
        "thresholds": threshold_summary,
    }

    # Evaluate selected thresholds on the test split.
    for label, (metric_key, values) in criteria.items():
        if not values:
            continue
        threshold = float(values[0][0])
        test_metrics = classification_metrics(y_test, test_probs, threshold)
        out_path = ARTIFACTS_METRICS / f"test_eval_{label}.json"
        save_json(out_path, test_metrics)
        summary.setdefault("test_thresholds", {})[label] = test_metrics

        y_pred = (test_probs >= threshold).astype(int)
        plot_cm(
            y_test,
            y_pred,
            ARTIFACTS_PLOTS / f"cm_test_{label}_raw.png",
            f"Test Confusion Matrix ({label}, raw)",
            normalize=False,
        )
        plot_cm(
            y_test,
            y_pred,
            ARTIFACTS_PLOTS / f"cm_test_{label}_norm.png",
            f"Test Confusion Matrix ({label}, normalized)",
            normalize=True,
        )

    save_json(ARTIFACTS_METRICS / "report.json", summary)
    print(f"Summary report saved to {ARTIFACTS_METRICS / 'report.json'}")

    print("Key metrics:")
    print(f"  Validation ROC-AUC: {roc_val:.3f} | PR-AUC: {pr_val:.3f}")
    print(f"  Test ROC-AUC: {roc_test:.3f} | PR-AUC: {pr_test:.3f}")


if __name__ == "__main__":
    main()
