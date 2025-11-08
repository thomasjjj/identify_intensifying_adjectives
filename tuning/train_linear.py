#!/usr/bin/env python3
"""Train and evaluate a simple linear (logistic) classifier on exported feature tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


ARTIFACTS_DIR = Path("artifacts/metrics")
FEATURE_TEMPLATE = "features_{split}.csv"
LABEL_MAP = {"human": 0, "ai": 1}
NON_FEATURE_COLUMNS = {
    "label",
    "text",
    "intensifying_words",
    "intensified_pairs",
    "original_label",
    "source_id",
    "data_source",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the classifier.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Feature splits to evaluate (expects CSVs under artifacts/metrics).",
    )
    parser.add_argument(
        "--penalty",
        choices=["l2", "l1"],
        default="l2",
        help="Regularization penalty for logistic regression.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="When >0, perform StratifiedKFold cross-validation with the specified number of folds.",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default="",
        help="Optional path to persist the trained pipeline (e.g., artifacts/models/logreg.joblib).",
    )
    parser.add_argument(
        "--show-top-k",
        type=int,
        default=10,
        help="Show the top-K positive/negative feature weights.",
    )
    return parser.parse_args()


def load_features(split: str) -> pd.DataFrame:
    path = ARTIFACTS_DIR / FEATURE_TEMPLATE.format(split=split)
    if not path.exists():
        raise FileNotFoundError(f"Missing features at {path}. Run tuning.evaluate_dataset first.")
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"'label' column missing in {path}.")
    df["y"] = df["label"].map(LABEL_MAP)
    if df["y"].isna().any():
        raise ValueError(f"Unexpected label values in {path}.")
    return df


def prepare_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, List[str]]:
    feature_df = df.drop(columns=list(NON_FEATURE_COLUMNS), errors="ignore").copy()
    feature_df = feature_df.drop(columns=["label", "y"], errors="ignore")

    for column in feature_df.columns:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")

    feature_df = feature_df.fillna(0.0)
    feature_names = feature_df.columns.tolist()
    X = feature_df.to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    return X, y, feature_names


def build_pipeline(penalty: str, c_value: float, seed: int) -> Pipeline:
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    clf = LogisticRegression(
        penalty=penalty,
        C=c_value,
        solver=solver,
        max_iter=200,
        random_state=seed,
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", clf),
        ]
    )
    return pipeline


def train_model(X: np.ndarray, y: np.ndarray, penalty: str, c_value: float, seed: int) -> Pipeline:
    pipeline = build_pipeline(penalty, c_value, seed)
    pipeline.fit(X, y)
    return pipeline


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    penalty: str,
    c_value: float,
    seed: int,
    folds: int,
) -> None:
    if folds <= 1:
        return
    print(f"\nRunning {folds}-fold StratifiedKFold cross-validation...")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    pipeline = build_pipeline(penalty, c_value, seed)
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=skf,
        scoring=scoring,
        n_jobs=None,
        return_train_score=False,
    )
    for metric, scores in results.items():
        if not metric.startswith("test_"):
            continue
        name = metric.replace("test_", "")
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"  {name:>9}: {mean:.4f} ± {std:.4f}")


def show_top_features(model: Pipeline, feature_names: List[str], top_k: int) -> None:
    if top_k <= 0:
        return
    clf = model.named_steps["clf"]
    weights = clf.coef_[0]
    pairs = list(zip(feature_names, weights))
    top_positive = sorted(pairs, key=lambda item: item[1], reverse=True)[:top_k]
    top_negative = sorted(pairs, key=lambda item: item[1])[:top_k]
    print(f"\nTop {top_k} features pushing toward AI label:")
    for name, weight in top_positive:
        print(f"  + {name:<30} {weight:+.4f}")
    print(f"\nTop {top_k} features pushing toward human label:")
    for name, weight in top_negative:
        print(f"  - {name:<30} {weight:+.4f}")


def evaluate_model(model: Pipeline, X: np.ndarray, y: np.ndarray, split: str) -> Dict[str, float]:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }
    if len(np.unique(y)) > 1:
        metrics["roc_auc"] = roc_auc_score(y, prob)
    else:
        metrics["roc_auc"] = float("nan")
    print(f"\n[{split}] metrics:")
    for key, value in metrics.items():
        print(f"  {key:>9}: {value:.4f}")
    return metrics


def main() -> None:
    args = parse_args()
    train_df = load_features("train")
    X_train, y_train, feature_names = prepare_matrix(train_df)
    print(f"Loaded train split with {X_train.shape[0]} rows and {X_train.shape[1]} features.")

    if args.cv_folds and args.cv_folds > 1:
        run_cross_validation(
            X_train,
            y_train,
            penalty=args.penalty,
            c_value=args.c,
            seed=args.seed,
            folds=args.cv_folds,
        )

    model = train_model(X_train, y_train, penalty=args.penalty, c_value=args.c, seed=args.seed)
    evaluate_model(model, X_train, y_train, "train")
    show_top_features(model, feature_names, args.show_top_k)

    if args.save_model:
        model_path = Path(args.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_names": feature_names}, model_path)
        print(f"\nSaved trained pipeline to {model_path}")

    for split in args.splits:
        if split == "train":
            continue
        df = load_features(split)
        X_split, y_split, _ = prepare_matrix(df)
        evaluate_model(model, X_split, y_split, split)


if __name__ == "__main__":
    main()
