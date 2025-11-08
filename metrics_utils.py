#!/usr/bin/env python3
"""Shared helpers for dataset validation scripts."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def set_seed(seed: int = 42) -> None:
    """Seed python and numpy RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)


def safe_makedirs(path: os.PathLike | str) -> None:
    """Create directories recursively if missing."""

    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: os.PathLike | str, obj: Any) -> None:
    """Persist JSON to disk with helpful defaults."""

    target = Path(path)
    safe_makedirs(target.parent)
    with target.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def plot_roc(
    y_true: Iterable[float],
    y_prob: Iterable[float],
    out_path: os.PathLike | str,
    title: str,
) -> float:
    """Plot ROC curve and return the computed AUC."""

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = np.trapz(tpr, fpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = Path(out_path)
    safe_makedirs(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)

    return auc_score


def plot_pr(
    y_true: Iterable[float],
    y_prob: Iterable[float],
    out_path: os.PathLike | str,
    title: str,
) -> float:
    """Plot precision-recall curve and return area under curve."""

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_score = np.trapz(precision, recall)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"Area = {auc_score:.3f}")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()

    out_path = Path(out_path)
    safe_makedirs(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)

    return auc_score


def plot_cm(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    out_path: os.PathLike | str,
    title: str,
    normalize: bool = False,
) -> None:
    """Plot confusion matrix heatmap."""

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            cm = np.divide(cm, row_sums, where=row_sums != 0)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Human", "AI"])
    ax.set_yticklabels(["Human", "AI"])

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = format(cm[i, j], fmt)
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, value, ha="center", va="center", color=color)

    fig.tight_layout()
    out_path = Path(out_path)
    safe_makedirs(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)


__all__ = [
    "plot_cm",
    "plot_pr",
    "plot_roc",
    "safe_makedirs",
    "save_json",
    "set_seed",
]
