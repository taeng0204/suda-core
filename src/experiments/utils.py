"""Shared utilities for SUDA experiment benchmarks."""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def compute_gmean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute G-mean (geometric mean of TPR and TNR)."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] < 2 or cm.shape[1] < 2:
        return 0.0
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    return float(np.sqrt(tpr * tnr))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1 score for the positive (attack) class."""
    return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))


def compute_phase_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, phase_boundaries: list[int]
) -> list[dict]:
    """Compute per-phase metrics for drift stream evaluation."""
    boundaries = [0] + phase_boundaries + [len(y_true)]
    phase_metrics = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        yt = y_true[start:end]
        yp = y_pred[start:end]
        if len(yt) == 0:
            continue
        phase_metrics.append({
            "phase": i,
            "start": start,
            "end": end,
            "gmean": compute_gmean(yt, yp),
            "f1": compute_f1(yt, yp),
            "balanced_acc": float(balanced_accuracy_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.0,
            "attack_recall": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
            "benign_recall": float(recall_score(yt, yp, pos_label=0, zero_division=0)),
            "attack_ratio": float(yt.mean()),
            "n_samples": len(yt),
        })
    return phase_metrics
