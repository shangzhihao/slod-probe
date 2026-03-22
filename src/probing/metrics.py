"""Classification metrics for the SLoD probe."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np


LABELS = ("macro", "meso", "micro")


def labels_to_indices(labels: Iterable[str]) -> np.ndarray:
    """Map string labels into the fixed class order used throughout the probe.

    Labels are mapped to: 'macro' -> 0, 'meso' -> 1, 'micro' -> 2.

    Args:
        labels: An iterable of label strings.

    Returns:
        A NumPy array of integer indices.
    """
    mapping = {label: index for index, label in enumerate(LABELS)}
    return np.array([mapping[label] for label in labels], dtype=np.int64)


def baseline_predictions(train_y: np.ndarray, count: int) -> np.ndarray:
    """Predict the majority train label for every requested output position.

    Args:
        train_y: NumPy array of training label indices.
        count: Number of predictions to generate (matching the test set size).

    Returns:
        A NumPy array filled with the majority label index.
    """
    majority = int(np.bincount(train_y, minlength=len(LABELS)).argmax())
    return np.full(count, majority, dtype=np.int64)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute accuracy, per-class F1, and a confusion matrix.

    Args:
        y_true: Ground truth label indices.
        y_pred: Predicted label indices.

    Returns:
        A dictionary containing macro F1, accuracy, per-class metrics
        (precision, recall, f1, support), and the confusion matrix.
    """
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        cm[true_label, pred_label] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    for index, label in enumerate(LABELS):
        tp = int(cm[index, index])
        fp = int(cm[:, index].sum() - tp)
        fn = int(cm[index, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )
        f1_scores.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[index, :].sum()),
        }

    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def label_counts(labels: Iterable[str]) -> dict[str, int]:
    """Count raw occurrences per label.

    Args:
        labels: An iterable of label strings.

    Returns:
        A dictionary mapping labels to their counts.
    """
    return dict(Counter(labels))
