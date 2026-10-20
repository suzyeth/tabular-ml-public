"""Scoring functions."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import ADJACENT_FACIES


def adjacent_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    adjacency: Mapping[int, Sequence[int]] = ADJACENT_FACIES,
) -> float:
    """Share of predictions that are the true facies or a geologically adjacent one.

    Adjacent facies grade into each other, so confusing them is a much smaller
    mistake than, say, calling a sandstone a dolomite.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("empty input")
    ok = [t == p or p in adjacency.get(int(t), ()) for t, p in zip(y_true, y_pred)]
    return float(np.mean(ok))


def score(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Accuracy (= micro-F1, the contest metric), macro-F1 and adjacent accuracy."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "adjacent_accuracy": adjacent_accuracy(y_true, y_pred),
    }
