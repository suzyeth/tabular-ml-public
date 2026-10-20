"""Out-of-fold evaluation for every (feature set, model, CV scheme) combination."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score

from .config import DEPTH, GROUP, TARGET
from .cv import make_splits, neighbour_copy_predictions
from .metrics import score
from .models import ModelFactory


@dataclass(frozen=True)
class FoldResult:
    oof_pred: np.ndarray
    fold_accuracy: list[float]
    fitted: list[ClassifierMixin] = field(repr=False)
    test_indices: list[np.ndarray] = field(repr=False)


def out_of_fold(
    factory: ModelFactory,
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> FoldResult:
    """Fit a fresh model per fold; every row is predicted by a model that never saw it."""
    oof = np.empty(len(y), dtype=y.dtype)
    fold_acc, fitted, tests = [], [], []
    for train_idx, test_idx in splits:
        model = factory()
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx])
        oof[test_idx] = pred
        fold_acc.append(float(accuracy_score(y[test_idx], pred)))
        fitted.append(model)
        tests.append(test_idx)
    return FoldResult(oof, fold_acc, fitted, tests)


def run_grid(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    factories: dict[str, ModelFactory],
    schemes: tuple[str, ...] = ("kfold", "group"),
    skip: set[tuple[str, str]] = frozenset(),
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], FoldResult]]:
    """Return a long results table plus the raw fold results for later analysis.

    `skip` holds (feature_set, model) pairs not worth running (e.g. the majority
    baseline is identical for every feature set).
    """
    y = df[TARGET].to_numpy()
    groups = df[GROUP].to_numpy()
    splits = {s: make_splits(s, groups) for s in schemes}
    rows, raw = [], {}
    for fs_name, cols in feature_sets.items():
        X = df[cols]
        for model_name, factory in factories.items():
            if (fs_name, model_name) in skip:
                continue
            for scheme in schemes:
                res = out_of_fold(factory, X, y, splits[scheme])
                raw[(fs_name, model_name, scheme)] = res
                rows.append({
                    "features": fs_name,
                    "model": model_name,
                    "cv": scheme,
                    **score(y, res.oof_pred),
                    "fold_acc_min": min(res.fold_accuracy),
                    "fold_acc_max": max(res.fold_accuracy),
                })
    return pd.DataFrame(rows), raw


def leak_probe(df: pd.DataFrame) -> pd.DataFrame:
    """Score the copy-the-nearest-neighbour "model" under each CV scheme."""
    y = df[TARGET].to_numpy()
    groups, depth = df[GROUP].to_numpy(), df[DEPTH].to_numpy()
    rows = []
    for scheme in ("kfold", "group"):
        pred = neighbour_copy_predictions(make_splits(scheme, groups), groups, depth, y)
        covered = ~np.isnan(pred)
        rows.append({
            "cv": scheme,
            "rows_with_same_well_neighbour_in_train": float(covered.mean()),
            "accuracy_on_those_rows": (
                float((pred[covered] == y[covered]).mean()) if covered.any() else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def pivot_gap(results: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    """Wide table: K-fold vs group score and the inflation between them."""
    wide = results.pivot_table(index=["features", "model"], columns="cv", values=metric, sort=False)
    wide = wide.rename(columns={"kfold": f"kfold_{metric}", "group": f"group_{metric}"})
    return wide.assign(inflation=wide[f"kfold_{metric}"] - wide[f"group_{metric}"]).reset_index()
