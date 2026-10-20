"""Feature importance and error analysis on held-out wells."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from .config import ADJACENT_FACIES, FACIES_NAMES, GROUP, NON_MARINE_FACIES, RANDOM_STATE, TARGET
from .evaluate import FoldResult
from .metrics import adjacent_accuracy

LABELS = sorted(FACIES_NAMES)


def feature_importance(
    df: pd.DataFrame, features: list[str], result: FoldResult, n_repeats: int = 5
) -> pd.DataFrame:
    """Two views of importance for the group-CV LightGBM models.

    * gain share: how much each feature was used to split, averaged over folds
      (cheap, but measured on training data and biased toward continuous features);
    * permutation drop: accuracy lost on the *held-out well* when the feature is
      shuffled (slower, but measured where it matters).
    """
    X, y, wells = df[features], df[TARGET].to_numpy(), df[GROUP].to_numpy()
    if any(len(np.unique(wells[idx])) != 1 for idx in result.test_indices):
        raise ValueError("feature_importance expects leave-one-well-out results")
    gains, perms = [], []
    for model, test_idx in zip(result.fitted, result.test_indices):
        gain = model.booster_.feature_importance(importance_type="gain")
        gains.append(gain / gain.sum())
        perm = permutation_importance(
            model, X.iloc[test_idx], y[test_idx],
            scoring="accuracy", n_repeats=n_repeats, random_state=RANDOM_STATE,
        )
        perms.append(perm.importances_mean)
    gains, perms = np.vstack(gains), np.vstack(perms)
    table = pd.DataFrame({
        "feature": features,
        "gain_share": gains.mean(axis=0),
        "perm_acc_drop": perms.mean(axis=0),
        "perm_acc_drop_std_across_wells": perms.std(axis=0),
    })
    return table.sort_values("perm_acc_drop", ascending=False).reset_index(drop=True)


def per_well(df: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Accuracy per held-out well for each model's out-of-fold predictions."""
    y = df[TARGET].to_numpy()
    rows = []
    for well, idx in df.groupby(GROUP, sort=True).indices.items():
        row = {"well": well, "rows": len(idx), "facies_present": int(len(np.unique(y[idx])))}
        for name, pred in preds.items():
            row[f"{name}_acc"] = float((pred[idx] == y[idx]).mean())
        first = next(iter(preds.values()))
        row["adjacent_acc"] = adjacent_accuracy(y[idx], first[idx])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"{next(iter(preds))}_acc").reset_index(drop=True)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    names = [FACIES_NAMES[k] for k in LABELS]
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return pd.DataFrame(cm, index=[f"true {n}" for n in names], columns=names)


def per_class(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    support = pd.Series(y_true).value_counts().reindex(LABELS, fill_value=0)
    return pd.DataFrame({
        "facies": [FACIES_NAMES[k] for k in LABELS],
        "support": support.to_numpy(),
        "recall": recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0),
        "f1": f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0),
    })


def error_summary(y_true: np.ndarray, y_pred: np.ndarray, top: int = 5) -> dict:
    wrong = y_true != y_pred
    adjacent = np.array([p in ADJACENT_FACIES[int(t)] for t, p in zip(y_true, y_pred)])
    crosses = np.isin(y_true, NON_MARINE_FACIES) != np.isin(y_pred, NON_MARINE_FACIES)
    pairs = (
        pd.DataFrame({"true": y_true[wrong], "pred": y_pred[wrong]})
        .value_counts()
        .head(top)
    )
    return {
        "error_rate": float(wrong.mean()),
        "share_of_errors_that_are_adjacent": float(adjacent[wrong].mean()) if wrong.any() else 0.0,
        "share_of_errors_crossing_marine_boundary": (
            float(crosses[wrong].mean()) if wrong.any() else 0.0
        ),
        "top_confusions": [
            {"true": FACIES_NAMES[t], "pred": FACIES_NAMES[p], "count": int(c)}
            for (t, p), c in pairs.items()
        ],
    }
