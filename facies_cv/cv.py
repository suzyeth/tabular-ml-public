"""Cross-validation splitters and the checks that keep them honest."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.model_selection import GroupKFold, KFold

from .config import RANDOM_STATE

SCHEMES = ("kfold", "group")


class GroupLeakError(AssertionError):
    """A group appears on both sides of a train/test split."""


def make_splits(
    scheme: str,
    groups: np.ndarray,
    n_splits: int | None = None,
    seed: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialise (train_idx, test_idx) pairs for a scheme.

    Both schemes default to n_splits = number of wells so that each fold trains
    on roughly the same amount of data; the only difference is *which* rows end
    up in the test fold. With n_splits = n_wells, "group" is leave-one-well-out.
    """
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    k = n_splits or n_groups
    placeholder = np.zeros(len(groups))
    if scheme == "kfold":
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        splits = list(splitter.split(placeholder))
    elif scheme == "group":
        if k > n_groups:
            raise ValueError(f"n_splits={k} > number of groups={n_groups}")
        splits = list(GroupKFold(n_splits=k).split(placeholder, groups=groups))
        assert_no_group_overlap(splits, groups)
    else:
        raise ValueError(f"Unknown scheme {scheme!r}; expected one of {SCHEMES}")
    return splits


def shared_groups(
    splits: Iterable[tuple[np.ndarray, np.ndarray]], groups: np.ndarray
) -> list[set]:
    """Per fold, the set of groups present in both train and test."""
    groups = np.asarray(groups)
    return [set(groups[tr]) & set(groups[te]) for tr, te in splits]


def assert_no_group_overlap(
    splits: Iterable[tuple[np.ndarray, np.ndarray]], groups: np.ndarray
) -> None:
    for fold, overlap in enumerate(shared_groups(splits, groups)):
        if overlap:
            raise GroupLeakError(f"Fold {fold}: groups in train and test: {sorted(overlap)}")


def neighbour_copy_predictions(
    splits: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    depth: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """A deliberately dumb "model" that exposes the leak.

    For each test row, copy the label of the nearest-in-depth *training* row from
    the same well. It uses no logs at all. Rows with no same-well training row
    (always the case under group CV) get NaN.
    """
    groups, depth, y = np.asarray(groups), np.asarray(depth, float), np.asarray(y)
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in splits:
        for well in np.unique(groups[test_idx]):
            tr = train_idx[groups[train_idx] == well]
            te = test_idx[groups[test_idx] == well]
            if len(tr) == 0:
                continue
            if len(tr) == 1:
                preds[te] = y[tr[0]]
                continue
            order = np.argsort(depth[tr])
            tr_depth, tr_y = depth[tr][order], y[tr][order]
            pos = np.clip(np.searchsorted(tr_depth, depth[te]), 1, len(tr_depth) - 1)
            left_closer = (depth[te] - tr_depth[pos - 1]) <= (tr_depth[pos] - depth[te])
            preds[te] = tr_y[np.where(left_closer, pos - 1, pos)]
    return preds
