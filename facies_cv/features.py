"""Depth-window features, computed strictly within each well."""
from __future__ import annotations

import pandas as pd

from .config import GROUP, LOG_COLUMNS, WINDOW_SIZES


def add_window_features(
    df: pd.DataFrame,
    logs: list[str] = LOG_COLUMNS,
    windows: tuple[int, ...] = WINDOW_SIZES,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (new frame, new column names) with per-well depth context.

    For every log: the first difference down the hole and centred rolling means.
    Only the *inputs* are smoothed (never the label) and only within a well, so
    these features do not move information between wells. They still make
    neighbouring rows look even more alike, which matters for random K-fold.
    Expects rows sorted by well and depth (see data.clean).
    """
    new_cols: dict[str, pd.Series] = {}
    grouped = df.groupby(GROUP, sort=False)
    for col in logs:
        new_cols[f"{col}_diff"] = grouped[col].diff().fillna(0.0)
        for w in windows:
            new_cols[f"{col}_mean{w}"] = grouped[col].transform(
                lambda s, w=w: s.rolling(w, center=True, min_periods=1).mean()
            )
    added = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, added], axis=1), list(new_cols)
