"""Load, verify and clean the SEG 2016 facies data."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from .config import (
    DATA_PATH,
    DATA_SHA256,
    DATA_URL,
    DEPTH,
    EXPECTED_COLUMNS,
    GROUP,
    PSEUDO_WELLS,
    TARGET,
)


class DataIntegrityError(ValueError):
    """Raised when the data file does not match what the analysis expects."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(path: Path = DATA_PATH, url: str = DATA_URL) -> Path:
    """Fetch the CSV from the contest repository (only needed if data/ is missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)  # noqa: S310 - fixed https URL
    return path


def load_raw(path: Path = DATA_PATH, expected_sha256: str | None = DATA_SHA256) -> pd.DataFrame:
    """Read the raw CSV, checking the checksum and schema before anything else."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m facies_cv.data` to download it."
        )
    if expected_sha256 is not None:
        actual = sha256_of(path)
        if actual != expected_sha256:
            raise DataIntegrityError(
                f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual}"
            )
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise DataIntegrityError(f"Missing columns: {missing}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame ready for modelling.

    * drops rows with no PE log (two whole wells have none, so imputing it would
      mean inventing an entire log for those wells);
    * drops the synthetic "Recruit F9" pseudo-well;
    * sorts by well then depth so depth-window features are well defined.
    """
    out = df.dropna(subset=["PE"])
    out = out[~out[GROUP].isin(PSEUDO_WELLS)]
    out = out.sort_values([GROUP, DEPTH], kind="mergesort").reset_index(drop=True)
    if out[TARGET].isna().any():
        raise DataIntegrityError("Target column contains missing values")
    return out


def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


def bed_run_lengths(df: pd.DataFrame) -> pd.Series:
    """Length (in samples) of each contiguous run of one facies within a well.

    Long runs are why random K-fold leaks: a test sample's depth neighbours,
    0.5 ft away, are almost always in the training fold with the same label.
    """
    lengths = []
    for _, well in df.groupby(GROUP, sort=False):
        labels = well[TARGET].to_numpy()
        run_id = (labels[1:] != labels[:-1]).cumsum()
        run_id = pd.Series([0, *run_id])
        lengths.extend(run_id.value_counts(sort=False).tolist())
    return pd.Series(lengths, name="run_length")


def summarise(df: pd.DataFrame) -> dict:
    runs = bed_run_lengths(df)
    return {
        "rows": int(len(df)),
        "wells": int(df[GROUP].nunique()),
        "rows_per_well_min": int(df.groupby(GROUP).size().min()),
        "rows_per_well_max": int(df.groupby(GROUP).size().max()),
        "classes": int(df[TARGET].nunique()),
        "majority_class_share": float(df[TARGET].value_counts(normalize=True).iloc[0]),
        "median_bed_run_samples": float(runs.median()),
        "mean_bed_run_samples": float(runs.mean()),
    }


if __name__ == "__main__":
    print(f"Downloading {DATA_URL} -> {DATA_PATH}")
    download()
    print("sha256:", sha256_of(DATA_PATH))
