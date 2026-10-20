import shutil

import pandas as pd
import pytest

from facies_cv.config import DATA_PATH, EXPECTED_COLUMNS, GROUP, PSEUDO_WELLS, TARGET
from facies_cv.data import DataIntegrityError, bed_run_lengths, clean, load_clean, load_raw


def test_load_raw_has_expected_schema_and_size():
    df = load_raw()
    assert list(df.columns[: len(EXPECTED_COLUMNS)]) == EXPECTED_COLUMNS
    assert len(df) == 4149
    assert set(df[TARGET].unique()) == set(range(1, 10))


def test_load_raw_rejects_tampered_file(tmp_path):
    copy = tmp_path / "facies_vectors.csv"
    shutil.copy(DATA_PATH, copy)
    with open(copy, "a", encoding="utf-8") as fh:
        fh.write("1,X,FAKE WELL,1,1,1,1,1,1,1,1\n")
    with pytest.raises(DataIntegrityError, match="Checksum"):
        load_raw(copy)


def test_load_raw_missing_file_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        load_raw(tmp_path / "nope.csv")


def test_clean_drops_missing_pe_and_pseudo_well_without_mutating_input():
    raw = load_raw()
    before = raw.copy()
    out = clean(raw)
    pd.testing.assert_frame_equal(raw, before)  # input untouched
    assert out["PE"].notna().all()
    assert not out[GROUP].isin(PSEUDO_WELLS).any()
    assert out[GROUP].nunique() == 7
    assert len(out) == 3164


def test_clean_sorts_by_well_then_depth():
    df = load_clean()
    for _, well in df.groupby(GROUP):
        assert well["Depth"].is_monotonic_increasing
        # rows of one well are contiguous
        assert well.index.to_series().diff().dropna().eq(1).all()


def test_bed_run_lengths_counts_contiguous_runs():
    toy = pd.DataFrame({GROUP: ["A"] * 6 + ["B"] * 2, TARGET: [1, 1, 2, 2, 2, 1, 1, 1]})
    assert sorted(bed_run_lengths(toy).tolist()) == [1, 2, 2, 3]
