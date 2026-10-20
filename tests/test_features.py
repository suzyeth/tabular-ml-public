import pandas as pd

from facies_cv.config import GROUP, LOG_COLUMNS
from facies_cv.features import add_window_features


def _toy() -> pd.DataFrame:
    rows = []
    for well, offset in (("A", 0.0), ("B", 1000.0)):
        for i in range(6):
            rows.append({GROUP: well, "Depth": float(i), **{c: offset + i for c in LOG_COLUMNS}})
    return pd.DataFrame(rows)


def test_window_features_do_not_mutate_input_and_add_expected_columns():
    toy = _toy()
    before = toy.copy()
    out, cols = add_window_features(toy, windows=(3,))
    pd.testing.assert_frame_equal(toy, before)
    assert len(cols) == len(LOG_COLUMNS) * 2
    assert set(cols) <= set(out.columns)


def test_window_features_never_mix_wells():
    toy = _toy()
    out, _ = add_window_features(toy, windows=(3,))
    a = out[out[GROUP] == "A"]
    b = out[out[GROUP] == "B"]
    # Well A's values are 0..5 and B's are 1000..1005; any bleed would show up here.
    assert a["GR_mean3"].between(0, 5).all()
    assert b["GR_mean3"].between(1000, 1005).all()
    # First row of each well has no predecessor, so its difference is 0, not a jump of 1000.
    assert a["GR_diff"].iloc[0] == 0 and b["GR_diff"].iloc[0] == 0
