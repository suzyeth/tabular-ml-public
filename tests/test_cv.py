import numpy as np
import pytest

from facies_cv.config import GROUP
from facies_cv.cv import (
    GroupLeakError,
    assert_no_group_overlap,
    make_splits,
    neighbour_copy_predictions,
    shared_groups,
)
from facies_cv.data import load_clean


@pytest.fixture(scope="module")
def wells():
    return load_clean()[GROUP].to_numpy()


def test_group_splits_never_share_a_well(wells):
    splits = make_splits("group", wells)
    assert all(not s for s in shared_groups(splits, wells))


def test_group_splits_hold_out_each_well_exactly_once(wells):
    splits = make_splits("group", wells)
    held_out = [tuple(np.unique(wells[te])) for _, te in splits]
    assert sorted(w for (w,) in held_out) == sorted(np.unique(wells))


@pytest.mark.parametrize("scheme", ["kfold", "group"])
def test_splits_partition_every_row_once(wells, scheme):
    splits = make_splits(scheme, wells)
    test_rows = np.concatenate([te for _, te in splits])
    assert np.array_equal(np.sort(test_rows), np.arange(len(wells)))
    for tr, te in splits:
        assert not set(tr) & set(te)


def test_random_kfold_does_leak_wells(wells):
    """Documents the problem the project is about: every K-fold test fold shares wells with training."""
    splits = make_splits("kfold", wells)
    assert all(len(s) == len(np.unique(wells)) for s in shared_groups(splits, wells))


def test_overlap_check_catches_a_leaky_split():
    groups = np.array(["a", "a", "b", "b"])
    leaky = [(np.array([0, 2]), np.array([1, 3]))]
    with pytest.raises(GroupLeakError):
        assert_no_group_overlap(leaky, groups)


def test_unknown_scheme_and_too_many_splits_raise():
    groups = np.array(["a", "b"])
    with pytest.raises(ValueError):
        make_splits("stratified", groups)
    with pytest.raises(ValueError):
        make_splits("group", groups, n_splits=3)


def test_neighbour_copy_uses_nearest_same_well_training_row():
    groups = np.array(["a", "a", "a", "b"])
    depth = np.array([1.0, 2.0, 10.0, 1.5])
    y = np.array([7, 8, 9, 5])
    splits = [(np.array([0, 2, 3]), np.array([1]))]
    pred = neighbour_copy_predictions(splits, groups, depth, y)
    assert pred[1] == 7  # depth 1.0 is nearer than 10.0; well b is ignored
    assert np.isnan(pred[[0, 2, 3]]).all()  # not in a test fold


def test_neighbour_copy_has_nothing_to_copy_under_group_cv(wells):
    df = load_clean()
    splits = make_splits("group", wells)
    pred = neighbour_copy_predictions(splits, wells, df["Depth"].to_numpy(), df["Facies"].to_numpy())
    assert np.isnan(pred).all()
