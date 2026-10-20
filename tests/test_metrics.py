import pytest

from facies_cv.config import ADJACENT_FACIES, FACIES_NAMES
from facies_cv.metrics import adjacent_accuracy, score


def test_adjacency_matches_source_including_its_one_asymmetry():
    """The contest table is symmetric except BS(9)->D(7): D does not list BS.

    We keep the published definition so scores stay comparable with the contest,
    and pin the quirk here so nobody "fixes" it silently.
    """
    assert set(ADJACENT_FACIES) == set(FACIES_NAMES)
    one_way = sorted(
        (facies, n)
        for facies, neighbours in ADJACENT_FACIES.items()
        for n in neighbours
        if facies not in ADJACENT_FACIES[n]
    )
    assert one_way == [(9, 7)]
    assert all(f not in ns for f, ns in ADJACENT_FACIES.items())


def test_adjacent_accuracy_counts_exact_and_adjacent_hits():
    y_true = [1, 2, 4, 6, 9]
    y_pred = [1, 3, 9, 8, 1]  # exact, adjacent, wrong, adjacent, wrong
    assert adjacent_accuracy(y_true, y_pred) == pytest.approx(3 / 5)


def test_adjacent_accuracy_validates_input():
    with pytest.raises(ValueError):
        adjacent_accuracy([1, 2], [1])
    with pytest.raises(ValueError):
        adjacent_accuracy([], [])


def test_score_on_perfect_and_constant_predictions():
    y = [1, 1, 2, 3]
    assert score(y, y) == {"accuracy": 1.0, "macro_f1": 1.0, "adjacent_accuracy": 1.0}
    constant = score(y, [1, 1, 1, 1])
    assert constant["accuracy"] == pytest.approx(0.5)
    # macro-F1 punishes ignoring minority classes: F1 is 2/3 for class 1, 0 for 2 and 3
    assert constant["macro_f1"] == pytest.approx((2 / 3) / 3)
    assert constant["adjacent_accuracy"] == pytest.approx(0.75)  # 2 is adjacent to 1
