"""End-to-end smoke test of run.py with a small model (about 10 s)."""
import pytest

import run


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("results")


@pytest.fixture(scope="module")
def outputs(out_dir):
    return run.run(out_dir, n_estimators=30)


def test_run_writes_all_tables(outputs, out_dir):
    for name in ("results", "gap_macro_f1", "importance", "per_well", "confusion"):
        assert (out_dir / f"{name}.csv").exists()
    assert (out_dir / "results.md").read_text(encoding="utf-8").startswith("# Results")


def test_majority_baseline_equals_majority_share(outputs):
    res = outputs["results"]
    acc = res.query("model == 'majority' and cv == 'group'")["accuracy"].item()
    assert acc == pytest.approx(outputs["summary"]["majority_class_share"])


def test_models_beat_majority_even_under_group_cv(outputs):
    group = outputs["results"].query("cv == 'group' and features == 'base'").set_index("model")
    assert group.loc["logistic", "macro_f1"] > group.loc["majority", "macro_f1"] + 0.2
    assert group.loc["lightgbm", "macro_f1"] > group.loc["majority", "macro_f1"] + 0.2


def test_random_kfold_inflates_lightgbm(outputs):
    gap = outputs["gap_macro_f1"].set_index(["features", "model"])
    assert gap.loc[("base", "lightgbm"), "inflation"] > 0.1
    assert gap.loc[("base", "majority"), "inflation"] == pytest.approx(0.0)


def test_leak_probe_only_works_with_random_kfold(outputs):
    probe = outputs["leak_probe"].set_index("cv")
    assert probe.loc["kfold", "rows_with_same_well_neighbour_in_train"] == 1.0
    assert probe.loc["group", "rows_with_same_well_neighbour_in_train"] == 0.0
