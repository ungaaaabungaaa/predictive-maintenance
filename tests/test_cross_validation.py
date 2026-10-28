import json

import numpy as np
import pytest
from test_features import trajectories

from predictive_maintenance import cli
from predictive_maintenance.pipeline import ExperimentConfig, cross_validate_engines


def test_group_cv_is_disjoint_exhaustive_and_reproducible():
    train = trajectories(10, 30)
    result = cross_validate_engines(train, ExperimentConfig(trees=3), 3)
    assert result == cross_validate_engines(train, ExperimentConfig(trees=3), 3)
    held_out = []
    for fold in result["folds"]:
        assert set(fold["fit_units"]).isdisjoint(fold["validation_units"])
        assert set(fold["fit_units"]) | set(fold["validation_units"]) == set(range(1, 11))
        held_out.extend(fold["validation_units"])
        assert [row["unit"] for row in fold["validation_snapshots"]] == fold["validation_units"]
        assert all(1 < row["cycle"] < 30 for row in fold["validation_snapshots"])
        assert set(fold["metrics"]) == {"mean_baseline", "ridge", "random_forest"}
    assert sorted(held_out) == list(range(1, 11))
    for metrics in result["pooled_metrics"].values():
        assert metrics["n_observations"] == 10
        assert metrics["rmse_cycles"] >= 0
    assert result["config"]["seed"] == 42


@pytest.mark.parametrize("folds", [1, 11, True, 2.5])
def test_cv_rejects_invalid_folds(folds):
    with pytest.raises(ValueError):
        cross_validate_engines(trajectories(10, 30), ExperimentConfig(trees=3), folds)


def test_cv_cli_reads_training_only_and_preserves_existing_output(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    np.savetxt(source / "train_FD001.txt", trajectories(10, 30).to_numpy())
    output = tmp_path / "cv.json"
    args = ["cross-validate", "--data-dir", str(source), "--output", str(output),
            "--folds", "2", "--trees", "3"]
    assert cli.main(args) == 0
    result = json.loads(output.read_text())
    assert result["training_engines"] == 10
    assert len(result["training_data_sha256"]) == 64
    assert len(result["folds"]) == 2
    original = output.read_bytes()
    assert cli.main(args) == 1
    assert output.read_bytes() == original
    assert not (source / "test_FD001.txt").exists()
    assert not (source / "RUL_FD001.txt").exists()
