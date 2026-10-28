import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_features import trajectories

from predictive_maintenance.experiment import run_experiment
from predictive_maintenance.pipeline import ExperimentConfig


def fixture_data(path: Path):
    path.mkdir()
    train = trajectories(10, 30)
    test = trajectories(5, 15)
    np.savetxt(path / "train_FD001.txt", train.to_numpy())
    np.savetxt(path / "test_FD001.txt", test.to_numpy())
    np.savetxt(path / "RUL_FD001.txt", [10, 15, 20, 25, 30])


def test_full_experiment_writes_measured_report_and_refuses_overwrite(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    fixture_data(source)
    result = run_experiment(source, output, ExperimentConfig(trees=5))
    assert result["dataset_status"] == "custom-unverified"
    assert result["counts"] == {"training_engines": 10, "training_rows": 300,
                               "test_engines": 5, "test_rows": 75}
    assert result["sensor_drift"]["reference_rows"] == 300
    assert result["sensor_drift"]["observed_rows"] == 75
    assert len(result["sensor_drift"]["sensors"]) == 21
    assert result["test_metrics_uncapped"]["rmse_cycles"] >= 0
    assert {"metrics.json", "predictions.csv", "report.html", "rul-predictions.png",
            "maintenance-bands.png"}.issubset({p.name for p in output.iterdir()})
    assert json.loads((output / "metrics.json").read_text()) == result
    html = (output / "report.html").read_text()
    assert "custom-unverified" in html
    assert "not calibrated failure probabilities" in html
    with pytest.raises(FileExistsError):
        run_experiment(source, output, ExperimentConfig(trees=5))


def test_cli_help_version_and_controlled_input_error(tmp_path):
    run = lambda args: subprocess.run([sys.executable, "-m", "predictive_maintenance", *args],
                                      capture_output=True, text=True, check=False)
    assert "download" in run(["--help"]).stdout
    assert run(["--version"]).stdout.strip() == "0.1.0"
    invalid = run(["run", "--data-dir", str(tmp_path / "absent"),
                   "--output", str(tmp_path / "out")])
    assert invalid.returncode == 1
    assert "Error:" in invalid.stderr and "Traceback" not in invalid.stderr
    assert run(["run", "--trees", "0"]).returncode == 1
