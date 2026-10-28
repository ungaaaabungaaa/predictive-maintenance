"""A reproducible run records the exact data, allocation, settings and measured results."""

import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .data import COLUMNS, FILE_SHA256, digest, read_offsets, read_trajectories
from .evaluation import maintenance_bands, regression_metrics, sensor_drift
from .features import endpoints
from .pipeline import ExperimentConfig, fit_and_select, predict_last
from .report import write_report


def run_experiment(data_dir: Path, output: Path, config: ExperimentConfig) -> dict:
    """Run once into a new folder. Test RUL never enters model selection or fitting."""
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new folder: {output}")
    train = read_trajectories(data_dir / "train_FD001.txt")
    fitted, selection = fit_and_select(train, config)
    test = read_trajectories(data_dir / "test_FD001.txt")
    actual = read_offsets(data_dir / "RUL_FD001.txt", test.unit.nunique())
    predicted = predict_last(fitted, test, config)
    actual_capped = np.minimum(actual, config.cap)
    hashes = {name: digest(data_dir / name) for name in FILE_SHA256}
    predictions = test.loc[endpoints(test), ["unit", "cycle"]].reset_index(drop=True)
    predictions["actual_rul_cycles"] = actual
    predictions["actual_capped_rul_cycles"] = actual_capped
    predictions["predicted_rul_cycles"] = predicted
    predictions = pd.concat([predictions, maintenance_bands(predicted, config.cap)], axis=1)
    packages = ["numpy", "pandas", "scikit-learn", "matplotlib"]
    if config.xgboost:
        packages.append("xgboost")
    sources = sorted(Path(__file__).parent.glob("*.py"))
    metrics = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "source_sha256": sha256(b"".join(p.name.encode() + p.read_bytes() for p in sources)).hexdigest(),
        "dataset_status": "verified-nasa-fd001" if hashes == FILE_SHA256 else "custom-unverified",
        "data_sha256": hashes,
        "python": platform.python_version(),
        "dependency_versions": {name: version(name) for name in packages},
        "config": asdict(config),
        "counts": {"training_engines": int(train.unit.nunique()), "training_rows": len(train),
                   "test_engines": int(test.unit.nunique()), "test_rows": len(test)},
        "selection": selection,
        "sensor_drift": sensor_drift(train[COLUMNS[5:]], test[COLUMNS[5:]]),
        "test_metrics_uncapped": regression_metrics(actual, predicted),
        "test_metrics_capped": regression_metrics(actual_capped, predicted),
        "heuristic_bands": {"inspect_at_or_below_cycles": 20, "plan_at_or_below_cycles": 50,
                            "meaning": "Illustrative thresholds, not calibrated failure probabilities"},
    }
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    write_report(output, metrics, predictions)
    return metrics
