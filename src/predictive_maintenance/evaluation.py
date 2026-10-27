"""Measured regression errors and explicitly heuristic maintenance labels."""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if (actual.ndim != 1 or actual.shape != predicted.shape or not len(actual)
            or not np.isfinite(actual).all() or not np.isfinite(predicted).all()):
        raise ValueError("Metrics require equal, nonempty, finite one-dimensional arrays.")
    error = predicted - actual
    exponent = np.where(error < 0, -error / 13, error / 10)
    if np.max(exponent) > 700:
        raise ValueError("Prediction errors are too large for the asymmetric score.")
    return {
        "n_observations": len(actual),
        "mae_cycles": float(np.abs(error).mean()),
        "rmse_cycles": float(np.sqrt(np.square(error).mean())),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
        "asymmetric_score_sum": float(np.expm1(exponent).sum()),
    }


def maintenance_bands(predicted: np.ndarray, cap: float = 125) -> pd.DataFrame:
    if not np.isfinite(predicted).all() or not np.isfinite(cap) or cap <= 0:
        raise ValueError("Maintenance labels require finite predictions and a positive cap.")
    return pd.DataFrame({
        "band": np.select([predicted <= 20, predicted <= 50], ["inspect", "plan"], default="monitor"),
        "rul_index_percent": np.clip(predicted / cap * 100, 0, 100),
    })
