import numpy as np
import pandas as pd
import pytest

from predictive_maintenance.data import COLUMNS
from predictive_maintenance.features import build_features, endpoints, split_units, training_targets


def trajectories(units=5, cycles=10):
    return pd.DataFrame([[u, c, *([float(c + u)] * 24)]
                         for u in range(1, units + 1)
                         for c in range(1, cycles + 1)], columns=COLUMNS)


def test_features_use_only_current_and_past_readings():
    frame = trajectories()
    features = build_features(frame, window=4)
    prefix = frame[(frame.unit == 1) & (frame.cycle <= 5)].copy()
    pd.testing.assert_frame_equal(features.iloc[:5], build_features(prefix, window=4))
    assert "unit" not in features and "rul" not in features
    assert features.iloc[10].sensor_1_mean == 3
    assert features.iloc[0].sensor_1_delta == 0
    assert np.isfinite(features.to_numpy()).all()


def test_rul_uses_each_training_engines_failure_and_explicit_cap():
    result = training_targets(trajectories(2, 10), cap=4)
    np.testing.assert_array_equal(result[:10], [4, 4, 4, 4, 4, 4, 3, 2, 1, 0])
    np.testing.assert_array_equal(result[:10], result[10:])


def test_engine_split_and_truncation_are_repeatable_and_disjoint():
    frame = trajectories(10, 20)
    fit, valid, indices = split_units(frame, seed=42)
    assert len(fit) == 8 and len(valid) == 2
    assert not set(fit) & set(valid)
    assert set(fit) | set(valid) == set(range(1, 11))
    assert frame.loc[indices, "unit"].tolist() == valid
    assert frame.loc[indices, "cycle"].between(6, 18).all()
    assert split_units(frame, seed=42) == (fit, valid, indices)
    assert endpoints(frame).tolist() == [19, 39, 59, 79, 99, 119, 139, 159, 179, 199]


@pytest.mark.parametrize("cap", [0, -1, float("nan")])
def test_invalid_cap(cap):
    with pytest.raises(ValueError):
        training_targets(trajectories(), cap)


def test_small_fleet_and_bad_window_rejected():
    with pytest.raises(ValueError):
        split_units(trajectories(2))
    with pytest.raises(ValueError):
        build_features(trajectories(), 0)

