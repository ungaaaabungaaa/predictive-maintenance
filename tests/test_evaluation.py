import numpy as np
import pytest

from predictive_maintenance.evaluation import maintenance_bands, regression_metrics


def test_measured_metrics_and_asymmetric_late_penalty():
    exact = regression_metrics(np.array([10., 20.]), np.array([10., 20.]))
    assert exact == {"n_observations": 2, "mae_cycles": 0., "rmse_cycles": 0., "r2": 1., "asymmetric_score_sum": 0.}
    late = regression_metrics(np.array([10., 20.]), np.array([20., 30.]))
    early = regression_metrics(np.array([10., 20.]), np.array([0., 10.]))
    assert late["mae_cycles"] == late["rmse_cycles"] == 10
    assert late["asymmetric_score_sum"] > early["asymmetric_score_sum"]


def test_heuristic_band_boundaries_and_health_index():
    result = maintenance_bands(np.array([0., 20., 20.1, 50., 50.1, 125.]), cap=125)
    assert result["band"].tolist() == ["inspect", "inspect", "plan", "plan", "monitor", "monitor"]
    assert result.iloc[0].rul_index_percent == 0
    assert result.iloc[-1].rul_index_percent == 100
    assert "probability" not in " ".join(result.columns)


@pytest.mark.parametrize("actual,predicted", [([], []), ([1, 2], [1]), ([1, 2], [1, np.nan])])
def test_metrics_reject_invalid_arrays(actual, predicted):
    with pytest.raises(ValueError):
        regression_metrics(np.array(actual), np.array(predicted))
