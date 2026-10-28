import numpy as np
import pandas as pd
import pytest

from predictive_maintenance.evaluation import sensor_drift


def test_drift_reports_shift_and_constant_reference_without_infinities():
    reference = pd.DataFrame({"sensor_1": [0., 2.], "sensor_2": [3., 3.]})
    observed = pd.DataFrame({"sensor_1": [2., 4.], "sensor_2": [4., 4.]})
    report = sensor_drift(reference, observed)
    assert report["reference_rows"] == report["observed_rows"] == 2
    assert report["sensors"]["sensor_1"]["standardized_mean_shift"] == 2.
    assert report["sensors"]["sensor_2"] == {
        "reference_mean": 3., "observed_mean": 4.,
        "standardized_mean_shift": None, "constant_reference_changed": True,
    }
    assert sensor_drift(reference, reference)["sensors"]["sensor_2"]["constant_reference_changed"] is False


@pytest.mark.parametrize("observed", [pd.DataFrame(), pd.DataFrame({"other": [1.]}),
                                     pd.DataFrame({"sensor_1": [np.nan]})])
def test_drift_rejects_empty_mismatched_or_nonfinite_inputs(observed):
    with pytest.raises(ValueError):
        sensor_drift(pd.DataFrame({"sensor_1": [0., 1.]}), observed)
