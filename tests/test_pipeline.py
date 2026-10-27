import numpy as np
import pytest
from test_features import trajectories

from predictive_maintenance.pipeline import ExperimentConfig, fit_and_select, predict_last


def test_model_selection_uses_heldout_engines_and_baseline():
    train = trajectories(10, 30)
    config = ExperimentConfig(trees=5)
    fitted, selection = fit_and_select(train, config)
    assert set(selection["fit_units"]).isdisjoint(selection["validation_units"])
    assert len(selection["validation_units"]) == 2
    assert set(selection["validation_metrics"]) == {"mean_baseline", "ridge", "random_forest"}
    expected = min(selection["validation_metrics"],
                   key=lambda name: selection["validation_metrics"][name]["rmse_cycles"])
    assert selection["selected_model"] == expected
    assert selection["candidate_parameters"]["random_forest"]["max_depth"] == 12
    prediction = predict_last(fitted, train, config)
    assert prediction.shape == (10,)
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 125)).all()
    _, repeated = fit_and_select(train, config)
    assert selection == repeated


@pytest.mark.parametrize("options", [{"cap": 0}, {"trees": 0}, {"window": 101}, {"seed": -1}])
def test_bad_experiment_options_rejected(options):
    with pytest.raises(ValueError):
        ExperimentConfig(**options)


def test_optional_xgboost_candidate():
    pytest.importorskip("xgboost")
    _, selection = fit_and_select(trajectories(10, 30), ExperimentConfig(trees=5, xgboost=True))
    assert "xgboost" in selection["validation_metrics"]
    assert selection["candidate_parameters"]["xgboost"]["missing"] == "nan"
