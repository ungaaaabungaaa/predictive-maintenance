"""Select on held-out training engines, then refit without using test labels."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import regression_metrics
from .features import build_features, endpoints, split_units, training_targets


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    cap: int = 125
    window: int = 10
    trees: int = 120
    xgboost: bool = False

    def __post_init__(self):
        for name, minimum, maximum in [("seed", 0, 2**32 - 1), ("cap", 1, 1000),
                                       ("window", 1, 100), ("trees", 1, 1000)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer from {minimum} to {maximum}.")


def candidates(config: ExperimentConfig) -> dict:
    models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(VarianceThreshold(), StandardScaler(), Ridge(alpha=10.)),
        "random_forest": RandomForestRegressor(
            n_estimators=config.trees, max_depth=12, min_samples_leaf=5,
            max_features=.7, random_state=config.seed, n_jobs=1,
        ),
    }
    if config.xgboost:
        try:
            from xgboost import XGBRegressor
        except (ImportError, OSError) as exc:
            raise ValueError("XGBoost is unavailable. Install the [xgboost] extra and its platform runtime.") from exc
        models["xgboost"] = XGBRegressor(
            n_estimators=config.trees, max_depth=4, learning_rate=.05,
            subsample=.8, colsample_bytree=.8, reg_lambda=1.,
            objective="reg:squarederror", tree_method="hist", random_state=config.seed, n_jobs=1,
        )
    return models


def fit_and_select(train: pd.DataFrame, config: ExperimentConfig):
    """This function receives no official test inputs or test RUL values."""
    fit_units, validation_units, indices = split_units(train, config.seed)
    features = build_features(train, config.window)
    targets = training_targets(train, config.cap)
    fit_mask = train.unit.isin(fit_units)
    x_fit, y_fit = features.loc[fit_mask], targets[fit_mask]
    x_validation, y_validation = features.loc[indices], targets[indices]
    metrics = {}
    models = candidates(config)
    for name, model in models.items():
        model.fit(x_fit, y_fit)
        predicted = np.clip(model.predict(x_validation), 0, config.cap)
        metrics[name] = regression_metrics(y_validation, predicted)
    selected = min(metrics, key=lambda name: metrics[name]["rmse_cycles"])
    fitted = clone(models[selected]).fit(features, targets)
    selection = {
        "fit_units": fit_units,
        "validation_units": validation_units,
        "validation_snapshots": train.loc[indices, ["unit", "cycle"]].to_dict(orient="records"),
        "validation_metrics": metrics,
        "candidate_parameters": {
            name: {key: str(value) if isinstance(value, float) and not np.isfinite(value) else value
                   for key, value in model.get_params(deep=True).items()
                   if isinstance(value, (bool, int, float, str, type(None)))}
            for name, model in models.items()
        },
        "selected_model": selected,
        "selection_rule": "lowest validation snapshot RMSE against capped RUL; ties use model order",
        "refit_units": sorted(train.unit.unique().tolist()),
        "feature_names": list(features.columns),
    }
    return fitted, selection


def predict_last(model, frame: pd.DataFrame, config: ExperimentConfig) -> np.ndarray:
    features = build_features(frame, config.window).loc[endpoints(frame)]
    return np.clip(model.predict(features), 0, config.cap)
