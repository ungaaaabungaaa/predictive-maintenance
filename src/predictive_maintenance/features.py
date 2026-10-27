"""Past-and-current sensor features and leakage-resistant engine allocation."""

import numpy as np
import pandas as pd

from .data import COLUMNS


def build_features(frame: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Use current signals, trailing means, and trailing deltas. No unit ID."""
    if not isinstance(window, int) or not 1 <= window <= 100:
        raise ValueError("Feature window must be an integer from 1 to 100.")
    signals = COLUMNS[5:]
    grouped = frame.groupby("unit", sort=False)
    result = frame[["cycle", *COLUMNS[2:]]].astype(float).copy()
    means = grouped[signals].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    previous = grouped[signals].shift(window - 1)
    first = grouped[signals].transform("first")
    deltas = frame[signals] - previous.fillna(first)
    return pd.concat([result, means.add_suffix("_mean"), deltas.add_suffix("_delta")], axis=1)


def training_targets(frame: pd.DataFrame, cap: float = 125) -> np.ndarray:
    """Capped cycles until the final (failure) observation in each training unit."""
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("RUL cap must be a finite positive number.")
    return np.minimum(frame.groupby("unit").cycle.transform("max") - frame.cycle, cap).to_numpy()


def split_units(frame: pd.DataFrame, seed: int = 42) -> tuple[list[int], list[int], list[int]]:
    """Hold out 20% of engines, one seeded 30–90% lifetime snapshot per engine."""
    units = sorted(frame.unit.unique().tolist())
    if len(units) < 5:
        raise ValueError("At least five training engines are needed for a held-out split.")
    rng = np.random.default_rng(seed)
    valid = sorted(rng.choice(units, max(1, int(np.ceil(len(units) * .2))), replace=False).tolist())
    fit = sorted(set(units) - set(valid))
    indices = []
    for unit in valid:
        group = frame[frame.unit == unit]
        if len(group) < 3:
            raise ValueError("Validation engines must contain at least three cycles.")
        position = int(np.clip(np.floor(len(group) * rng.uniform(.3, .9)), 1, len(group) - 2))
        indices.append(int(group.index[position]))
    return fit, valid, indices


def endpoints(frame: pd.DataFrame) -> np.ndarray:
    """Index of each unit's last observed cycle, in ascending unit order."""
    return frame.groupby("unit", sort=True).tail(1).index.to_numpy()
