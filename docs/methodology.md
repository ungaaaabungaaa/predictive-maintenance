# Experiment method

FD001 training sequences run to failure. Official test sequences stop earlier; their RUL vector gives remaining cycles after each last observation, in ascending unit order. IDs are local to each split: train unit 1 and test unit 1 are different engines.

## Fixed protocol

Use seed 42, a 10-observation trailing window, and a 125-cycle target cap. A NumPy generator selects 20 of 100 training engines for validation without replacement. The same generator selects one snapshot at roughly 30–90% of each validation engine's lifetime. Exact engine/cycle allocations are saved. This is an evaluation design choice, not a model of real inspection timing.

Fit candidates on all cycles of the other 80 engines. Features are cycle, three settings, 21 current sensors, 21 trailing sensor means, and 21 trailing sensor deltas: 67 features. A delta subtracts the reading nine cycles earlier for a full 10-cycle window. At early cycles it subtracts that engine's first reading. Features use only the past and current observations within one engine. This is a time availability rule, not causal inference. Unit identity and RUL are excluded.

All fit rows have equal weight, so longer trajectories contribute more observations. Validation/test engines have equal metric weight because each contributes one endpoint. This difference is explicit and can be studied in a future controlled experiment.

## Models

- Mean baseline: mean of capped fit-row RUL.
- Ridge: remove zero-variance features, standardize using fit engines only, Ridge alpha 10.
- Random Forest: 120 trees, maximum depth 12, minimum leaf size 5, feature fraction 0.7, seed 42, one worker.
- Optional XGBoost: 120 trees, depth 4, learning rate 0.05, row/column sampling 0.8, lambda 1, squared-error objective, histogram tree method, seed 42, one worker.

Clip predictions to 0–125. Select the lowest validation snapshot RMSE against capped RUL; a tie uses model insertion order. Refit a fresh copy on all 100 training engines. Test inputs and RUL are read only after fitting. Parameters, feature names, allocation, package versions, code digest, and data hashes are in `metrics.json`.

There is one split and seed, with no hyperparameter search. Twenty validation observations can give unstable rankings. This is not a state-of-the-art claim. Repeating the unchanged protocol to verify reproducibility is distinct from tuning against test results.

## Metrics

MAE and RMSE use cycle errors. R² requires at least two observations. The asymmetric score sums `exp(-error/13)-1` for negative error and `exp(error/10)-1` otherwise, where `error = prediction - actual`. Late estimates receive a larger penalty. It is a sum over the stated observation count, not a probability or percentage accuracy.

Selection uses capped validation RUL. Test results report both original and capped RUL. Predictions stay clipped in both comparisons. Compare results with the same cap and protocol. Cycles are not hours or guaranteed safe operating intervals.

The maintenance confusion matrix uses the same illustrative 20/50-cycle thresholds on actual and predicted RUL. Thresholds were not optimized on the test set. Neither these bands nor the linear RUL index measures calibrated physical health or failure probability.

## Sources

- [NASA dataset and attribution](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
- [scikit-learn: data leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html).
- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).
