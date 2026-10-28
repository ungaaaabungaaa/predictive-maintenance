"""Actual-run figures and an offline HTML summary; no generated product imagery."""

from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from .evaluation import maintenance_bands


def write_report(output: Path, metrics: dict, predictions) -> None:
    actual = predictions.actual_rul_cycles.to_numpy()
    predicted = predictions.predicted_rul_cycles.to_numpy()
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        limit = max(float(actual.max()), float(predicted.max()), metrics["config"]["cap"]) + 5
        axes[0].scatter(actual, predicted, color="#272727", alpha=.65, s=23)
        axes[0].plot([0, limit], [0, limit], color="#777777", linestyle="--", linewidth=1)
        axes[0].set(xlabel="Actual RUL (cycles, uncapped)", ylabel="Predicted RUL (cycles)",
                    title="Test endpoints", xlim=(0, limit), ylim=(0, limit))
        axes[1].scatter(actual, predicted - actual, color="#272727", alpha=.65, s=23)
        axes[1].axhline(0, color="#777777", linestyle="--", linewidth=1)
        axes[1].set(xlabel="Actual RUL (cycles, uncapped)", ylabel="Prediction − actual (cycles)",
                    title="Positive error means a late estimate")
        fig.suptitle(f"FD001 · {metrics['selection']['selected_model']} · {metrics['dataset_status']}")
        fig.savefig(output / "rul-predictions.png", dpi=170)
        plt.close(fig)

        labels = ["inspect", "plan", "monitor"]
        actual_bands = maintenance_bands(actual, metrics["config"]["cap"])["band"]
        counts = confusion_matrix(actual_bands, predictions["band"], labels=labels)
        fig, ax = plt.subplots(figsize=(5.8, 5), layout="constrained")
        ax.imshow(counts, cmap="Greys", vmin=0)
        for row, column in np.ndindex(counts.shape):
            ax.text(column, row, str(counts[row, column]), ha="center", va="center",
                    color="white" if counts[row, column] > counts.max() / 2 else "black")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=labels, yticklabels=labels,
               xlabel="Predicted heuristic band", ylabel="Actual RUL threshold band",
               title="Heuristic labels · 20 / 50 cycle cutoffs")
        fig.savefig(output / "maintenance-bands.png", dpi=170)
        plt.close(fig)

    rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{value['rmse_cycles']:.2f}</td>"
        f"<td>{value['mae_cycles']:.2f}</td></tr>"
        for name, value in metrics["selection"]["validation_metrics"].items()
    )
    drift_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{value['reference_mean']:.3f}</td>"
        f"<td>{value['observed_mean']:.3f}</td><td>"
        + (f"{value['standardized_mean_shift']:.3f}"
           if value["standardized_mean_shift"] is not None else "constant reference")
        + "</td></tr>" for name, value in metrics["sensor_drift"]["sensors"].items()
    )
    test = metrics["test_metrics_uncapped"]
    html = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>FD001 maintenance experiment</title>
<style>body{{font:16px/1.6 system-ui;max-width:960px;margin:40px auto;padding:0 20px;color:#222}}
img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}code{{overflow-wrap:anywhere}}</style>
<main><h1>Predictive Maintenance</h1><p>C-MAPSS FD001 · {escape(metrics['dataset_status'])}</p>
<p>Selected on {len(metrics['selection']['validation_units'])} held-out training engines:
<strong>{escape(metrics['selection']['selected_model'])}</strong>.
Refit on all {metrics['counts']['training_engines']} training engines before test evaluation.</p>
<h2>Validation, capped RUL</h2><table><tr><th>Model</th><th>RMSE (cycles)</th><th>MAE (cycles)</th></tr>
{rows}</table><h2>Test endpoints, uncapped RUL</h2>
<p>{metrics['counts']['test_engines']} engines · RMSE {test['rmse_cycles']:.2f} cycles ·
MAE {test['mae_cycles']:.2f} cycles.</p>
<img src="rul-predictions.png" alt="Actual versus predicted remaining cycles and signed errors">
<img src="maintenance-bands.png" alt="Counts of predicted versus actual threshold bands">
<h2>Sensor distribution diagnostics</h2>
<p>All training rows versus all observed test rows; standardized shifts use training-only
population standard deviations. Constant references have no standardized score;
metrics.json records whether their observed values changed. Different cycle mixtures can
produce shifts. These are descriptive comparisons, not fault alarms or significance tests.</p>
<table><tr><th>Sensor</th><th>Training mean</th><th>Test mean</th><th>Mean shift (training SD)</th></tr>
{drift_rows}</table>
<h2>Limits</h2><p>Simulated engine data. RUL is measured in operating cycles, not hours.
The {metrics['config']['cap']}-cycle target cap is a modeling choice. Bands and the RUL index are heuristic labels,
not calibrated failure probabilities. This experiment is not an operational safety system.</p>
<p>See <a href="metrics.json">complete metrics and provenance</a> and
<a href="predictions.csv">per-engine predictions</a>.</p></main></html>
"""
    (output / "report.html").write_text(html, encoding="utf-8")
