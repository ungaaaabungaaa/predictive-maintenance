"""Explicit download and offline experiment commands."""

import argparse
import json
import sys
from pathlib import Path
from urllib.error import URLError

from . import __version__
from .data import download_fd001
from .experiment import run_experiment
from .pipeline import ExperimentConfig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="C-MAPSS FD001 remaining-useful-life experiment.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download and verify NASA-linked FD001 data.")
    download.add_argument("--data-dir", type=Path, default=Path("data/fd001"))
    run = commands.add_parser("run", help="Train, select on held-out engines, and report test results.")
    run.add_argument("--data-dir", type=Path, default=Path("data/fd001"))
    run.add_argument("--output", type=Path, default=Path("artifacts/fd001"))
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--cap", type=int, default=125)
    run.add_argument("--window", type=int, default=10)
    run.add_argument("--trees", type=int, default=120)
    run.add_argument("--xgboost", action="store_true", help="Also compare optional XGBoost.")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            print(json.dumps(download_fd001(args.data_dir), indent=2))
        else:
            config = ExperimentConfig(args.seed, args.cap, args.window, args.trees, args.xgboost)
            result = run_experiment(args.data_dir, args.output, config)
            print(f"Data: {result['dataset_status']}")
            print(f"Selected: {result['selection']['selected_model']}")
            print(f"Test endpoint RMSE (uncapped): {result['test_metrics_uncapped']['rmse_cycles']:.3f} cycles")
            print(f"Report: {args.output / 'report.html'}")
        return 0
    except (ValueError, OSError, URLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
