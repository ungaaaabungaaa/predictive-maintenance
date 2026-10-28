"""Explicit download and offline experiment commands."""

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from urllib.error import URLError

from . import __version__
from .data import digest, download_fd001, read_trajectories
from .experiment import run_experiment
from .pipeline import ExperimentConfig, cross_validate_engines


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
    cv = commands.add_parser("cross-validate", help="Compare candidates on training-engine folds only.")
    cv.add_argument("--data-dir", type=Path, default=Path("data/fd001"))
    cv.add_argument("--output", type=Path, required=True, help="New JSON file; never overwrite.")
    cv.add_argument("--folds", type=int, default=5)
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--cap", type=int, default=125)
    cv.add_argument("--window", type=int, default=10)
    cv.add_argument("--trees", type=int, default=120)
    cv.add_argument("--xgboost", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            print(json.dumps(download_fd001(args.data_dir), indent=2))
        elif args.command == "cross-validate":
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError("Choose a new cross-validation output file.")
            config = ExperimentConfig(args.seed, args.cap, args.window, args.trees, args.xgboost)
            source = args.data_dir / "train_FD001.txt"
            result = cross_validate_engines(read_trajectories(source), config, args.folds)
            result["training_data_sha256"] = digest(source)
            result["package_version"] = __version__
            modules = sorted(Path(__file__).parent.glob("*.py"))
            result["source_sha256"] = sha256(b"".join(p.name.encode() + p.read_bytes() for p in modules)).hexdigest()
            text = json.dumps(result, indent=2, allow_nan=False) + "\n"
            with args.output.open("x", encoding="utf-8", newline="\n") as output:
                output.write(text)
            print(f"Training-engine cross-validation: {len(result['folds'])} folds; {result['training_engines']} engines.")
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
