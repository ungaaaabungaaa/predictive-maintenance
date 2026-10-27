# Verification record

Local checks on 27 October 2026 used Python 3.12.14 on macOS ARM64 and the versions in `requirements-benchmark.txt`.

- Ruff passed on `src` and `tests`.
- All 35 pytest tests passed, including optional XGBoost. Branch-enabled total coverage was 94% (313 statements, 72 branches; 15 uncovered statements).
- Both source distribution and wheel built successfully. The wheel was installed into a temporary target and imported from that target. Its version command, complete synthetic train/select/evaluate/report flow, and verified FD001 reuse passed.
- The real CLI downloader fetched the official NASA-linked archive into a fresh temporary folder and verified the three expected hashes.
- The unchanged fixed experiment was repeated after final source/provenance edits. It selected the same Random Forest and reproduced test RMSE 20.208668433569287 uncapped / 19.080376729623165 capped, with 100 endpoints each. No model/feature/threshold changes were chosen from test results.
- Generated scientific figures were visually inspected; they show actual run outputs. Notebook code cells parsed successfully; the notebook itself was not executed through Jupyter.
- `git diff --check` passed. No raw data, private inputs, model binaries, or pending generated cover were added to Git.

Local checks do not establish CI success. CI status is recorded on the pull request and must pass before merge. Hardware/field operation is outside this simulation benchmark and has not been tested.
