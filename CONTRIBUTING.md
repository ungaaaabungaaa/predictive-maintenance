# Contributing

Use Python 3.12 or newer in a virtual environment. Install `.[dev,xgboost]`, then run Ruff and all pytest tests. Tests use small synthetic trajectories. Do not commit raw NASA data, model binaries, credentials, or private equipment readings.

Keep model changes separate from protocol changes. Add behavior tests for data leakage, feature availability, readers, and metrics. Save seeds, engine allocations, cap, package versions, and data hashes. Use a new output folder. Label custom and synthetic inputs.

Select improvements on training/validation data. Do not tune against published test results and call that an independent test. State comparison limits. Match the existing style, keep changes focused, and disclose AI assistance where required. Do not represent generated work as physical testing.
