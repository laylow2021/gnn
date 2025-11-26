# gnn

Base scaffold on `master` for a graph-focused project. Use this branch for minimal, stable elements; do active work on `develop` and feature branches.

## Getting Started
- Python: 3.11 (see `.venv` for a ready virtual environment scaffold).
- Install dependencies (once listed): `.venv\\Scripts\\python.exe -m pip install -r requirements.txt -r requirements-dev.txt`.
- Run tests: `.venv\\Scripts\\python.exe -m pytest`.

## Branching Model
- `master`: minimal, stable baseline suitable for bootstrapping new work.
- `develop`: integration branch for ongoing work.
- `feature/*`: short-lived branches for specific tasks; merge into `develop` when ready.

## Layout
- `src/gnn/`: package code.
- `tests/`: pytest suites mirroring `src/`.
- `scripts/`: helper CLIs or maintenance scripts.
- `docs/`: design or architecture notes.
- `data/`: small, versioned sample data (keep large/raw data out of Git).
