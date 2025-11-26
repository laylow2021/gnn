# Repository Guidelines

This repository is currently empty; use this guide to keep early work consistent as you add code and tools.

## Project Structure & Module Organization
- Place all production code in `src/`; mirror package/module names to the feature being implemented (e.g., `src/graph/loader.py`).
- Keep tests in `tests/` with the same relative structure as `src/`.
- Store one-off utilities in `scripts/`; keep long-running assets or sample data in `data/` (small, versionable files only).
- Add `docs/` for design notes and deeper explanations that do not belong in code comments.

## Build, Test, and Development Commands
- Create a virtual environment: `python -m venv .venv && .venv\\Scripts\\activate` (PowerShell: `. .venv\\Scripts\\Activate.ps1`).
- Install dependencies (once `requirements*.txt` exists): `pip install -r requirements.txt -r requirements-dev.txt`.
- Run tests: `pytest` from the repo root.
- Optional local checks (add when tools are configured): `ruff check` for lint, `black .` for formatting, `mypy src` for typing.

## Coding Style & Naming Conventions
- Use Python 3.10+ features; prefer type hints everywhere and `typing`-friendly APIs.
- Formatting: target `black` defaults (88 columns) and `isort` imports; keep indentation at 4 spaces.
- Naming: `snake_case` for modules/functions/variables, `PascalCase` for classes, `CONSTANT_CASE` for constants.
- Keep functions small and pure when possible; extract side effects into explicit helper functions.

## Testing Guidelines
- Testing framework: `pytest`.
- Name files `test_*.py` and functions `test_<behavior>`; group by feature rather than by object.
- Aim for meaningful coverage (≥85%) and include regression tests for every bug fix.
- Use fixtures for setup; avoid sharing mutable global state between tests.

## Commit & Pull Request Guidelines
- Commit messages: imperative mood (“Add loader for node features”); optionally prefix with Conventional Commit types (`feat:`, `fix:`, `chore:`) for clarity.
- Keep commits focused; avoid mixing refactors with behavior changes without explanation.
- Pull requests should describe the change, note risks, list tests run (`pytest`, lint/format), and link issues or design notes in `docs/` where relevant.

## Security & Configuration
- Do not commit secrets; use `.env.example` to document required environment variables and add real `.env` to `.gitignore`.
- Prefer configuration via environment variables or `pyproject.toml`; keep defaults safe for local development.
