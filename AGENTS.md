# Repository Guidelines

## Project Structure & Module Organization
- `chatbuy/` – core Python package. Add new modules here as the library grows.
- `scripts/` – runnable demos/utilities (e.g., `eth_dual_ma_vectorbt.py`).
- `data/` – small local CSV samples (large files ignored by VCS).
- `tests/` – unit tests named `test_*.py`.
- Root: `README.md`, `pyproject.toml`, `uv.lock`.

## Build, Test, and Development Commands (uv)
- Create env & install deps: `uv venv && source .venv/bin/activate && uv sync --frozen`.
- Add/remove deps: `uv add <pkg>` / `uv remove <pkg>` (updates `uv.lock`).
- Lint & format (ruff): `uv run ruff check . --fix` and `uv run ruff format .`.
- Run tests: `pytest -q`.
- Run demo (CSV): `uv run python scripts/eth_dual_ma_vectorbt.py --source csv --csv-path data/ETH-USD.csv`.
- Run demo (Yahoo): `uv add yfinance && uv run python scripts/eth_dual_ma_vectorbt.py --source yfinance --symbol ETH-USD`.

## Coding Style & Naming Conventions
- Python 3.12+, 4-space indentation, small single‑purpose functions.
- Type hints required; Google‑style docstrings (pydocstyle config in `pyproject.toml`).
- Strings use double quotes; imports sorted (isort via ruff).
- Naming: modules `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`.

## Testing Guidelines
- Framework: `pytest`.
- Location: `tests/` with files `test_*.py`.
- Keep tests deterministic; prefer fixtures/fakes over network calls.
- Aim for meaningful coverage on new code and critical paths.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject. Prefer prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Before PR: run `uv run ruff check . --fix`, `uv run ruff format .`, and verify demos run.
- PRs: include summary, rationale, usage notes, and link issues; add screenshots/outputs when helpful.

## Security & Configuration Tips
- Do not commit secrets; use `.env`/environment variables.
- Keep large datasets out of VCS; store only small samples in `data/`.
- Lockfile policy: commit `uv.lock`; use `uv sync --frozen` for reproducible installs.

## Agent‑Specific Instructions
- This AGENTS.md applies to the entire repository tree.
- Follow the style above; keep diffs minimal and focused.
- Do not change file names/structure unless necessary and documented in PR.
