# Repository Guidelines

## Project Structure & Module Organization
- `chatbuy/` – Python package (core library code lives here as the project grows).
- `scripts/` – runnable demos and utilities, e.g. `eth_dual_ma_vectorbt.py` for ETH dual‑MA backtests.
- `data/` – local datasets (ignored by VCS if large). Prefer small CSV samples.
- `README.md` – quickstart and usage. `pyproject.toml` – packaging, deps, lint config.

## Build, Test, and Development Commands (uv)
- Create env & install deps (uses `uv.lock`):
  - `uv venv && source .venv/bin/activate`
  - `uv sync --frozen`
- Add/remove deps:
  - `uv add <package>` / `uv remove <package>` (updates lockfile)
- Lint & format (ruff):
  - `uv run ruff check . --fix`
  - `uv run ruff format .`
- Run demo locally:
  - CSV: `uv run python scripts/eth_dual_ma_vectorbt.py --source csv --csv-path data/ETH-USD.csv`
  - Yahoo: `uv add yfinance && uv run python scripts/eth_dual_ma_vectorbt.py --source yfinance --symbol ETH-USD`

## Coding Style & Naming Conventions
- Python 3.12+, 4‑space indentation, max reasonable complexity per function.
- Use type hints and Google‑style docstrings (see `pyproject.toml` pydocstyle config).
- Strings use double quotes; imports sorted (isort via ruff). Keep functions small and single‑purpose.
- Naming: modules `snake_case.py`, classes `PascalCase`, functions/vars `snake_case`.

## Testing Guidelines
- Framework: `pytest` (add as needed).
- Location: `tests/` with files named `test_*.py`.
- Run: `pytest -q`. Target meaningful unit coverage for new code and critical paths.
- Prefer deterministic tests; use fixtures/fakes over network calls.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject. Suggested prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Before PR: run `uv run ruff check . --fix` and `uv run ruff format .`; ensure demos run.
- PRs: include summary of changes, rationale, usage notes, and any screenshots/outputs if relevant; link issues.
- Keep PRs focused; update `README.md` and this guide when behavior or structure changes.

## Security & Configuration Tips
- Never commit secrets; store in `.env`/environment variables. Keep large datasets out of VCS.
- Prefer real data sources (yfinance by default) or user-provided CSV; document required extras (e.g., `yfinance`).
 - Lockfile policy: commit changes to `uv.lock` with dependency updates; use `uv sync --frozen` in CI/local to ensure reproducibility.
