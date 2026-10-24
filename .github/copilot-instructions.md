# Copilot instructions for MTechProject

This repository contains a Semester 3 demand-forecasting study under `semester3-demand-forecasting/`.

## Primary goal

Help with **reproducible retail demand forecasting experiments** on a Walmart M5 subset and comparison of:

- Seasonal naive baseline
- LightGBM baseline
- Chronos-2 zero-shot model

## Tech stack and structure

- Language: Python (3.10+)
- Packaging: `pyproject.toml`
- Tests: `pytest`
- App: Streamlit (`app/dashboard.py`)
- Main package: `src/demand_forecast/`
- Config: `config/semester3.yaml`

## Key engineering rules

1. **No data leakage**
   - Features must use only information available at or before each forecast origin.
   - Keep chronological train/test separation.
   - Avoid future price usage unless explicitly enabled in config and justified.

2. **Respect experiment protocol**
   - Forecast origins are rolling cutoffs; for each origin $t_o$, models train only on data up to and including $t_o$.
   - Each evaluation window is a non-overlapping 28-day horizon: $h \in \{1,\dots,28\}$ for dates $t_o + h$.
   - MASE uses seasonal period $m=7$ (daily data with weekly seasonality), with scaling computed only from pre-origin history.
   - Chronos-2 prediction intervals are evaluated from quantiles (for example $q_{0.1}$ and $q_{0.9}$): enforce quantile order, clip to non-negative sales, and compute coverage and interval width only on held-out horizons.

3. **Prefer minimal, targeted edits**
   - Preserve public APIs and existing CLI behavior where possible.
   - Follow existing style and naming in nearby files.

4. **Keep outputs reproducible**
   - Use deterministic seeds where sampling or randomization is involved.
   - Keep output paths consistent with README documentation.
   - Inspect existing repository files and conventions before editing; do not invent paths, APIs, or experimental outcomes.

5. **Respect scope and evidence**
   - Do not fabricate experimental metrics, tables, plots, or runtime claims.
   - Clearly distinguish demo/synthetic outputs from results generated using real M5 data.
   - Keep Semester 3 conclusions within forecasting accuracy, uncertainty quality, and documented limitations.

## What to do when generating or editing code

- Add/adjust tests in `tests/` for behavior changes.
- For model or metric changes, update both implementation and tests.
- For config-sensitive logic, confirm defaults in `config/semester3.yaml`.
- Prefer small helper functions over large monolithic blocks.

## Useful run commands (zsh/macOS)

Run from `semester3-demand-forecasting/`:

- Create env: `python3 -m venv .venv`
- Activate: `source .venv/bin/activate`
- Install: `python -m pip install -e ".[dev,dashboard]"`
- Quick demo: `python -m demand_forecast.cli demo`
- Tests: `pytest`
- Baseline-only forecast: `python -m demand_forecast.cli forecast --skip-chronos`
- Dashboard: `streamlit run app/dashboard.py`

If Chronos is needed:

- `python -m pip install -e ".[chronos]"`

## Output artifacts to keep in sync

- `data/processed/panel.parquet`
- `data/processed/series_catalog.csv`
- `data/processed/forecasts.csv`
- `reports/eda_summary.md`
- `reports/experiment_summary.md`
- `reports/figures/*`

## Communication expectations for generated docs/comments

- Use clear, concise academic-project language.
- Do not claim inventory/cost savings from this Semester 3 scope.
- Mention constraints and assumptions explicitly.
