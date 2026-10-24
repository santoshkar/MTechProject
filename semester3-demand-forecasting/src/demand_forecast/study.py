"""Run the Semester 3 forecasting study."""

from __future__ import annotations

import json
import logging

import pandas as pd

from demand_forecast.data.prepare import load_prepared_panel, prepare_dataset, processed_dir
from demand_forecast.eda import run_eda
from demand_forecast.evaluation.backtest import run_backtest
from demand_forecast.evaluation.report import write_evaluation_outputs
from demand_forecast.paths import FORECASTS_FILENAME, METADATA_FILENAME, project_root, resolve_path

logger = logging.getLogger(__name__)


def reports_dir(config: dict):
    return resolve_path(project_root(), config["output"]["reports_dir"])


def run_prepare(config: dict) -> pd.DataFrame:
    return prepare_dataset(config)


def run_eda_stage(config: dict, rebuild: bool = False) -> None:
    panel = load_prepared_panel(config, rebuild=rebuild)
    run_eda(panel, config, reports_dir(config), processed_dir(config))
    logger.info("EDA figures written to %s", reports_dir(config) / "figures")


def run_forecast_study(config: dict, rebuild: bool = False) -> pd.DataFrame:
    panel = load_prepared_panel(config, rebuild=rebuild)
    forecasts, runtimes = run_backtest(panel, config)
    out_dir = processed_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(out_dir / FORECASTS_FILENAME, index=False)
    metadata = json.loads((out_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    write_evaluation_outputs(forecasts, runtimes, config, reports_dir(config), metadata)
    logger.info("Forecasts saved to %s", out_dir / FORECASTS_FILENAME)
    logger.info("Experiment summary saved to %s", reports_dir(config) / "experiment_summary.md")
    if not config["models"].get("chronos", False):
        logger.warning("Chronos-2 was not included. Enable models.chronos before the Semester 3 comparison is treated as complete.")
    if metadata.get("is_synthetic"):
        logger.warning("This run used synthetic data. Do not report it as an M5 result.")
    return forecasts
