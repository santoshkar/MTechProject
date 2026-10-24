"""Rolling-origin evaluation. Splits are chronological, never random."""

from __future__ import annotations

import logging
import time
import tracemalloc

import pandas as pd

from demand_forecast.evaluation.metrics import seasonal_naive_scale
from demand_forecast.models.chronos_forecaster import ChronosForecaster
from demand_forecast.models.lightgbm_forecaster import LightGBMForecaster
from demand_forecast.models.seasonal_naive import SeasonalNaiveForecaster

logger = logging.getLogger(__name__)


def forecast_origins(panel: pd.DataFrame, config: dict) -> list[pd.Timestamp]:
    horizon = int(config["forecast"]["horizon"])
    n_windows = int(config["forecast"]["n_windows"])
    min_history = int(config["forecast"]["min_history_days"])
    max_date = pd.Timestamp(panel["date"].max())
    min_date = pd.Timestamp(panel["date"].min())
    available_dates = set(pd.to_datetime(panel["date"].unique()))
    origins = [max_date - pd.Timedelta(days=horizon * step) for step in range(1, n_windows + 1)]
    origins = sorted(origins)
    missing = [origin for origin in origins if origin not in available_dates]
    if missing:
        raise ValueError(f"Forecast origins are not in the panel: {missing}")
    earliest = origins[0]
    history_days = (earliest - min_date).days + 1
    if history_days < min_history:
        raise ValueError(
            f"The earliest origin {earliest.date()} has {history_days} history days, "
            f"below min_history_days={min_history}. Reduce n_windows or min_history_days."
        )
    return origins


def split_at_origin(panel: pd.DataFrame, origin: pd.Timestamp, horizon: int, assume_future_price_known: bool):
    future_end = origin + pd.Timedelta(days=horizon)
    history = panel.loc[panel["date"] <= origin].copy()
    future = panel.loc[(panel["date"] > origin) & (panel["date"] <= future_end)].copy()
    expected = panel["series_id"].nunique() * horizon
    if len(future) != expected:
        raise ValueError(
            f"Origin {origin.date()} produced {len(future)} future rows, expected {expected}. "
            "The panel is not a complete daily grid through the horizon."
        )
    if not assume_future_price_known:
        last_price = history.groupby("series_id")["sell_price"].last()
        future["sell_price"] = future["series_id"].map(last_price)
    return history, future


def mase_scales(history: pd.DataFrame, season: int) -> pd.Series:
    scales = {}
    for series_id, group in history.groupby("series_id", sort=False):
        sales = group.sort_values("date")["sales"].to_numpy(dtype=float)
        scales[series_id] = seasonal_naive_scale(sales, season)
    return pd.Series(scales, name="mase_scale")


def enabled_models(config: dict) -> list:
    models = []
    if config["models"].get("seasonal_naive", True):
        models.append(SeasonalNaiveForecaster(config))
    if config["models"].get("lightgbm", True):
        models.append(LightGBMForecaster(config))
    if config["models"].get("chronos", False):
        models.append(ChronosForecaster(config))
    if not models:
        raise ValueError("No models are enabled in the config.")
    return models


def run_backtest(panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = forecast_origins(panel, config)
    models = enabled_models(config)
    for model in models:
        if hasattr(model, "load"):
            model.load()
    horizon = int(config["forecast"]["horizon"])
    season = int(config["forecast"]["seasonality"])
    assume_price = bool(config["features"]["assume_future_price_known"])
    forecast_frames = []
    runtime_rows = []
    actual_cols = ["series_id", "date", "sales", "demand_class", "store_id", "cat_id", "dept_id", "item_id"]
    for index, origin in enumerate(origins, start=1):
        logger.info("Forecast window %s/%s, origin %s.", index, len(origins), origin.date())
        history, future = split_at_origin(panel, origin, horizon, assume_price)
        actuals = future[actual_cols].rename(columns={"sales": "y_true"})
        scales = mase_scales(history, season)
        for model in models:
            tracemalloc.start()
            started = time.perf_counter()
            if hasattr(model, "fit"):
                model.fit(history)
            prediction = model.predict(history, future)
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            prediction["model"] = model.name
            prediction["origin"] = origin
            prediction = prediction.merge(actuals, on=["series_id", "date"], how="left")
            prediction["mase_scale"] = prediction["series_id"].map(scales)
            if prediction["y_true"].isna().any():
                raise ValueError(f"{model.name} produced dates that do not match actual sales.")
            forecast_frames.append(prediction)
            runtime_rows.append(
                {
                    "model": model.name,
                    "origin": origin,
                    "seconds": round(elapsed, 3),
                    "peak_python_memory_mb": round(peak / (1024 * 1024), 2),
                }
            )
            logger.info("%s finished in %.1f seconds.", model.name, elapsed)
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    return forecasts, pd.DataFrame(runtime_rows)
