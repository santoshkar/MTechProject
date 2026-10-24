"""Seasonal-naive point forecasts and empirical residual intervals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from demand_forecast.models.common import enforce_nonnegative_quantile_order, quantile_frame


def seasonal_naive_point(history: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """Predict y_{T+h} with y_{T+h-k*season}, where k = ceil(h / season)."""
    if season < 1:
        raise ValueError("season must be positive")
    history = np.asarray(history, dtype=float)
    if len(history) < season:
        raise ValueError(f"Seasonal naive needs at least {season} observations, found {len(history)}.")
    point = np.empty(horizon, dtype=float)
    for step in range(1, horizon + 1):
        cycles = math.ceil(step / season)
        days_back = cycles * season - step
        point[step - 1] = history[-(days_back + 1)]
    return point


def residual_quantiles(
    history: np.ndarray,
    horizon: int,
    season: int,
    quantiles: list[float],
    step: int | None = None,
) -> np.ndarray:
    """In-sample multi-step seasonal-naive residual quantiles, shape (horizon, n_quantiles)."""
    history = np.asarray(history, dtype=float)
    step = season if step is None else max(1, step)
    max_origin = len(history) - horizon - 1
    min_origin = season - 1
    levels = np.asarray(quantiles, dtype=float)
    if max_origin < min_origin:
        return np.zeros((horizon, len(quantiles)), dtype=float)
    errors = []
    for origin in range(min_origin, max_origin + 1, step):
        point = seasonal_naive_point(history[: origin + 1], horizon, season)
        actual = history[origin + 1 : origin + 1 + horizon]
        errors.append(actual - point)
    error_matrix = np.vstack(errors)
    # np.quantile returns shape (n_quantiles, horizon) when quantiles is a sequence.
    estimated = np.quantile(error_matrix, levels, axis=0)
    if estimated.ndim == 1:
        estimated = estimated.reshape(1, -1)
    return estimated.T


class SeasonalNaiveForecaster:
    """Weekly seasonal naive, with residual quantiles estimated only from history."""

    name = "seasonal_naive"

    def __init__(self, config: dict):
        self.config = config

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        horizon = int(self.config["forecast"]["horizon"])
        season = int(self.config["forecast"]["seasonality"])
        quantiles = [float(q) for q in self.config["forecast"]["quantiles"]]
        frames = []
        future_groups = {sid: group.sort_values("date") for sid, group in future.groupby("series_id", sort=False)}
        for series_id, hist in history.groupby("series_id", sort=False):
            fut = future_groups[series_id]
            if len(fut) != horizon:
                raise ValueError(f"{series_id} future window has {len(fut)} rows, expected {horizon}.")
            sales = hist.sort_values("date")["sales"].to_numpy(dtype=float)
            point = np.maximum(seasonal_naive_point(sales, horizon, season), 0.0)
            resid = residual_quantiles(sales, horizon, season, quantiles)
            raw = {q: point + resid[:, index] for index, q in enumerate(quantiles)}
            ordered = enforce_nonnegative_quantile_order(raw)
            frame = pd.DataFrame(
                {
                    "series_id": series_id,
                    "date": fut["date"].to_numpy(),
                    "horizon": np.arange(1, horizon + 1),
                    "y_pred": point,
                    **quantile_frame(ordered),
                }
            )
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)
