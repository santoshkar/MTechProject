"""Forecast metrics for the Semester 3 study.

MAPE is intentionally not used. Retail series contain many zero-sales days,
and MAPE divides by the actual value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_column(q: float) -> str:
    return f"q_{q:.2f}"


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.nanmean(np.abs(y_true - y_pred)))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Mean pinball loss. For the median, this is half of MAE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1.0) * error)
    if len(loss) == 0:
        return float("nan")
    return float(np.nanmean(loss))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= np.asarray(lower, dtype=float)) & (y_true <= np.asarray(upper, dtype=float))
    if len(inside) == 0:
        return float("nan")
    return float(np.mean(inside))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    width = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    if len(width) == 0:
        return float("nan")
    return float(np.nanmean(width))


def seasonal_naive_scale(history_sales: np.ndarray, season: int) -> float:
    """Hyndman MASE scale: mean absolute seasonal difference on the training history."""
    history_sales = np.asarray(history_sales, dtype=float)
    if len(history_sales) <= season:
        return float("nan")
    scale = np.mean(np.abs(history_sales[season:] - history_sales[:-season]))
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    return float(scale)


def mase(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    if scale is None or not np.isfinite(scale) or scale == 0:
        return float("nan")
    return mean_absolute_error(y_true, y_pred) / float(scale)


def series_origin_metrics(forecasts: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    if forecasts.empty:
        return pd.DataFrame()
    low_q = min(q for q in quantiles if q < 0.5)
    high_q = max(q for q in quantiles if q > 0.5)
    low_col = quantile_column(low_q)
    high_col = quantile_column(high_q)
    rows: list[dict] = []
    group_cols = ["model", "series_id", "origin"]
    for keys, group in forecasts.groupby(group_cols, sort=False):
        model, series_id, origin = keys
        row = {
            "model": model,
            "series_id": series_id,
            "origin": origin,
            "demand_class": group["demand_class"].iloc[0],
            "n_points": int(len(group)),
            "mae": mean_absolute_error(group["y_true"], group["y_pred"]),
            "mase": mase(group["y_true"], group["y_pred"], float(group["mase_scale"].iloc[0])),
            "coverage": interval_coverage(group["y_true"], group[low_col], group[high_col]),
            "interval_width": interval_width(group[low_col], group[high_col]),
            "interval_low": low_q,
            "interval_high": high_q,
        }
        for q in quantiles:
            row[f"pinball_{q:.2f}"] = pinball_loss(group["y_true"], group[quantile_column(q)], q)
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(frame: pd.DataFrame, quantiles: list[float], extra_group: list[str]) -> pd.DataFrame:
    metric_cols = ["mae", "mase", "coverage", "interval_width"] + [f"pinball_{q:.2f}" for q in quantiles]
    grouped = frame.groupby(["model", *extra_group], dropna=False)
    summary = grouped[metric_cols].mean().reset_index()
    counts = grouped.agg(n_series=("series_id", "nunique"), n_windows=("origin", "nunique"), n_points=("n_points", "sum"))
    summary = summary.merge(counts.reset_index(), on=["model", *extra_group], how="left")
    summary["aggregation"] = "macro_series_origin"
    return summary


def aggregate_metrics(series_metrics: pd.DataFrame, quantiles: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = _aggregate(series_metrics, quantiles, [])
    by_class = _aggregate(series_metrics, quantiles, ["demand_class"])
    return overall, by_class
