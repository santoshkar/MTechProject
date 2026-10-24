"""Leakage-safe features for the LightGBM direct multi-horizon model.

Every feature is computed from sales and prices observed at the forecast origin,
plus calendar fields of the target date. Target-date sales are never read here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

NUMERIC_CALENDAR = ["dow", "month", "weekofyear", "is_event", "snap"]
CATEGORICAL_FEATURES = ["series_id", "dept_id", "cat_id", "store_id", "state_id"]


def numeric_feature_names(config: dict) -> list[str]:
    names = [f"lag_{lag}" for lag in config["features"]["lags"]]
    for window in config["features"]["rolling_windows"]:
        names.extend([f"roll_mean_{window}", f"roll_std_{window}", f"zero_rate_{window}"])
    names.extend(["days_since_last_sale", "sell_price", "horizon", *NUMERIC_CALENDAR])
    return names


def history_numeric_features(sales: np.ndarray, price: float, config: dict) -> dict[str, float]:
    sales = np.asarray(sales, dtype=float)
    features: dict[str, float] = {}
    for lag in config["features"]["lags"]:
        features[f"lag_{lag}"] = float(sales[-lag]) if len(sales) >= lag else np.nan
    for window in config["features"]["rolling_windows"]:
        if len(sales) < window:
            features[f"roll_mean_{window}"] = np.nan
            features[f"roll_std_{window}"] = np.nan
            features[f"zero_rate_{window}"] = np.nan
            continue
        recent = sales[-window:]
        features[f"roll_mean_{window}"] = float(np.mean(recent))
        features[f"roll_std_{window}"] = float(np.std(recent))
        features[f"zero_rate_{window}"] = float(np.mean(recent == 0))
    nonzero = np.flatnonzero(sales > 0)
    if len(nonzero) == 0:
        features["days_since_last_sale"] = float(len(sales))
    else:
        features["days_since_last_sale"] = float(len(sales) - 1 - nonzero[-1])
    features["sell_price"] = float(price)
    return features


def _static_values(history: pd.DataFrame) -> dict[str, Any]:
    last = history.iloc[-1]
    return {column: last[column] for column in CATEGORICAL_FEATURES}


def rows_for_origin(
    history: pd.DataFrame,
    future: pd.DataFrame,
    config: dict,
    include_target: bool,
) -> list[dict[str, Any]]:
    horizon = int(config["forecast"]["horizon"])
    future = future.sort_values("date")
    history = history.sort_values("date")
    if len(future) != horizon:
        series_id = history["series_id"].iloc[0]
        raise ValueError(f"{series_id} future window has {len(future)} rows, expected {horizon}.")
    sales = history["sales"].to_numpy(dtype=float)
    origin_price = float(history["sell_price"].iloc[-1])
    static = _static_values(history)
    origin_date = history["date"].iloc[-1]
    assume_price = bool(config["features"]["assume_future_price_known"])
    base = history_numeric_features(sales, origin_price, config)
    rows: list[dict[str, Any]] = []
    for step, fut in enumerate(future.itertuples(index=False), start=1):
        row = dict(base)
        row.update(static)
        row["origin_date"] = origin_date
        row["date"] = fut.date
        row["horizon"] = step
        row["dow"] = int(pd.Timestamp(fut.date).day_of_week) + 1
        row["month"] = int(fut.month)
        row["weekofyear"] = int(fut.weekofyear)
        row["is_event"] = int(fut.is_event)
        row["snap"] = int(fut.snap)
        if assume_price:
            row["sell_price"] = float(fut.sell_price)
        if include_target:
            row["target"] = float(fut.sales)
        rows.append(row)
    return rows


def build_training_frame(history: pd.DataFrame, config: dict) -> pd.DataFrame:
    horizon = int(config["forecast"]["horizon"])
    min_history = int(config["forecast"]["min_history_days"])
    step = int(config["features"]["train_origin_step"])
    records: list[dict[str, Any]] = []
    for _, group in history.groupby("series_id", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        last_origin = len(group) - horizon - 1
        first_origin = min_history - 1
        if last_origin < first_origin:
            continue
        for origin_pos in range(first_origin, last_origin + 1, step):
            hist = group.iloc[: origin_pos + 1]
            fut = group.iloc[origin_pos + 1 : origin_pos + 1 + horizon]
            records.extend(rows_for_origin(hist, fut, config, include_target=True))
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    if not (frame["date"] > frame["origin_date"]).all():
        raise AssertionError("Leakage check failed: a training target date is not after its origin.")
    return frame


def build_prediction_frame(history: pd.DataFrame, future: pd.DataFrame, config: dict) -> pd.DataFrame:
    missing = set(history["series_id"].unique()) - set(future["series_id"].unique())
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(f"Future frame is missing {len(missing)} series, including {sample}.")
    future_groups = {sid: group for sid, group in future.groupby("series_id", sort=False)}
    records: list[dict[str, Any]] = []
    for series_id, hist in history.groupby("series_id", sort=False):
        records.extend(rows_for_origin(hist, future_groups[series_id], config, include_target=False))
    return pd.DataFrame.from_records(records)
