"""Demand-pattern labels used to build a balanced study sample.

Labels are descriptive. They are not model features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def adi_and_cv2(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    nonzero_idx = np.flatnonzero(values > 0)
    if len(nonzero_idx) <= 1:
        adi = float(len(values)) if len(values) else float("nan")
    else:
        adi = float(np.mean(np.diff(nonzero_idx)))
    nonzero = values[values > 0]
    if len(nonzero) <= 1 or float(np.mean(nonzero)) == 0:
        cv2 = float("nan")
    else:
        cv2 = float((np.std(nonzero, ddof=1) / np.mean(nonzero)) ** 2)
    return adi, cv2


def syntetos_boylan_class(adi: float, cv2: float, adi_threshold: float = 1.32, cv2_threshold: float = 0.49) -> str:
    intermittent = adi >= adi_threshold
    erratic = np.isfinite(cv2) and cv2 >= cv2_threshold
    if intermittent and erratic:
        return "lumpy"
    if intermittent:
        return "intermittent"
    if erratic:
        return "erratic"
    return "smooth"


def project_demand_class(mean_sales: float, zero_rate: float, adi: float, config: dict) -> str:
    subset = config["subset"]
    if zero_rate >= float(subset["intermittent_min_zero_rate"]) or adi >= float(subset["intermittent_min_adi"]):
        return "intermittent"
    if mean_sales >= float(subset["fast_min_mean"]):
        return "fast"
    if mean_sales <= float(subset["slow_max_mean"]):
        return "slow"
    return "medium"


def stats_for_window(values: np.ndarray, config: dict) -> dict[str, float | str]:
    values = np.asarray(values, dtype=float)
    mean_sales = float(np.mean(values)) if len(values) else float("nan")
    zero_rate = float(np.mean(values == 0)) if len(values) else float("nan")
    adi, cv2 = adi_and_cv2(values)
    return {
        "mean_sales": mean_sales,
        "zero_rate": zero_rate,
        "adi": adi,
        "cv2": cv2,
        "demand_class": project_demand_class(mean_sales, zero_rate, adi, config),
        "sb_class": syntetos_boylan_class(adi, cv2),
    }


def classification_bounds(n_days: int, config: dict) -> tuple[int, int]:
    """Return the [start, end) day slice that excludes the evaluation holdout."""
    holdout = int(config["forecast"]["horizon"]) * int(config["forecast"]["n_windows"])
    if n_days <= holdout:
        raise ValueError(
            f"The series has {n_days} days, which is not longer than the {holdout}-day evaluation holdout."
        )
    end = n_days - holdout
    start = max(0, end - int(config["subset"]["lookback_days"]))
    if end - start < 14:
        raise ValueError("The pre-holdout classification window is shorter than 14 days.")
    return start, end


def classify_wide(sales: pd.DataFrame, day_cols: list[str], config: dict) -> pd.DataFrame:
    start, end = classification_bounds(len(day_cols), config)
    window_cols = day_cols[start:end]
    values = sales[window_cols].to_numpy(dtype=float)
    rows = []
    id_cols = [column for column in ["series_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] if column in sales.columns]
    identifiers = sales[id_cols]
    for position in range(len(sales)):
        stats = stats_for_window(values[position], config)
        row = {column: identifiers.iloc[position][column] for column in id_cols}
        row.update(stats)
        rows.append(row)
    catalog = pd.DataFrame(rows)
    catalog["classification_start_day"] = day_cols[start]
    catalog["classification_end_day"] = day_cols[end - 1]
    return catalog
