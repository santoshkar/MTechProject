import numpy as np
import pandas as pd

from demand_forecast.models.common import enforce_nonnegative_quantile_order
from demand_forecast.models.seasonal_naive import SeasonalNaiveForecaster, seasonal_naive_point


def test_seasonal_naive_repeats_the_previous_season():
    history = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=float)
    point = seasonal_naive_point(history, horizon=8, season=7)
    assert point.tolist() == [8, 9, 10, 11, 12, 13, 14, 8]


def test_quantile_repair_is_nonnegative_and_ordered():
    repaired = enforce_nonnegative_quantile_order(
        {0.9: np.array([-1.0, 3.0]), 0.1: np.array([4.0, -2.0]), 0.5: np.array([1.0, 1.0])}
    )
    assert repaired[0.1].tolist() == [0.0, 0.0]
    assert repaired[0.5].tolist() == [1.0, 1.0]
    assert repaired[0.9].tolist() == [4.0, 3.0]
    assert np.all(repaired[0.1] <= repaired[0.5])
    assert np.all(repaired[0.5] <= repaired[0.9])


def test_forecaster_uses_only_history_for_the_point_forecast():
    dates = pd.date_range("2020-01-01", periods=21, freq="D")
    panel = pd.DataFrame(
        {
            "series_id": "A",
            "date": dates,
            "sales": np.arange(1, 22, dtype=float),
            "sell_price": 1.0,
            "dow": dates.day_of_week + 1,
            "month": dates.month,
            "weekofyear": dates.isocalendar().week.astype(int),
            "is_event": 0,
            "snap": 0,
        }
    )
    history = panel.iloc[:14]
    future = panel.iloc[14:]
    config = {"forecast": {"horizon": 7, "seasonality": 7, "quantiles": [0.1, 0.5, 0.9]}}
    forecast = SeasonalNaiveForecaster(config).predict(history, future)
    assert forecast["y_pred"].tolist() == [8, 9, 10, 11, 12, 13, 14]
