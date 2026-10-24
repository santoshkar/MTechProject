import numpy as np
import pandas as pd

from demand_forecast.models.features import build_prediction_frame, build_training_frame


def _panel(n_days: int = 50) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "series_id": "ITEM_CA_1",
            "item_id": "ITEM",
            "dept_id": "FOODS_1",
            "cat_id": "FOODS",
            "store_id": "CA_1",
            "state_id": "CA",
            "date": dates,
            "sales": np.arange(n_days, dtype=float),
            "sell_price": 3.0,
            "month": dates.month,
            "weekofyear": dates.isocalendar().week.astype(int),
            "is_event": 0,
            "snap": 0,
        }
    )


def _config() -> dict:
    return {
        "forecast": {"horizon": 7, "min_history_days": 21},
        "features": {
            "lags": [1, 7],
            "rolling_windows": [7],
            "train_origin_step": 7,
            "assume_future_price_known": False,
        },
    }


def test_future_sales_do_not_change_features():
    panel = _panel()
    history = panel.iloc[:40]
    future = panel.iloc[40:47]
    changed = future.copy()
    changed["sales"] = 999.0
    config = _config()
    original = build_prediction_frame(history, future, config)
    mutated = build_prediction_frame(history, changed, config)
    pd.testing.assert_frame_equal(original, mutated)


def test_last_observed_sale_is_lag_1_and_target_is_after_origin():
    panel = _panel()
    config = _config()
    train = build_training_frame(panel.iloc[:47], config)
    assert not train.empty
    assert (train["date"] > train["origin_date"]).all()
    last_origin = train["origin_date"].max()
    row = train.loc[train["origin_date"].eq(last_origin) & train["horizon"].eq(1)].iloc[0]
    history_end = panel.loc[panel["date"].eq(last_origin), "sales"].iloc[0]
    assert row["lag_1"] == history_end
    assert row["sell_price"] == 3.0
