import numpy as np
import pandas as pd

from demand_forecast.data.classify import classify_wide, project_demand_class
from demand_forecast.data.sample import sample_series


def _config() -> dict:
    return {
        "seed": 7,
        "subset": {
            "lookback_days": 20,
            "counts": {"fast": 1, "slow": 1, "intermittent": 1},
            "fast_min_mean": 1.0,
            "slow_max_mean": 0.5,
            "intermittent_min_zero_rate": 0.5,
            "intermittent_min_adi": 1.32,
            "fill_from_medium": False,
        },
        "forecast": {"horizon": 7, "n_windows": 1},
    }


def test_holdout_sales_do_not_change_the_class_label():
    config = _config()
    early = np.array([5.0] * 20)
    holdout_quiet = np.array([0.0] * 7)
    holdout_busy = np.array([9.0] * 7)
    day_cols = [f"d_{i}" for i in range(1, 28)]
    frame = pd.DataFrame(
        [
            {"series_id": "quiet_later", "item_id": "A", "dept_id": "D", "cat_id": "C", "store_id": "CA_1", "state_id": "CA"},
            {"series_id": "busy_later", "item_id": "B", "dept_id": "D", "cat_id": "C", "store_id": "CA_1", "state_id": "CA"},
        ]
    )
    for column, left, right in zip(day_cols, list(early) + list(holdout_quiet), list(early) + list(holdout_busy)):
        frame.loc[0, column] = left
        frame.loc[1, column] = right
    catalog = classify_wide(frame, day_cols, config)
    assert set(catalog["demand_class"]) == {"fast"}
    assert catalog["mean_sales"].nunique() == 1


def test_sampling_is_reproducible_and_does_not_relabel_medium_fill():
    config = _config()
    config["subset"]["fill_from_medium"] = True
    config["subset"]["counts"] = {"fast": 1, "slow": 2, "intermittent": 1}
    rows = []
    for index, label, mean, zero_rate, adi in [
        (1, "fast", 4.0, 0.0, 1.0),
        (2, "slow", 0.2, 0.0, 1.0),
        (3, "intermittent", 0.4, 0.8, 5.0),
        (4, "medium", 0.8, 0.1, 1.1),
    ]:
        rows.append(
            {
                "series_id": f"S{index}",
                "mean_sales": mean,
                "zero_rate": zero_rate,
                "adi": adi,
                "demand_class": project_demand_class(mean, zero_rate, adi, config),
            }
        )
    catalog = pd.DataFrame(rows)
    assert catalog["demand_class"].tolist() == ["fast", "slow", "intermittent", "medium"]
    first = sample_series(catalog, config)
    second = sample_series(catalog, config)
    assert first == second
    assert "S4" in first
