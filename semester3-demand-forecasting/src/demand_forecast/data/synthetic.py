"""Small M5-shaped data for pipeline checks. These are not Walmart observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_synthetic_m5(raw_dir: Path, config: dict) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    n_days = int(config["data"]["synthetic_days"])
    per_class = int(config["data"]["synthetic_series_per_class"])
    rng = np.random.default_rng(int(config["seed"]))
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    day_cols = [f"d_{i}" for i in range(1, n_days + 1)]

    specs = []
    stores = [("CA_1", "CA"), ("TX_1", "TX")]
    for index in range(per_class):
        store_id, state_id = stores[index % len(stores)]
        specs.append(("fast", f"FOODS_1_{index + 1:03d}", "FOODS_1", "FOODS", store_id, state_id))
        specs.append(("slow", f"HOBBIES_1_{index + 1:03d}", "HOBBIES_1", "HOBBIES", store_id, state_id))
        specs.append(("intermittent", f"HOUSEHOLD_1_{index + 1:03d}", "HOUSEHOLD_1", "HOUSEHOLD", store_id, state_id))

    sales_rows = []
    price_rows = []
    weekday = dates.day_of_week.to_numpy()
    for kind, item_id, dept_id, cat_id, store_id, state_id in specs:
        if kind == "fast":
            weekly = np.array([5, 6, 6, 7, 8, 10, 9], dtype=float)[weekday]
            sales = np.maximum(0, rng.poisson(weekly))
        elif kind == "slow":
            sales = 0.25 + 0.05 * np.sin(np.arange(n_days) / 6.0)
            sales = np.clip(sales, 0.1, 0.45)
        else:
            sales = np.zeros(n_days, dtype=float)
            spike_days = np.arange(2, n_days, 8)
            sales[spike_days] = rng.integers(2, 6, size=len(spike_days))
        series_id = f"{item_id}_{store_id}"
        row = {
            "id": f"{series_id}_evaluation",
            "item_id": item_id,
            "dept_id": dept_id,
            "cat_id": cat_id,
            "store_id": store_id,
            "state_id": state_id,
        }
        row.update({column: value for column, value in zip(day_cols, sales)})
        sales_rows.append(row)
        base_price = 2.0 + (index % 5) * 0.8
        weeks = dates.isocalendar()
        week_keys = sorted({(int(year), int(week)) for year, week in zip(weeks.year, weeks.week)})
        for year, week in week_keys:
            price_rows.append(
                {
                    "store_id": store_id,
                    "item_id": item_id,
                    "wm_yr_wk": year * 100 + week,
                    "sell_price": round(base_price + (0.1 if week % 5 == 0 else 0.0), 2),
                }
            )

    sales = pd.DataFrame(sales_rows)
    calendar = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "d": day_cols})
    iso = dates.isocalendar()
    calendar["wm_yr_wk"] = (iso.year.to_numpy(dtype=int) * 100) + iso.week.to_numpy(dtype=int)
    calendar["wday"] = dates.day_of_week + 1
    calendar["month"] = dates.month
    calendar["year"] = dates.year
    calendar["weekday"] = dates.day_name()
    calendar["event_name_1"] = np.where(np.arange(n_days) % 28 == 0, "Sporting", "")
    calendar["event_type_1"] = np.where(calendar["event_name_1"] == "", "", "Sporting")
    calendar["snap_CA"] = (dates.day_of_week == 0).astype(int)
    calendar["snap_TX"] = (dates.day_of_week == 2).astype(int)
    calendar["snap_WI"] = (dates.day_of_week == 4).astype(int)

    sales.to_csv(raw_dir / config["data"]["sales_file"], index=False)
    calendar.to_csv(raw_dir / config["data"]["calendar_file"], index=False)
    pd.DataFrame(price_rows).to_csv(raw_dir / config["data"]["prices_file"], index=False)
