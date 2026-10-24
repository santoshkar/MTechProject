"""Load M5-shaped files and build the study panel."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DAY_COLUMN = re.compile(r"^d_(\d+)$")
ID_COLUMNS = ["series_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def download_instructions() -> str:
    return """
M5 files were not found. Semester 3 uses the Walmart M5 Forecasting - Accuracy data.

1. Create a free Kaggle account.
2. Open https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
   and accept the competition rules.
3. Install the Kaggle CLI and put your API token at %USERPROFILE%\\.kaggle\\kaggle.json
4. From the project folder, either run:
     python -m demand_forecast.cli download
   or download manually and unzip these files into data/raw:
     sales_train_evaluation.csv
     calendar.csv
     sell_prices.csv

To check the code without M5, run:
  python -m demand_forecast.cli demo
""".strip()


def day_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if DAY_COLUMN.match(str(column))]
    return sorted(columns, key=lambda column: int(DAY_COLUMN.match(str(column)).group(1)))


def load_sales(path: Path) -> pd.DataFrame:
    logger.info("Reading sales file %s", path)
    frame = pd.read_csv(path).copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    required = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if "id" in frame.columns:
        frame["series_id"] = (
            frame["id"].astype(str).str.replace("_evaluation", "", regex=False).str.replace("_validation", "", regex=False)
        )
    else:
        frame["series_id"] = frame["item_id"].astype(str) + "_" + frame["store_id"].astype(str)
    columns = day_columns(frame)
    if not columns:
        raise ValueError(f"{path} has no d_1, d_2, ... sales columns.")
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="coerce")
    return frame


def load_calendar(path: Path) -> pd.DataFrame:
    calendar = pd.read_csv(path)
    calendar.columns = [str(column).strip() for column in calendar.columns]
    required = ["d", "date", "wm_yr_wk"]
    missing = [column for column in required if column not in calendar.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["wm_yr_wk"] = pd.to_numeric(calendar["wm_yr_wk"], errors="coerce").astype("int64")
    if "wday" not in calendar.columns:
        calendar["wday"] = calendar["date"].dt.day_of_week + 1
    if "month" not in calendar.columns:
        calendar["month"] = calendar["date"].dt.month
    if "year" not in calendar.columns:
        calendar["year"] = calendar["date"].dt.year
    calendar["weekofyear"] = calendar["date"].dt.isocalendar().week.astype(int)
    event = calendar["event_name_1"] if "event_name_1" in calendar.columns else ""
    event_text = pd.Series(event, index=calendar.index).fillna("").astype(str).str.strip()
    calendar["is_event"] = (~event_text.str.lower().isin(["", "nan", "none"])).astype(int)
    for column in ["snap_CA", "snap_TX", "snap_WI"]:
        if column not in calendar.columns:
            calendar[column] = 0
        calendar[column] = pd.to_numeric(calendar[column], errors="coerce").fillna(0).astype(int)
    return calendar


def load_prices(path: Path, item_ids: set[str], store_ids: set[str]) -> pd.DataFrame:
    logger.info("Reading prices file %s", path)
    prices = pd.read_csv(path, low_memory=False)
    prices.columns = [str(column).strip() for column in prices.columns]
    required = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    missing = [column for column in required if column not in prices.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    prices = prices.loc[prices["item_id"].isin(item_ids) & prices["store_id"].isin(store_ids), required].copy()
    prices["wm_yr_wk"] = pd.to_numeric(prices["wm_yr_wk"], errors="coerce").astype("int64")
    prices["sell_price"] = pd.to_numeric(prices["sell_price"], errors="coerce")
    return prices.drop_duplicates(["store_id", "item_id", "wm_yr_wk"], keep="last")


def melt_selected(sales: pd.DataFrame, selected_ids: list[str], day_cols: list[str]) -> pd.DataFrame:
    selected = sales.loc[sales["series_id"].isin(selected_ids), ID_COLUMNS + day_cols].copy()
    long = selected.melt(id_vars=ID_COLUMNS, value_vars=day_cols, var_name="d", value_name="sales")
    long["sales"] = pd.to_numeric(long["sales"], errors="coerce").fillna(0.0)
    return long


def join_calendar_and_prices(sales_long: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    keep = ["d", "date", "wm_yr_wk", "wday", "month", "year", "weekofyear", "is_event", "snap_CA", "snap_TX", "snap_WI"]
    panel = sales_long.merge(calendar[keep], on="d", how="left")
    if panel["date"].isna().any():
        raise ValueError("Some sales day columns did not match calendar.d.")
    panel = panel.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    panel = panel.sort_values(["series_id", "date"]).reset_index(drop=True)
    panel["sell_price"] = panel.groupby("series_id", sort=False)["sell_price"].ffill().bfill()
    missing_price = int(panel["sell_price"].isna().sum())
    if missing_price:
        fill = float(panel["sell_price"].median()) if panel["sell_price"].notna().any() else 1.0
        logger.warning("Filled %s missing prices with %s.", missing_price, fill)
        panel["sell_price"] = panel["sell_price"].fillna(fill)
    # Date weekday, not the M5 `wday` code. In M5, wday 1 is Saturday.
    panel["dow"] = panel["date"].dt.day_of_week + 1
    snap = np.zeros(len(panel), dtype=int)
    for state, column in [("CA", "snap_CA"), ("TX", "snap_TX"), ("WI", "snap_WI")]:
        mask = panel["state_id"].astype(str) == state
        snap[mask.to_numpy()] = panel.loc[mask, column].to_numpy()
    panel["snap"] = snap
    panel = panel.drop(columns=["snap_CA", "snap_TX", "snap_WI", "d"])
    return panel


def assert_complete_daily_panel(panel: pd.DataFrame) -> None:
    if panel.duplicated(["series_id", "date"]).any():
        raise ValueError("Panel has duplicate series/date rows.")
    grouped = panel.groupby("series_id")["date"]
    counts = grouped.nunique()
    span = (grouped.max() - grouped.min()).dt.days + 1
    if not counts.equals(span):
        bad = counts.index[counts.to_numpy() != span.to_numpy()].tolist()[:5]
        raise ValueError(f"Series have date gaps, including {bad}.")
