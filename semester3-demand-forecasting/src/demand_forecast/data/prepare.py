"""Build the reproducible study panel."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from demand_forecast.data.classify import classify_wide
from demand_forecast.data.m5 import (
    assert_complete_daily_panel,
    day_columns,
    download_instructions,
    join_calendar_and_prices,
    load_calendar,
    load_prices,
    load_sales,
    melt_selected,
)
from demand_forecast.data.sample import sample_series
from demand_forecast.data.synthetic import write_synthetic_m5
from demand_forecast.paths import (
    CATALOG_ALL_FILENAME,
    CATALOG_FILENAME,
    METADATA_FILENAME,
    PANEL_FILENAME,
    project_root,
    resolve_path,
)

logger = logging.getLogger(__name__)


def config_fingerprint(config: dict) -> str:
    relevant = {
        "seed": config["seed"],
        "subset": {**config["subset"], "all_series": bool(config["subset"].get("all_series", False))},
        "forecast": {
            "horizon": config["forecast"]["horizon"],
            "n_windows": config["forecast"]["n_windows"],
        },
        "data": {
            "source": config["data"]["source"],
            "synthetic_days": config["data"]["synthetic_days"],
            "synthetic_series_per_class": config["data"]["synthetic_series_per_class"],
        },
    }
    payload = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def source_signature(raw_dir: Path, config: dict) -> dict[str, Any]:
    if config["data"]["source"] == "synthetic":
        return {
            "source": "synthetic",
            "seed": int(config["seed"]),
            "days": int(config["data"]["synthetic_days"]),
            "per_class": int(config["data"]["synthetic_series_per_class"]),
        }
    signature: dict[str, Any] = {"source": "m5"}
    for key in ["sales_file", "calendar_file", "prices_file"]:
        path = raw_dir / config["data"][key]
        stat = path.stat()
        signature[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return signature


def processed_dir(config: dict) -> Path:
    return resolve_path(project_root(), config["data"]["processed_dir"])


def raw_dir(config: dict) -> Path:
    return resolve_path(project_root(), config["data"]["raw_dir"])


def prepare_dataset(config: dict) -> pd.DataFrame:
    source_dir = raw_dir(config)
    if config["data"]["source"] == "synthetic":
        logger.warning("Writing synthetic M5-shaped data. Do not report these results as Walmart findings.")
        write_synthetic_m5(source_dir, config)
    sales_path = source_dir / config["data"]["sales_file"]
    calendar_path = source_dir / config["data"]["calendar_file"]
    prices_path = source_dir / config["data"]["prices_file"]
    if not sales_path.exists() or not calendar_path.exists() or not prices_path.exists():
        raise FileNotFoundError(download_instructions())

    sales = load_sales(sales_path)
    columns = day_columns(sales)
    logger.info("Loaded %s series and %s daily columns.", len(sales), len(columns))
    catalog_all = classify_wide(sales, columns, config)
    logger.info("Demand class counts before sampling:\n%s", catalog_all["demand_class"].value_counts().to_string())
    selected_ids = sample_series(catalog_all, config)
    catalog = catalog_all.loc[catalog_all["series_id"].isin(selected_ids)].copy()
    logger.info("Selected %s series.", len(selected_ids))

    sales_long = melt_selected(sales, selected_ids, columns)
    calendar = load_calendar(calendar_path)
    prices = load_prices(prices_path, set(catalog["item_id"]), set(catalog["store_id"]))
    panel = join_calendar_and_prices(sales_long, calendar, prices)
    panel = panel.merge(catalog[["series_id", "demand_class"]], on="series_id", how="left")
    assert_complete_daily_panel(panel)

    out_dir = processed_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_dir / PANEL_FILENAME, index=False)
    catalog.to_csv(out_dir / CATALOG_FILENAME, index=False)
    catalog_all.to_csv(out_dir / CATALOG_ALL_FILENAME, index=False)
    metadata = {
        "fingerprint": config_fingerprint(config),
        "source_signature": source_signature(source_dir, config),
        "is_synthetic": config["data"]["source"] == "synthetic",
        "n_series": int(panel["series_id"].nunique()),
        "n_days": int(panel.groupby("series_id")["date"].nunique().iloc[0]),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "class_counts": catalog["demand_class"].value_counts().to_dict(),
        "assume_future_price_known": bool(config["features"]["assume_future_price_known"]),
    }
    (out_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved panel to %s", out_dir / PANEL_FILENAME)
    return panel


def load_prepared_panel(config: dict, rebuild: bool = False) -> pd.DataFrame:
    out_dir = processed_dir(config)
    panel_path = out_dir / PANEL_FILENAME
    meta_path = out_dir / METADATA_FILENAME
    if rebuild or not panel_path.exists() or not meta_path.exists():
        return prepare_dataset(config)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    current = {
        "fingerprint": config_fingerprint(config),
        "source_signature": source_signature(raw_dir(config), config) if config["data"]["source"] == "m5" else metadata.get("source_signature"),
    }
    stale = metadata.get("fingerprint") != current["fingerprint"]
    if config["data"]["source"] == "m5":
        stale = stale or metadata.get("source_signature") != current["source_signature"]
    if config["data"]["source"] == "synthetic":
        stale = stale or metadata.get("source_signature") != source_signature(raw_dir(config), config)
    if stale:
        logger.info("Processed panel is stale. Rebuilding.")
        return prepare_dataset(config)
    logger.info("Loading existing panel from %s", panel_path)
    return pd.read_parquet(panel_path)
