"""Reproducible series selection."""

from __future__ import annotations

import logging
import random

import pandas as pd

logger = logging.getLogger(__name__)


def sample_series(catalog: pd.DataFrame, config: dict) -> list[str]:
    if config["subset"].get("all_series", False):
        selected = sorted(catalog["series_id"].astype(str).unique())
        logger.info("Including all %s input series in the study.", len(selected))
        return selected

    eligible = catalog.loc[catalog["mean_sales"] > 0].copy()
    dropped = len(catalog) - len(eligible)
    if dropped:
        logger.info("Dropped %s series with zero mean sales in the classification window.", dropped)
    rng = random.Random(int(config["seed"]))
    selected: list[str] = []
    shortfall = 0
    for label, count in config["subset"]["counts"].items():
        pool = sorted(eligible.loc[eligible["demand_class"] == label, "series_id"].astype(str))
        take = min(int(count), len(pool))
        shortfall += int(count) - take
        if take < int(count):
            logger.warning("Only %s '%s' series are available; requested %s.", take, label, count)
        chosen = pool if take == len(pool) else rng.sample(pool, take)
        selected.extend(chosen)
    if shortfall and config["subset"].get("fill_from_medium", True):
        pool = sorted(set(eligible.loc[eligible["demand_class"] == "medium", "series_id"].astype(str)) - set(selected))
        extra_n = min(shortfall, len(pool))
        extra = pool if extra_n == len(pool) else rng.sample(pool, extra_n)
        if extra:
            logger.warning(
                "Added %s medium series because a requested demand class was short. Their label is still medium.",
                len(extra),
            )
        selected.extend(extra)
    if not selected:
        raise ValueError("No series were selected. Relax subset thresholds or check the input data.")
    return sorted(set(selected))
