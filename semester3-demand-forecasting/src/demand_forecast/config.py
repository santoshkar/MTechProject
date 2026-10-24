"""Load and validate the Semester 3 experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from demand_forecast.paths import project_root

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "data": {
        "source": "m5",
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "sales_file": "sales_train_evaluation.csv",
        "calendar_file": "calendar.csv",
        "prices_file": "sell_prices.csv",
        "synthetic_days": 160,
        "synthetic_series_per_class": 4,
    },
    "subset": {
        "lookback_days": 365,
        "counts": {"fast": 40, "slow": 40, "intermittent": 40},
        "fast_min_mean": 1.0,
        "slow_max_mean": 0.5,
        "intermittent_min_zero_rate": 0.50,
        "intermittent_min_adi": 1.32,
        "fill_from_medium": True,
    },
    "forecast": {
        "horizon": 28,
        "seasonality": 7,
        "n_windows": 4,
        "min_history_days": 180,
        "quantiles": [0.1, 0.5, 0.9],
    },
    "features": {
        "lags": [1, 7, 14, 28, 56, 364],
        "rolling_windows": [7, 28, 56],
        "train_origin_step": 7,
        "assume_future_price_known": False,
    },
    "models": {
        "seasonal_naive": True,
        "lightgbm": True,
        "chronos": True,
        "chronos_model_id": "autogluon/chronos-2-small",
        "chronos_device": "cpu",
        "chronos_batch_size": 32,
        "chronos_context_length": 512,
        "chronos_cross_learning": False,
        "lightgbm_params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
        },
    },
    "output": {"reports_dir": "reports"},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or (project_root() / "config" / "semester3.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return validate_config(deep_merge(DEFAULTS, loaded))


def apply_demo_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Small synthetic run so the pipeline can be checked without the M5 download."""
    overlay: dict[str, Any] = {
        "data": {
            "source": "synthetic",
            "raw_dir": "data/demo/raw",
            "processed_dir": "data/demo/processed",
            "synthetic_days": 160,
            "synthetic_series_per_class": 4,
        },
        "subset": {
            "lookback_days": 90,
            "counts": {"fast": 4, "slow": 4, "intermittent": 4},
        },
        "forecast": {
            "horizon": 7,
            "n_windows": 2,
            "min_history_days": 28,
        },
        "features": {
            "lags": [1, 7, 14, 28],
            "rolling_windows": [7, 28],
            "train_origin_step": 7,
            "assume_future_price_known": False,
        },
        "models": {
            "seasonal_naive": True,
            "lightgbm": True,
            "chronos": False,
            "lightgbm_params": {
                "n_estimators": 30,
                "learning_rate": 0.1,
                "num_leaves": 15,
                "min_child_samples": 5,
            },
        },
        "output": {"reports_dir": "reports/demo"},
    }
    return validate_config(deep_merge(config, overlay))


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    quantiles = [float(q) for q in config["forecast"]["quantiles"]]
    if any(q <= 0 or q >= 1 for q in quantiles):
        raise ValueError("forecast.quantiles must be strictly between 0 and 1.")
    if 0.5 not in quantiles:
        raise ValueError(
            "forecast.quantiles must include 0.5. LightGBM and Chronos-2 use the "
            "median as the point forecast."
        )
    config["forecast"]["quantiles"] = quantiles
    horizon = int(config["forecast"]["horizon"])
    windows = int(config["forecast"]["n_windows"])
    season = int(config["forecast"]["seasonality"])
    if horizon < 1 or windows < 1 or season < 1:
        raise ValueError("horizon, n_windows, and seasonality must be positive.")
    if config["data"]["source"] not in {"m5", "synthetic"}:
        raise ValueError("data.source must be 'm5' or 'synthetic'.")
    counts = config["subset"]["counts"]
    if not counts or any(int(v) < 1 for v in counts.values()):
        raise ValueError("subset.counts must contain at least one positive class count.")
    return config


def resolve_config_arg(arg: str) -> Path:
    candidate = Path(arg)
    if candidate.exists():
        return candidate.resolve()
    rooted = project_root() / arg
    if rooted.exists():
        return rooted.resolve()
    raise FileNotFoundError(f"Config file not found: {arg}")
