"""Shared forecast post-processing."""

from __future__ import annotations

import numpy as np

from demand_forecast.evaluation.metrics import quantile_column


def enforce_nonnegative_quantile_order(quantile_map: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Clip sales forecasts at zero and restore quantile order after clipping."""
    levels = sorted(quantile_map)
    stacked = np.vstack([np.maximum(np.asarray(quantile_map[q], dtype=float), 0.0) for q in levels])
    stacked.sort(axis=0)
    return {q: stacked[i] for i, q in enumerate(levels)}


def quantile_frame(levels: dict[float, np.ndarray]) -> dict[str, np.ndarray]:
    return {quantile_column(q): values for q, values in levels.items()}
