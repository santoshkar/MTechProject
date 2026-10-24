"""Project paths and output filenames."""

from __future__ import annotations

from pathlib import Path

PANEL_FILENAME = "panel.parquet"
CATALOG_FILENAME = "series_catalog.csv"
CATALOG_ALL_FILENAME = "series_catalog_all.csv"
METADATA_FILENAME = "run_metadata.json"
FORECASTS_FILENAME = "forecasts.csv"
METRICS_OVERALL_FILENAME = "metrics_overall.csv"
METRICS_BY_CLASS_FILENAME = "metrics_by_class.csv"
METRICS_BY_SERIES_FILENAME = "metrics_by_series.csv"
RUNTIME_FILENAME = "runtime.csv"
EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.md"
EDA_SUMMARY_FILENAME = "eda_summary.md"


def project_root() -> Path:
    """Return the semester 3 project root (the directory that contains pyproject.toml)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path
