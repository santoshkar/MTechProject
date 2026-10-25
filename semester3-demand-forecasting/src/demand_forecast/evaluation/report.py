"""Write metric tables, comparison plots, and the experiment summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from demand_forecast import RESEARCH_QUESTION
from demand_forecast.evaluation.metrics import aggregate_metrics, series_origin_metrics
from demand_forecast.paths import (
    EXPERIMENT_SUMMARY_FILENAME,
    METRICS_BY_CLASS_FILENAME,
    METRICS_BY_SERIES_FILENAME,
    METRICS_OVERALL_FILENAME,
    RUNTIME_FILENAME,
)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(map(str, display.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *body])


def plot_mae_by_class(by_class: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if by_class.empty:
        return
    pivot = by_class.pivot(index="demand_class", columns="model", values="mae")
    ax = pivot.plot(kind="bar", figsize=(8, 4), rot=0)
    ax.set_ylabel("Macro MAE")
    ax.set_xlabel("")
    ax.set_title("Point-forecast MAE by demand class")
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(figures_dir / "mae_by_class.png", dpi=140)
    plt.close()


def write_evaluation_outputs(
    forecasts: pd.DataFrame,
    runtimes: pd.DataFrame,
    config: dict,
    reports_dir: Path,
    metadata: dict,
) -> dict[str, pd.DataFrame]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    quantiles = [float(q) for q in config["forecast"]["quantiles"]]
    by_series = series_origin_metrics(forecasts, quantiles)
    overall, by_class = aggregate_metrics(by_series, quantiles)
    by_series.to_csv(reports_dir / METRICS_BY_SERIES_FILENAME, index=False)
    overall.to_csv(reports_dir / METRICS_OVERALL_FILENAME, index=False)
    by_class.to_csv(reports_dir / METRICS_BY_CLASS_FILENAME, index=False)
    runtimes.to_csv(reports_dir / RUNTIME_FILENAME, index=False)
    plot_mae_by_class(by_class, reports_dir / "figures")
    summary = _summary_markdown(overall, by_class, runtimes, config, metadata)
    (reports_dir / EXPERIMENT_SUMMARY_FILENAME).write_text(summary, encoding="utf-8")
    return {"overall": overall, "by_class": by_class, "by_series": by_series}


def _summary_markdown(
    overall: pd.DataFrame,
    by_class: pd.DataFrame,
    runtimes: pd.DataFrame,
    config: dict,
    metadata: dict,
) -> str:
    synthetic = bool(metadata.get("is_synthetic"))
    banner = ""
    if synthetic:
        banner = (
            "> **Synthetic data.** These numbers only check that the pipeline runs. "
            "Do not put them in the Semester 3 report as empirical results.\n\n"
        )
    models = ", ".join(overall["model"].astype(str).tolist()) if not overall.empty else "none"
    chronos_note = (
        f"Chronos checkpoint: `{config['models']['chronos_model_id']}`."
        if config["models"].get("chronos")
        else "Chronos-2 was not run. The Semester 3 comparison is incomplete until it is."
    )
    metric_cols = [
        column
        for column in ["model", "mae", "mase", "coverage", "interval_width", "pinball_0.10", "pinball_0.50", "pinball_0.90", "n_series"]
        if column in overall.columns
    ]
    class_cols = [column for column in ["model", "demand_class", "mae", "mase", "coverage", "interval_width"] if column in by_class.columns]
    runtime = (
        runtimes.groupby("model", as_index=False)
        .agg(total_seconds=("seconds", "sum"), max_python_memory_mb=("peak_python_memory_mb", "max"))
        .round(3)
    )
    return f"""# Semester 3 experiment summary

{banner}Research question:

{RESEARCH_QUESTION}

## Protocol

- Data source: `{metadata.get("source_signature", {}).get("source", config["data"]["source"])}`.
- Series in the study sample: {metadata.get("n_series", "unknown")}.
- Series selection: {"all input series" if config["subset"].get("all_series", False) else "configured demand-class sample"}.
- Class counts: `{json.dumps(metadata.get("class_counts", {}))}`.
- Horizon: {config["forecast"]["horizon"]} days.
- Non-overlapping chronological windows: {config["forecast"]["n_windows"]}.
- Future price treated as known: {config["features"]["assume_future_price_known"]}.
- Models in this run: {models}.
- {chronos_note}
- LightGBM is refit on each origin using only earlier days. Chronos-2, when enabled, is zero-shot.
- Macro metrics average series-level scores. A few fast sellers do not dominate MASE.
- `peak_python_memory_mb` is Python traced memory, not full process RAM. Check Task Manager for Chronos.

## Point accuracy and uncertainty

{_markdown_table(overall[metric_cols] if metric_cols else overall)}

## By demand class

{_markdown_table(by_class[class_cols] if class_cols else by_class)}

## Runtime

{_markdown_table(runtime)}

## Do not claim

- A zero-sales day is a stockout.
- M5 sales are unconstrained customer demand.
- These forecast scores are inventory savings. Inventory simulation is Semester 4.
- A fixed improvement percentage before this table has been filled by a real M5 run.
- That Chronos-2 has never seen related public retail series during pretraining. If overlap cannot be ruled out from the model card, say so.
"""
