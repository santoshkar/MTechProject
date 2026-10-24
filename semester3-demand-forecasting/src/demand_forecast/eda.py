"""Exploratory plots for seasonality and intermittent sales."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from demand_forecast.paths import CATALOG_ALL_FILENAME, CATALOG_FILENAME, EDA_SUMMARY_FILENAME


def run_eda(panel: pd.DataFrame, config: dict, reports_dir: Path, processed_dir: Path) -> None:
    figures = reports_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(processed_dir / CATALOG_FILENAME)
    all_catalog_path = processed_dir / CATALOG_ALL_FILENAME
    all_catalog = pd.read_csv(all_catalog_path) if all_catalog_path.exists() else catalog

    _plot_class_counts(all_catalog, catalog, figures / "demand_class_counts.png")
    _plot_weekday(panel, figures / "weekday_seasonality.png")
    _plot_zero_rate(catalog, figures / "zero_rate.png")
    _plot_examples(panel, figures / "example_series.png")

    summary = catalog.groupby("demand_class").agg(
        n_series=("series_id", "nunique"),
        mean_sales=("mean_sales", "mean"),
        zero_rate=("zero_rate", "mean"),
        adi=("adi", "mean"),
    )
    lines = [
        "# Exploratory summary",
        "",
        "Class labels use only the pre-holdout window. They are not model features.",
        "",
        "Selected sample:",
        "",
        "```",
        summary.round(3).to_string(),
        "```",
        "",
        "Figures:",
        "",
        "- figures/demand_class_counts.png",
        "- figures/weekday_seasonality.png",
        "- figures/zero_rate.png",
        "- figures/example_series.png",
        "",
    ]
    if config["data"]["source"] == "synthetic":
        lines.insert(2, "This file describes synthetic data, not Walmart M5.")
    (reports_dir / EDA_SUMMARY_FILENAME).write_text("\n".join(lines), encoding="utf-8")


def _plot_class_counts(all_catalog: pd.DataFrame, selected: pd.DataFrame, path: Path) -> None:
    labels = ["fast", "medium", "slow", "intermittent"]
    before = all_catalog["demand_class"].value_counts()
    after = selected["demand_class"].value_counts()
    frame = pd.DataFrame({"before_sampling": before, "selected": after}).reindex(labels).fillna(0)
    ax = frame.plot(kind="bar", figsize=(8, 4), rot=0)
    ax.set_ylabel("Series")
    ax.set_title("Demand classes before and after sampling")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_weekday(panel: pd.DataFrame, path: Path) -> None:
    grouped = panel.groupby(["demand_class", "dow"], as_index=False)["sales"].mean()
    plt.figure(figsize=(8, 4))
    for label, group in grouped.groupby("demand_class"):
        plt.plot(group["dow"], group["sales"], marker="o", label=label)
    plt.xticks(range(1, 8), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.ylabel("Mean sales")
    plt.title("Average sales by weekday")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_zero_rate(catalog: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    for label, group in catalog.groupby("demand_class"):
        plt.hist(group["zero_rate"], bins=10, alpha=0.45, label=label)
    plt.xlabel("Zero-sales rate in the classification window")
    plt.ylabel("Series")
    plt.title("Intermittency in the study sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_examples(panel: pd.DataFrame, path: Path) -> None:
    examples = []
    for label, group in panel.groupby("demand_class"):
        series_ids = sorted(group["series_id"].unique())[:2]
        examples.extend((label, series_id) for series_id in series_ids)
    if not examples:
        return
    columns = 2
    rows = (len(examples) + 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(10, 2.4 * rows), sharex=False)
    flat_axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    for ax, (label, series_id) in zip(flat_axes, examples):
        series = panel.loc[panel["series_id"] == series_id].sort_values("date").tail(120)
        ax.plot(series["date"], series["sales"], color="#1f4e79", linewidth=1)
        ax.set_title(f"{label}: {series_id}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in flat_axes[len(examples) :]:
        ax.axis("off")
    fig.suptitle("Recent sales for example series", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
