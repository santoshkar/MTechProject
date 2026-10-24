"""Small Semester 3 dashboard: actual sales, forecasts, and uncertainty.

Inventory recommendations are not shown. That extension belongs to Semester 4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from demand_forecast.evaluation.metrics import quantile_column

st.set_page_config(page_title="Semester 3 demand forecasts", layout="wide")
st.title("Semester 3 demand forecasts")
st.caption("Actual sales, point forecasts, and an 80% prediction interval. This is not an inventory-ordering screen.")


def _load_csv(path: Path, dates: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in dates:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def _source_paths(source: str) -> dict[str, Path]:
    if source == "Demo (synthetic)":
        processed = ROOT / "data" / "demo" / "processed"
        reports = ROOT / "reports" / "demo"
    else:
        processed = ROOT / "data" / "processed"
        reports = ROOT / "reports"
    return {
        "forecasts": processed / "forecasts.csv",
        "panel": processed / "panel.parquet",
        "metrics": reports / "metrics_overall.csv",
        "by_class": reports / "metrics_by_class.csv",
        "metadata": processed / "run_metadata.json",
    }


source = st.sidebar.radio("Results", ["M5 study", "Demo (synthetic)"], index=0)
paths = _source_paths(source)
if not paths["forecasts"].exists():
    command = "python -m demand_forecast.cli demo" if source.startswith("Demo") else "python -m demand_forecast.cli all"
    st.warning(f"No forecasts found. From the project folder, run `{command}`.")
    st.stop()

metadata = {}
if paths["metadata"].exists():
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
if metadata.get("is_synthetic"):
    st.error("These results are synthetic. They are a pipeline check, not evidence about Walmart.")

forecasts = _load_csv(paths["forecasts"], ["date", "origin"])
metrics = _load_csv(paths["metrics"], []) if paths["metrics"].exists() else pd.DataFrame()
by_class = _load_csv(paths["by_class"], []) if paths["by_class"].exists() else pd.DataFrame()
panel = pd.read_parquet(paths["panel"]) if paths["panel"].exists() else pd.DataFrame()

models = sorted(forecasts["model"].unique())
stores = sorted(forecasts["store_id"].dropna().unique()) if "store_id" in forecasts.columns else []
classes = sorted(forecasts["demand_class"].dropna().unique())
model = st.sidebar.selectbox("Model", models)
store = st.sidebar.selectbox("Store", ["All", *stores])
demand_class = st.sidebar.selectbox("Demand class", ["All", *classes])
filtered = forecasts.loc[forecasts["model"] == model]
if store != "All":
    filtered = filtered.loc[filtered["store_id"] == store]
if demand_class != "All":
    filtered = filtered.loc[filtered["demand_class"] == demand_class]
series_ids = sorted(filtered["series_id"].unique())
if not series_ids:
    st.warning("No series match these filters.")
    st.stop()
series_id = st.sidebar.selectbox("Series", series_ids)
origins = sorted(filtered.loc[filtered["series_id"] == series_id, "origin"].unique())
origin = st.sidebar.selectbox("Forecast origin", origins, format_func=lambda value: pd.Timestamp(value).date().isoformat())

window = filtered.loc[(filtered["series_id"] == series_id) & (filtered["origin"] == origin)].sort_values("date")
low_col = quantile_column(0.1) if quantile_column(0.1) in window.columns else None
high_col = quantile_column(0.9) if quantile_column(0.9) in window.columns else None

left, right = st.columns([2, 1])
with left:
    st.subheader(series_id)
    if not panel.empty:
        history = panel.loc[(panel["series_id"] == series_id) & (panel["date"] <= origin)].sort_values("date").tail(90)
        chart = pd.DataFrame({"date": history["date"], "actual history": history["sales"]})
        future_chart = pd.DataFrame(
            {
                "date": window["date"],
                "actual": window["y_true"],
                "forecast": window["y_pred"],
            }
        )
        if low_col and high_col:
            future_chart["lower"] = window[low_col]
            future_chart["upper"] = window[high_col]
        combined = pd.concat([chart, future_chart], ignore_index=True)
        st.line_chart(combined, x="date", y=[column for column in ["actual history", "actual", "forecast"] if column in combined.columns])
        if low_col and high_col:
            st.caption("The line chart shows the point forecast. The table includes the prediction interval.")
    st.dataframe(window[["date", "horizon", "y_true", "y_pred", low_col, high_col]].round(3), use_container_width=True)
with right:
    st.subheader("Study metrics")
    if not metrics.empty:
        show = [column for column in ["model", "mae", "mase", "coverage", "interval_width"] if column in metrics.columns]
        st.dataframe(metrics[show].round(3), use_container_width=True)
    if not by_class.empty:
        st.subheader("By demand class")
        show = [column for column in ["model", "demand_class", "mae", "mase", "coverage"] if column in by_class.columns]
        st.dataframe(by_class[show].round(3), use_container_width=True)
    st.info("Semester 4 will turn these uncertainty estimates into simulated replenishment decisions. That is not implemented here.")
