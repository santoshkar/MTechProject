"""Explore forecast runs and compare their evaluation results."""

from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from demand_forecast.evaluation.metrics import quantile_column


st.set_page_config(page_title="Retail demand forecast review", page_icon="📈", layout="wide")
st.set_option("client.toolbarMode", "developer")
st.markdown(
    "<style>.stAppDeployButton {display: none !important;}</style>",
    unsafe_allow_html=True,
)
st.title("Retail demand forecast review")
st.caption("Explore held-out forecasts, compare models, and inspect prediction uncertainty.")

M5_REQUIRED_FILES = (
    "sales_train_evaluation.csv",
    "calendar.csv",
    "sell_prices.csv",
)


def _load_csv(path: Path, date_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in date_columns:
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
        "mae_plot": reports / "figures" / "mae_by_class.png",
    }


def _run_pipeline(commands: list[list[str]]) -> tuple[int, list[str]]:
    """Run CLI stages while showing their logs in the dashboard."""
    output: list[str] = []
    log_view = st.empty()
    for command in commands:
        output.append(f"$ {' '.join(command)}")
        log_view.code("\n".join(output[-80:]), language="text")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output.append(line.rstrip())
            log_view.code("\n".join(output[-80:]), language="text")
        return_code = process.wait()
        if return_code:
            return return_code, output
    return 0, output


with st.expander("Run a study", expanded=False):
    st.caption("Start the synthetic demo, M5 baselines, or the full three-model M5 comparison.")
    raw_dir = ROOT / "data" / "raw"
    missing_m5 = [name for name in M5_REQUIRED_FILES if not (raw_dir / name).is_file()]
    chronos_available = importlib.util.find_spec("chronos") is not None
    m5_ready = not missing_m5
    full_m5_ready = m5_ready and chronos_available

    run_columns = st.columns(3)
    demo_clicked = run_columns[0].button("Run synthetic demo", type="primary", use_container_width=True)
    baseline_clicked = run_columns[1].button("Run M5 baselines", disabled=not m5_ready, use_container_width=True)
    full_clicked = run_columns[2].button("Run M5 with Chronos-2", disabled=not full_m5_ready, use_container_width=True)

    if missing_m5:
        st.caption("M5 runs are disabled until these files are added to data/raw: " + ", ".join(missing_m5))
    if m5_ready and not chronos_available:
        st.caption("The full comparison also needs Chronos-2. Install it with `python -m pip install -e '.[chronos]'`.")

    run_option = None
    if demo_clicked:
        run_option = "Synthetic demo"
        commands = [[sys.executable, "-m", "demand_forecast.cli", "demo"]]
    elif baseline_clicked:
        run_option = "M5 baselines only"
        cli = [sys.executable, "-m", "demand_forecast.cli"]
        commands = [cli + ["prepare"], cli + ["eda"], cli + ["forecast", "--skip-chronos"]]
    elif full_clicked:
        run_option = "M5 full comparison with Chronos-2"
        commands = [[sys.executable, "-m", "demand_forecast.cli", "all"]]

    if run_option:
        with st.status(f"Running {run_option}…", expanded=True) as run_status:
            try:
                return_code, run_output = _run_pipeline(commands)
            except OSError as exc:
                return_code, run_output = 1, [str(exc)]
            if return_code == 0:
                run_status.update(label="Run complete", state="complete")
                st.session_state["last_run_notice"] = f"{run_option} completed successfully. Results have been refreshed."
                st.rerun()
            run_status.update(label="Run failed", state="error")
            st.error("The run stopped with an error. Review the command output above.")
            if run_output:
                st.code("\n".join(run_output[-80:]), language="text")

if notice := st.session_state.pop("last_run_notice", None):
    st.success(notice)


m5_forecasts = ROOT / "data" / "processed" / "forecasts.csv"
demo_forecasts = ROOT / "data" / "demo" / "processed" / "forecasts.csv"
default_source = "M5 study" if m5_forecasts.exists() or not demo_forecasts.exists() else "Demo (synthetic)"
source = st.sidebar.radio("Result set", ["M5 study", "Demo (synthetic)"], index=0 if default_source == "M5 study" else 1)
paths = _source_paths(source)

if not paths["forecasts"].exists():
    command = "python -m demand_forecast.cli demo" if source == "Demo (synthetic)" else "python -m demand_forecast.cli all"
    st.info(f"No forecasts found for this result set. From the project folder, run `{command}`.")
    st.stop()

metadata = {}
if paths["metadata"].exists():
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
forecasts = _load_csv(paths["forecasts"], ("date", "origin"))
metrics = _load_csv(paths["metrics"]) if paths["metrics"].exists() else pd.DataFrame()
by_class = _load_csv(paths["by_class"]) if paths["by_class"].exists() else pd.DataFrame()
panel = pd.read_parquet(paths["panel"]) if paths["panel"].exists() else pd.DataFrame()

if metadata.get("is_synthetic") or source == "Demo (synthetic)":
    st.warning("Synthetic demo data: this run checks the pipeline and is not evidence about Walmart sales.")

models = sorted(forecasts["model"].dropna().unique())
classes = sorted(forecasts["demand_class"].dropna().unique())
stores = sorted(forecasts["store_id"].dropna().unique()) if "store_id" in forecasts else []
window_lengths = forecasts.groupby(["series_id", "origin"])["date"].nunique()
horizon = int(window_lengths.mode().iloc[0]) if not window_lengths.empty else 0
n_origins = forecasts["origin"].nunique()
n_series = forecasts["series_id"].nunique()

overview = st.columns(4)
overview[0].metric("Series", f"{n_series:,}")
overview[1].metric("Models", str(len(models)))
overview[2].metric("Forecast windows", str(n_origins))
overview[3].metric("Days per forecast", str(horizon))

tab_forecast, tab_compare, tab_details = st.tabs(["Forecast explorer", "Model comparison", "Run details"])

with tab_forecast:
    st.subheader("Inspect one forecast")
    model_col, class_col, store_col = st.columns(3)
    model = model_col.selectbox("Model", models)
    demand_classes = ["All", *classes]
    demand_class = class_col.selectbox("Demand class", demand_classes)
    store_options = ["All", *stores]
    store = store_col.selectbox("Store", store_options)

    filtered = forecasts.loc[forecasts["model"] == model]
    if demand_class != "All":
        filtered = filtered.loc[filtered["demand_class"] == demand_class]
    if store != "All":
        filtered = filtered.loc[filtered["store_id"] == store]
    series_ids = sorted(filtered["series_id"].dropna().unique())
    if not series_ids:
        st.info("No series match these filters.")
    else:
        series_id = st.selectbox("Product–store series", series_ids)
        origins = sorted(filtered.loc[filtered["series_id"] == series_id, "origin"].dropna().unique())
        origin = st.selectbox("Forecast origin", origins, format_func=lambda value: pd.Timestamp(value).date().isoformat())
        window = filtered.loc[(filtered["series_id"] == series_id) & (filtered["origin"] == origin)].sort_values("date").copy()

        low_col = quantile_column(0.1)
        high_col = quantile_column(0.9)
        has_interval = low_col in window.columns and high_col in window.columns
        actual = pd.to_numeric(window["y_true"], errors="coerce")
        predicted = pd.to_numeric(window["y_pred"], errors="coerce")
        window_mae = float((actual - predicted).abs().mean())
        quick_metrics = st.columns(3)
        quick_metrics[0].metric("Window MAE", f"{window_mae:.2f}")
        quick_metrics[1].metric("Actual sales", f"{actual.sum():.1f}")
        quick_metrics[2].metric("Forecast sales", f"{predicted.sum():.1f}")

        chart_parts = []
        if not panel.empty:
            history = panel.loc[(panel["series_id"] == series_id) & (panel["date"] <= origin)].sort_values("date").tail(90)
            if not history.empty:
                chart_parts.append(pd.DataFrame({"date": history["date"], "Actual history": history["sales"]}))
        forecast_chart = pd.DataFrame({"date": window["date"], "Actual": actual, "Forecast": predicted})
        if has_interval:
            forecast_chart["Lower 80% bound"] = pd.to_numeric(window[low_col], errors="coerce")
            forecast_chart["Upper 80% bound"] = pd.to_numeric(window[high_col], errors="coerce")
        chart_parts.append(forecast_chart)
        chart = pd.concat(chart_parts, ignore_index=True).sort_values("date").set_index("date")
        st.line_chart(chart, height=380)
        if has_interval:
            st.caption("The lower and upper lines mark the model’s 10th and 90th percentile forecasts (an 80% prediction interval).")
        else:
            st.caption("This forecast run does not include prediction interval columns.")

        display_columns = [column for column in ["date", "horizon", "y_true", "y_pred", low_col, high_col] if column in window.columns]
        st.dataframe(window[display_columns].rename(columns={"y_true": "actual", "y_pred": "forecast"}).round(3), use_container_width=True, hide_index=True)
        st.download_button(
            "Download this forecast as CSV",
            data=window.to_csv(index=False).encode("utf-8"),
            file_name=f"{series_id}_{pd.Timestamp(origin).date()}_{model}.csv",
            mime="text/csv",
        )

with tab_compare:
    st.subheader("Overall model scores")
    if metrics.empty:
        st.info("No overall metrics file was found for this run.")
    else:
        columns = [column for column in ["model", "mae", "mase", "coverage", "interval_width", "pinball_0.10", "pinball_0.50", "pinball_0.90"] if column in metrics]
        st.dataframe(metrics[columns].round(3), use_container_width=True, hide_index=True)
        st.caption("Lower MAE, MASE, pinball loss, and interval width are generally better; coverage should be interpreted alongside interval width.")

    if not by_class.empty:
        st.subheader("Scores by demand class")
        class_columns = [column for column in ["model", "demand_class", "mae", "mase", "coverage", "interval_width"] if column in by_class]
        st.dataframe(by_class[class_columns].round(3), use_container_width=True, hide_index=True)
    if paths["mae_plot"].exists():
        st.image(str(paths["mae_plot"]), caption="Macro MAE by demand class")

with tab_details:
    st.subheader("Run information")
    details = {
        "Result set": source,
        "Data source": metadata.get("source_signature", {}).get("source", "unknown"),
        "Series": metadata.get("n_series", n_series),
        "Date range": f"{metadata.get('date_min', 'unknown')} to {metadata.get('date_max', 'unknown')}",
        "Class counts": metadata.get("class_counts", {}),
        "Synthetic data": bool(metadata.get("is_synthetic", source == "Demo (synthetic)")),
    }
    st.json(details)
    st.caption("This page is for forecast review. It does not calculate reorder quantities or estimate inventory savings.")
