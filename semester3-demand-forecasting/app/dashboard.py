"""Explore forecast runs and compare their evaluation results."""

from __future__ import annotations

import json
import importlib.util
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st
import yaml

from demand_forecast.config import load_config
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
MAX_UPLOAD_SERIES = 10_000


def _minimum_upload_days() -> int:
    config = load_config()
    forecast = config["forecast"]
    return int(forecast["min_history_days"]) + int(forecast["horizon"]) * (int(forecast["n_windows"]) + 1)


def _load_csv(path: Path, date_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in date_columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def _source_paths(source: str) -> dict[str, Path]:
    if source == "Demo (synthetic)":
        raw = ROOT / "data" / "demo" / "raw"
        processed = ROOT / "data" / "demo" / "processed"
        reports = ROOT / "reports" / "demo"
    elif source.startswith("Uploaded: "):
        run_dir = ROOT / "data" / "uploads" / source.removeprefix("Uploaded: ")
        raw = run_dir / "raw"
        processed = run_dir / "processed"
        reports = ROOT / "reports" / "uploads" / run_dir.name
    else:
        raw = ROOT / "data" / "raw"
        processed = ROOT / "data" / "processed"
        reports = ROOT / "reports"
    return {
        "raw_sales": raw / "sales_train_evaluation.csv",
        "raw_calendar": raw / "calendar.csv",
        "raw_prices": raw / "sell_prices.csv",
        "forecasts": processed / "forecasts.csv",
        "panel": processed / "panel.parquet",
        "catalog_all": processed / "series_catalog_all.csv",
        "metrics": reports / "metrics_overall.csv",
        "by_class": reports / "metrics_by_class.csv",
        "metadata": processed / "run_metadata.json",
        "summary": reports / "experiment_summary.md",
        "mae_plot": reports / "figures" / "mae_by_class.png",
    }


def _prepare_uploaded_workbook(uploaded_file, include_all_series: bool) -> tuple[str, Path, dict[str, int]]:
    """Validate an M5-shaped workbook, save its three sheets, and write run config."""
    workbook = pd.ExcelFile(uploaded_file, engine="openpyxl")
    available = {re.sub(r"[^a-z0-9]", "", sheet.casefold()): sheet for sheet in workbook.sheet_names}
    aliases = {
        "sales": ("sales", "salestrainevaluation", "salestrainvalidation"),
        "calendar": ("calendar",),
        "prices": ("sellprices", "prices"),
    }
    selected_sheets = {}
    for key, names in aliases.items():
        match = next((available[name] for name in names if name in available), None)
        if match is None:
            raise ValueError(
                "Workbook must have three sheets: sales (or sales_train_evaluation), "
                "calendar, and sell_prices (or prices)."
            )
        selected_sheets[key] = match

    sales = pd.read_excel(workbook, sheet_name=selected_sheets["sales"], engine="openpyxl")
    calendar = pd.read_excel(workbook, sheet_name=selected_sheets["calendar"], engine="openpyxl")
    prices = pd.read_excel(workbook, sheet_name=selected_sheets["prices"], engine="openpyxl")
    for frame in (sales, calendar, prices):
        frame.columns = [str(column).strip() for column in frame.columns]
    if len(sales) > MAX_UPLOAD_SERIES:
        raise ValueError(
            f"The sales sheet has {len(sales):,} series rows. The current upload limit is {MAX_UPLOAD_SERIES:,}."
        )

    sales_required = {"item_id", "dept_id", "cat_id", "store_id", "state_id"}
    calendar_required = {"d", "date", "wm_yr_wk"}
    prices_required = {"store_id", "item_id", "wm_yr_wk", "sell_price"}
    day_columns = [column for column in sales.columns if re.fullmatch(r"d_\d+", column)]
    missing = {
        "sales": sorted(sales_required - set(sales.columns)),
        "calendar": sorted(calendar_required - set(calendar.columns)),
        "sell_prices": sorted(prices_required - set(prices.columns)),
    }
    missing = {name: columns for name, columns in missing.items() if columns}
    if missing:
        details = "; ".join(f"{name} is missing {', '.join(columns)}" for name, columns in missing.items())
        raise ValueError(f"Workbook columns do not match the M5 format: {details}.")
    if not day_columns:
        raise ValueError("The sales sheet needs daily columns named d_1, d_2, and so on.")
    day_columns.sort(key=lambda column: int(column.split("_", 1)[1]))
    minimum_days = _minimum_upload_days()
    if len(day_columns) < minimum_days:
        raise ValueError(f"The sales sheet has {len(day_columns)} days; at least {minimum_days} daily columns are required for the configured backtest.")
    if sales.empty or calendar.empty or prices.empty:
        raise ValueError("The sales, calendar, and sell_prices sheets must each contain data rows.")
    expected_days = [f"d_{index}" for index in range(1, len(day_columns) + 1)]
    if day_columns != expected_days:
        raise ValueError("Sales day columns must be consecutive and start at d_1.")
    calendar_days = set(calendar["d"].astype(str).str.strip())
    missing_days = [day for day in day_columns if day not in calendar_days]
    if missing_days:
        raise ValueError(f"The calendar sheet is missing sales dates for {len(missing_days)} day IDs, such as {missing_days[:3]}.")
    if calendar["d"].astype(str).duplicated().any():
        raise ValueError("The calendar sheet has duplicate day IDs in column d.")
    if "id" in sales.columns:
        duplicate_series = sales["id"].duplicated().any()
    else:
        duplicate_series = sales.duplicated(["item_id", "store_id"]).any()
    if duplicate_series:
        raise ValueError("The sales sheet has duplicate series IDs or duplicate item_id/store_id pairs.")

    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(uploaded_file.name).stem).strip("_")[:36] or "workbook"
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{safe_stem}"
    run_dir = ROOT / "data" / "uploads" / run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)
    sales.to_csv(raw_dir / "sales_train_evaluation.csv", index=False)
    calendar.to_csv(raw_dir / "calendar.csv", index=False)
    prices.to_csv(raw_dir / "sell_prices.csv", index=False)

    processed_dir = run_dir / "processed"
    reports_dir = ROOT / "reports" / "uploads" / run_id
    config = {
        "data": {
            "source": "m5",
            "raw_dir": str(raw_dir.relative_to(ROOT)),
            "processed_dir": str(processed_dir.relative_to(ROOT)),
        },
        "subset": {"all_series": bool(include_all_series)},
        "output": {"reports_dir": str(reports_dir.relative_to(ROOT))},
    }
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return run_id, config_path, {"series": len(sales), "days": len(day_columns)}


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
    st.caption("Run the synthetic demo or an M5 comparison using local files or an uploaded workbook.")
    raw_dir = ROOT / "data" / "raw"
    missing_m5 = [name for name in M5_REQUIRED_FILES if not (raw_dir / name).is_file()]
    chronos_available = importlib.util.find_spec("chronos") is not None
    excel_available = importlib.util.find_spec("openpyxl") is not None

    m5_input = st.radio(
        "M5 data source",
        ["Files in data/raw", "Upload Excel workbook"],
        index=0 if not missing_m5 else 1,
        horizontal=True,
    )
    uploaded_file = None
    if m5_input == "Upload Excel workbook":
        st.markdown("#### Workbook format")
        minimum_upload_days = _minimum_upload_days()
        st.caption(
            "Use one .xlsx workbook with sales, calendar, and sell_prices sheets. "
            f"The sales sheet needs M5 columns plus at least {minimum_upload_days} consecutive d_1…d_N daily columns."
        )
        st.caption(
            "Required columns: sales = item_id, dept_id, cat_id, store_id, state_id; "
            "calendar = d, date, wm_yr_wk; sell_prices = store_id, item_id, wm_yr_wk, sell_price."
        )
        st.caption(f"Upload limit: {MAX_UPLOAD_SERIES:,} series rows in the sales sheet.")
        series_scope = st.radio(
            "Series to forecast",
            ["All uploaded series", "Research sample"],
            index=0,
            horizontal=True,
            help="All uploaded series includes zero-sales series too and may take much longer. Research sample uses the configured demand-class targets.",
        )
        include_all_series = series_scope == "All uploaded series"
        if include_all_series:
            st.warning("All-series mode can take much longer and use substantially more memory, especially for thousands of series.")
        uploaded_file = st.file_uploader("Excel workbook (.xlsx)", type=["xlsx"], key="m5_workbook_upload")
        if not excel_available:
            st.caption("Excel support is not installed. Install the dashboard extra with `python -m pip install -e '.[dashboard]'`.")
        elif uploaded_file is None:
            st.caption("Upload a workbook to enable the M5 run buttons. The file is kept in a separate run folder.")
        m5_ready = uploaded_file is not None and excel_available
    else:
        m5_ready = not missing_m5
        if missing_m5:
            st.caption("Add these files to data/raw or switch to Upload Excel workbook: " + ", ".join(missing_m5))

    full_m5_ready = m5_ready and chronos_available

    run_columns = st.columns(3)
    demo_clicked = run_columns[0].button("Run synthetic demo", type="primary", use_container_width=True)
    baseline_clicked = run_columns[1].button("Run M5 baselines", disabled=not m5_ready, use_container_width=True)
    full_clicked = run_columns[2].button("Run M5 with Chronos-2", disabled=not full_m5_ready, use_container_width=True)

    if m5_ready and not chronos_available:
        st.caption("The full comparison also needs Chronos-2. Install it with `python -m pip install -e '.[chronos]'`.")

    run_option = None
    upload_error = None
    if demo_clicked:
        run_option = "Synthetic demo"
        commands = [[sys.executable, "-m", "demand_forecast.cli", "demo"]]
    elif baseline_clicked or full_clicked:
        is_full_run = full_clicked
        run_option = "M5 full comparison with Chronos-2" if is_full_run else "M5 baselines only"
        if m5_input == "Upload Excel workbook":
            try:
                run_id, upload_config, workbook_info = _prepare_uploaded_workbook(uploaded_file, include_all_series)
                st.session_state["last_uploaded_run_id"] = run_id
                cli = [sys.executable, "-m", "demand_forecast.cli", "--config", str(upload_config)]
                commands = [cli + ["all"]] if is_full_run else [
                    cli + ["prepare"], cli + ["eda"], cli + ["forecast", "--skip-chronos"]
                ]
                scope_name = "all series" if include_all_series else "research sample"
                run_option += f" on uploaded data ({scope_name}; {workbook_info['series']} input series, {workbook_info['days']} days)"
            except Exception as exc:
                upload_error = str(exc)
        else:
            cli = [sys.executable, "-m", "demand_forecast.cli"]
            commands = [cli + ["all"]] if is_full_run else [
                cli + ["prepare"], cli + ["eda"], cli + ["forecast", "--skip-chronos"]
            ]

    if upload_error:
        st.error(upload_error)

    if run_option and not upload_error:
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


upload_root = ROOT / "data" / "uploads"
upload_runs = sorted(
    [
        path
        for path in upload_root.iterdir()
        if (path / "processed" / "forecasts.csv").exists()
        and (ROOT / "reports" / "uploads" / path.name / "experiment_summary.md").exists()
    ],
    reverse=True,
) if upload_root.exists() else []
m5_forecasts = ROOT / "data" / "processed" / "forecasts.csv"
demo_forecasts = ROOT / "data" / "demo" / "processed" / "forecasts.csv"
source_options = ["M5 study", "Demo (synthetic)", *(f"Uploaded: {path.name}" for path in upload_runs)]
last_uploaded_id = st.session_state.get("last_uploaded_run_id")
last_uploaded_label = f"Uploaded: {last_uploaded_id}" if last_uploaded_id else None
if last_uploaded_label in source_options:
    default_source = last_uploaded_label
elif m5_forecasts.exists():
    default_source = "M5 study"
elif upload_runs:
    default_source = f"Uploaded: {upload_runs[0].name}"
elif demo_forecasts.exists():
    default_source = "Demo (synthetic)"
else:
    default_source = "M5 study"
source = st.sidebar.radio("Result set", source_options, index=source_options.index(default_source))
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
catalog_all = _load_csv(paths["catalog_all"]) if source.startswith("Uploaded: ") and paths["catalog_all"].exists() else pd.DataFrame()

if metadata.get("is_synthetic") or source == "Demo (synthetic)":
    st.warning("Synthetic demo data: this run checks the pipeline and is not evidence about Walmart sales.")

models = sorted(forecasts["model"].dropna().unique())
classes = sorted(forecasts["demand_class"].dropna().unique())
stores = sorted(forecasts["store_id"].dropna().unique()) if "store_id" in forecasts else []
window_lengths = forecasts.groupby(["series_id", "origin"])["date"].nunique()
horizon = int(window_lengths.mode().iloc[0]) if not window_lengths.empty else 0
n_origins = forecasts["origin"].nunique()
n_series = forecasts["series_id"].nunique()
uploaded_series_count = int(catalog_all["series_id"].nunique()) if not catalog_all.empty else n_series

overview = st.columns(4)
overview[0].metric(
    "Forecasted / uploaded series" if source.startswith("Uploaded: ") else "Series",
    f"{n_series:,} / {uploaded_series_count:,}" if source.startswith("Uploaded: ") else f"{n_series:,}",
)
overview[1].metric("Models", str(len(models)))
overview[2].metric("Forecast windows", str(n_origins))
overview[3].metric("Days per forecast", str(horizon))
if source.startswith("Uploaded: ") and uploaded_series_count > n_series:
    st.info(
        f"This run forecast {n_series:,} of {uploaded_series_count:,} uploaded series. "
        "The study uses a demand-class sample; the Product–store series list contains only series with forecasts. "
        "Open Uploaded data → Series catalog to inspect every uploaded series."
    )

tab_names = ["Forecast explorer", "Model comparison", "Run details"]
if source.startswith("Uploaded: "):
    tab_names.append("Uploaded data")
tabs = st.tabs(tab_names)
tab_forecast, tab_compare, tab_details = tabs[:3]

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
        if source.startswith("Uploaded: ") and uploaded_series_count > n_series:
            st.caption(f"This selector lists the {n_series:,} forecasted sample series, not all {uploaded_series_count:,} input series.")
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

        st.caption("Forecast table: one selected series and one forecast window. Open Uploaded data to inspect the workbook tables.")
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
        "Uploaded source series": uploaded_series_count,
        "Forecasted series": n_series,
        "Synthetic data": bool(metadata.get("is_synthetic", source == "Demo (synthetic)")),
    }
    st.json(details)
    st.caption("This page is for forecast review. It does not calculate reorder quantities or estimate inventory savings.")

if source.startswith("Uploaded: "):
    with tabs[3]:
        st.subheader("Uploaded workbook tables")
        table_paths = {
            "Sales": paths["raw_sales"],
            "Calendar": paths["raw_calendar"],
            "Sell prices": paths["raw_prices"],
            "Series catalog (all uploaded)": paths["catalog_all"],
        }
        table_name = st.selectbox("Workbook sheet", list(table_paths))
        table_path = table_paths[table_name]
        if not table_path.exists():
            st.error(f"Uploaded source table not found: {table_path.name}")
        elif st.checkbox("Load the complete sheet into the table", value=True, help="Large sheets can take a moment to load."):
            full_table = pd.read_csv(table_path)
            st.caption(f"Showing all {len(full_table):,} rows and {len(full_table.columns):,} columns from {table_name.lower()}.")
            st.dataframe(full_table, use_container_width=True, height=600, hide_index=True)
            st.download_button(
                f"Download full {table_name.lower()} sheet as CSV",
                data=table_path.read_bytes(),
                file_name=f"{source.removeprefix('Uploaded: ')}_{table_name.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )
