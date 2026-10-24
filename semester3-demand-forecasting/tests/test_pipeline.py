import numpy as np
import pandas as pd
import pytest

from demand_forecast.config import apply_demo_overrides, load_config
from demand_forecast.models.chronos_forecaster import normalize_chronos_predictions
from demand_forecast.study import run_eda_stage, run_forecast_study, run_prepare


def test_chronos_output_normalizer_reads_quantile_columns():
    pred = pd.DataFrame(
        {
            "series_id": ["A", "A"],
            "date": pd.to_datetime(["2020-02-01", "2020-02-02"]),
            "predictions": [1.5, -1.0],
            "0.1": [-2.0, -3.0],
            "0.5": [1.2, 0.4],
            "0.9": [3.0, 2.0],
        }
    )
    normalized = normalize_chronos_predictions(pred, [0.1, 0.5, 0.9], "series_id", "date")
    assert normalized["q_0.10"].min() >= 0
    assert np.all(normalized["q_0.10"] <= normalized["q_0.50"])
    assert np.all(normalized["q_0.50"] <= normalized["q_0.90"])


@pytest.mark.skipif(pytest.importorskip("lightgbm") is None, reason="lightgbm is not installed")
def test_synthetic_study_runs_without_using_future_as_history(tmp_path):
    config = apply_demo_overrides(load_config())
    config["data"]["raw_dir"] = str(tmp_path / "raw")
    config["data"]["processed_dir"] = str(tmp_path / "processed")
    config["output"]["reports_dir"] = str(tmp_path / "reports")
    config["models"]["chronos"] = False
    panel = run_prepare(config)
    assert panel["series_id"].nunique() == 12
    assert set(panel["demand_class"]) >= {"fast", "slow", "intermittent"}
    forecasts = run_forecast_study(config)
    assert set(forecasts["model"]) == {"seasonal_naive", "lightgbm"}
    assert forecasts["y_pred"].ge(0).all()
    assert (forecasts["date"] > forecasts["origin"]).all()
    run_eda_stage(config)
    assert (tmp_path / "reports" / "experiment_summary.md").exists()
    assert (tmp_path / "reports" / "figures" / "weekday_seasonality.png").exists()
    assert "Synthetic data" in (tmp_path / "reports" / "experiment_summary.md").read_text(encoding="utf-8")
