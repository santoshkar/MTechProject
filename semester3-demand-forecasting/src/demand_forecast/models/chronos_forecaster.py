"""Zero-shot Chronos-2 forecasts.

The model is pretrained. This project does not train a neural network.
Calendar covariates are treated as known. Price is a future covariate only when
the experiment config explicitly allows that assumption.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from demand_forecast.models.common import enforce_nonnegative_quantile_order, quantile_frame

logger = logging.getLogger(__name__)

KNOWN_FUTURE_COVARIATES = ["dow", "month", "weekofyear", "is_event", "snap"]


def quantile_key_candidates(q: float) -> list[str]:
    return [str(q), f"{q:.1f}", f"{q:.2f}", f"{q:g}"]


def extract_quantile(frame: pd.DataFrame, q: float) -> np.ndarray:
    for name in quantile_key_candidates(q):
        if name in frame.columns:
            return frame[name].to_numpy(dtype=float)
    raise KeyError(f"Chronos output has no column for quantile {q}. Columns: {list(frame.columns)}")


def normalize_chronos_predictions(
    pred_df: pd.DataFrame,
    quantiles: list[float],
    id_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    ordered_input = {q: extract_quantile(pred_df, q) for q in quantiles}
    ordered = enforce_nonnegative_quantile_order(ordered_input)
    if "predictions" in pred_df.columns:
        point = np.maximum(pred_df["predictions"].to_numpy(dtype=float), 0.0)
    else:
        point = ordered[0.5]
    # Keep the reported point forecast inside the repaired interval.
    point = np.minimum(np.maximum(point, ordered[min(quantiles)]), ordered[max(quantiles)])
    return pd.DataFrame(
        {
            "series_id": pred_df[id_column].to_numpy(),
            "date": pred_df[timestamp_column].to_numpy(),
            "y_pred": point,
            **quantile_frame(ordered),
        }
    )


class ChronosForecaster:
    name = "chronos2"

    def __init__(self, config: dict):
        self.config = config
        self.pipeline = None

    def load(self) -> None:
        if self.pipeline is not None:
            return
        try:
            import torch
            from chronos import Chronos2Pipeline
        except ImportError as exc:
            raise ImportError(
                "Chronos-2 is enabled but not installed. From the project folder run: "
                'pip install -e ".[chronos]"   or set models.chronos to false for a baseline-only run.'
            ) from exc
        device = str(self.config["models"]["chronos_device"])
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA was requested but is not available. Using CPU.")
            device = "cpu"
        model_id = self.config["models"]["chronos_model_id"]
        kwargs = {"device_map": device}
        if device == "cpu":
            kwargs["torch_dtype"] = torch.float32
        logger.info("Loading pretrained Chronos-2 model %s on %s.", model_id, device)
        logger.info("The first run downloads weights from Hugging Face and can take several minutes.")
        self.pipeline = Chronos2Pipeline.from_pretrained(model_id, **kwargs)

    def _context_and_future(self, history: pd.DataFrame, future: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        context_length = int(self.config["models"]["chronos_context_length"])
        assume_price = bool(self.config["features"]["assume_future_price_known"])
        context_cols = ["series_id", "date", "sales", *KNOWN_FUTURE_COVARIATES]
        future_cols = ["series_id", "date", *KNOWN_FUTURE_COVARIATES]
        if assume_price:
            context_cols.append("sell_price")
            future_cols.append("sell_price")
        else:
            # Past-only price. Chronos treats a column as known into the future
            # only when it is also present in future_df.
            context_cols.append("sell_price")
        context = history.loc[:, context_cols].sort_values(["series_id", "date"])
        context = context.groupby("series_id", sort=False).tail(context_length)
        future_df = future.loc[:, future_cols].sort_values(["series_id", "date"])
        return context, future_df

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        self.load()
        horizon = int(self.config["forecast"]["horizon"])
        quantiles = [float(q) for q in self.config["forecast"]["quantiles"]]
        context, future_df = self._context_and_future(history, future)
        logger.info("Running Chronos-2 on %s series.", context["series_id"].nunique())
        pred = self.pipeline.predict_df(
            context,
            future_df=future_df,
            prediction_length=horizon,
            quantile_levels=quantiles,
            id_column="series_id",
            timestamp_column="date",
            target="sales",
            batch_size=int(self.config["models"]["chronos_batch_size"]),
            context_length=int(self.config["models"]["chronos_context_length"]),
            cross_learning=bool(self.config["models"]["chronos_cross_learning"]),
            freq="D",
        )
        normalized = normalize_chronos_predictions(pred, quantiles, "series_id", "date")
        # predict_df may repeat ids in model order. Attach horizon from the date order.
        normalized["date"] = pd.to_datetime(normalized["date"])
        normalized = normalized.sort_values(["series_id", "date"]).reset_index(drop=True)
        normalized["horizon"] = normalized.groupby("series_id").cumcount() + 1
        return normalized
