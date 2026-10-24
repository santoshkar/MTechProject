"""Global LightGBM quantile models, refit on each forecast origin's past only."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from demand_forecast.evaluation.metrics import quantile_column
from demand_forecast.models.common import enforce_nonnegative_quantile_order, quantile_frame
from demand_forecast.models.features import (
    CATEGORICAL_FEATURES,
    build_prediction_frame,
    build_training_frame,
    numeric_feature_names,
)

logger = logging.getLogger(__name__)


class LightGBMForecaster:
    name = "lightgbm"

    def __init__(self, config: dict):
        self.config = config
        self.models_: dict[float, object] = {}
        self.category_maps_: dict[str, pd.Index] = {}
        self.feature_columns_: list[str] = []

    def fit(self, history: pd.DataFrame) -> None:
        import lightgbm as lgb

        logger.info("Building LightGBM training rows from history available at this origin.")
        train = build_training_frame(history, self.config)
        if train.empty:
            raise ValueError(
                "LightGBM has no training rows. Reduce min_history_days or horizon, "
                "or check that each series is long enough."
            )
        numeric = numeric_feature_names(self.config)
        self.feature_columns_ = numeric + CATEGORICAL_FEATURES
        features = train[self.feature_columns_].copy()
        for column in CATEGORICAL_FEATURES:
            features[column] = features[column].astype("category")
            self.category_maps_[column] = features[column].cat.categories
        target = train["target"].to_numpy(dtype=float)
        params = dict(self.config["models"]["lightgbm_params"])
        quantiles = [float(q) for q in self.config["forecast"]["quantiles"]]
        self.models_ = {}
        for quantile in quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=int(params.get("n_estimators", 200)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                num_leaves=int(params.get("num_leaves", 31)),
                min_child_samples=int(params.get("min_child_samples", 20)),
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=int(self.config["seed"]),
                n_jobs=-1,
                verbose=-1,
            )
            logger.info("Fitting LightGBM quantile %.2f on %s rows.", quantile, len(train))
            model.fit(features, target, categorical_feature=CATEGORICAL_FEATURES)
            self.models_[quantile] = model

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        if not self.models_:
            raise RuntimeError("LightGBMForecaster.predict called before fit.")
        frame = build_prediction_frame(history, future, self.config)
        features = frame[self.feature_columns_].copy()
        for column in CATEGORICAL_FEATURES:
            features[column] = pd.Categorical(features[column], categories=self.category_maps_[column])
        raw = {quantile: model.predict(features) for quantile, model in self.models_.items()}
        ordered = enforce_nonnegative_quantile_order(raw)
        median = ordered[0.5]
        return pd.DataFrame(
            {
                "series_id": frame["series_id"].to_numpy(),
                "date": frame["date"].to_numpy(),
                "horizon": frame["horizon"].to_numpy(),
                "y_pred": median,
                **quantile_frame(ordered),
            }
        )


def median_column() -> str:
    return quantile_column(0.5)
