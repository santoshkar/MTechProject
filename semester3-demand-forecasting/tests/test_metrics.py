import numpy as np
import pytest

from demand_forecast.evaluation.metrics import (
    interval_coverage,
    mase,
    mean_absolute_error,
    pinball_loss,
    seasonal_naive_scale,
)


def test_pinball_underforecast_and_median_identity():
    actual = np.array([0.0, 4.0])
    forecast = np.array([2.0, 4.0])
    assert pinball_loss(actual, forecast, 0.9) == pytest.approx(0.1)
    assert pinball_loss(actual, forecast, 0.5) == pytest.approx(mean_absolute_error(actual, forecast) / 2)


def test_mase_is_undefined_for_a_constant_seasonal_scale():
    history = np.ones(10)
    assert np.isnan(seasonal_naive_scale(history, 7))
    assert np.isnan(mase(np.array([1.0]), np.array([2.0]), seasonal_naive_scale(history, 7)))


def test_interval_coverage():
    actual = np.array([0.5, 2.0])
    assert interval_coverage(actual, np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.5
