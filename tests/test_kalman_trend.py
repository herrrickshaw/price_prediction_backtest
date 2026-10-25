"""Tests for price_prediction.kalman_trend."""
import numpy as np
import pandas as pd
import pytest

from price_prediction.kalman_trend import KalmanTrendModel


def _series(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_constant_series_has_zero_slope_and_innovation():
    out = KalmanTrendModel().filter(_series(np.full(50, 4.0)))
    assert np.allclose(out["slope"], 0.0)
    assert np.allclose(out["innovation"], 0.0)
    assert np.allclose(out["level"], 4.0)


def test_noise_free_linear_trend_slope_converges_to_drift():
    drift = 0.001
    y = 4.0 + drift * np.arange(2000)
    slope = KalmanTrendModel().predict_return(_series(y))
    assert slope.iloc[-1] == pytest.approx(drift, rel=1e-3)


def test_filter_is_causal():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 0.01, 400)) + 4
    full = KalmanTrendModel().filter(_series(y))
    head = KalmanTrendModel().filter(_series(y[:250]))
    pd.testing.assert_frame_equal(full.iloc[:250], head)


def test_explicit_variances_are_used():
    m = KalmanTrendModel(q_level=1e-5, q_slope=1e-7, r_obs=1e-3)
    assert np.allclose(np.diag(m.Q), [1e-5, 1e-7]) and m.R == 1e-3


def test_filter_is_scale_equivariant():
    # The gain depends only on Q, R and P0 (never on y), so the filter is linear in y:
    # filtering k * log-price gives k * slope. (Fixed Q/R therefore act as a fixed signal-to-noise
    # ratio, which is what the docstring's "variances set from the series' own variance" amounts to.)
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(0.0005, 0.02, 600))
    k = 10.0
    s1 = KalmanTrendModel().predict_return(_series(y))
    sk = KalmanTrendModel().predict_return(_series(k * y))
    assert np.allclose(sk.values, k * s1.values, rtol=1e-6, atol=1e-12)
