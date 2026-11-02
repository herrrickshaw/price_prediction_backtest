from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from price_prediction.kalman_trend import KalmanTrendModel


def test_filter_matches_baseline(symbol_frame, baseline):
    out = KalmanTrendModel().filter(np.log(symbol_frame["Close"]))
    assert list(out.columns) == ["level", "slope", "innovation"]
    np.testing.assert_allclose(out.iloc[-1].values, baseline["kalman"]["last"], atol=1e-9)
    assert out["slope"].sum() == pytest.approx(baseline["kalman"]["slope_sum"], abs=1e-9)


def test_filter_is_causal():
    """Changing the future must not change earlier filtered values."""
    idx = pd.bdate_range("2020-01-01", periods=120)
    y = pd.Series(np.linspace(4.0, 4.5, 120), index=idx)
    a = KalmanTrendModel().filter(y)
    y2 = y.copy()
    y2.iloc[100:] += 1.0
    b = KalmanTrendModel().filter(y2)
    pd.testing.assert_frame_equal(a.iloc[:100], b.iloc[:100])


def test_slope_tracks_linear_trend():
    idx = pd.bdate_range("2020-01-01", periods=500)
    y = pd.Series(np.arange(500) * 0.001, index=idx)
    slope = KalmanTrendModel().predict_return(y)
    assert slope.iloc[-1] == pytest.approx(0.001, rel=0.05)
