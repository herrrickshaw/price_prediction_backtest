"""Tests for price_prediction.evaluate._metrics."""
import numpy as np
import pandas as pd
import pytest

from price_prediction.evaluate import _metrics


def _s(v):
    return pd.Series(v, index=pd.bdate_range("2020-01-01", periods=len(v)))


def test_metrics_hand_computed():
    p = _s([0.01, -0.02, 0.0, 0.03])
    a = _s([0.02, 0.01, -0.01, 0.01])
    m = _metrics(p, a)
    assert m["n"] == 4
    assert m["dir_acc"] == pytest.approx(2 / 3)     # zero prediction excluded
    assert m["rmse"] == pytest.approx(np.sqrt((0.01**2 + 0.03**2 + 0.01**2 + 0.02**2) / 4))
    strat = np.array([0.02, -0.01, 0.0, 0.01])
    assert m["strat_ann_ret"] == pytest.approx(strat.mean() * 252)
    assert m["bh_ann_ret"] == pytest.approx(np.mean([0.02, 0.01, -0.01, 0.01]) * 252)


def test_metrics_aligns_on_index_and_drops_nan():
    p = _s([0.01, np.nan, 0.02])
    a = _s([0.01, 0.02, -0.02])
    m = _metrics(p, a)
    assert m["n"] == 2 and m["dir_acc"] == pytest.approx(0.5)
