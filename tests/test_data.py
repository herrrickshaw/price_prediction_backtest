"""Tests for price_prediction.data (synthetic panels only)."""
import numpy as np
import pandas as pd
import pytest

from price_prediction.data import symbol_series, top_liquid_symbols


def _panel():
    d = pd.bdate_range("2010-01-01", periods=2300)
    rows = []
    for sym, px, vol in [("A", 10.0, 1e6), ("B", 50.0, 1e6), ("C", 5.0, 1e3)]:
        rows.append(pd.DataFrame({"Date": d, "Symbol": sym, "Open": px, "High": px, "Low": px,
                                  "Close": px, "Volume": vol}))
    short = pd.DataFrame({"Date": d[:300], "Symbol": "S", "Open": 1e4, "High": 1e4, "Low": 1e4,
                          "Close": 1e4, "Volume": 1e9})
    return pd.concat(rows + [short], ignore_index=True)


def test_top_liquid_ranks_by_dollar_volume_and_requires_history():
    assert top_liquid_symbols(_panel(), n=2) == ["B", "A"]


def test_symbol_series_drops_bad_ticks_and_nonpositive_closes():
    d = pd.bdate_range("2020-01-01", periods=6)
    p = pd.DataFrame({"Date": d, "Symbol": "X", "Close": [10, 10.5, 0, 31.5, 11, 11.5]})
    s = symbol_series(p, "X")
    assert 0 not in s["Close"].values
    assert 31.5 not in s["Close"].values           # +100% one-day jump treated as bad tick
    assert (s["log_ret"].abs() < 0.5).all()
    assert s["log_ret"].iloc[-1] == pytest.approx(np.log(11.5 / 11))


def test_symbol_series_unflagged_split_leaves_no_absurd_return():
    # 2:1 split on day 3 that the data vendor did not adjust: the jump must not survive in log_ret
    d = pd.bdate_range("2020-01-01", periods=6)
    p = pd.DataFrame({"Date": d, "Symbol": "X", "Close": [100, 101, 50, 50.5, 51, 51.5]})
    s = symbol_series(p, "X")
    assert (s["log_ret"].abs() < 0.5).all(), s["log_ret"].round(3).tolist()
    assert s["log_ret"].iloc[-1] == pytest.approx(np.log(51.5 / 51))
