from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from price_prediction import data


def test_top_liquid_symbols_matches_baseline(small_panel, baseline):
    assert data.top_liquid_symbols(small_panel, n=3, min_years=3.0) == baseline["top_liquid"]


def test_top_liquid_symbols_requires_history(small_panel):
    # 1000 business days is < 8 years, so the default min_years excludes everything
    assert data.top_liquid_symbols(small_panel, n=3) == []


def test_symbol_series_matches_baseline(symbol_frame, baseline):
    b = baseline["symbol_series"]
    assert len(symbol_frame) == b["n"]
    assert str(symbol_frame.index[0].date()) == b["first"]
    assert symbol_frame["log_ret"].sum() == pytest.approx(b["logret_sum"], abs=1e-9)


def test_symbol_series_drops_bad_ticks_and_dupes():
    dates = pd.bdate_range("2020-01-01", periods=6)
    close = [100.0, 101.0, 500.0, 102.0, 103.0, 103.0]  # 500 is an absurd jump
    df = pd.DataFrame({"Date": list(dates) + [dates[-1]], "Symbol": "X",
                       "Close": close + [104.0], "Volume": 1})
    out = data.symbol_series(df, "X")
    assert not out.index.duplicated().any()
    assert out["Close"].max() < 500
    assert np.isfinite(out["log_ret"]).all()
    assert out["Close"].iloc[-1] == 104.0  # keep="last" on duplicate dates
