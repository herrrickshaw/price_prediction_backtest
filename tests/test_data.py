import datetime as dt

import numpy as np
import pandas as pd
import pytest

from price_prediction import data


def panel_rows(sym, start, n, close=100.0, vol=1000.0):
    d = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Date": d, "Symbol": sym, "Open": close, "High": close, "Low": close,
                         "Close": close, "Volume": vol})


def test_load_panel_reads_market_file_sorts_and_parses_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "LTM_DIR", str(tmp_path))
    df = pd.concat([panel_rows("B", "2020-01-01", 3), panel_rows("A", "2020-01-01", 3)]).iloc[::-1]
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")                  # stored as strings
    df.to_parquet(tmp_path / "US.parquet")
    out = data.load_panel("us")                                      # market name is upper-cased
    assert pd.api.types.is_datetime64_any_dtype(out["Date"])
    assert list(out["Symbol"]) == ["A"] * 3 + ["B"] * 3
    assert out["Date"].is_monotonic_increasing is False              # sorted within symbol, not globally
    assert out.groupby("Symbol")["Date"].apply(lambda s: s.is_monotonic_increasing).all()
    assert list(out.index) == list(range(6))


def test_load_panel_missing_market(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "LTM_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        data.load_panel("XX")


def test_ltm_dir_comes_from_the_environment(monkeypatch):
    import importlib
    monkeypatch.setenv("LTM_DIR", "/custom/ltm")
    try:
        assert importlib.reload(data).LTM_DIR == "/custom/ltm"
    finally:
        monkeypatch.delenv("LTM_DIR")
        importlib.reload(data)


def test_top_liquid_symbols_ranks_by_median_dollar_volume():
    n = 8 * 365 // 7 * 5 + 400                            # > 8 calendar years of business days
    frames = [panel_rows("LOW", "2000-01-03", n, 10, 100), panel_rows("HIGH", "2000-01-03", n, 100, 5000),
              panel_rows("MID", "2000-01-03", n, 50, 400)]
    panel = pd.concat(frames, ignore_index=True)
    assert data.top_liquid_symbols(panel, n=3) == ["HIGH", "MID", "LOW"]
    assert data.top_liquid_symbols(panel, n=2) == ["HIGH", "MID"]
    assert data.top_liquid_symbols(panel) == ["HIGH", "MID", "LOW"]


def test_top_liquid_symbols_requires_span_and_observation_count():
    long_n = 2100
    long_ = panel_rows("LONG", "2000-01-03", long_n, 10, 10)
    short = panel_rows("SHORT", "2000-01-03", 500, 1000, 1e6)                   # richest, but < 8 years
    sparse = pd.concat([panel_rows("SPARSE", "2000-01-03", 1, 1000, 1e6),
                        panel_rows("SPARSE", "2009-01-05", 1, 1000, 1e6)])      # long span, 2 observations
    panel = pd.concat([long_, short, sparse], ignore_index=True)
    assert data.top_liquid_symbols(panel, n=5) == ["LONG"]


def test_top_liquid_symbols_span_boundary():
    days = 8 * 365
    d = [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-01") + dt.timedelta(days=days)]
    base = pd.bdate_range("2000-01-03", periods=1700)
    exact = pd.DataFrame({"Date": list(base[:-1]) + [d[0] + dt.timedelta(days=days)], "Symbol": "EXACT"})
    exact["Date"] = [pd.Timestamp("2000-01-01")] + list(pd.bdate_range("2000-01-04", periods=len(exact) - 2)) + [d[1]]
    for c in ("Close", "Volume"):
        exact[c] = 10.0
    assert data.top_liquid_symbols(exact, n=1, min_years=8.0) == ["EXACT"]            # span == 8y kept
    assert data.top_liquid_symbols(exact, n=1, min_years=8.0 + 1e-3) == []            # just short


def test_symbol_series_cleaning():
    d = pd.bdate_range("2021-01-04", periods=8)
    close = [100, 101, 0, 102, 102, 103, 104, 105]
    df = pd.DataFrame({"Date": d, "Symbol": "X", "Close": close, "Open": 1, "High": 1, "Low": 1, "Volume": 1})
    dup = df.iloc[[3]].copy()
    dup["Close"] = 102.5                                         # a later duplicate wins
    other = df.assign(Symbol="Y", Close=5.0)
    panel = pd.concat([df, dup, other], ignore_index=True)
    s = data.symbol_series(panel, "X")
    assert "Y" not in set(s.get("Symbol", []))
    assert (s["Close"] > 0).all()
    assert s.index.is_unique and s.index.is_monotonic_increasing
    assert d[2] not in s.index                                   # the Close == 0 bar is dropped
    assert s.loc[d[3], "Close"] == 102.5
    assert s.loc[d[3], "log_ret"] == pytest.approx(np.log(102.5 / 101.0))   # return spans the dropped bar
    assert s["log_ret"].notna().all()


def test_symbol_series_log_returns_and_first_row_dropped():
    d = pd.bdate_range("2021-01-04", periods=4)
    panel = pd.DataFrame({"Date": d, "Symbol": "X", "Close": [100.0, 110.0, 99.0, 99.0],
                          "Open": 1, "High": 1, "Low": 1, "Volume": 1})
    s = data.symbol_series(panel, "X")
    assert list(s.index) == list(d[1:])
    assert s["log_ret"].tolist() == pytest.approx([np.log(1.1), np.log(0.9), 0.0])


@pytest.mark.parametrize("jump,kept", [(np.exp(0.49), True), (np.exp(0.5), False), (np.exp(-0.49), True),
                                       (np.exp(-0.5), False)])
def test_symbol_series_drops_absurd_moves_at_half_a_log_point(jump, kept):
    d = pd.bdate_range("2021-01-04", periods=4)
    close = [100.0, 100.0 * jump, 100.0 * jump, 100.0 * jump]
    panel = pd.DataFrame({"Date": d, "Symbol": "X", "Close": close, "Open": 1, "High": 1, "Low": 1, "Volume": 1})
    s = data.symbol_series(panel, "X")
    assert (d[1] in s.index) is kept
    # after a dropped bar the return is recomputed across the gap, so it is not a phantom zero
    assert s["log_ret"].notna().all()


def test_symbol_series_bad_tick_removes_the_tick_and_the_bar_after_it():
    # +300% then -75%: both moves exceed the jump limit, so both bars go, and the surviving
    # return is recomputed across the gap (pinned current behaviour)
    d = pd.bdate_range("2021-01-04", periods=4)
    panel = pd.DataFrame({"Date": d, "Symbol": "X", "Close": [100.0, 400.0, 101.0, 102.0],
                          "Open": 1, "High": 1, "Low": 1, "Volume": 1})
    s = data.symbol_series(panel, "X")
    assert list(s.index) == [d[3]]
    assert s["log_ret"].iloc[0] == pytest.approx(np.log(102.0 / 100.0))


def test_top_liquid_symbols_observation_count_boundary():
    # 8 calendar years of span but exactly min_years * 200 = 1600 observations: kept; 1599: dropped
    def sparse(sym, n_obs):
        d = pd.date_range("2000-01-03", "2008-02-01", periods=n_obs)
        return pd.DataFrame({"Date": d, "Symbol": sym, "Close": 10.0, "Volume": 10.0})
    panel = pd.concat([sparse("EXACT", 1600), sparse("SHORT", 1599)], ignore_index=True)
    assert data.top_liquid_symbols(panel, n=5) == ["EXACT"]
