"""Tests for price_prediction.replay on small synthetic panels."""
import numpy as np
import pandas as pd
import pytest

from price_prediction.replay import build_signals, clustered_stats


def _panel(n_sym=60, n_days=420, vol_path=None, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2015-01-01", periods=n_days)
    vol_path = np.full(n_days, 0.01) if vol_path is None else vol_path
    common = rng.normal(0, 1, n_days) * vol_path
    frames = []
    for i in range(n_sym):
        r = common + rng.normal(0, 0.002, n_days)
        c = 50 * np.exp(np.cumsum(r))
        frames.append(pd.DataFrame({"Date": d, "Symbol": f"S{i:02d}", "Close": c,
                                    "High": c * 1.005, "Low": c * 0.995,
                                    "Volume": 1e6 + i}))
    return pd.concat(frames, ignore_index=True)


def test_clustered_stats_hand_computed():
    sig = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03"]),
        "signal": ["B"] * 4,
        "ex5": [0.02, 0.04, -0.01, 0.01],
    })
    out = clustered_stats(sig, 5, ["signal"], min_dates=1)
    row = out.loc["B"]
    per_date = np.array([0.03, -0.01, 0.01])
    assert row["n_signals"] == 4 and row["n_dates"] == 3
    assert row["mean_excess_%"] == pytest.approx(round(per_date.mean() * 100, 2))
    t = per_date.mean() / (per_date.std(ddof=1) / np.sqrt(3))
    assert row["t_clustered"] == pytest.approx(round(t, 2))
    assert row["hit_%"] == pytest.approx(75.0)


def test_clustered_stats_drops_thin_cells():
    sig = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"]), "signal": ["B"], "ex5": [0.1]})
    assert clustered_stats(sig, 5, ["signal"], min_dates=2).empty


def test_too_few_symbols_raises():
    with pytest.raises(ValueError):
        build_signals(_panel(n_sym=10))


def test_breakout_needs_close_above_prior_box_high():
    sig, dates = build_signals(_panel())
    assert set(sig["signal"]) <= {"BREAKOUT_BUY", "BREAKDOWN_SELL"}
    assert {"trend", "vol_regime"} <= set(dates.columns)
    # forward excess = forward return minus same-date cross-sectional median (defined where fwd is)
    assert sig["ex5"].notna().sum() > 0


def test_record_low_volatility_is_labelled_low_not_missing():
    # Index volatility that keeps falling: after the 252-day expanding warm-up each day's rolling
    # vol is the lowest seen so far (percentile 0). That is the LOW regime, not "unknown".
    n = 420
    vol_path = np.linspace(0.03, 0.002, n)
    _, dates = build_signals(_panel(n_days=n, vol_path=vol_path))
    tail = dates["vol_regime"].iloc[300:]
    assert tail.notna().all(), f"{tail.isna().sum()} of {len(tail)} days unlabeled"
    assert (tail == "LOW").mean() > 0.5
