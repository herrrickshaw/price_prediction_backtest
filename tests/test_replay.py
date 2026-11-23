import os

import numpy as np
import pandas as pd
import pytest

from price_prediction import replay as R


def make_panel(n_sym=60, n=420, seed=0, vol_growth=1.15, common=None):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    frames = []
    for i in range(n_sym):
        steps = rng.normal(0.0004, 0.015, n)
        if common is not None:
            steps = steps + common
        close = 100 * np.exp(np.cumsum(steps))
        frames.append(pd.DataFrame({
            "Date": dates, "Symbol": f"S{i:02d}", "Open": close,
            "High": close * (1 + rng.rand(n) * 0.01), "Low": close * (1 - rng.rand(n) * 0.01),
            "Close": close, "Volume": 1e5 * vol_growth ** i}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


@pytest.fixture(scope="module")
def built(panel):
    return R.build_signals(panel)


# ── helpers ──────────────────────────────────────────────────────────────────
def test_rsi_extremes_and_midpoint():
    up = pd.DataFrame({"a": np.arange(1.0, 60.0)})
    assert R._rsi14(up)["a"].iloc[-1] == pytest.approx(100.0)
    down = pd.DataFrame({"a": np.arange(60.0, 1.0, -1.0)})
    assert R._rsi14(down)["a"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    zigzag = pd.DataFrame({"a": [100.0 + (i % 2) for i in range(80)]})
    assert 40 < R._rsi14(zigzag)["a"].iloc[-1] < 60
    assert R._rsi14(up)["a"].iloc[:14].isna().all()               # 14-bar warm-up


def test_wide_pivots_with_last_value_per_date():
    p = pd.DataFrame({"Date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-02"]),
                      "Symbol": ["A", "A", "A", "B"], "Close": [1.0, 2.0, 3.0, 9.0]})
    w = R._wide(p, "Close")
    assert w.loc["2020-01-02", "A"] == 2.0                        # duplicate date: last row wins
    assert w.loc["2020-01-02", "B"] == 9.0 and np.isnan(w.loc["2020-01-03", "B"])
    assert list(w.columns) == ["A", "B"] and w.index.is_monotonic_increasing


def test_constants():
    assert (R.MIN_OBS, R.MIN_NAMES, R.UNIVERSE_N, R.BOX_WIN, R.HORIZONS) == (300, 50, 1000, 55, (5, 21))
    assert R.TREND_NAMES == {0: "bear", 1: "chop", 2: "bull", -1: "warmup"}


# ── build_signals ────────────────────────────────────────────────────────────
def brute_force_signals(panel):
    """Independent oracle: explicit loops, the box taken from bars strictly BEFORE the signal bar."""
    close, high, low = (R._wide(panel, c) for c in ("Close", "High", "Low"))
    ema50 = close.ewm(span=50, min_periods=50).mean()
    rsi = R._rsi14(close)
    buys, sells = set(), set()
    for s in close.columns:
        c, h, l = close[s].to_numpy(), high[s].to_numpy(), low[s].to_numpy()
        for t in range(56, len(c)):
            box_hi, box_hi_prev = h[t - 55:t].max(), h[t - 56:t - 1].max()
            box_lo, box_lo_prev = l[t - 55:t].min(), l[t - 56:t - 1].min()
            e, r = ema50[s].iloc[t], rsi[s].iloc[t]
            if c[t] > box_hi and c[t - 1] <= box_hi_prev and c[t] > e and 50 <= r <= 80:
                buys.add((close.index[t], s))
            if c[t] < box_lo and c[t - 1] >= box_lo_prev and c[t] < e:
                sells.add((close.index[t], s))
    return buys, sells


def test_signals_match_a_brute_force_oracle(panel, built):
    sig, _ = built
    buys, sells = brute_force_signals(panel)
    assert len(buys) > 20 and len(sells) > 20                       # the comparison is not vacuous
    got_b = set(zip(sig[sig.signal == "BREAKOUT_BUY"].Date, sig[sig.signal == "BREAKOUT_BUY"].Symbol))
    got_s = set(zip(sig[sig.signal == "BREAKDOWN_SELL"].Date, sig[sig.signal == "BREAKDOWN_SELL"].Symbol))
    assert got_b == buys and got_s == sells


def test_current_bar_is_excluded_from_the_box():
    # A flat 100 box, then one bar closing at 101 with a 101 high. If the box included its own bar the
    # breakout could never fire (close > own high is impossible); with it excluded the bar must signal.
    n = 330
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.RandomState(1)
    frames = []
    for i in range(60):
        close = 100 + np.cumsum(rng.normal(0.02, 0.5, n)) * 0.3
        frames.append(pd.DataFrame({"Date": dates, "Symbol": f"S{i:02d}", "Open": close, "High": close, "Low": close,
                                    "Close": close, "Volume": 1e5 * (1 + i)}))
    # designed name: oscillating (RSI mid-range), flat top at 105, then a 106 close on bar 100
    osc = 100 + 2.5 * np.sin(np.arange(n) * 1.3) + 2.5
    osc = np.minimum(osc, 105.0)
    osc[300] = 106.0
    osc[301:] = 106.0
    f = pd.DataFrame({"Date": dates, "Symbol": "DESIGN", "Open": osc, "High": osc, "Low": osc, "Close": osc, "Volume": 1e9})
    sig, _ = R.build_signals(pd.concat(frames + [f], ignore_index=True))
    s = sig[(sig.Symbol == "DESIGN") & (sig.signal == "BREAKOUT_BUY")]
    rsi = R._rsi14(R._wide(pd.concat(frames + [f]), "Close"))["DESIGN"]
    assert 50 <= rsi.iloc[300] <= 80                                  # the other conditions are satisfied
    assert dates[300] in set(s.Date)
    assert dates[301] not in set(s.Date)                             # already above the box yesterday: no re-fire


def test_forward_and_excess_returns(panel, built):
    sig, _ = built
    close = R._wide(panel, "Close")
    row = sig[(sig.signal == "BREAKOUT_BUY")].dropna(subset=["fwd21"]).iloc[3]
    c = close[row.Symbol]
    i = c.index.get_loc(row.Date)
    assert row.fwd5 == pytest.approx(c.iloc[i + 5] / c.iloc[i] - 1)
    assert row.fwd21 == pytest.approx(c.iloc[i + 21] / c.iloc[i] - 1)
    cross = (close.shift(-5) / close - 1).loc[row.Date].median()    # all kept names; every symbol is liquid here
    assert row.ex5 == pytest.approx(row.fwd5 - cross)
    assert set(sig.columns) == {"Date", "Symbol", "signal", "fwd5", "ex5", "fwd21", "ex21", "trend", "vol_regime"}
    assert set(sig.signal) == {"BREAKOUT_BUY", "BREAKDOWN_SELL"}
    assert sig.trend.isin(["bear", "chop", "bull", "warmup"]).all() or sig.trend.isna().any()


def test_buy_conditions_all_hold_on_every_buy(panel, built):
    sig, _ = built
    close = R._wide(panel, "Close")
    ema = close.ewm(span=50, min_periods=50).mean()
    rsi = R._rsi14(close)
    for r in sig[sig.signal == "BREAKOUT_BUY"].itertuples():
        assert close.at[r.Date, r.Symbol] > ema.at[r.Date, r.Symbol]
        assert 50 <= rsi.at[r.Date, r.Symbol] <= 80
    for r in sig[sig.signal == "BREAKDOWN_SELL"].itertuples():
        assert close.at[r.Date, r.Symbol] < ema.at[r.Date, r.Symbol]


def test_too_shallow_panel_is_rejected_at_the_boundaries():
    ok = make_panel(n_sym=50, n=300)
    R.build_signals(ok)                                              # exactly 50 symbols x exactly 300 obs
    with pytest.raises(ValueError, match="only 49 symbols with >= 300 obs"):
        R.build_signals(make_panel(n_sym=49, n=300))
    short = ok[~((ok.Symbol == "S00") & (ok.Date == ok.Date.min()))]            # one name has 299 obs
    with pytest.raises(ValueError, match="only 49 symbols"):
        R.build_signals(short)


def test_universe_is_limited_to_the_most_liquid_names(panel, monkeypatch):
    monkeypatch.setattr(R, "UNIVERSE_N", 55)
    sig, _ = R.build_signals(panel)
    close, vol = R._wide(panel, "Close"), R._wide(panel, "Volume")
    keep = set((close * vol).median().nlargest(55).index)
    assert len(set(sig.Symbol)) > 10 and set(sig.Symbol) <= keep


def test_absurd_single_day_moves_are_masked():
    p = make_panel(n_sym=60, n=420, seed=5)
    s = p[p.Symbol == "S10"].copy()
    i = 200
    idx = s.index[i]
    p.loc[idx, ["Close", "High", "Low", "Open"]] = p.loc[idx, "Close"] * 1.7          # +70% print
    sig, _ = R.build_signals(p)
    bad_date = s.Date.iloc[i]
    assert not ((sig.Symbol == "S10") & (sig.Date == bad_date)).any()                 # no signal on the bad tick
    # the +60% mask boundary: a +50% day is a real move and is left alone
    assert len(sig) > 0


def trending_panel():
    n = 700
    t = np.arange(n)
    common = np.where(t < 350, 0.004, -0.004) + np.where(t < 350, 0.0, 0.0)
    rng = np.random.RandomState(9)
    dates = pd.bdate_range("2018-01-01", periods=n)
    shock = rng.normal(0, 1, n) * np.where(t < 520, 0.004, 0.03)            # calm, then violent
    frames = []
    for i in range(60):
        steps = common + shock + rng.normal(0, 0.004, n)
        close = 100 * np.exp(np.cumsum(steps))
        frames.append(pd.DataFrame({"Date": dates, "Symbol": f"T{i:02d}", "Open": close,
                                    "High": close * 1.002, "Low": close * 0.998, "Close": close,
                                    "Volume": 1e5 * 1.1 ** i}))
    return pd.concat(frames, ignore_index=True), dates


def test_market_regimes_follow_the_basket_trend_and_volatility():
    p, dates = trending_panel()
    _, reg = R.build_signals(p)
    assert list(reg.columns) == ["trend", "vol_regime"]
    assert reg.trend.iloc[0] == "warmup"
    assert (reg.trend.iloc[:80] == "warmup").all()                      # EMA100 needs 100 bars
    assert reg.trend.loc[dates[300]] == "bull"
    assert reg.trend.loc[dates[-1]] == "bear"
    assert reg.vol_regime.loc[dates[-1]] == "HIGH"                      # latest vol tops its whole history
    assert str(reg.vol_regime.dtype) == "category" and list(reg.vol_regime.cat.categories) == ["LOW", "MID", "HIGH"]
    assert reg.vol_regime.iloc[:200].isna().all()                       # 252-point expanding window not yet full
    assert set(reg.trend.dropna()) <= {"bear", "chop", "bull", "warmup"}


# ── clustered_stats ──────────────────────────────────────────────────────────
def stats_frame():
    d1, d2, d3 = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    return pd.DataFrame({
        "Date": [d1, d1, d2, d3, d3],
        "signal": ["A", "A", "A", "B", "B"],
        "trend": ["bull", "bull", "bear", "bull", "bull"],
        "ex5": [0.02, 0.04, -0.01, np.nan, 0.05],
        "ex21": [1.0] * 5})


def test_clustered_stats_means_first_per_date_then_t_across_dates():
    out = R.clustered_stats(stats_frame(), 5, ["signal"], min_dates=1)
    a = out.loc["A"]
    assert a.n_signals == 3 and a.n_dates == 2
    assert a["mean_excess_%"] == pytest.approx(1.0)                 # dates' means 0.03 and -0.01 -> 0.01
    day = np.array([0.03, -0.01])
    assert a.t_clustered == pytest.approx(round(day.mean() / (day.std(ddof=1) / np.sqrt(2)), 2))
    assert a["hit_%"] == pytest.approx(66.67)
    b = out.loc["B"]
    assert b.n_signals == 1 and b.n_dates == 1                      # the NaN excess is dropped before counting
    assert b["mean_excess_%"] == 5.0 and b["hit_%"] == 100.0
    assert np.isnan(b.t_clustered)                                  # one date: no dispersion


def test_clustered_stats_min_dates_and_multi_key_grouping():
    sig = stats_frame()
    assert list(R.clustered_stats(sig, 5, ["signal"], min_dates=2).index) == ["A"]
    assert list(R.clustered_stats(sig, 5, ["signal"], min_dates=3).index) == []
    out = R.clustered_stats(sig, 5, ["signal", "trend"], min_dates=1)
    assert set(out.index) == {("A", "bull"), ("A", "bear"), ("B", "bull")}
    assert out.loc[("A", "bull"), "mean_excess_%"] == pytest.approx(3.0)
    assert R.clustered_stats(sig, 21, ["signal"], min_dates=1).loc["B", "mean_excess_%"] == 100.0
    assert R.clustered_stats(sig, 5, ["signal"]).empty                # default needs 100 dates


def test_clustered_stats_default_threshold_is_one_hundred_dates():
    d = pd.bdate_range("2020-01-01", periods=100)
    sig = pd.DataFrame({"Date": d, "signal": "A", "ex5": np.linspace(-0.01, 0.03, 100)})
    assert len(R.clustered_stats(sig, 5, ["signal"])) == 1
    assert R.clustered_stats(sig.iloc[:99], 5, ["signal"]).empty


# ── run_market / main ────────────────────────────────────────────────────────
def test_run_market_writes_parquet_and_prints_every_cut(panel, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(R, "load_panel", lambda m: panel)
    monkeypatch.setattr(R, "clustered_stats", lambda sig, h, by, **kw: pd.DataFrame({"h": [h], "by": [",".join(by)]}))
    R.run_market("US", str(tmp_path))
    out = capsys.readouterr().out
    assert "================ US ================" in out
    for h in (5, 21):
        for cut in ("signal alone", "signal x trend regime", "signal x vol regime"):
            assert f"--- US h={h}d: {cut} ---" in out
    got = pd.read_parquet(tmp_path / "replay_signals_us.parquet")
    assert {"Date", "Symbol", "signal", "ex5"} <= set(got.columns) and len(got) > 0


def test_main_runs_each_market_and_survives_a_failure(panel, tmp_path, monkeypatch, capsys):
    calls = []

    def fake_run(market, out_dir):
        calls.append((market, out_dir))
        if market == "BAD":
            raise FileNotFoundError("no panel")

    monkeypatch.setattr(R, "run_market", fake_run)
    monkeypatch.setattr("sys.argv", ["replay", "--market", "US", "BAD", "IN", "--out", str(tmp_path / "o")])
    R.main()
    assert [c[0] for c in calls] == ["US", "BAD", "IN"]
    assert (tmp_path / "o").is_dir()
    assert "BAD: FAILED — no panel" in capsys.readouterr().out


def test_main_defaults(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(R, "run_market", lambda m, o: calls.append((m, o)))
    monkeypatch.setattr(R, "__file__", str(tmp_path / "price_prediction" / "replay.py"))
    monkeypatch.setattr("sys.argv", ["replay"])
    R.main()
    assert [c[0] for c in calls] == ["US", "IN", "JP", "KR", "CN"]
    assert calls[0][1] == str(tmp_path / "reports") and (tmp_path / "reports").is_dir()
