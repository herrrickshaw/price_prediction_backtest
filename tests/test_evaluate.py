import numpy as np
import pandas as pd
import pytest

from price_prediction import evaluate as E
from price_prediction import lstm_model
from price_prediction.data import symbol_series
from price_prediction.kalman_trend import KalmanTrendModel


def ser(vals):
    return pd.Series(vals, index=pd.bdate_range("2021-01-04", periods=len(vals)), dtype=float)


# ── _metrics ─────────────────────────────────────────────────────────────────
def test_metrics_hand_checked():
    p = ser([0.01, -0.02, 0.0, 0.03])
    a = ser([0.02, -0.01, 0.05, -0.04])
    m = E._metrics(p, a)
    assert m["n"] == 4
    assert m["dir_acc"] == pytest.approx(2 / 3)                    # the zero prediction is not scored
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean((p - a) ** 2)))
    strat = np.sign(p) * a                                         # [0.02, 0.01, 0, -0.04]
    assert m["strat_ann_ret"] == pytest.approx(strat.mean() * 252)
    assert m["strat_sharpe"] == pytest.approx(strat.mean() / (strat.std() + 1e-12) * np.sqrt(252))
    assert m["bh_ann_ret"] == pytest.approx(a.mean() * 252)
    assert m["bh_sharpe"] == pytest.approx(a.mean() / (a.std() + 1e-12) * np.sqrt(252))


def test_metrics_drop_missing_pairs_and_handle_all_zero_predictions():
    p = ser([0.01, np.nan, 0.02, 0.0])
    a = ser([0.01, 0.5, np.nan, 0.01])
    m = E._metrics(p, a)
    assert m["n"] == 2
    m0 = E._metrics(ser([0.0, 0.0, 0.0]), ser([0.1, -0.1, 0.2]))
    assert np.isnan(m0["dir_acc"]) and m0["strat_ann_ret"] == 0.0


def test_metrics_perfect_and_inverted_forecasts():
    a = ser([0.01, -0.02, 0.03, -0.01, 0.02])
    assert E._metrics(a, a)["dir_acc"] == 1.0 and E._metrics(a, a)["rmse"] == 0.0
    assert E._metrics(-a, a)["dir_acc"] == 0.0
    assert E._metrics(-a, a)["strat_ann_ret"] == pytest.approx(-np.abs(a).mean() * 252)


# ── evaluate_symbol ──────────────────────────────────────────────────────────
def make_df(n=820, seed=0):
    rng = np.random.RandomState(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    panel = pd.DataFrame({"Date": pd.bdate_range("2018-01-01", periods=n), "Symbol": "X",
                          "Close": close, "High": close * 1.005, "Low": close * 0.995,
                          "Open": close, "Volume": 1e5 + rng.randint(0, 1000, n).astype(float)})
    return symbol_series(panel, "X")


FAST = {"lstm": dict(epochs=1, hidden=4, lookback=8, batch=128)}


def test_all_models_scored_over_the_test_window():
    df = make_df()
    res = E.evaluate_symbol(df, test_days=40, refit_every=20, seq_models=FAST)
    assert list(res.index) == ["naive_zero", "naive_yesterday", "naive_histmean", "kalman", "markov", "lstm"]
    assert (res["n"] == 39).all()                         # last test day has no next-day return yet
    assert set(res.columns) == {"n", "dir_acc", "rmse", "strat_ann_ret", "strat_sharpe", "bh_ann_ret", "bh_sharpe"}


def test_baselines_are_computed_causally():
    df = make_df()
    res = E.evaluate_symbol(df, test_days=40, refit_every=20, seq_models=FAST)
    ret = df["log_ret"]
    test_idx = ret.index[-40:]
    actual = ret.shift(-1).reindex(test_idx)
    # always-long baseline earns the buy-and-hold return
    z = res.loc["naive_zero"]
    assert z.strat_ann_ret == pytest.approx(z.bh_ann_ret)
    assert z.dir_acc == pytest.approx((actual.dropna() > 0).mean())
    # yesterday's return forecasts tomorrow's
    y = E._metrics(ret.reindex(test_idx), actual)
    assert res.loc["naive_yesterday", "rmse"] == pytest.approx(y["rmse"])
    # historical mean at t uses only days strictly before t
    hist = pd.Series([ret.loc[:t].iloc[:-1].mean() for t in test_idx], index=test_idx)
    manual = np.sqrt(((hist - actual).dropna() ** 2).mean())
    assert res.loc["naive_histmean", "rmse"] == pytest.approx(manual)
    assert hist.iloc[0] == pytest.approx(ret.iloc[:len(ret) - 40].mean())    # every day before the first test day
    # Kalman row equals the online filter's slope scored directly
    k = KalmanTrendModel().predict_return(np.log(df["Close"])).reindex(test_idx)
    assert res.loc["kalman", "rmse"] == pytest.approx(E._metrics(k, actual)["rmse"])


def test_refits_use_an_expanding_window_that_ends_before_each_block(monkeypatch):
    df = make_df()
    fits = []

    class SpyMarkov(E.MarkovRegimeModel):
        def fit(self, states, log_ret):
            fits.append(("markov", states.index.max(), log_ret.index.max(), len(log_ret)))
            return super().fit(states, log_ret)

    class SpySeq:
        lookback = 5

        def __init__(self, cell, **kw):
            self.cell = cell

        def fit(self, feats, target):
            fits.append(("seq:" + self.cell, feats.index.max(), target.index.max(), len(feats)))
            return self

        def predict(self, ctx):
            return pd.Series(0.001, index=ctx.index[self.lookback:])

    monkeypatch.setattr(E, "MarkovRegimeModel", SpyMarkov)
    monkeypatch.setattr(lstm_model, "SeqModel", SpySeq)
    res = E.evaluate_symbol(df, test_days=40, refit_every=15, seq_models={"lstm": {}, "gru": {}})
    idx = df.index[-40:]
    starts = [0, 15, 30]                                              # ceil(40 / 15) blocks
    markov = [f for f in fits if f[0] == "markov"]
    assert len(markov) == 3
    for (name, last_state, last_ret, n_train), s in zip(markov, starts):
        prev_day = df.index[df.index.get_loc(idx[s]) - 1]
        assert last_state == prev_day and last_ret == prev_day          # day t itself is never in training
    seq = [f for f in fits if f[0] == "seq:lstm"]
    for (name, last_feat, last_target, n_train), s in zip(seq, starts):
        prev_day = df.index[df.index.get_loc(idx[s]) - 1]
        assert last_feat == prev_day and last_target == prev_day
    sizes = [f[3] for f in markov]
    assert sizes[1] == sizes[0] + 15 and sizes[2] == sizes[1] + 15       # window expands, never slides
    assert [f[0] for f in fits if f[0].startswith("seq:")] == ["seq:lstm", "seq:gru"] * 3
    assert set(res.index) == {"naive_zero", "naive_yesterday", "naive_histmean", "kalman", "markov", "lstm", "gru"}


def test_history_length_boundary():
    df = make_df(n=40 + 750 + 1)                       # symbol_series drops the first bar
    assert len(df) == 790
    E.evaluate_symbol(df, test_days=40, refit_every=20, seq_models={})        # exactly test_days + 750: allowed
    with pytest.raises(ValueError, match="history too short"):
        E.evaluate_symbol(df.iloc[1:], test_days=40, refit_every=20, seq_models={})


def test_empty_seq_models_runs_the_classical_models_only():
    res = E.evaluate_symbol(make_df(), test_days=30, refit_every=30, seq_models={})
    assert list(res.index) == ["naive_zero", "naive_yesterday", "naive_histmean", "kalman", "markov"]


# ── evaluate_universe ────────────────────────────────────────────────────────
def test_evaluate_universe_skips_failures_and_tags_symbols(monkeypatch, capsys):
    d = pd.bdate_range("2020-01-01", periods=5)
    panel = pd.concat([pd.DataFrame({"Date": d, "Symbol": s, "Close": 100.0 + np.arange(5),
                                     "Open": 1, "High": 1, "Low": 1, "Volume": 1}) for s in ("A", "BAD", "C")])
    seen = []

    def fake(df, **kw):
        seen.append(kw)
        if df["Close"].iloc[0] == 101.0 and len(seen) == 2:
            raise ValueError("history too short")
        return pd.DataFrame({"n": [1, 2]}, index=pd.Index(["m1", "m2"], name="model"))

    monkeypatch.setattr(E, "evaluate_symbol", fake)
    out = E.evaluate_universe(panel, ["A", "BAD", "C"], test_days=11)
    text = capsys.readouterr().out
    assert "skip BAD: history too short" in text and "done A" in text and "done C" in text
    assert list(out["symbol"]) == ["A", "A", "C", "C"] and list(out["model"]) == ["m1", "m2", "m1", "m2"]
    assert out.index.tolist() == [0, 1, 2, 3]
    assert all(kw == {"test_days": 11} for kw in seen)                     # extra kwargs are forwarded
