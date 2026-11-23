import numpy as np
import pandas as pd
import pytest

from price_prediction.kalman_trend import KalmanTrendModel
from price_prediction.markov_regime import MarkovRegimeModel, STATE_NAMES, label_states


def series(vals, start="2020-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)), dtype=float)


# ── Kalman ───────────────────────────────────────────────────────────────────
def reference_filter(y, q_level, q_slope, r):
    """Hand-rolled 2x2 local-linear-trend filter (explicit algebra, no matrix operators)."""
    lvl, slp = y[0], 0.0
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0
    out = []
    for obs in y:
        lvl, slp = lvl + slp, slp
        # P = F P F' + Q
        n00 = p00 + p01 + p10 + p11 + q_level
        n01 = p01 + p11
        n10 = p10 + p11
        n11 = p11 + q_slope
        innov = obs - lvl
        s = n00 + r
        k0, k1 = n00 / s, n10 / s
        lvl, slp = lvl + k0 * innov, slp + k1 * innov
        p00, p01 = n00 - k0 * n00, n01 - k0 * n01
        p10, p11 = n10 - k1 * n00, n11 - k1 * n01
        out.append((lvl, slp, innov))
    return np.array(out)


def test_filter_matches_a_hand_rolled_reference():
    rng = np.random.RandomState(3)
    y = series(np.cumsum(rng.normal(0.01, 0.02, 60)))
    m = KalmanTrendModel(q_level=1e-4, q_slope=1e-6, r_obs=1e-3)
    got = m.filter(y)
    assert list(got.columns) == ["level", "slope", "innovation"]
    assert got.index.equals(y.index)
    np.testing.assert_allclose(got.to_numpy(), reference_filter(y.values, 1e-4, 1e-6, 1e-3), rtol=1e-9, atol=1e-12)


def test_first_innovation_is_zero_and_second_is_the_first_difference():
    y = series([1.0, 1.3, 1.1])
    f = KalmanTrendModel().filter(y)
    assert f["innovation"].iloc[0] == 0.0 and f["slope"].iloc[0] == 0.0
    assert f["innovation"].iloc[1] == pytest.approx(0.3)           # prior level y0, slope 0


def test_constant_series_has_zero_slope():
    f = KalmanTrendModel().filter(series([2.0] * 50))
    assert np.allclose(f["slope"], 0.0) and np.allclose(f["level"], 2.0)


def test_linear_trend_slope_converges_to_the_true_slope():
    y = series(0.01 * np.arange(400))
    slope = KalmanTrendModel(q_level=1e-6, q_slope=1e-6, r_obs=1e-6).predict_return(y)
    assert slope.iloc[-1] == pytest.approx(0.01, rel=0.02)
    assert (slope.iloc[200:] > 0).all()
    down = KalmanTrendModel(q_level=1e-6, q_slope=1e-6, r_obs=1e-6).predict_return(series(-0.01 * np.arange(400)))
    assert down.iloc[-1] == pytest.approx(-0.01, rel=0.02)


def test_prediction_at_t_uses_only_data_through_t():
    rng = np.random.RandomState(4)
    y = series(np.cumsum(rng.normal(0, 0.02, 80)))
    full = KalmanTrendModel().predict_return(y)
    head = KalmanTrendModel().predict_return(y.iloc[:50])
    np.testing.assert_allclose(full.iloc[:50].to_numpy(), head.to_numpy())
    altered = y.copy()
    altered.iloc[60:] += 5.0
    np.testing.assert_allclose(KalmanTrendModel().predict_return(altered).iloc[:60].to_numpy(), full.iloc[:60].to_numpy())


def test_defaults_and_matrices():
    m = KalmanTrendModel()
    assert m.R == 1e-3 and m.Q[0, 0] == 1e-5 and m.Q[1, 1] == 1e-7 and m.Q[0, 1] == 0
    assert m.F.tolist() == [[1.0, 1.0], [0.0, 1.0]] and m.H.tolist() == [[1.0, 0.0]]


# ── label_states ─────────────────────────────────────────────────────────────
def test_label_states_trend_bands_with_zero_volatility():
    # constant returns -> rolling std 0 -> band is 0: sign of the mean decides
    pos = label_states(series([0.01] * 6), trend_win=3, vol_win=3)
    assert pos.tolist() == [-1, -1, 2, 2, 2, 2]
    neg = label_states(series([-0.01] * 6), trend_win=3, vol_win=3)
    assert neg.tolist() == [-1, -1, 0, 0, 0, 0]
    zero = label_states(series([0.0] * 6), trend_win=3, vol_win=3)
    assert zero.tolist() == [-1, -1, 1, 1, 1, 1]           # mu == 0 is neither strictly above nor below


def test_label_states_uses_the_band_around_the_mean():
    ret = series([0.0, 0.02, -0.02, 0.01, 0.03])
    s = label_states(ret, trend_win=3, vol_win=3, trend_band=0.25)
    mu = ret.rolling(3).mean()
    se = ret.rolling(3).std() / np.sqrt(3)
    for i in range(2, 5):
        want = 2 if mu.iloc[i] > 0.25 * se.iloc[i] else 0 if mu.iloc[i] < -0.25 * se.iloc[i] else 1
        assert s.iloc[i] == want
    wide = label_states(ret, trend_win=3, vol_win=3, trend_band=100.0)
    assert (wide.iloc[2:] == 1).all()                       # a huge band leaves everything in "chop"
    assert s.dtype.kind == "i"


def test_label_states_warmup_is_minus_one_and_names():
    s = label_states(series(np.linspace(0, 0.01, 30)))
    assert (s.iloc[:20] == -1).all() and (s.iloc[20:] >= 0).all()
    assert STATE_NAMES == ["bear", "chop", "bull"]


# ── Markov chain ─────────────────────────────────────────────────────────────
def test_defaults_before_fit():
    m = MarkovRegimeModel()
    assert m.T.tolist() == [[1 / 3] * 3] * 3 and m.state_ret.tolist() == [0.0] * 3
    assert m.predict_return(1) == 0.0
    assert MarkovRegimeModel(n_states=2).T.shape == (2, 2)


def test_fit_counts_transitions_with_laplace_smoothing():
    states = series([0, 1, 1, 2, 0, 1], "2020-01-01").astype(int)
    ret = series([0.0] * 6)
    m = MarkovRegimeModel(laplace=1.0).fit(states, ret)
    # transitions: 0->1, 1->1, 1->2, 2->0, 0->1  (+1 smoothing everywhere)
    counts = np.ones((3, 3))
    for a, b in [(0, 1), (1, 1), (1, 2), (2, 0), (0, 1)]:
        counts[a, b] += 1
    np.testing.assert_allclose(m.T, counts / counts.sum(axis=1, keepdims=True))
    assert np.allclose(m.T.sum(axis=1), 1.0)
    m0 = MarkovRegimeModel(laplace=0.0).fit(states, ret)
    assert m0.T[0].tolist() == [0.0, 1.0, 0.0] and m0.T[1].tolist() == pytest.approx([0.0, 0.5, 0.5])


def test_fit_ignores_warmup_states():
    states = series([-1, -1, 0, 1, 0, 1], "2020-01-01").astype(int)
    m = MarkovRegimeModel(laplace=0.0).fit(states, series([0.0] * 6))
    assert m.T[0].tolist() == [0.0, 1.0, 0.0]                  # the -1 -> 0 hop is not a transition
    assert m.T[1].tolist() == [1.0, 0.0, 0.0]


def test_state_returns_are_next_day_returns_conditional_on_todays_state():
    states = series([0, 2, 0, 2, 1], "2020-01-01").astype(int)
    ret = series([0.00, 0.10, 0.20, 0.30, 0.40], "2020-01-01")
    m = MarkovRegimeModel().fit(states, ret)
    # state 0 days: idx 0, 2 -> next-day returns 0.10, 0.30 -> mean 0.20; state 2: idx 1, 3 -> 0.20, 0.40
    assert m.state_ret[0] == pytest.approx(0.20)
    assert m.state_ret[2] == pytest.approx(0.30)
    assert m.state_ret[1] == 0.0                                 # state 1 only on the last day: no next return


def test_predict_return_and_proba():
    m = MarkovRegimeModel()
    m.T = np.array([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.2, 0.2, 0.6]])
    m.state_ret = np.array([-0.01, 0.0, 0.02])
    assert m.predict_return(0) == pytest.approx(-0.005)
    assert m.predict_return(2) == pytest.approx(0.2 * -0.01 + 0.6 * 0.02)
    assert isinstance(m.predict_return(0), float)
    assert m.predict_return(-1) == 0.0
    assert m.predict_proba(2).tolist() == [0.2, 0.2, 0.6]
    assert m.predict_proba(-1).tolist() == [1 / 3] * 3
