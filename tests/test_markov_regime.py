"""Tests for price_prediction.markov_regime."""
import numpy as np
import pandas as pd
import pytest

from price_prediction.markov_regime import MarkovRegimeModel, label_states


def _s(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_label_states_warmup_bull_bear():
    r = np.r_[np.full(30, 0.01), np.full(30, -0.01)] + np.tile([1e-4, -1e-4], 30)
    st = label_states(_s(r))
    assert (st.iloc[:20] == -1).all()           # 21-day window warm-up
    assert st.iloc[25] == 2                      # steady gains -> bull
    assert st.iloc[59] == 0                      # steady losses -> bear


def test_transition_matrix_hand_computed():
    states = _s([0, 0, 1, 2, 2, 1])
    rets = _s([0.0] * 6)
    m = MarkovRegimeModel(laplace=1.0).fit(states, rets)
    # transitions: 0->0, 0->1, 1->2, 2->2, 2->1 ; +1 Laplace in each cell
    assert m.T[0] == pytest.approx([2 / 5, 2 / 5, 1 / 5])
    assert m.T[1] == pytest.approx([1 / 4, 1 / 4, 2 / 4])
    assert m.T[2] == pytest.approx([1 / 5, 2 / 5, 2 / 5])
    assert np.allclose(m.T.sum(axis=1), 1.0)


def test_state_returns_are_next_day_returns_and_prediction_is_expectation():
    states = _s([0, 1, 0, 1, 2])
    rets = _s([0.0, 0.01, 0.02, 0.03, 0.04])
    m = MarkovRegimeModel(laplace=0.0).fit(states, rets)
    # E[next ret | state 0] = mean(ret after days 0 and 2) = mean(0.01, 0.03)
    assert m.state_ret[0] == pytest.approx(0.02)
    assert m.state_ret[1] == pytest.approx(0.03)   # mean(0.02, 0.04)
    assert m.state_ret[2] == 0.0                   # no next day
    assert m.predict_return(0) == pytest.approx(m.T[0] @ m.state_ret)


def test_warmup_state_predicts_zero_and_uniform():
    m = MarkovRegimeModel()
    assert m.predict_return(-1) == 0.0
    assert m.predict_proba(-1) == pytest.approx([1 / 3] * 3)
