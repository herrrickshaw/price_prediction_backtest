from __future__ import annotations

import numpy as np
import pytest

from price_prediction.markov_regime import MarkovRegimeModel, label_states


def test_label_states_matches_baseline(symbol_frame, baseline):
    st = label_states(symbol_frame["log_ret"])
    counts = {str(k): int(v) for k, v in st.value_counts().sort_index().items()}
    assert counts == baseline["states"]["counts"]
    assert set(st.unique()) <= {-1, 0, 1, 2}


def test_markov_fit_predict_matches_baseline(symbol_frame, baseline):
    st = label_states(symbol_frame["log_ret"])
    m = MarkovRegimeModel().fit(st, symbol_frame["log_ret"])
    b = baseline["markov"]
    np.testing.assert_allclose(m.T, np.array(b["T"]), atol=1e-9)
    np.testing.assert_allclose(m.state_ret, np.array(b["state_ret"]), atol=1e-9)
    assert [m.predict_return(k) for k in (-1, 0, 1, 2)] == pytest.approx(b["pred"], abs=1e-9)
    np.testing.assert_allclose(m.predict_proba(-1), b["proba_neg"])


def test_transition_rows_sum_to_one(symbol_frame):
    st = label_states(symbol_frame["log_ret"])
    m = MarkovRegimeModel().fit(st, symbol_frame["log_ret"])
    np.testing.assert_allclose(m.T.sum(axis=1), 1.0)
    assert (m.T > 0).all()  # Laplace smoothing keeps every transition possible
