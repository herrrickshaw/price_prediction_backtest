from __future__ import annotations

import pandas as pd
import pytest

from price_prediction import evaluate
from price_prediction.baselines import BASELINES


class StubSeq:
    """Deterministic stand-in for the PyTorch SeqModel (constant forecast)."""
    lookback = 42

    def __init__(self, cell="lstm", **kw):
        self.cell = cell

    def fit(self, X, y):
        return self

    def predict(self, ctx):
        value = 1e-5 if self.cell == "lstm" else -1e-5
        return pd.Series(value, index=ctx.index[self.lookback:])


def test_metrics_matches_baseline(symbol_frame, baseline):
    import numpy as np

    idx = symbol_frame.index[:50]
    pred = pd.Series(np.sin(np.arange(50)), index=idx)
    act = pd.Series(np.cos(np.arange(50)), index=idx)
    got = evaluate.metrics(pred, act)
    assert got["n"] == baseline["metrics"]["n"]
    for k, v in baseline["metrics"].items():
        if k != "n":
            assert got[k] == pytest.approx(v, abs=1e-9), k
    assert evaluate._metrics is evaluate.metrics


def test_evaluate_symbol_matches_baseline(symbol_frame, baseline):
    res = evaluate.evaluate_symbol(symbol_frame, test_days=100, refit_every=50,
                                   seq_model_factory=lambda cell, **kw: StubSeq(cell, **kw))
    assert list(res.index) == list(baseline["evaluate_symbol"])
    for model, row in baseline["evaluate_symbol"].items():
        for col, v in row.items():
            got = res.loc[model, col]
            if v != v:  # NaN in fixture
                assert got != got, (model, col)
            else:
                assert got == pytest.approx(v, abs=1e-9), (model, col)


def test_evaluate_symbol_can_skip_nets(symbol_frame):
    res = evaluate.evaluate_symbol(symbol_frame, test_days=100, refit_every=50, seq_models={})
    assert set(res.index) == set(BASELINES) | {"kalman", "markov"}


def test_history_too_short_raises(symbol_frame):
    with pytest.raises(ValueError, match="history too short"):
        evaluate.evaluate_symbol(symbol_frame.iloc[:300], test_days=100)


def test_evaluate_universe_skips_bad_symbols(small_panel, caplog):
    res = evaluate.evaluate_universe(small_panel, ["S00", "S01"], test_days=100, refit_every=50,
                                     seq_models={})
    assert set(res["symbol"]) == {"S00", "S01"}
    short = small_panel[small_panel["Date"] < "2016-01-01"]
    with pytest.raises(ValueError):  # every symbol skipped -> nothing to concat
        evaluate.evaluate_universe(short, ["S00"], test_days=100, seq_models={})
    assert "skip S00" in caplog.text
