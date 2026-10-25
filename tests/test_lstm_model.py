"""Tests for price_prediction.lstm_model windowing / alignment.

torch (>300 MB installed) is not installed in the review venv; when it is missing a minimal stub
is injected so the numpy windowing and the date labels of SeqModel.predict can still be tested
with a deterministic stand-in network. With real torch installed the same tests run unchanged."""
import contextlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - depends on environment
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    nn.Module = object
    torch.nn = nn
    torch.manual_seed = lambda *a, **k: None
    torch.float32 = "float32"
    torch.no_grad = contextlib.nullcontext
    torch.tensor = lambda x, dtype=None, device=None: np.asarray(x)
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn

from price_prediction.lstm_model import SeqModel, build_features, make_windows  # noqa: E402


class _Out:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class LastFeatureNet:
    """Predicts the value of feature 0 on the last row of each window."""

    def eval(self):
        return self

    def __call__(self, x):
        return _Out(np.asarray(x)[:, -1, 0])


def _model(lookback=3):
    m = SeqModel(lookback=lookback)
    m.mu, m.sd, m.target_sd = np.array([0.0]), np.array([1.0]), 1.0
    m.net = LastFeatureNet()
    return m


def test_make_windows_pairs_window_with_next_target():
    feat = np.arange(6, dtype=float).reshape(-1, 1)
    X, Y = make_windows(feat, np.arange(6) * 10.0, lookback=2)
    assert X.shape == (4, 2, 1)
    # window rows [i-2, i-1] are paired with target[i]: the day after the window's last row
    assert X[0, :, 0].tolist() == [0, 1] and Y[0] == 20


def test_prediction_labelled_t_uses_data_through_t():
    # evaluate.py scores the prediction labelled t against ret[t+1] and promises
    # "every prediction at day t uses data through day t". The forecast labelled t must
    # therefore come from the window that ENDS at row t (and so forecast t+1, as trained).
    idx = pd.bdate_range("2021-01-01", periods=8)
    feat = pd.DataFrame({"x": np.arange(8, dtype=float)}, index=idx)
    pred = _model(lookback=3).predict(feat)
    for t, v in pred.items():
        assert v == feat.loc[t, "x"], f"forecast labelled {t.date()} built from data ending {v}"


def test_prediction_covers_the_last_row():
    # the last day of a walk-forward block must get a forecast too
    idx = pd.bdate_range("2021-01-01", periods=8)
    feat = pd.DataFrame({"x": np.arange(8, dtype=float)}, index=idx)
    pred = _model(lookback=3).predict(feat)
    assert pred.index[-1] == idx[-1]
    assert len(pred) == 8 - 3 + 1


def test_build_features_columns_and_warmup():
    n = 100
    idx = pd.bdate_range("2021-01-01", periods=n)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * 0.99,
                       "Volume": rng.integers(1e5, 2e5, n).astype(float)}, index=idx)
    df["log_ret"] = np.log(df["Close"]).diff()
    f = build_features(df)
    assert list(f.columns) == ["ret", "vol21", "range", "vol_z", "mom5"]
    assert f.index[0] == idx[62]                   # 63-row rolling volume window warm-up
    assert f["range"].iloc[0] == pytest.approx(0.02)
