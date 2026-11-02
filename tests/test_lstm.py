from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # heavy optional dependency; skipped in CI

from price_prediction.features import build_features  # noqa: E402
from price_prediction.lstm_model import SeqModel, make_windows  # noqa: E402


def test_make_windows_shapes():
    feat = np.arange(20 * 3, dtype=float).reshape(20, 3)
    X, Y = make_windows(feat, np.arange(20, dtype=float), lookback=5)
    assert X.shape == (15, 5, 3) and Y.shape == (15,)
    assert X.dtype == np.float32
    np.testing.assert_array_equal(X[0], feat[0:5])
    assert Y[0] == 5


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_seqmodel_fit_predict_aligns_index(symbol_frame, cell):
    feats = build_features(symbol_frame).iloc[:300]
    target = symbol_frame["log_ret"].reindex(feats.index)
    sm = SeqModel(cell=cell, lookback=10, hidden=4, epochs=1).fit(feats, target)
    pred = sm.predict(feats.iloc[-50:])
    assert len(pred) == 40
    assert pred.index.equals(feats.index[-40:])
    assert np.isfinite(pred.values).all()
