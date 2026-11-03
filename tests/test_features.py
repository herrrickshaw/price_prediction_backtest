from __future__ import annotations

import numpy as np

from price_prediction.features import FEATURE_COLS, build_features


def test_build_features_matches_baseline(symbol_frame, baseline):
    f = build_features(symbol_frame)
    b = baseline["features"]
    assert len(f) == b["n"]
    assert list(f.columns) == b["cols"] == list(FEATURE_COLS)
    np.testing.assert_allclose(f.sum().values, b["colsum"], atol=1e-7)
    assert not f.isna().any().any()
