import numpy as np
import pandas as pd
import pytest
import torch

from price_prediction import lstm_model as L


def symbol_frame(n=200, seed=0):
    rng = np.random.RandomState(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * 0.99,
                       "Volume": 1000 + rng.randint(0, 500, n).astype(float)}, index=idx)
    df["log_ret"] = np.log(df["Close"]).diff()
    return df.dropna(subset=["log_ret"])


def test_build_features_columns_and_formulas():
    df = symbol_frame()
    f = L.build_features(df)
    assert list(f.columns) == ["ret", "vol21", "range", "vol_z", "mom5"]
    assert f.notna().all().all()
    assert len(f) == len(df) - 62                                  # the 63-bar volume window sets the warm-up
    t = f.index[10]
    assert f.loc[t, "ret"] == df.loc[t, "log_ret"]
    assert f.loc[t, "range"] == pytest.approx((df.loc[t, "High"] - df.loc[t, "Low"]) / df.loc[t, "Close"])
    assert f.loc[t, "vol21"] == pytest.approx(df["log_ret"].rolling(21).std().loc[t])
    assert f.loc[t, "mom5"] == pytest.approx(df["log_ret"].rolling(5).sum().loc[t])
    vm, vs = df["Volume"].rolling(63).mean().loc[t], df["Volume"].rolling(63).std().loc[t]
    assert f.loc[t, "vol_z"] == pytest.approx((df.loc[t, "Volume"] - vm) / vs)


def test_features_use_only_past_and_present_bars():
    df = symbol_frame()
    full = L.build_features(df)
    cut = L.build_features(df.iloc[:150])
    pd.testing.assert_frame_equal(full.loc[cut.index], cut)


def test_constant_volume_rows_are_dropped_not_infinite():
    df = symbol_frame()
    df["Volume"] = 1000.0                                          # zero rolling std -> NaN z-score -> row dropped
    assert L.build_features(df).empty


def test_make_windows_alignment_and_dtype():
    feat = np.arange(20, dtype=float).reshape(10, 2)
    target = np.arange(10, dtype=float) * 10
    X, Y = L.make_windows(feat, target, lookback=3)
    assert X.shape == (7, 3, 2) and Y.shape == (7,)
    assert X.dtype == np.float32 and Y.dtype == np.float32
    assert X[0].tolist() == feat[0:3].tolist() and Y[0] == 30            # window [i-3, i) predicts target[i]
    assert X[-1].tolist() == feat[6:9].tolist() and Y[-1] == 90
    X0, Y0 = L.make_windows(feat, target, lookback=10)
    assert len(X0) == 0 and len(Y0) == 0


@pytest.mark.parametrize("cell,cls", [("lstm", torch.nn.LSTM), ("gru", torch.nn.GRU), ("anything", torch.nn.GRU)])
def test_seqnet_cell_and_output_shape(cell, cls):
    net = L.SeqNet(5, hidden=7, cell=cell)
    assert isinstance(net.rnn, cls) and net.rnn.hidden_size == 7
    out = net(torch.zeros(4, 10, 5))
    assert out.shape == (4,)


def test_defaults():
    m = L.SeqModel()
    assert (m.cell, m.lookback, m.hidden, m.epochs, m.lr, m.batch) == ("lstm", 42, 32, 30, 1e-3, 128)
    assert m.net is None and m.target_sd == 1.0
    assert L.DEVICE == "cpu"


def learnable(n=260, seed=1):
    rng = np.random.RandomState(seed)
    x = rng.normal(0, 1, n)
    feat = pd.DataFrame({"a": x, "b": rng.normal(0, 1, n)}, index=pd.bdate_range("2020-01-01", periods=n))
    target = pd.Series(0.01 * np.roll(x, 0), index=feat.index)        # target depends on the window's last bar
    return feat, target


def test_fit_standardises_with_training_statistics_only():
    feat, target = learnable()
    m = L.SeqModel(lookback=5, hidden=4, epochs=1).fit(feat * 3 + 7, target)
    assert m.mu == pytest.approx((feat * 3 + 7).values.mean(0))
    assert m.sd == pytest.approx((feat * 3 + 7).values.std(0) + 1e-9)
    assert m.target_sd == pytest.approx(target.values.std() + 1e-9)
    assert isinstance(m.net, L.SeqNet)


def test_predict_returns_one_value_per_full_window_with_the_right_index():
    feat, target = learnable()
    m = L.SeqModel(lookback=5, hidden=4, epochs=1).fit(feat, target)
    p = m.predict(feat.iloc[100:130])
    assert isinstance(p, pd.Series) and len(p) == 25
    assert p.index.equals(feat.index[105:130])                       # first `lookback` rows have no full window
    assert np.isfinite(p.values).all()
    assert not m.net.training


def test_training_is_reproducible_under_a_fixed_seed():
    feat, target = learnable()
    runs = []
    for _ in range(2):
        torch.manual_seed(123)
        runs.append(L.SeqModel(lookback=5, hidden=4, epochs=2).fit(feat, target).predict(feat.iloc[:60]))
    pd.testing.assert_series_equal(runs[0], runs[1])


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_network_learns_a_simple_signal(cell):
    feat, target = learnable(n=400)
    torch.manual_seed(7)
    m = L.SeqModel(cell=cell, lookback=4, hidden=16, epochs=60, lr=5e-3, batch=64).fit(feat, target)
    pred = m.predict(feat)
    truth = target.reindex(pred.index)
    assert np.corrcoef(pred.values, truth.values)[0, 1] > 0.5
    assert abs(pred.std() - truth.std()) < truth.std()               # predictions are on the return scale, not z-scores


def test_prediction_scale_follows_target_sd():
    feat, target = learnable()
    torch.manual_seed(0)
    small = L.SeqModel(lookback=5, hidden=4, epochs=1).fit(feat, target * 1.0).predict(feat)
    torch.manual_seed(0)
    big = L.SeqModel(lookback=5, hidden=4, epochs=1).fit(feat, target * 100.0).predict(feat)
    np.testing.assert_allclose(big.values, small.values * 100.0, rtol=1e-3, atol=1e-8)
