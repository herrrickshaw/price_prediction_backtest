from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from price_prediction import replay


@pytest.fixture(scope="module")
def signals(wide_panel):
    return replay.build_signals(wide_panel)


def test_build_signals_matches_baseline(signals, baseline):
    sig, dates = signals
    b = baseline["replay"]
    assert len(sig) == b["n_sig"]
    assert {k: int(v) for k, v in sig["signal"].value_counts().items()} == b["by_signal"]
    assert {str(k): int(v) for k, v in dates["trend"].value_counts().items()} == b["trend_counts"]
    vol_counts = {str(k): int(v) for k, v in dates["vol_regime"].value_counts().items()}
    assert vol_counts == b["vol_counts"]
    assert sig["ex5"].sum() == pytest.approx(b["ex5_sum"], abs=1e-8)
    assert sig["ex21"].sum() == pytest.approx(b["ex21_sum"], abs=1e-8)


def test_clustered_stats_matches_baseline(signals, baseline):
    sig, _ = signals
    cs = replay.clustered_stats(sig, 21, ["signal"], min_dates=5)
    got = {idx: {c: float(v) for c, v in row.items()} for idx, row in cs.to_dict("index").items()}
    assert got == baseline["clustered"]


def test_rsi14_matches_baseline(wide_panel, baseline):
    close = wide_panel.pivot_table(index="Date", columns="Symbol", values="Close", aggfunc="last")
    rsi = replay._rsi14(close)
    assert rsi["S00"].iloc[-1] == pytest.approx(baseline["rsi"]["last_S00"], abs=1e-9)
    assert rsi.stack().mean() == pytest.approx(baseline["rsi"]["mean"], abs=1e-9)
    assert rsi.stack().dropna().between(0, 100).all()


def test_box_excludes_current_bar():
    """Owner's rule: the Darvas box is formed from the prior BOX_WIN bars only.

    A bar that is itself the highest high must still be able to close above
    the box (a breakout); if the current bar were inside the box this could
    never happen.
    """
    n = replay.BOX_WIN + 10
    idx = pd.bdate_range("2020-01-01", periods=n)
    high = pd.DataFrame({"A": np.full(n, 10.0)}, index=idx)
    high.iloc[-1] = 20.0
    box_hi = high.rolling(replay.BOX_WIN).max().shift(1)
    assert box_hi.iloc[-1, 0] == 10.0  # today's 20 is not in today's box


def test_build_signals_rejects_shallow_panel(small_panel):
    with pytest.raises(ValueError, match="panel too shallow"):
        replay.build_signals(small_panel)


def test_run_market_with_injected_loader(wide_panel, tmp_path, capsys):
    pytest.importorskip("pyarrow")
    replay.run_market("XX", tmp_path, load=lambda market: wide_panel)
    assert (tmp_path / "replay_signals_xx.parquet").exists()
    out = capsys.readouterr().out
    assert "XX h=21d: signal x trend regime" in out


def test_main_logs_failed_market(monkeypatch, tmp_path, caplog):
    def boom(market):
        raise ValueError("no panel")

    monkeypatch.setattr(replay, "load_panel", boom)
    replay.main(["--market", "ZZ", "--out", str(tmp_path)])
    assert "ZZ: FAILED" in caplog.text
