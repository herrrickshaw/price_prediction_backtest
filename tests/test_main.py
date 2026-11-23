import importlib
import os
import sys

import pandas as pd
import pytest

M = importlib.import_module("price_prediction.__main__")


def fake_results():
    return pd.DataFrame({
        "model": ["a", "b", "a", "b"], "symbol": ["X", "X", "Y", "Y"],
        "dir_acc": [0.5, 0.6, 0.5, 0.7], "rmse": [0.01] * 4,
        "strat_sharpe": [0.1, 0.9, 0.3, 0.5], "strat_ann_ret": [0.0] * 4, "bh_sharpe": [0.2] * 4})


def wire(monkeypatch, symbols=("S1", "S2")):
    calls = {}
    monkeypatch.setattr(M, "load_panel", lambda m: calls.setdefault("market", m) and "PANEL")
    monkeypatch.setattr(M, "top_liquid_symbols", lambda panel, n: calls.update(top=n) or list(symbols))
    monkeypatch.setattr(M, "evaluate_universe", lambda panel, syms, **kw: calls.update(syms=syms, kw=kw) or fake_results())
    return calls


def test_defaults_summary_and_output_file(tmp_path, monkeypatch, capsys):
    calls = wire(monkeypatch)
    monkeypatch.setattr(M, "__file__", str(tmp_path / "price_prediction" / "__main__.py"))
    monkeypatch.setattr(sys, "argv", ["prog"])
    M.main()
    out = capsys.readouterr().out
    assert calls["market"] == "US" and calls["top"] == 20 and calls["syms"] == ["S1", "S2"]
    assert calls["kw"] == {"test_days": 500, "refit_every": 125}
    assert "US: evaluating 2 symbols, 500 test days, refit every 125" in out
    path = tmp_path / "reports" / "price_prediction_us.csv"
    assert pd.read_csv(path).shape == (4, 7)
    assert f"per-symbol rows -> {path}" in out
    summary = out.split("=== mean across symbols")[1]
    rows = {l.split()[0]: l.split() for l in summary.splitlines() if l[:2] in ("a ", "b ")}
    assert summary.index("\nb ") < summary.index("\na ")            # sorted by mean Sharpe, best first
    assert float(rows["b"][3]) == pytest.approx(0.7) and float(rows["a"][3]) == pytest.approx(0.2)   # strat_sharpe means
    assert float(rows["b"][1]) == pytest.approx(0.65) and float(rows["a"][1]) == pytest.approx(0.5)  # dir_acc means


def test_flags_override_defaults(tmp_path, monkeypatch, capsys):
    calls = wire(monkeypatch)
    out = tmp_path / "x" / "y" / "res.csv"
    monkeypatch.setattr(sys, "argv", ["prog", "--market", "IN", "--top", "3", "--test-days", "50",
                                      "--refit-every", "10", "--out", str(out)])
    M.main()
    assert calls["market"] == "IN" and calls["top"] == 3
    assert calls["kw"] == {"test_days": 50, "refit_every": 10}
    assert out.exists()                                              # missing directories are created
    assert "IN: evaluating 2 symbols, 50 test days, refit every 10" in capsys.readouterr().out


def test_explicit_symbols_bypass_the_universe_selection(tmp_path, monkeypatch, capsys):
    calls = wire(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["prog", "--market", "IN", "--symbols", "RELIANCE", "TCS", "--out", str(tmp_path / "o.csv")])
    M.main()
    assert calls["syms"] == ["RELIANCE", "TCS"] and "top" not in calls
    calls2 = wire(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["prog", "--symbols", "--out", str(tmp_path / "o2.csv")])
    M.main()                                                          # an empty --symbols list falls back to --top
    assert calls2["top"] == 20 and calls2["syms"] == ["S1", "S2"]
