from __future__ import annotations

import subprocess
import sys


def _help(module: str) -> str:
    return subprocess.run([sys.executable, "-m", module, "--help"], check=True,
                          capture_output=True, text=True).stdout


def test_main_help():
    out = _help("price_prediction")
    for flag in ("--market", "--top", "--test-days", "--refit-every", "--symbols", "--out"):
        assert flag in out


def test_replay_help():
    out = _help("price_prediction.replay")
    assert "--market" in out and "--out" in out


def test_main_end_to_end_with_stubbed_nets(monkeypatch, tmp_path, small_panel, capsys):
    """Full CLI path on a synthetic panel: load -> select -> evaluate -> CSV + summary."""
    from price_prediction import __main__ as cli
    from price_prediction import evaluate
    from tests.test_evaluate import StubSeq

    monkeypatch.setattr(cli, "load_panel", lambda market: small_panel)
    monkeypatch.setattr(evaluate, "_default_seq_model_factory",
                        lambda cell, **kw: StubSeq(cell, **kw))
    out = tmp_path / "res.csv"
    cli.main(["--market", "ZZ", "--symbols", "S00", "S01", "--test-days", "100",
              "--refit-every", "50", "--out", str(out)])
    assert out.exists()
    text = capsys.readouterr().out
    assert "mean across symbols" in text and "markov" in text
