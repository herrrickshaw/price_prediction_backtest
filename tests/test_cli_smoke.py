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
