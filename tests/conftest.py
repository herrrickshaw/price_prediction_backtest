"""Shared fixtures: small seeded synthetic OHLCV panels (never real market data)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def make_panel(n_sym: int, n_days: int, seed: int) -> pd.DataFrame:
    """Long panel [Date, Symbol, OHLCV] of seeded geometric random walks."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    rows = []
    for i in range(n_sym):
        lr = rng.normal(0.0003 * (i % 3), 0.01 + 0.002 * (i % 5), n_days)
        close = 100 * np.exp(np.cumsum(lr))
        high = close * (1 + rng.uniform(0, 0.02, n_days))
        low = close * (1 - rng.uniform(0, 0.02, n_days))
        opn = (high + low) / 2
        vol = rng.integers(1000, 100000, n_days) * (i + 1)
        rows.append(pd.DataFrame({"Date": dates, "Symbol": f"S{i:02d}", "Open": opn,
                                  "High": high, "Low": low, "Close": close, "Volume": vol}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def baseline() -> dict:
    """Outputs captured from the pre-refactor code on the same synthetic panels."""
    return json.loads((FIXTURES / "baseline.json").read_text())


@pytest.fixture(scope="session")
def small_panel() -> pd.DataFrame:
    return make_panel(5, 1000, 1)


@pytest.fixture(scope="session")
def wide_panel() -> pd.DataFrame:
    return make_panel(60, 600, 2)


@pytest.fixture(scope="session")
def symbol_frame(small_panel):
    from price_prediction.data import symbol_series

    return symbol_series(small_panel, "S01")
