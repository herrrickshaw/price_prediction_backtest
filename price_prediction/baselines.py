"""Naive next-day-return baselines every model must beat to matter.

Each baseline is a pure function ``(ret, test_idx) -> pd.Series`` of
predictions indexed by ``test_idx``, where ``ret`` is the full log-return
series (so expanding-window baselines can look back before the test period).
``BASELINES`` is the registry the walk-forward harness iterates over; add a
new baseline by registering a function here, callers need no change.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

ALWAYS_LONG_EPS = 1e-12   # a tiny positive forecast == always long / buy & hold

BaselineFn = Callable[[pd.Series, pd.DatetimeIndex], pd.Series]


def naive_zero(ret: pd.Series, test_idx: pd.DatetimeIndex) -> pd.Series:
    """Predict ~0 (sign positive): equivalent to always-long / buy & hold."""
    return pd.Series(ALWAYS_LONG_EPS, index=test_idx)


def naive_yesterday(ret: pd.Series, test_idx: pd.DatetimeIndex) -> pd.Series:
    """1-day momentum: yesterday's return is today's forecast."""
    return ret.reindex(test_idx)


def naive_histmean(ret: pd.Series, test_idx: pd.DatetimeIndex) -> pd.Series:
    """Expanding historical mean return through the day before each test day."""
    return pd.Series([ret.loc[:t].iloc[:-1].mean() for t in test_idx], index=test_idx)


BASELINES: dict[str, BaselineFn] = {
    "naive_zero": naive_zero,
    "naive_yesterday": naive_yesterday,
    "naive_histmean": naive_histmean,
}
