"""Panel loading + universe selection for prediction models.

I/O lives here (parquet read); everything downstream takes DataFrames.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_LTM_DIR = Path("~/repos/global-stock-screener/cache_seed/ltm").expanduser()
LTM_DIR = str(Path(os.environ.get("LTM_DIR", DEFAULT_LTM_DIR)).expanduser())

PANEL_COLUMNS = ("Date", "Symbol", "Open", "High", "Low", "Close", "Volume")
MIN_HISTORY_YEARS = 8.0        # a symbol must span this many years to be eligible
CALENDAR_DAYS_PER_YEAR = 365
MIN_OBS_PER_YEAR = 200         # ~trading days/yr; guards against gappy series
MAX_ABS_LOG_RET = 0.5          # larger single-day moves are bad ticks / unflagged splits


def panel_path(market: str, ltm_dir: str | os.PathLike | None = None) -> Path:
    """``<ltm_dir>/<MARKET>.parquet``; ``ltm_dir`` defaults to ``$LTM_DIR``."""
    return Path(ltm_dir or LTM_DIR) / f"{market.upper()}.parquet"


def load_panel(market: str, ltm_dir: str | os.PathLike | None = None) -> pd.DataFrame:
    """Long panel [Date, Symbol, Open, High, Low, Close, Volume] for a market."""
    df = pd.read_parquet(panel_path(market, ltm_dir))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    return df


def top_liquid_symbols(panel: pd.DataFrame, n: int = 20,
                       min_years: float = MIN_HISTORY_YEARS) -> list[str]:
    """Most-liquid symbols (median daily dollar volume) with a long history."""
    g = panel.groupby("Symbol")
    span_days = (g["Date"].max() - g["Date"].min()).dt.days
    long_enough = span_days[span_days >= min_years * CALENDAR_DAYS_PER_YEAR].index
    sub = panel[panel["Symbol"].isin(long_enough)]
    dollar_vol = (sub["Close"] * sub["Volume"]).groupby(sub["Symbol"]).median()
    # Guard against data glitches: require a sane, mostly-continuous series
    counts = sub.groupby("Symbol")["Date"].count()
    ok = counts[counts >= min_years * MIN_OBS_PER_YEAR].index
    return dollar_vol.loc[dollar_vol.index.isin(ok)].nlargest(n).index.tolist()


def symbol_series(panel: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Clean single-symbol frame indexed by Date with a ``log_ret`` column."""
    df = panel[panel["Symbol"] == symbol].set_index("Date").sort_index()
    df = df[df["Close"] > 0]
    df = df[~df.index.duplicated(keep="last")]
    df["log_ret"] = np.log(df["Close"]).diff()
    # Drop absurd jumps (bad ticks / unflagged splits) rather than model them
    df = df[df["log_ret"].abs().fillna(0) < MAX_ABS_LOG_RET]
    df["log_ret"] = np.log(df["Close"]).diff()
    return df.dropna(subset=["log_ret"])
