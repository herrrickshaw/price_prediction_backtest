"""Torch-free feature engineering for the sequence models.

Kept separate from ``lstm_model`` so the walk-forward harness (and its tests)
can build features without importing PyTorch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VOL_WIN = 21        # rolling std window of log returns
VOLUME_WIN = 63     # rolling window for the volume z-score
MOM_WIN = 5         # short momentum window (sum of log returns)
FEATURE_COLS = ("ret", "vol21", "range", "vol_z", "mom5")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day features from a symbol frame (needs log_ret, High, Low, Close, Volume).

    Returns a frame with columns ``FEATURE_COLS``; warm-up rows are dropped.
    """
    f = pd.DataFrame(index=df.index)
    f["ret"] = df["log_ret"]
    f["vol21"] = df["log_ret"].rolling(VOL_WIN).std()
    f["range"] = (df["High"] - df["Low"]) / df["Close"]
    vol_mu = df["Volume"].rolling(VOLUME_WIN).mean()
    vol_sd = df["Volume"].rolling(VOLUME_WIN).std()
    f["vol_z"] = (df["Volume"] - vol_mu) / vol_sd.replace(0, np.nan)
    f["mom5"] = df["log_ret"].rolling(MOM_WIN).sum()
    return f.dropna()
