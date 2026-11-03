"""Full historical replay: screener recommends, regime gates, outcomes judge.

Replays Darvas-style breakout/breakdown signals over the full OHLCV panel,
labels each signal date with the market's point-in-time trend regime (Markov
state of the equal-weight market index) and volatility regime (expanding
percentile of rolling vol), then scores forward 5d/21d excess returns.

Statistics are clustered by signal date: per-date mean excess first, then a
t-stat across dates — signals sharing a date share market moves, so
per-signal t-stats would be badly overstated (the defect in the small-sample
live tracking this replay extends).

Signal definitions (documented proxies for the live daily_scanner):
  BREAKOUT_BUY   close crosses above the prior 55-day box high (current bar
                 excluded from the box), close > EMA50, RSI14 in [50, 80]
  BREAKDOWN_SELL close crosses below the prior 55-day box low, close < EMA50

Pipeline (each stage is a pure function on wide Date x Symbol frames):
  load_panel -> select_universe -> darvas_signals + market_regimes
             -> attach_outcomes (= build_signals) -> clustered_stats

Run:  python -m price_prediction.replay --market US IN JP KR EU UK CN HK
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .data import load_panel

log = logging.getLogger(__name__)

TREND_NAMES = {0: "bear", 1: "chop", 2: "bull", -1: "warmup"}
MIN_OBS = 300          # symbol must have this many days to participate
MIN_NAMES = 50         # market index needs this many names per day
UNIVERSE_N = 1000      # top names by median dollar volume
BOX_WIN = 55           # Darvas box lookback; the current bar is always excluded
HORIZONS = (5, 21)     # forward-return horizons (trading days)
BAD_TICK_ABS_RET = 0.6     # mask single-day moves larger than this (bad ticks / splits)
EMA_SPAN = 50              # trend filter on the stock
RSI_WIN = 14
RSI_BUY_RANGE = (50, 80)   # RSI14 band required for BREAKOUT_BUY
BASKET_N = 20              # blue-chip basket size for the market index
BASKET_MIN_COVERAGE = 0.95 # a basket name needs prices on this share of days
BASKET_MIN_NAMES = 4       # trimmed mean needs more than this many names
INDEX_EMA_SPAN = 100       # market trend: index vs EMA100 ...
INDEX_SLOPE_LAG = 5        # ... and the EMA's 5-day slope
INDEX_VOL_WIN = 21         # rolling vol window of the index
VOL_PCT_MIN_HISTORY = 252  # expanding percentile needs a year of history
VOL_REGIME_BINS = [0, 1 / 3, 2 / 3, 1.0001]
VOL_REGIME_LABELS = ["LOW", "MID", "HIGH"]
MIN_DATES_PER_CELL = 100   # clustered_stats drops cells with fewer signal dates
DEFAULT_MARKETS = ["US", "IN", "JP", "KR", "CN"]
REPO_ROOT = Path(__file__).resolve().parent.parent

PanelLoader = Callable[[str], pd.DataFrame]


def _rsi14(close: pd.DataFrame) -> pd.DataFrame:
    """Wilder RSI over ``RSI_WIN`` days, column-wise."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / RSI_WIN, min_periods=RSI_WIN).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / RSI_WIN, min_periods=RSI_WIN).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def _wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """Pivot a long panel column to Date x Symbol."""
    return panel.pivot_table(index="Date", columns="Symbol", values=col,
                             aggfunc="last").sort_index()


def select_universe(close: pd.DataFrame, vol: pd.DataFrame) -> tuple[pd.Index, pd.Series]:
    """Top-``UNIVERSE_N`` symbols by median dollar volume among long-history names.

    Returns (kept symbols, median dollar volume per eligible symbol). Kills
    penny-stock artifacts in the breadth index and benchmark, and trims (not
    eliminates) delisting-survivorship in the sell leg. Full-sample liquidity
    ranking is mild lookahead — documented tradeoff.
    """
    hist_ok = close.columns[close.notna().sum() >= MIN_OBS]
    if len(hist_ok) < MIN_NAMES:
        raise ValueError(
            f"only {len(hist_ok)} symbols with >= {MIN_OBS} obs — panel too shallow")
    dv = (close[hist_ok] * vol[hist_ok]).median()
    return dv.nlargest(UNIVERSE_N).index, dv


def mask_bad_ticks(close: pd.DataFrame) -> pd.DataFrame:
    """Blank out closes whose single-day move exceeds ``BAD_TICK_ABS_RET``."""
    return close.mask(close.pct_change().abs() > BAD_TICK_ABS_RET)


def darvas_signals(close: pd.DataFrame, high: pd.DataFrame,
                   low: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Boolean (buy, sell) frames. The box is built from the prior ``BOX_WIN``
    bars — the current bar is always excluded (``.shift(1)``); do not change this."""
    ema50 = close.ewm(span=EMA_SPAN, min_periods=EMA_SPAN).mean()
    rsi = _rsi14(close)
    box_hi = high.rolling(BOX_WIN).max().shift(1)   # current bar excluded
    box_lo = low.rolling(BOX_WIN).min().shift(1)
    rsi_lo, rsi_hi = RSI_BUY_RANGE
    buy = (close > box_hi) & (close.shift(1) <= box_hi.shift(1)) \
        & (close > ema50) & (rsi >= rsi_lo) & (rsi <= rsi_hi)
    sell = (close < box_lo) & (close.shift(1) >= box_lo.shift(1)) \
        & (close < ema50)
    return buy, sell


def market_regimes(close: pd.DataFrame, ret: pd.DataFrame, dv: pd.Series) -> pd.DataFrame:
    """Per-date trend ('bear'/'chop'/'bull'/'warmup') and vol regime (LOW/MID/HIGH).

    Market index for regimes: trimmed mean of the top-20 turnover names with
    near-complete coverage (blue-chip basket). The full-universe breadth
    median compounds hugely negative in IN/KR/CN (sporadic microcaps +
    decliner-heavy turnover tail) and mislabels the whole decade "bear";
    the basket tracks each market's real index to within reason.
    """
    cov_full = close.columns[close.notna().mean() >= BASKET_MIN_COVERAGE]
    basket = (dv.reindex(cov_full)).nlargest(BASKET_N).index
    bret = ret[basket]
    n_b = bret.notna().sum(axis=1)
    trimmed = (bret.sum(axis=1) - bret.max(axis=1) - bret.min(axis=1)) / (n_b - 2)
    idx_ret = trimmed.where(n_b > BASKET_MIN_NAMES).dropna()
    # market trend: index vs its EMA100 and the EMA's slope (balanced, PIT)
    idx = (1 + idx_ret).cumprod()
    ema = idx.ewm(span=INDEX_EMA_SPAN, min_periods=INDEX_EMA_SPAN).mean()
    slope = ema.diff(INDEX_SLOPE_LAG)
    trend = pd.Series(1, index=idx.index)               # chop
    trend[(idx > ema) & (slope > 0)] = 2                # bull
    trend[(idx < ema) & (slope < 0)] = 0                # bear
    trend[ema.isna()] = -1
    vol21 = idx_ret.rolling(INDEX_VOL_WIN).std()
    vol_pct = vol21.expanding(VOL_PCT_MIN_HISTORY).apply(
        lambda w: (w[:-1] <= w[-1]).mean() if len(w) > 1 else np.nan, raw=True)
    vol_reg = pd.cut(vol_pct, VOL_REGIME_BINS, labels=VOL_REGIME_LABELS)
    return pd.DataFrame({
        "trend": trend.map(TREND_NAMES).reindex(idx_ret.index),
        "vol_regime": vol_reg,
    })


def attach_outcomes(close: pd.DataFrame, buy: pd.DataFrame, sell: pd.DataFrame,
                    dates: pd.DataFrame) -> pd.DataFrame:
    """Long frame of signals with forward returns, excess vs the per-date market median, regimes."""
    fwd = {h: close.shift(-h) / close - 1 for h in HORIZONS}
    mkt_med = {h: fwd[h].median(axis=1) for h in HORIZONS}
    frames = []
    for name, mask in [("BREAKOUT_BUY", buy), ("BREAKDOWN_SELL", sell)]:
        s = mask.stack()
        s = s[s]
        f = pd.DataFrame(index=s.index)
        f["signal"] = name
        for h in HORIZONS:
            f[f"fwd{h}"] = fwd[h].stack().reindex(s.index)
            f[f"ex{h}"] = f[f"fwd{h}"] - mkt_med[h].reindex(
                s.index.get_level_values(0)).values
        frames.append(f)
    sig = pd.concat(frames).reset_index().rename(
        columns={"level_0": "Date", "level_1": "Symbol"})
    return sig.merge(dates, left_on="Date", right_index=True, how="left")


def build_signals(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (signals long-frame, per-date market/regime frame)."""
    close, high, low, vol = (_wide(panel, c) for c in ("Close", "High", "Low", "Volume"))
    keep, dv = select_universe(close, vol)
    close, high, low = close[keep], high[keep], low[keep]
    close = mask_bad_ticks(close)
    ret = close.pct_change()
    buy, sell = darvas_signals(close, high, low)
    dates = market_regimes(close, ret, dv)
    return attach_outcomes(close, buy, sell, dates), dates


def clustered_stats(sig: pd.DataFrame, h: int, by: list[str],
                    min_dates: int = MIN_DATES_PER_CELL) -> pd.DataFrame:
    """Per-date means first, then t across dates. Cells under min_dates dropped."""
    col = f"ex{h}"
    d = sig.dropna(subset=[col])
    day = d.groupby(["Date"] + by, observed=True)[col].mean().reset_index()
    g = day.groupby(by, observed=True)[col]
    out = pd.DataFrame({
        "n_signals": d.groupby(by, observed=True)[col].size(),
        "n_dates": g.size(),
        "mean_excess_%": g.mean() * 100,
        "t_clustered": g.mean() / (g.std() / np.sqrt(g.size().clip(lower=2))),
        "hit_%": d.groupby(by, observed=True)[col].apply(lambda x: (x > 0).mean() * 100),
    })
    return out[out.n_dates >= min_dates].round(2)


def run_market(market: str, out_dir: str | os.PathLike,
               load: PanelLoader = load_panel) -> None:
    """Replay one market, write ``replay_signals_<market>.parquet``, print the stat tables."""
    print(f"\n================ {market} ================")
    sig, _ = build_signals(load(market))
    sig.to_parquet(Path(out_dir) / f"replay_signals_{market.lower()}.parquet")
    for h in HORIZONS:
        print(f"\n--- {market} h={h}d: signal alone ---")
        print(clustered_stats(sig, h, ["signal"]).to_string())
        print(f"\n--- {market} h={h}d: signal x trend regime ---")
        print(clustered_stats(sig, h, ["signal", "trend"]).to_string())
        print(f"\n--- {market} h={h}d: signal x vol regime ---")
        print(clustered_stats(sig, h, ["signal", "vol_regime"]).to_string())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--market", nargs="+", default=DEFAULT_MARKETS)
    ap.add_argument("--out", default=str(REPO_ROOT / "reports"))
    return ap


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for m in args.market:
        try:
            run_market(m, args.out)
        except Exception as e:  # noqa: BLE001 — one bad market must not stop the others
            log.error("%s: FAILED — %s", m, e)


if __name__ == "__main__":
    main()
