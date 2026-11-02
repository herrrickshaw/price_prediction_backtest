"""Walk-forward evaluation of prediction models against naive baselines.

Protocol: the final `test_days` trading days are the out-of-sample period.
Trainable models (Markov, LSTM, GRU) are refit every `refit_every` days on an
expanding window ending at the refit date; the Kalman filter is inherently
online/causal. Every prediction at day t uses data through day t only and is
scored against the realised return of day t+1.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from .baselines import BASELINES
from .features import build_features
from .kalman_trend import KalmanTrendModel
from .markov_regime import MarkovRegimeModel, label_states

log = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
MIN_TRAIN_DAYS = 750          # history required before the first test day
DEFAULT_SEQ_MODELS: dict[str, dict] = {"lstm": {}, "gru": {}}
EPS = 1e-12                   # guards zero-variance Sharpe denominators

# A sequence model: fit(features, target) -> self; predict(features) -> Series;
# exposes `.lookback` (rows of context needed before the first prediction).
SeqModelFactory = Callable[..., Any]


def _default_seq_model_factory(cell: str, **kw: Any) -> Any:
    """Build the PyTorch SeqModel; torch is imported only when first needed."""
    from .lstm_model import SeqModel

    return SeqModel(cell=cell, **kw)


def metrics(pred: pd.Series, actual: pd.Series) -> dict:
    """Score next-day-return predictions: hit rate, RMSE, frictionless Sharpe vs B&H."""
    df = pd.concat({"p": pred, "a": actual}, axis=1).dropna()
    p, a = df["p"], df["a"]
    nonzero = p != 0
    hit = float((np.sign(p[nonzero]) == np.sign(a[nonzero])).mean()) if nonzero.any() else np.nan
    strat = np.sign(p) * a  # long/short next-day strategy, no costs
    ann = np.sqrt(TRADING_DAYS_PER_YEAR)
    return {
        "n": len(df),
        "dir_acc": hit,
        "rmse": float(np.sqrt(((p - a) ** 2).mean())),
        "strat_ann_ret": float(strat.mean() * TRADING_DAYS_PER_YEAR),
        "strat_sharpe": float(strat.mean() / (strat.std() + EPS) * ann),
        "bh_ann_ret": float(a.mean() * TRADING_DAYS_PER_YEAR),
        "bh_sharpe": float(a.mean() / (a.std() + EPS) * ann),
    }


_metrics = metrics  # backwards-compatible alias


def kalman_predictions(df: pd.DataFrame, test_idx: pd.DatetimeIndex) -> pd.Series:
    """Online Kalman slope on log close, one causal pass, restricted to test days."""
    return KalmanTrendModel().predict_return(np.log(df["Close"])).reindex(test_idx)


def walk_forward_predictions(df: pd.DataFrame, test_idx: pd.DatetimeIndex, refit_every: int,
                             seq_models: dict[str, dict],
                             seq_model_factory: SeqModelFactory) -> dict[str, pd.Series]:
    """Markov + sequence-net predictions with expanding-window refits every ``refit_every`` days."""
    ret = df["log_ret"]
    states_all = label_states(ret)
    feats_all = build_features(df)
    markov_p: list[pd.Series] = []
    seq_p: dict[str, list[pd.Series]] = {k: [] for k in seq_models}
    for s in range(0, len(test_idx), refit_every):
        block = test_idx[s:s + refit_every]
        train_end = block[0]
        tr_ret = ret.loc[:train_end].iloc[:-1]

        m = MarkovRegimeModel().fit(states_all.loc[tr_ret.index], tr_ret)
        markov_p.append(pd.Series(
            [m.predict_return(int(states_all.loc[t])) for t in block], index=block))

        tr_feats = feats_all.loc[:train_end].iloc[:-1]
        for name, kw in seq_models.items():
            sm = seq_model_factory(name, **kw).fit(tr_feats, ret.reindex(tr_feats.index))
            # feed enough history before the block to fill the lookback window
            ctx = feats_all.loc[:block[-1]].iloc[-(len(block) + sm.lookback):]
            seq_p[name].append(sm.predict(ctx).reindex(block))

    preds = {"markov": pd.concat(markov_p)}
    for name in seq_models:
        preds[name] = pd.concat(seq_p[name])
    return preds


def evaluate_symbol(df: pd.DataFrame, test_days: int = 500, refit_every: int = 125,
                    seq_models: dict[str, dict] | None = None,
                    seq_model_factory: SeqModelFactory | None = None) -> pd.DataFrame:
    """Metric rows per model for one symbol frame (from ``data.symbol_series``).

    ``seq_models`` maps sequence-model name -> constructor kwargs (default:
    ``{"lstm": {}, "gru": {}}``); pass ``{}`` to skip the nets entirely.
    ``seq_model_factory`` builds them (default: the PyTorch ``SeqModel``) and
    can be swapped for a stub in tests.
    """
    ret = df["log_ret"]
    if len(ret) < test_days + MIN_TRAIN_DAYS:
        raise ValueError("history too short")
    actual = ret.shift(-1)  # realised next-day return, aligned to prediction day
    test_idx = ret.index[-test_days:]
    specs = DEFAULT_SEQ_MODELS if seq_models is None else seq_models
    factory = seq_model_factory or _default_seq_model_factory

    preds: dict[str, pd.Series] = {name: fn(ret, test_idx) for name, fn in BASELINES.items()}
    preds["kalman"] = kalman_predictions(df, test_idx)
    preds.update(walk_forward_predictions(df, test_idx, refit_every, specs, factory))

    rows = [{"model": name, **metrics(p, actual.reindex(test_idx))} for name, p in preds.items()]
    return pd.DataFrame(rows).set_index("model")


def evaluate_universe(panel: pd.DataFrame, symbols: list[str], **kw: Any) -> pd.DataFrame:
    """Run ``evaluate_symbol`` per symbol; symbols with short/bad data are skipped, not fatal."""
    from .data import symbol_series

    out = []
    for sym in symbols:
        try:
            res = evaluate_symbol(symbol_series(panel, sym), **kw)
        except Exception as e:  # noqa: BLE001 — short history, bad data: skip, don't die
            log.warning("skip %s: %s", sym, e)
            continue
        res["symbol"] = sym
        out.append(res.reset_index())
        log.info("done %s", sym)
    return pd.concat(out, ignore_index=True)
