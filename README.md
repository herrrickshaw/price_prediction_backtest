# price_prediction_backtest — walk-forward next-day return models and a regime-gated Darvas replay

Research harness for daily equity price prediction on deep 10-year OHLCV panels
(US, India, Japan, Korea, China). It evaluates three sequential model families
(Markov regime chain, Kalman trend filter, LSTM/GRU) walk-forward against naive
baselines, and separately replays Darvas-style breakout/breakdown signals through
point-in-time market regimes. Output: per-symbol metric CSVs and per-market
signal parquet files in `reports/`, plus the write-ups in `docs/`.

> Status: research-only / educational. Historical data, frictionless P&L, no
> transaction costs. Not investment advice.

<!-- how-to-use-with-ai -->
> **New here?** See [How to use this repo with AI (Claude, Gemini, ChatGPT)](HOW_TO_USE_WITH_AI.md).

## What it does
- **Markov regime chain** (`markov_regime.py`): bear/chop/bull states from rolling trend vs. vol, Laplace-smoothed first-order transition matrix, E[next-day return] read-out.
- **Kalman trend filter** (`kalman_trend.py`): local-linear-trend state space on log price, pure numpy; the filtered slope is the next-day forecast, fully online/causal.
- **LSTM / GRU** (`lstm_model.py`, optional PyTorch): 42-day windows of (return, vol, range, volume-z, momentum); feature scaling fit on the training window only.
- **Walk-forward harness** (`evaluate.py`): last `--test-days` out-of-sample, expanding-window refits every `--refit-every` days, scored against three naive baselines (`baselines.py`).
- **Regime-gated Darvas replay** (`replay.py`): 55-day box breakouts/breakdowns (current bar always excluded from the box), labelled with point-in-time trend and vol regimes, forward 5d/21d excess returns with date-clustered t-stats.

## Quick start
```bash
git clone https://github.com/herrrickshaw/price_prediction_backtest.git
cd price_prediction_backtest
python -m venv .venv && source .venv/bin/activate
pip install -e ".[nets]"          # or: pip install -e .   (Markov + Kalman + baselines only)
python -m price_prediction --market US --top 20 --test-days 500
```

The harness expects `LTM_DIR` (default `~/repos/global-stock-screener/cache_seed/ltm`)
to contain `<MARKET>.parquet` long panels with columns `Date, Symbol, Open, High, Low, Close, Volume`.
Per-symbol results land in `reports/price_prediction_<market>.csv`. Other invocations:

```bash
python -m price_prediction --market IN --symbols RELIANCE TCS
python -m price_prediction.replay --market US IN JP KR CN
```

## Headline result (US, top-20 liquid, 500 OOS days ending 2026-07)

| model | dir. accuracy | long/short Sharpe (no costs) | buy & hold Sharpe |
|---|---|---|---|
| always-long baseline | 0.528 | 0.69 | 0.69 |
| markov | **0.532** | 0.63 | 0.69 |
| kalman | 0.503 | 0.17 | 0.69 |
| gru | 0.509 | −0.01 | 0.69 |
| lstm | 0.510 | −0.06 | 0.69 |

No model beats buy & hold before costs on daily US large-caps. The Markov regime
layer is the only competitive piece — useful as a regime gate for other signals,
not as a price forecaster. The five-market comparison is in
[docs/WALKFORWARD_MULTIMARKET.md](docs/WALKFORWARD_MULTIMARKET.md); the regime-gated
Darvas replay (~470k signals, bull-gated breakouts positive in every market, breakdown
"sells" contrarian) is in [docs/HISTORICAL_REPLAY.md](docs/HISTORICAL_REPLAY.md).

## How it works

| Stage | Module | Output |
|---|---|---|
| Load long OHLCV panel from `$LTM_DIR/<MARKET>.parquet` | `data.load_panel` | DataFrame |
| Pick top-N by median dollar volume with ≥ 8y history; clean one symbol (bad ticks dropped, log returns) | `data.top_liquid_symbols`, `data.symbol_series` | symbol frame |
| Baselines: `naive_zero` (always long), `naive_yesterday` (1-day momentum), `naive_histmean` | `baselines.BASELINES` | prediction series |
| Kalman slope, one causal pass | `evaluate.kalman_predictions` | prediction series |
| Markov + LSTM/GRU, expanding-window refits per block | `evaluate.walk_forward_predictions` | prediction series |
| Directional accuracy, RMSE, frictionless sign-strategy Sharpe vs buy & hold | `evaluate.metrics` | metric rows per model |
| CLI: run universe, write CSV, print mean-per-model summary | `__main__` | `reports/price_prediction_<mkt>.csv` |

Replay pipeline (`replay.py`): `select_universe` (top-1000 by dollar volume) →
`mask_bad_ticks` → `darvas_signals` + `market_regimes` (trimmed top-20 basket index:
trend = EMA100 position + slope, vol = expanding percentile of 21d vol) →
`attach_outcomes` → `clustered_stats` (per-date means first, then t across dates).

Design decisions:
- Every prediction at day *t* uses data through *t* only and is scored on *t+1*.
- The Darvas box is built from the prior 55 bars with `.shift(1)` — the current bar is never part of its own box. Do not change this.
- `features.py` is torch-free so the harness and tests run without PyTorch; `SeqModel` is built through an injectable factory and torch is imported only when the nets actually run.
- Signals sharing a date share market moves, so replay t-stats are clustered by date rather than per signal.

## Repository layout
| Path | Purpose |
|---|---|
| `price_prediction/__main__.py` | CLI for the walk-forward evaluation |
| `price_prediction/data.py` | parquet loading, universe selection, per-symbol cleaning |
| `price_prediction/baselines.py` | naive baseline registry |
| `price_prediction/markov_regime.py`, `kalman_trend.py` | numpy/pandas models |
| `price_prediction/features.py`, `lstm_model.py` | feature windows and the optional PyTorch nets |
| `price_prediction/evaluate.py` | walk-forward harness and metrics |
| `price_prediction/replay.py` | regime-gated Darvas replay CLI |
| `price_prediction/README.md` | model/protocol details |
| `docs/` | result write-ups (multi-market walk-forward, historical replay, accuracy/reliability matrix, watchlist validation) |
| `reports/` | generated outputs (tracked snapshots from the 2026-07 runs) |
| `tests/` | pytest suite on seeded synthetic panels; `tests/fixtures/baseline.json` pins pre-refactor outputs |
| `graphify-out/` | generated knowledge-graph artefacts |

## Data sources
| Source | What is taken | Access route | Caveat |
|---|---|---|---|
| `global-stock-screener` cache (`cache_seed/ltm/<MARKET>.parquet`) | 10y daily OHLCV long panels | local path via `LTM_DIR` | Not in this repo. India panel carries raw-Close split risk on some names; EU/UK/HK panels hold ~1 year and are excluded from the replay. |

## Configuration
| Variable | Default | Meaning |
|---|---|---|
| `LTM_DIR` | `~/repos/global-stock-screener/cache_seed/ltm` | directory of `<MARKET>.parquet` panels |

No secrets are needed or read.

## Testing
```bash
pip install pytest ruff
ruff check .
pytest -q        # LSTM tests are skipped automatically when torch is not installed
```
Tests use small seeded synthetic series only; nothing is downloaded.

## Limitations and known issues
- No transaction costs, slippage or borrow in `strat_sharpe`; a daily sign-flipping strategy trades constantly, so real costs would eat most or all of any small edge.
- Daily close-to-close returns are close to unpredictable; expect directional accuracy near 0.5. The harness exists to measure that fairly, not to assume a model works.
- Universe selection ranks liquidity over the full sample (mild lookahead in membership, not in signals or regimes).
- Replay: overlapping 21d windows make date-clustered t's still somewhat optimistic; cross-market consistency is the evidence, not any single t.
- India panel: unadjusted corporate actions on some names (single-day |move| > 60% masked; 1:1 bonuses at −50% survive).
- `naive_zero` is a tiny positive constant, i.e. always long; `dir_acc` for it is the share of up days.

## Related repositories
- `global-stock-screener` — produces the `cache_seed/ltm` parquet panels this repo reads.

## Licence
No licence file yet — all rights reserved.
