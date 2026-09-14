# price_prediction_backtest

Walk-forward backtest of three sequential price-prediction families on deep
10-year OHLCV panels, scored honestly against naive baselines:

- **Markov regime chain** — bear/chop/bull states from rolling trend vs. vol,
  first-order transition matrix, E[next-day return] read-out
- **Kalman trend filter** — local-linear-trend state space on log price;
  filtered slope = next-day forecast, fully online/causal
- **LSTM / GRU** — PyTorch nets on 42-day feature windows (return, vol,
  range, volume-z, momentum), scaling fit on train only

Full model/protocol details: [price_prediction/README.md](price_prediction/README.md).

## Headline result (US, top-20 liquid, 500 OOS days ending 2026-07)

| model | dir. accuracy | long/short Sharpe (no costs) | buy & hold Sharpe |
|---|---|---|---|
| always-long baseline | 0.528 | 0.69 | 0.69 |
| markov | **0.532** | 0.63 | 0.69 |
| kalman | 0.503 | 0.17 | 0.69 |
| gru | 0.509 | −0.01 | 0.69 |
| lstm | 0.510 | −0.06 | 0.69 |

No model beats buy & hold before costs on daily US large-caps. The Markov
regime layer is the only competitive piece — useful as a regime gate for other
signals, not as a price forecaster. Per-symbol rows in `reports/`.

## Run

```bash
pip install pandas numpy pyarrow torch
python -m price_prediction --market US --top 20 --test-days 500
```

Expects `LTM_DIR` (default `~/repos/global-stock-screener/cache_seed/ltm`)
containing `<MARKET>.parquet` long panels: Date, Symbol, OHLCV.

Not investment advice; research code on historical data, frictionless P&L.
