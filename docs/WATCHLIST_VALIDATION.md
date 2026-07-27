# Watchlist recommendations vs. actual performance — small-sample validation

Source: `market-pipeline` `reports/signal_outcomes.parquet` — 18,599 unique
scored Darvas signals (5d and 21d fixed horizons, forward return vs. each
market's median tracked name = excess). Signal dates 2026-06-12 → 06-26
(**only 6 unique dates** — signals within a date share market moves, so
per-signal t-stats of 4–6 are heavily overstated; read directionally).

## By market × horizon

| market | h | n | mean excess | hit rate |
|---|---|---|---|---|
| US | 5d | 11,709 | **+0.93%** | 52.3% |
| US | 21d | 1,982 | **+2.20%** | 59.6% |
| IN | 5d | 6,807 | −0.50% | 43.4% |
| IN | 21d | 4,402 | −0.05% | 47.6% |

## By signal type

| detail | h | n | mean excess | hit |
|---|---|---|---|---|
| BREAKOUT_BUY | 5d | 13,862 | +0.41% | 48.7% |
| BREAKOUT_BUY | 21d | 4,938 | +0.76% | 51.7% |
| BREAKDOWN_SELL | 5d | 4,682 | **+0.39%** | 50.1% |
| BREAKDOWN_SELL | 21d | 1,474 | **+0.30%** | 50.1% |

## By volatility regime at signal date

| regime | h | n | mean excess | hit |
|---|---|---|---|---|
| HIGH | 5d | 10,162 | **+0.79%** | 51.1% |
| HIGH | 21d | 2,456 | **+1.94%** | 58.6% |
| MID | 5d | 8,382 | −0.05% | 46.5% |
| MID | 21d | 3,956 | −0.14% | 46.9% |

## Findings

1. **The buy leg works in the US, not in India (this window).** US signals:
   +2.2% excess at 21d, 60% hit. India signals lagged their market. Consistent
   with the per-market playbook (breakout/momentum reads differently by market)
   — though six signal dates cannot settle that.
2. **BREAKDOWN_SELL adds nothing**: names flagged as breakdowns went on to
   *outperform* slightly (+0.3–0.4% excess). As a short/avoid signal it was
   wrong this window.
3. **The composite score does not rank outcomes.** Excess return by score
   quintile is non-monotonic (Q5-high ≈ Q1-low). Binary signal > score magnitude.
4. **Regime is the strongest conditioner** — HIGH-vol-regime signals beat
   MID-regime signals by ~2pp at 21d (+1.94% vs −0.14%). This independently
   corroborates the walk-forward backtest conclusion that the Markov regime
   layer is the component worth keeping: gate the screener, don't forecast price.
5. Separate mailer paper-track (2,500 picks, Jul 13–22, `paper_track.md`)
   points the same way: curation (grade-A + fundamentals) improved excess
   ~0.2pp and dodged the KOSDAQ crash, while raw volume of picks lagged.

## Design for the full historical extension

"Screener recommends, price prediction validates":

- Replay the Darvas/Piotroski screeners over the full 10y panel (signals are
  reproducible from OHLCV — no need to wait for live accumulation).
- At each historical signal date, compute the Markov regime state and Kalman
  slope from data up to that date; test excess returns of signal ∩ regime
  cells vs. signal alone.
- Fix the small-sample defects seen here: cluster standard errors by signal
  date (or block-bootstrap dates), require ≥100 signal dates per cell, and
  evaluate India on the split-adjusted series only.
