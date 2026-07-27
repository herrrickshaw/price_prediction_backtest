# Accuracy vs. Reliability Matrix — price-prediction techniques

Literature synthesis (July 2026) cross-checked against this repo's own
walk-forward results. **Reported accuracy** = what papers typically claim for
next-day direction; **honest OOS accuracy** = what survives leak-free
walk-forward evaluation on returns (not prices); **reliability** = does the
result reproduce across assets, periods, and independent labs.

| Technique | Reported accuracy | Honest OOS accuracy | Reliability | Verdict |
|---|---|---|---|---|
| ARIMA / GARCH | 55–60% | ~50–52% direction; vol forecasts genuinely good | **High** | Useless for direction, excellent for volatility — GARCH vol is among the most replicated results in finance. Use for position sizing |
| Kalman filters | 55–65% | ~50% direction; strong trend denoising | **High** | Deterministic, no tuning lottery. Trend/spread *estimator*, not predictor. Our runs: +1.2–1.4 Sharpe on trending names (GOOGL, GLD), blowups on choppy ones (NVDA −0.97) |
| Markov / HMM regime models | 60–70% (regime ID) | ~53% direction — best in our tests | **High** | The honest winner. Identifies *states*, not prices; reproduces across markets. Best used to gate other signals |
| Tree ensembles (XGBoost/RF) on cross-sections | Monthly OOS R² ~0.3–0.4% | Small but real cross-sectional premium | **Med-High** | Academic gold standard (Gu–Kelly–Xiu 2020): nonlinear interactions beat linear models — but ranking stocks against each other monthly, never single-name daily prices |
| LSTM / GRU (single asset, daily) | 85–95%+ in papers | ~51% (ours: 50.9–51.0% US; GRU beat B&H 13/20 in a falling India tape) | **Low** | Largest accuracy-vs-reliability gap in the field. Headline numbers come from predicting price *levels* (autocorrelation illusion), leakage, or cherry-picked assets |
| Transformers / TFT | "beats LSTM" | Asset-dependent, not dominant | **Low-Med** | 2026 comparisons: ARIMA and Random Forest remain strong baselines; transformers competitive, not uniformly better on daily equities |
| TS foundation models (Chronos, TimesFM, Kronos) | Strong zero-shot on generic series | Beat naive on ~1 of 5 test equities | **Med** | Useful priors in low-data settings; "not universal engines of statistically reliable alpha" per 2026 benchmarks |
| Sentiment / LLM-augmented | +7.4% median next-day accuracy gain claimed | Decays fast beyond next-day | **Low** | Of 44 reviewed studies: 5 report Sharpe, 1 accounts for transaction costs. Evidence base is thin |

## The pattern

Reliability is roughly **inverse** to reported accuracy. Methods with modest,
boring claims (GARCH vol, regime detection, cross-sectional ranking)
replicate; methods claiming 90%+ direction evaporate under leak-free
evaluation. Our own harness reproduced this: sub-53% direction everywhere,
with value living in regime gating and market character, not point forecasts.
A related empirical note: several models earn positive Sharpe with <50%
directional accuracy — P&L comes from being right on the big days.

## Sources

- Gu, Kelly, Xiu — *Empirical Asset Pricing via Machine Learning* (RFS 2020), <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577>
- *A Systematic Literature Review for Stock Price Prediction with ML* (2018–2024 corpus, 44 studies), <https://www.researchgate.net/publication/394711257>
- *Comparative Study of Transformer-Based and Classical Models for Financial TS Forecasting* (JRFM 2026), <https://www.mdpi.com/1911-8074/19/3/203>
- *Pretrained Time-Series Foundation Models for Financial Return Forecasting* (2026), <https://arxiv.org/pdf/2606.27100>
- *Deep Learning for Stock Market Prediction: A Systematic Review* (2025), <https://www.researchgate.net/publication/403314395>
