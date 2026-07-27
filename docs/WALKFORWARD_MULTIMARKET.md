# Walk-forward next-day prediction — five-market comparison

Same protocol per market: top-20 liquid (≥8y history), last 500 trading days
out-of-sample, expanding-window refits every 125 days, frictionless
sign-long/short Sharpe vs buy & hold. Means across symbols.

| market | B&H Sharpe | best model (Sharpe) | markov | kalman | lstm | gru | best naive |
|---|---|---|---|---|---|---|---|
| US | 0.69 | none beats B&H | 0.63 | 0.17 | −0.06 | −0.01 | always-long 0.69 |
| JP | 0.60 | none | 0.57 | −0.16 | 0.35 | 0.38 | always-long 0.60 |
| KR | 0.54 | none | 0.30 | 0.25 | 0.08 | 0.15 | always-long 0.54 |
| CN | 0.43 | none | 0.32 | 0.28 | 0.22 | 0.32 | always-long 0.43 |
| IN | −0.13 | 1-day momentum +0.15 | −0.07 | −0.17 | −0.20 | +0.04 | momentum-1 +0.15 |

## Read

- **Buy & hold beats every model in every up market** (US/JP/KR/CN). Daily
  close-to-close point forecasts add nothing on liquid large caps — matching
  the accuracy-vs-reliability literature matrix.
- **Markov is the consistent runner-up** (2nd in US/JP, near-2nd KR/CN):
  its value is stepping aside in bear states, i.e. it is a regime gate, not a
  forecaster. This is the model the production scanner/mailer gate now uses.
- **India, the one down-tape, is where models finally beat B&H**: 1-day
  momentum (+0.15) and GRU (+0.04, 13/20 names) — shorting ability matters
  only when the market falls, consistent with IN's momentum character.
- **1-day momentum is best in IN and worst everywhere else** (−0.18 to −0.25)
  — per-market character (IN momentum vs US/JP/KR mean-reversion) shows up
  even at the daily horizon.
- Directional accuracy sits at 0.48–0.53 in every cell; profitable cells get
  there on magnitude, not hit rate.
