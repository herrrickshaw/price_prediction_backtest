# Full historical replay — screener recommends, regime gates, outcomes judge

`price_prediction/replay.py` replays Darvas-style signals over the deep 10-year
panels (2016→2026) for **US, IN, JP, KR, CN** — the markets with full-depth
data (EU/UK/HK panels hold only ~1 year and are excluded). Universe = top-1000
by median dollar volume per market; regimes labeled point-in-time from a
trimmed top-20 blue-chip basket index (trend: EMA100 position+slope; vol:
expanding percentile of rolling 21d vol). Stats are **clustered by signal
date** (per-date means first, then t across dates); cells under 100 dates
dropped. ~470k signals scored.

## Headline: BUY × trend regime, 21-day excess vs market median (t clustered)

| market | bull | bear | bull−bear spread |
|---|---|---|---|
| KR | **+3.54%** (t 15.9) | +1.56% (t 3.3) | +2.0pp |
| CN | **+2.77%** (t 14.1) | +1.62% (t 4.5) | +1.1pp |
| IN | **+2.08%** (t 17.6) | +0.57% (t 1.4) | +1.5pp |
| JP | +0.80% (t 8.5) | −0.11% (t −0.4) | +0.9pp |
| US | +0.66% (t 5.7) | **−0.85% (t −1.9)** | +1.5pp |

**The regime gate works everywhere.** Breakout-buys earn positive excess only
(US, JP) or mostly (IN) in bull regimes; bear-regime breakouts are dead in
US/JP/IN. Ungated, the same signal averages ~half the bull-gated excess.

## BREAKDOWN_SELL is contrarian, not a short signal

Breakdown names *outperform* peers over the next 21d in bear regimes in every
market (US +1.29, JP +1.59, KR +1.63, CN +1.05, IN +0.96) and are ~flat in
bull regimes. The live small-sample finding (sell leg inverted) is confirmed
at scale: crashed names bounce relative to the median name, especially when
the whole market is falling. Do not short breakdowns; if anything they are a
mean-reversion *watch* list.

## Vol regime: LOW/MID vol is breakout-friendly; HIGH vol is not

LOW-vol breakout excess: KR +4.67 (t 8.2), CN +2.71, IN +2.25, JP +0.92,
US +0.50 — beats HIGH-vol cells in four of five markets (US: MID best).
**This contradicts the 6-date live sample** (which favoured HIGH-vol
signals): that sample sat inside one specific high-vol stretch and
generalises poorly — exactly why this replay was built.

## Cross-check vs the stored market playbook (MARKET_PLAYBOOK.md)

| playbook claim | replay verdict |
|---|---|
| IN momentum primary (long-only) | ✅ IN breakout×bull +2.08 t17.6 — strongest t in the study |
| KR breakout secondary (DSR 0.99) | ✅ KR breakout×bull +3.54 — largest excess |
| US momentum "fragile — light" | ✅ works only bull-gated, modest +0.66 |
| JP avoid momentum (whipsaws) | ◑ ungated JP breakout is weak (+0.62); bull-gated +0.80 t8.5 is real but the smallest ex-US |
| CN momentum "untested, character-implied" | ✅ now tested: +2.77 bull-gated — real at 21d |
| "~50% hit rate is normal; profit is magnitude" | ✅ every profitable cell sits at 45–55% hit |
| "never short a trending market" | ✅ sell leg fails everywhere; worst idea in bear regimes |

## Caveats

- Liquidity ranking uses full-sample dollar volume (mild lookahead in
  universe membership, not in signals or regimes).
- Overlapping 21d windows within a regime stretch: date-clustered t's are
  still somewhat optimistic (~√21 worst-case); the cross-market consistency,
  not any single t, is the evidence.
- India panel carries unadjusted corporate-action risk on some names
  (single-day |move|>60% masked; 1:1 bonuses at −50% survive the mask).
- Excess is vs the liquid-universe median name, frictionless; per the
  playbook, IN/CN round-trip costs are the worst.

## Reproduce

```bash
python -m price_prediction.replay --market US IN JP KR CN
```

Per-signal parquets: `reports/replay_signals_<mkt>.parquet`; full console
tables: `reports/replay_full_output.txt`.
