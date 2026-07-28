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

<!-- 
DATA LIBRARY LINK - Add this section to every repo README.md
This snippet provides discovery and documentation links.
-->

## 📊 Data Discovery

This repository is part of the **Global Data Library** — a unified catalog of 10,528 datasets across 40+ repositories.

### Quick Links

- **[Global Data Library README](.ruflo/DATA_LIBRARY_README.md)** — Full catalog, search API, and usage examples
- **[Data Library Python Interface](.ruflo/data-library/data_library.py)** — Query datasets programmatically
- **[Repository Scanner](.ruflo/data-library/repo_scanner.py)** — Reindex all repos to update the catalog

### Datasets in This Repository

The data catalog automatically inventories all datasets in this repo. To find your data:

```python
from data_library import DataLibrary

lib = DataLibrary()

# Search this repo's datasets
results = lib.search("", source="<repo-name>")

# Get dataset details
dataset = lib.get("<dataset_id>")
print(f"Rows: {dataset['row_count']}")
print(f"Freshness: {dataset['freshness_hours']} hours old")
print(f"Storage: {dataset['storage_tier']}")
```

### Browse the Full Catalog

**Market Coverage** (5 markets, 21,279 symbols):
- India (NSE/BSE): 2,364 instruments
- US (NASDAQ/NYSE): 7,442 instruments
- Europe (17 exchanges): 1,214 instruments
- Japan (TSE): 3,709 instruments
- Korea (KRX): 2,768 instruments

**Government Sources** (30+ ministries):
- MOSPI: 25 datasets (GDP, CPI, trade, agri, power)
- SEBI: 151,928 XBRL results + IPO pipeline
- PIB: 25+ ministry announcements
- DGFT: India trade data (monthly)
- Agmarknet: 300+ mandi prices (daily)
- NSE/MCX: Real-time derivatives chains

See [Global Data Library README](.ruflo/DATA_LIBRARY_README.md) for complete documentation.

### Finding Data Across All Repos

```python
# Find India OHLCV data (might be in multiple repos)
lib.search("india ohlcv", market="india")

# Get the fastest/freshest version
optimal = lib.get_optimal("india ohlcv", latency="<100ms", freshness="<1day")
# Returns: {"storage_tier": "cassandra", "path": "..."}

# Check data gaps
gaps = lib.gaps("india", date_from="2026-01-01")

# See which collectors are stale
status = lib.collectors_status()
```

---
