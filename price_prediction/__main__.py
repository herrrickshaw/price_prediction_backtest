"""CLI: walk-forward evaluation of Markov / Kalman / LSTM / GRU predictors.

    python -m price_prediction --market US --top 20 --test-days 500
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .data import load_panel, top_liquid_symbols
from .evaluate import evaluate_universe

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports"
SUMMARY_COLS = ["dir_acc", "rmse", "strat_sharpe", "strat_ann_ret", "bh_sharpe"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--market", default="US")
    ap.add_argument("--top", type=int, default=20, help="most-liquid symbols to test")
    ap.add_argument("--test-days", type=int, default=500)
    ap.add_argument("--refit-every", type=int, default=125)
    ap.add_argument("--symbols", nargs="*", help="explicit symbol list (overrides --top)")
    ap.add_argument("--out", default=None,
                    help="CSV path (default reports/price_prediction_<market>.csv)")
    return ap


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean metric per model across symbols, best frictionless Sharpe first."""
    return (results.groupby("model")[SUMMARY_COLS].mean()
            .sort_values("strat_sharpe", ascending=False))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    panel = load_panel(args.market)
    symbols = args.symbols or top_liquid_symbols(panel, n=args.top)
    print(f"{args.market}: evaluating {len(symbols)} symbols, "
          f"{args.test_days} test days, refit every {args.refit_every}")

    results = evaluate_universe(panel, symbols, test_days=args.test_days,
                                refit_every=args.refit_every)

    out = Path(args.out or REPORT_DIR / f"price_prediction_{args.market.lower()}.csv").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)

    print("\n=== mean across symbols (walk-forward, no costs) ===")
    print(summarise(results).round(4).to_string())
    print(f"\nper-symbol rows -> {out}")


if __name__ == "__main__":
    main()
