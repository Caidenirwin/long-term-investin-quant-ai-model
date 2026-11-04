"""Test the strategy on companies it was never tuned on.

If the swing and dividend rules only worked because they were fitted to the original ticker lists,
they should fall apart here. Same settings, different companies.

    python test_new_companies.py                          # built-in holdout lists below
    python test_new_companies.py --swing-tickers V MA COST NKE SHOP.TO
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest import run_backtest
from src.config import load_config, research_config
from src.data import download_price_data
from src.reporting import performance_metrics


# Large US and Canadian companies in 2012 that are NOT in the swing stock_universe.
# Companies that have since disappeared (acquired, bankrupt) can't be downloaded, which slightly flatters results.
HOLDOUT_SWING_STOCKS = [
    "CMCSA", "OXY", "COP", "MO", "AMGN", "BMY", "USB", "GS", "MMM", "CAT", "BA", "HON",
    "HPQ", "V", "MA", "NKE", "LOW", "TGT", "CVS", "FCX", "F", "GM", "COST", "SBUX",
    "AXP", "MDT", "LMT", "UNP", "EMR", "TXN", "ADP", "BLK", "MET", "EBAY", "GILD", "LLY",
    "TRI.TO", "MG.TO", "GIB-A.TO", "ATD.TO", "WN.TO", "SAP.TO", "CCO.TO", "K.TO",
    "AEM.TO", "FM.TO", "NA.TO", "IFC.TO", "BN.TO", "TOU.TO", "CTC-A.TO",
]

# A different dividend universe across the same 8 industries, with no overlap with config/settings.toml.
HOLDOUT_DIVIDEND_SECTORS = {
    "Financials": ["NA.TO", "IFC.TO", "GWO.TO", "IGM.TO", "USB", "MET"],
    "Energy / Pipelines": ["TOU.TO", "KEY.TO", "XOM", "COP"],
    "Utilities": ["ACO-X.TO", "CPX.TO", "DUK", "SO", "NEE"],
    "Telecom": ["T", "CMCSA", "QBR-B.TO"],
    "Consumer Staples": ["MO", "PM", "CL", "KMB", "SAP.TO"],
    "Industrials / Transport": ["MMM", "HON", "UNP", "EMR", "ADP", "LMT"],
    "Healthcare": ["AMGN", "BMY", "ABT", "MDT", "GILD"],
    "Real Estate": ["AP-UN.TO", "HR-UN.TO", "SRU-UN.TO", "O", "SPG"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the strategy on companies it was never tuned on.")
    parser.add_argument("--swing-tickers", nargs="+", help="Swing-test these tickers instead of the built-in holdout list.")
    args = parser.parse_args()

    base = research_config(load_config(Path("config/settings.toml")))
    holdout_swing = args.swing_tickers or HOLDOUT_SWING_STOCKS
    original_swing = list(base["swing"]["stock_universe"])
    overlap = sorted(set(holdout_swing) & set(original_swing))
    if overlap:
        print(f"Note: these are already in the tuned stock_universe: {overlap}")

    # One download covering every list (not saved, so the main data/ cache is untouched).
    download_config = copy.deepcopy(base)
    download_config["swing"]["stock_universe"] = sorted(set(original_swing) | set(holdout_swing))
    for sector, tickers in HOLDOUT_DIVIDEND_SECTORS.items():
        download_config["sectors"][sector] = sorted(set(base["sectors"][sector]) | set(tickers))
    print("Downloading data for original and new companies...")
    data = download_price_data(download_config, save=False)
    available = {ticker for ticker in data[0].columns if data[0][ticker].notna().any()}

    oos_start = pd.Timestamp(base.get("out_of_sample_start", "2021-01-01"))
    print("\n" + "=" * 100)
    print("SWING TEST: same tuned breakout rules, stocks only (no ETFs), 2012-today")
    print("=" * 100)
    swing_rows = [
        _swing_test("Original stocks (used for tuning)", base, original_swing, data, oos_start),
        _swing_test("NEW stocks (never seen)", base, [t for t in holdout_swing if t in available], data, oos_start),
    ]
    new_trades = swing_rows[1].pop("_trades")
    swing_rows[0].pop("_trades")
    print(pd.DataFrame(swing_rows).to_string(index=False))

    if not new_trades.empty:
        by_ticker = new_trades.groupby("ticker").agg(
            trades=("pnl", "size"), win_rate=("pnl", lambda x: (x > 0).mean()), avg_return=("return", "mean"), pnl=("pnl", "sum")
        ).sort_values("pnl", ascending=False)
        print("\nNew stocks, best and worst contributors:")
        print(pd.concat([by_ticker.head(5), by_ticker.tail(5)]).round(3).to_string())
        print(f"\n{(by_ticker['pnl'] > 0).mean():.0%} of new stocks were net profitable for the swing rules.")

    print("\n" + "=" * 100)
    print("DIVIDEND CORE TEST: same scoring rules, completely different dividend stocks (no swing sleeve)")
    print("=" * 100)
    holdout_sectors = {sector: [t for t in tickers if t in available] for sector, tickers in HOLDOUT_DIVIDEND_SECTORS.items()}
    core_rows = [
        _core_test("Original dividend list", base, base["sectors"], data, oos_start),
        _core_test("NEW dividend list (never seen)", base, holdout_sectors, data, oos_start),
    ]
    print(pd.DataFrame(core_rows).to_string(index=False))

    benchmark = base.get("benchmark_ticker", "XIC.TO")
    signals = data[2]
    if benchmark in signals.columns:
        stats = performance_metrics(signals.loc[base["start_date"]:, benchmark].dropna())
        print(f"\nBenchmark {benchmark}: CAGR {stats['CAGR']:.2%}, max drawdown {stats['max_drawdown']:.2%}")

    pd.DataFrame(swing_rows + core_rows).to_csv("outputs/new_companies_test.csv", index=False)
    print("\nSaved outputs/new_companies_test.csv")


def _swing_test(
    label: str, base: dict[str, Any], stocks: list[str], data: tuple, oos_start: pd.Timestamp
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["swing"].update(universe=[], stock_universe=stocks, use_stocks=True, enabled=True)
    results = run_backtest(config, *data)
    curve = results.equity_curve.set_index("date")["swing_contribution"]
    capital = config["starting_cash"] * float(config["swing"]["capital_fraction"])
    in_sample = performance_metrics(curve.loc[: oos_start - pd.Timedelta(days=1)], 0.0, capital)
    out_of_sample = performance_metrics(curve.loc[curve.index[curve.index < oos_start][-1] :])
    trades = results.swing_trades
    summary = results.performance_summary.iloc[0]
    return {
        "test": label,
        "stocks": len(stocks),
        "trades": len(trades),
        "win_rate": f"{(trades['pnl'] > 0).mean():.1%}" if len(trades) else "n/a",
        "avg_trade": f"{trades['return'].mean():+.2%}" if len(trades) else "n/a",
        "swing_pnl": f"C${trades['pnl'].sum():,.0f}" if len(trades) else "C$0",
        "swing_CAGR_2012-20": f"{in_sample['CAGR']:.2%}",
        "swing_CAGR_2021+": f"{out_of_sample['CAGR']:.2%}",
        "swing_max_DD": f"{min(in_sample['max_drawdown'], out_of_sample['max_drawdown']):.2%}",
        "portfolio_CAGR": f"{summary['CAGR']:.2%}",
        "_trades": trades,
    }


def _core_test(
    label: str, base: dict[str, Any], sectors: dict[str, list[str]], data: tuple, oos_start: pd.Timestamp
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["sectors"] = sectors
    config["swing"]["enabled"] = False
    results = run_backtest(config, *data)
    curve = results.equity_curve.set_index("date")["equity"]
    out_of_sample = performance_metrics(curve.loc[curve.index[curve.index < oos_start][-1] :])
    summary = results.performance_summary.iloc[0]
    return {
        "test": label,
        "stocks": sum(len(tickers) for tickers in sectors.values()),
        "CAGR": f"{summary['CAGR']:.2%}",
        "CAGR_2021+": f"{out_of_sample['CAGR']:.2%}",
        "max_DD": f"{summary['max_drawdown']:.2%}",
        "dividends_received": f"C${summary['cumulative_dividends']:,.0f}",
        "forward_income": f"C${summary['forward_annual_dividends']:,.0f}/yr",
        "core_trades": int((results.trades["sleeve"] == "core").sum()) if not results.trades.empty else 0,
    }


if __name__ == "__main__":
    main()
