import argparse
from pathlib import Path

from src.backtest import run_backtest
from src.config import load_config
from src.data import download_price_data, load_cached_price_data
from src.reporting import print_industry_breakdown, print_summary, save_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the swing + dividend core backtest.")
    parser.add_argument("--cached", action="store_true", help="Use data/ CSVs instead of downloading fresh data.")
    args = parser.parse_args()

    config = load_config(Path("config/settings.toml"))

    if args.cached:
        try:
            prices, dividends, signal_prices = load_cached_price_data(config)
        except FileNotFoundError:
            prices, dividends, signal_prices = download_price_data(config)
    else:
        prices, dividends, signal_prices = download_price_data(config)

    results = run_backtest(config, prices, dividends, signal_prices)
    save_outputs(results, Path("outputs"))

    print("Backtest complete. CSV files saved to outputs/.\n")
    print_summary(results.performance_summary)
    print_industry_breakdown(results.positions)


if __name__ == "__main__":
    main()
