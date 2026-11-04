from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from src.swing import swing_universe


def get_all_tickers(config: dict[str, Any], include_fx: bool = True) -> list[str]:
    tickers: set[str] = set()
    for sector_tickers in config["sectors"].values():
        tickers.update(sector_tickers)
    tickers.add(config.get("benchmark_ticker", "XIC.TO"))
    swing = config.get("swing", {})
    if swing.get("enabled", False):
        # Download the stock universe even when use_stocks is off so the optimizer can test both.
        tickers.update(swing_universe(config, include_all=True))
        if swing.get("market_filter_ticker"):
            tickers.add(swing["market_filter_ticker"])
    if include_fx and config.get("portfolio_currency", "CAD").upper() == "CAD":
        tickers.add(config.get("cad_per_usd_ticker", "CAD=X"))
    return sorted(tickers)


def download_price_data(
    config: dict[str, Any], save: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download prices, adjusted signal prices, and dividends from yfinance (save=False skips the data/ cache)."""
    Path("data").mkdir(exist_ok=True)
    cache_dir = Path("data/yfinance_cache")
    cache_dir.mkdir(exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))

    tickers = get_all_tickers(config)
    asset_tickers = get_all_tickers(config, include_fx=False)
    fx_ticker = config.get("cad_per_usd_ticker", "CAD=X")
    end_date = config.get("end_date") or datetime.today().strftime("%Y-%m-%d")

    # Extra history before start_date so moving averages and dividend history are ready on day one.
    warmup_years = int(config.get("warmup_years", 2))
    download_start = (pd.Timestamp(config["start_date"]) - pd.DateOffset(years=warmup_years)).strftime("%Y-%m-%d")

    data = download_with_retry(tickers, download_start, end_date)

    close_prices = _extract_field(data, "Close", tickers).dropna(how="all")
    signal_prices = _extract_field(data, "Adj Close", tickers).dropna(how="all")
    dividends = _extract_field(data, "Dividends", tickers).fillna(0.0)
    # The FX rate must come from the close prices: the dividends frame has 0 for CAD=X every day.
    cad_per_usd = close_prices[fx_ticker] if fx_ticker in close_prices.columns else None
    prices = _convert_to_portfolio_currency(close_prices, asset_tickers, cad_per_usd, fx_ticker, config)
    signal_prices = _convert_to_portfolio_currency(signal_prices, asset_tickers, cad_per_usd, fx_ticker, config)
    dividends = _convert_to_portfolio_currency(dividends, asset_tickers, cad_per_usd, fx_ticker, config)

    if prices.empty:
        raise RuntimeError(
            "No price data was downloaded. Check internet access, ticker symbols, "
            "or try again because Yahoo Finance can temporarily reject requests."
        )

    empty = [ticker for ticker in asset_tickers if prices[ticker].dropna().empty]
    if empty:
        print(f"Warning: no price data for {empty}. Check these ticker symbols.")

    if save:
        prices.to_csv("data/prices_cad.csv")
        signal_prices.to_csv("data/signal_prices_cad.csv")
        dividends.to_csv("data/dividends_cad.csv")
    return prices, dividends, signal_prices


def load_cached_price_data(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the latest CAD-converted data files if they already exist and cover the configured tickers."""
    paths = {
        "prices": Path("data/prices_cad.csv"),
        "dividends": Path("data/dividends_cad.csv"),
        "signals": Path("data/signal_prices_cad.csv"),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cached data files: {missing}")

    prices = pd.read_csv(paths["prices"], index_col=0, parse_dates=True)
    dividends = pd.read_csv(paths["dividends"], index_col=0, parse_dates=True)
    signal_prices = pd.read_csv(paths["signals"], index_col=0, parse_dates=True)

    if config is not None:
        missing_tickers = set(get_all_tickers(config, include_fx=False)) - set(prices.columns)
        if missing_tickers:
            raise FileNotFoundError(f"Cached data is missing tickers: {sorted(missing_tickers)}")
    return prices, dividends, signal_prices


def download_with_retry(
    tickers: list[str], start: str, end: str, attempts: int = 3, threads: bool = False
) -> pd.DataFrame:
    """Yahoo sometimes rejects part of a big request; re-download only the tickers that came back empty."""
    def fetch(batch: list[str]) -> pd.DataFrame:
        return yf.download(
            tickers=batch,
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
            progress=False,
            group_by="column",
            threads=threads,
            multi_level_index=True,
        )

    data = fetch(tickers)
    for attempt in range(1, attempts):
        close = data["Close"] if "Close" in data.columns.get_level_values(0) else pd.DataFrame()
        failed = [ticker for ticker in tickers if ticker not in close.columns or close[ticker].dropna().empty]
        if not failed:
            break
        print(f"Retrying {len(failed)} tickers that failed to download (attempt {attempt + 1}/{attempts})...")
        time.sleep(5 * attempt)
        retry = fetch(failed)
        if retry.empty:
            continue
        data = data.drop(columns=[c for c in data.columns if c[1] in failed])
        data = pd.concat([data, retry], axis=1).sort_index()
    return data


def is_usd_listed(ticker: str) -> bool:
    # This project treats Toronto-listed tickers as CAD and common US listings as USD.
    return not ticker.endswith(".TO")


def _extract_field(data: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        if field not in data.columns.get_level_values(0):
            return pd.DataFrame(index=data.index, columns=tickers, dtype=float)
        frame = data[field].copy()
    else:
        frame = data[[field]].copy() if field in data else pd.DataFrame(index=data.index)

    if len(tickers) == 1 and tickers[0] not in frame.columns:
        frame.columns = tickers

    return frame.reindex(columns=tickers).sort_index()


def _convert_to_portfolio_currency(
    frame: pd.DataFrame,
    asset_tickers: list[str],
    cad_per_usd: pd.Series | None,
    fx_ticker: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Convert USD-listed securities into CAD using CAD per USD from yfinance."""
    converted = frame.reindex(columns=asset_tickers).copy()
    if config.get("portfolio_currency", "CAD").upper() != "CAD":
        return converted

    if cad_per_usd is None or cad_per_usd.dropna().empty:
        raise RuntimeError(f"Missing FX ticker {fx_ticker}; cannot convert USD assets to CAD.")

    cad_per_usd = cad_per_usd.reindex(frame.index).ffill().bfill()
    for ticker in asset_tickers:
        if is_usd_listed(ticker):
            converted[ticker] = converted[ticker] * cad_per_usd
    return converted
