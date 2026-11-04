"""Market data and the features the AI learns from.

Every feature uses only information available on its date (rolling windows look backward), so the model
can't peek at the future. Labels look forward: they're what the model is trained to predict.
"""
from __future__ import annotations

import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import download_with_retry, is_usd_listed
from src.swing import rsi_frame, swing_universe
from src.universe import ETF_INFO, load_index_constituents

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FX_TICKER = "CAD=X"
MARKET_TICKERS = ["XIC.TO", "SPY"]


def load_ai_config() -> dict[str, Any]:
    with (ROOT / "config.toml").open("rb") as file:
        return tomllib.load(file)


def universe() -> tuple[list[str], dict[str, str], dict[str, str]]:
    """(tickers, ticker -> industry, ticker -> company name): index members plus everything in settings.toml."""
    ai = load_ai_config()
    settings = load_config(Path("config/settings.toml"))
    members = load_index_constituents(ai["data"]["indexes"])
    sectors = dict(zip(members["ticker"], members["sector"]))
    names = dict(zip(members["ticker"], members["name"]))
    for ticker, (name, sector) in ETF_INFO.items():
        sectors.setdefault(ticker, sector)
        names.setdefault(ticker, name)
    # Your dividend stocks keep the industry you gave them in settings.toml, in the standard (GICS) names.
    to_standard = {"Energy / Pipelines": "Energy", "Healthcare": "Health Care",
                   "Industrials / Transport": "Industrials", "Telecom": "Communication Services"}
    for sector, tickers in settings["sectors"].items():
        for ticker in tickers:
            sectors.setdefault(ticker, to_standard.get(sector, sector))
    own = [t for tickers in settings["sectors"].values() for t in tickers] + swing_universe(settings, include_all=True)
    for ticker in own:
        sectors.setdefault(ticker, "Other / Unknown")
        names.setdefault(ticker, ticker)
    tickers = sorted(set(members["ticker"]) | set(own) | set(MARKET_TICKERS))
    return tickers, sectors, names


GROUP_NAMES = {
    "sp500": "US large (S&P 500)", "sp400": "US mid (S&P 400)", "sp600": "US small (S&P 600)",
    "tsx60": "Canada large (TSX 60)", "tsx_composite": "Canada mid/small (TSX Composite)",
}


def index_groups() -> dict[str, str]:
    """ticker -> which kind of company it is (by the index it belongs to), for the per-group test results."""
    members = load_index_constituents(load_ai_config()["data"]["indexes"])
    return {t: GROUP_NAMES.get(i, i) for t, i in zip(members["ticker"], members["index"])}


def load_market_data(refresh: bool = False, start: str | None = None) -> dict[str, pd.DataFrame]:
    """Adjusted prices (for returns), raw closes (for dividend yields) and dividends, all in CAD.

    Cached in ai_investor/data/ so training doesn't re-download every time; re-downloaded automatically when
    the stock list grows. start overrides the config (the daily advisor only needs the last few years).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / ("market_recent.pkl" if start else "market_history.pkl")
    tickers, _, _ = universe()
    if cache.exists() and not refresh:
        data = pd.read_pickle(cache)
        covers_universe = not set(tickers) - set(data.get("requested", data["adjusted"].columns))
        if covers_universe and (start is None or data["fetched"] == datetime.now().strftime("%Y-%m-%d")):
            return data
    start = start or load_ai_config()["data"]["start_date"]
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Downloading {len(tickers)} tickers from {start} (this takes a few minutes the first time)...")
    raw = download_with_retry(sorted(set(tickers) | {FX_TICKER}), start, end, threads=True)

    def field(name: str) -> pd.DataFrame:
        return raw[name].reindex(columns=sorted(set(tickers) | {FX_TICKER})).sort_index()

    close, adjusted, dividends = field("Close"), field("Adj Close"), field("Dividends").fillna(0.0)
    cad_per_usd = close[FX_TICKER].ffill().bfill()
    usd = [t for t in tickers if is_usd_listed(t)]
    for frame in (close, adjusted, dividends):
        frame[usd] = frame[usd].mul(cad_per_usd, axis=0)
    keep = [t for t in tickers if close[t].notna().sum() > 300]
    data = {
        # float32 halves memory: plenty of precision for prices, and ~1,700 stocks x 18 years adds up.
        "adjusted": adjusted[keep].dropna(how="all").ffill(limit=5).astype("float32"),
        "close": close[keep].dropna(how="all").ffill(limit=5).astype("float32"),
        "dividends": dividends[keep].astype("float32"),
        "fetched": datetime.now().strftime("%Y-%m-%d"),
        "requested": tickers,
    }
    data["dividends"] = data["dividends"].reindex(data["close"].index).fillna(0.0)
    pd.to_pickle(data, cache)
    print(f"Saved {len(keep)} tickers to {cache}")
    return data


# ---------------------------------------------------------------------------- features

FEATURES = [
    # momentum and trend
    "ret_5", "ret_21", "ret_63", "ret_126", "ret_252", "mom_12_1",
    "dist_ma50", "dist_ma200", "ma50_vs_ma200", "drawdown_252", "breakout_20", "breakout_55",
    # short-term overbought/oversold
    "rsi_3", "rsi_14",
    # risk
    "vol_21", "vol_63", "vol_ratio", "beta_252",
    # dividends
    "div_yield", "div_growth", "pays_dividend",
    # the same, ranked against all stocks that week (0 = lowest, 1 = highest)
    "rank_ret_21", "rank_ret_126", "rank_mom_12_1", "rank_vol_63", "rank_dist_ma200", "rank_div_yield",
    # market conditions (identical for every stock on a date)
    "mkt_ret_63", "mkt_dist_ma200", "mkt_vol_21",
    # industry (categorical)
    "sector_code",
]
CATEGORICAL = ["sector_code"]


def _wide_features(adjusted: pd.DataFrame, close: pd.DataFrame, dividends: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every feature as a dates x tickers table."""
    a = adjusted
    returns = a.pct_change(fill_method=None)
    ma50, ma200 = a.rolling(50, min_periods=40).mean(), a.rolling(200, min_periods=150).mean()
    high252 = a.rolling(252, min_periods=200).max()
    vol21 = returns.rolling(21, min_periods=15).std() * np.sqrt(252)
    vol63 = returns.rolling(63, min_periods=45).std() * np.sqrt(252)

    market = a[MARKET_TICKERS[0]] if MARKET_TICKERS[0] in a else a.mean(axis=1)
    market_returns = market.pct_change(fill_method=None)
    # Rolling beta = cov(stock, market) / var(market), from rolling means (fast for hundreds of stocks).
    window = dict(window=252, min_periods=200)
    mean_xy = returns.mul(market_returns, axis=0).rolling(**window).mean()
    mean_x = returns.rolling(**window).mean()
    mean_y = market_returns.rolling(**window).mean()
    variance = (market_returns ** 2).rolling(**window).mean() - mean_y ** 2
    beta = (mean_xy - mean_x.mul(mean_y, axis=0)).div(variance, axis=0)

    # Trailing-12-month dividends vs today's raw price; growth vs the 12 months before.
    ttm = dividends.rolling(252, min_periods=1).sum()
    div_yield = (ttm / close).where(close > 0).clip(upper=0.25)
    previous_ttm = ttm.shift(252)
    div_growth = (ttm / previous_ttm - 1).where(previous_ttm > 0).clip(-1, 2)

    features = {
        "ret_5": a / a.shift(5) - 1,
        "ret_21": a / a.shift(21) - 1,
        "ret_63": a / a.shift(63) - 1,
        "ret_126": a / a.shift(126) - 1,
        "ret_252": a / a.shift(252) - 1,
        "mom_12_1": a.shift(21) / a.shift(252) - 1,  # classic momentum: last year excluding the last month
        "dist_ma50": a / ma50 - 1,
        "dist_ma200": a / ma200 - 1,
        "ma50_vs_ma200": ma50 / ma200 - 1,
        "drawdown_252": a / high252 - 1,
        "breakout_20": a / a.rolling(20).max().shift(1) - 1,
        "breakout_55": a / a.rolling(55).max().shift(1) - 1,
        "rsi_3": rsi_frame(a, 3),
        "rsi_14": rsi_frame(a, 14),
        "vol_21": vol21,
        "vol_63": vol63,
        "vol_ratio": vol21 / vol63,
        "beta_252": beta,
        "div_yield": div_yield.fillna(0.0),
        "div_growth": div_growth,
        "pays_dividend": (ttm > 0).astype(float),
    }
    for name, source in [("rank_ret_21", "ret_21"), ("rank_ret_126", "ret_126"), ("rank_mom_12_1", "mom_12_1"),
                         ("rank_vol_63", "vol_63"), ("rank_dist_ma200", "dist_ma200"), ("rank_div_yield", "div_yield")]:
        features[name] = features[source].rank(axis=1, pct=True)

    return features


def _market_features(adjusted: pd.DataFrame) -> pd.DataFrame:
    """Market conditions per date (the same for every stock), joined onto each row by date."""
    market = adjusted[MARKET_TICKERS[0]] if MARKET_TICKERS[0] in adjusted else adjusted.mean(axis=1)
    returns = market.pct_change(fill_method=None)
    return pd.DataFrame({
        "mkt_ret_63": market / market.shift(63) - 1,
        "mkt_dist_ma200": market / market.rolling(200, min_periods=150).mean() - 1,
        "mkt_vol_21": returns.rolling(21).std() * np.sqrt(252),
    })


def build_dataset(data: dict[str, pd.DataFrame], sample_every: int, horizons: dict[str, int],
                  sectors: dict[str, str], dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """One row per (date, ticker): features plus forward-looking labels for each horizon.

    Labels are the cross-sectional percentile of the forward return (0 = worst stock that period, 1 = best),
    which is what matters for picking stocks and is robust to market-wide swings.
    """
    adjusted, close, dividends = data["adjusted"], data["close"], data["dividends"]
    tradable = [t for t in adjusted.columns if t not in MARKET_TICKERS or t in sectors]
    wide = _wide_features(adjusted, close, dividends)

    if dates is None:
        dates = adjusted.index[260::sample_every]  # skip the first year: features need history
    sector_names = sorted(set(sectors.values()) | {"Other / Unknown"})
    sector_code = {t: sector_names.index(sectors.get(t, "Other / Unknown")) for t in tradable}

    frames = []
    for name, frame in wide.items():
        frames.append(frame.loc[dates, tradable].stack(future_stack=True).rename(name))
    for name, horizon in horizons.items():
        forward = adjusted.shift(-horizon) / adjusted - 1
        frames.append(forward.loc[dates, tradable].stack(future_stack=True).rename(f"fwd_{name}"))
        frames.append(forward.rank(axis=1, pct=True).loc[dates, tradable].stack(future_stack=True).rename(f"label_{name}"))
    table = pd.concat(frames, axis=1)
    table.index.names = ["date", "ticker"]
    table = table.reset_index()
    table = table.join(_market_features(adjusted).loc[dates], on="date")
    table["sector_code"] = table["ticker"].map(sector_code).astype(int)
    table["sector"] = table["ticker"].map(lambda t: sectors.get(t, "Other / Unknown"))
    # A row needs its core history to be usable.
    return table.dropna(subset=["ret_252", "dist_ma200", "vol_63"]).reset_index(drop=True)


def sector_names_list(sectors: dict[str, str]) -> list[str]:
    return sorted(set(sectors.values()) | {"Other / Unknown"})
