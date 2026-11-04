from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS = 252
DIVIDEND_FREQUENCIES = (1, 2, 4, 12)


def calculate_dividend_features(
    prices: pd.DataFrame,
    signal_prices: pd.DataFrame,
    dividends: pd.DataFrame,
    as_of_date: pd.Timestamp,
    ticker_to_sector: dict[str, str],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Dividend quality, trend, and risk features using only data up to as_of_date.

    Yield uses raw closes (adjusted closes would inflate historical yields);
    momentum and volatility use adjusted closes so they reflect total return.
    """
    config = config or {}
    cut_threshold = float(config.get("core", {}).get("cut_threshold", 0.15))
    raw_history = prices.loc[:as_of_date]
    signal_history = signal_prices.loc[:as_of_date].tail(TRADING_DAYS + 5)
    dividend_history = dividends.loc[as_of_date - pd.Timedelta(days=730) : as_of_date]
    one_year_ago = as_of_date - pd.Timedelta(days=365)
    rows = []

    for ticker, sector in ticker_to_sector.items():
        if ticker not in prices.columns:
            continue
        series = signal_history[ticker].dropna()
        if len(series) < 220:
            continue
        last_price = raw_history[ticker].dropna().iloc[-1]
        if last_price <= 0:
            continue

        payments = dividend_history[ticker][dividend_history[ticker] > 0]
        annual_dividend, dividend_growth, dividend_cut = _dividend_profile(payments, one_year_ago, cut_threshold)
        returns = series.pct_change().dropna()
        ma_200 = series.rolling(200).mean().iloc[-1]
        high_52w = series.tail(TRADING_DAYS).max()

        rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "price": float(last_price),
                "annual_dividend": annual_dividend,
                "dividend_yield": annual_dividend / last_price,
                "dividend_growth": dividend_growth,
                "dividend_cut": dividend_cut,
                "momentum_12m": series.iloc[-1] / series.iloc[0] - 1,
                "distance_ma_200": series.iloc[-1] / ma_200 - 1 if ma_200 > 0 else np.nan,
                "drawdown_52w": series.iloc[-1] / high_52w - 1 if high_52w > 0 else np.nan,
                "volatility": returns.tail(126).std() * np.sqrt(TRADING_DAYS),
            }
        )

    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


def score_dividend_features(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Rank stocks within their own sector so every industry gets its best dividend names."""
    if features.empty:
        return features
    core = config.get("core", {})
    weights = {
        "dividend_yield": 0.30,
        "dividend_growth": 0.25,
        "low_volatility": 0.15,
        "trend": 0.30,
        **core.get("score_weights", {}),
    }
    max_yield = float(core.get("max_yield", 0.10))

    scored = features.copy()
    by_sector = scored.groupby("sector")
    # Very high yields usually mean the market expects a cut, so they rank last instead of first.
    capped_yield = scored["dividend_yield"].where(scored["dividend_yield"] <= max_yield, 0.0)
    yield_rank = capped_yield.groupby(scored["sector"]).rank(pct=True)
    growth_rank = by_sector["dividend_growth"].rank(pct=True).fillna(0.5)
    low_vol_rank = by_sector["volatility"].rank(pct=True, ascending=False)
    momentum_rank = by_sector["momentum_12m"].rank(pct=True)
    above_trend = (scored["distance_ma_200"] > 0).astype(float)

    scored["score"] = (
        weights["dividend_yield"] * yield_rank
        + weights["dividend_growth"] * growth_rank
        + weights["low_volatility"] * low_vol_rank
        + weights["trend"] * (0.5 * momentum_rank + 0.5 * above_trend)
    )
    return scored.sort_values("score", ascending=False)


def _dividend_profile(
    payments: pd.Series, one_year_ago: pd.Timestamp, cut_threshold: float
) -> tuple[float, float, bool]:
    """Return (indicated annual dividend, year-over-year growth, cut flag) from the last two years of payments."""
    recent = payments[payments.index > one_year_ago]
    prior = payments[payments.index <= one_year_ago]
    if len(payments) < 2 or recent.empty:
        # Never paid, or stopped paying in the last year.
        return 0.0, np.nan, len(prior) > 0

    frequency = min(DIVIDEND_FREQUENCIES, key=lambda freq: abs(freq - len(payments) / 2))
    last = float(payments.iloc[-1])
    typical = float(payments.iloc[-5:-1].median())
    # Cap the last payment so a one-off special dividend does not inflate the yield.
    annual_dividend = min(last, 1.5 * typical) * frequency

    growth = float(recent.mean() / prior.mean() - 1) if len(prior) else np.nan
    is_cut = last < (1 - cut_threshold) * typical or (not np.isnan(growth) and growth < -cut_threshold)
    return annual_dividend, growth, bool(is_cut)
