"""Swing-trading setups shared by the backtest and the live scanner (scan.py).

Keeping the rules in one place guarantees the scanner shows exactly what the backtest trades.

Setups:
- pullback: long-term uptrend + sharp short-term dip (low RSI). Exit on the bounce above a short MA.
- breakout: new N-day closing high while the 50-day MA is above the 200-day MA. Exit on a trailing N-day low.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def swing_universe(config: dict[str, Any], include_all: bool = False) -> list[str]:
    """ETF universe, plus the large-cap stock universe when swing.use_stocks is on (or include_all for downloads)."""
    swing = config.get("swing", {})
    tickers = list(swing.get("universe", []))
    if include_all or swing.get("use_stocks", False):
        tickers += [ticker for ticker in swing.get("stock_universe", []) if ticker not in tickers]
    return tickers


def swing_slots(swing_value: float, swing: dict[str, Any]) -> int:
    """How many positions the sleeve can hold: max_positions, fewer while it's too small for each to reach
    min_trade_value (a C$250 sleeve can't sensibly hold three positions)."""
    max_positions = int(swing.get("max_positions", 3))
    min_trade = float(swing.get("min_trade_value", 500))
    if min_trade <= 0:
        return max_positions
    return max(0, min(max_positions, int(swing_value // min_trade)))


def swing_indicators(signal_prices: pd.DataFrame, config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Precompute every swing indicator for all tickers at once (rows = dates, columns = tickers)."""
    swing = config.get("swing", {})
    trend_days = int(swing.get("trend_ma_days", 200))
    fast_days = int(swing.get("fast_ma_days", 50))
    breakout_days = int(swing.get("breakout_days", 20))
    trail_days = int(swing.get("trail_days", 10))

    def rolling_mean(days: int) -> np.ndarray:
        return signal_prices.rolling(days, min_periods=days).mean().to_numpy(float)

    return {
        "price": signal_prices.to_numpy(float),
        "trend_ma": rolling_mean(trend_days),
        "fast_ma": rolling_mean(fast_days),
        "exit_ma": rolling_mean(int(swing.get("exit_ma_days", 10))),
        "rsi": rsi_frame(signal_prices, int(swing.get("rsi_days", 3))).to_numpy(float),
        # Highest/lowest close of the previous N days, not including today.
        "breakout_high": signal_prices.rolling(breakout_days, min_periods=breakout_days).max().shift(1).to_numpy(float),
        "trail_low": signal_prices.rolling(trail_days, min_periods=trail_days).min().shift(1).to_numpy(float),
        "momentum": (signal_prices / signal_prices.shift(63) - 1).to_numpy(float),
    }


def rsi_frame(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Wilder RSI for every column."""
    changes = prices.diff()
    gains = changes.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    losses = (-changes.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rsi = 100 - 100 / (1 + gains / losses)
    return rsi.where(losses > 0, 100.0).where(gains.notna())


def entry_setup(ind: dict[str, np.ndarray], k: int, j: int, swing: dict[str, Any]) -> tuple[str, float] | None:
    """Return (setup, priority) if ticker column j has an entry signal on row k. Higher priority is bought first."""
    price = ind["price"][k, j]
    if not price > ind["trend_ma"][k, j]:
        return None
    setups = swing.get("setups", ["pullback"])

    entry_rsi = float(swing.get("entry_rsi", 10))
    rsi = ind["rsi"][k, j]
    if "pullback" in setups and rsi < entry_rsi:
        # Pullbacks rank ahead of breakouts (priority 1-2 vs 0-1); deeper dips rank first.
        return "pullback", 1 + (entry_rsi - rsi) / entry_rsi

    if "breakout" in setups and price > ind["breakout_high"][k, j] and ind["fast_ma"][k, j] > ind["trend_ma"][k, j]:
        return "breakout", float(np.clip(ind["momentum"][k, j], 0, 0.99))
    return None


def entry_signals(ind: dict[str, np.ndarray], k: int, columns: list[int], swing: dict[str, Any]) -> list[tuple[int, str, float]]:
    """All (column, setup, priority) entry signals on row k, best first."""
    signals = []
    for j in columns:
        found = entry_setup(ind, k, j, swing)
        if found:
            signals.append((j, *found))
    signals.sort(key=lambda item: -item[2])
    return signals


def market_allows_entries(ind: dict[str, np.ndarray], k: int, market_col: int | None) -> bool:
    return market_col is None or bool(ind["price"][k, market_col] > ind["trend_ma"][k, market_col])


def exit_reason(
    ind: dict[str, np.ndarray],
    k: int,
    j: int,
    setup: str,
    entry_price: float,
    reference_price: float,
    held_days: int,
    swing: dict[str, Any],
) -> str | None:
    """Why a position should be sold on row k, or None to keep holding.

    reference_price and entry_price must be in the same units (both CAD in the backtest, both native in the scanner).
    """
    stop_loss = float(swing.get("stop_loss", 0.0))
    take_profit = float(swing.get("take_profit", 0.0))
    if stop_loss > 0 and reference_price <= entry_price * (1 - stop_loss):
        return "stop_loss"
    if take_profit > 0 and reference_price >= entry_price * (1 + take_profit):
        return "take_profit"

    price = ind["price"][k, j]
    if setup == "breakout":
        if price < ind["trail_low"][k, j]:
            return "trailing_stop"
        max_hold = int(swing.get("breakout_max_hold_days", 40))
    else:
        if price > ind["exit_ma"][k, j]:
            return "exit_signal"
        max_hold = int(swing.get("max_hold_days", 20))
    return "time_stop" if held_days >= max_hold else None


def setup_track_record(ind: dict[str, np.ndarray], j: int, swing: dict[str, Any], start: int = 0) -> pd.DataFrame:
    """Replay the setups on one ticker on its own (no position limits) to see how they have worked on it."""
    rows = []
    position: tuple[str, float, int] | None = None
    prices = ind["price"][:, j]
    for k in range(max(start, 1), len(prices)):
        if position is None:
            found = entry_setup(ind, k, j, swing)
            if found:
                position = (found[0], prices[k], k)
            continue
        setup, entry_price, entry_k = position
        reason = exit_reason(ind, k, j, setup, entry_price, prices[k], k - entry_k, swing)
        if reason:
            rows.append({"setup": setup, "return": prices[k] / entry_price - 1, "hold_days": k - entry_k})
            position = None
    return pd.DataFrame(rows, columns=["setup", "return", "hold_days"])
