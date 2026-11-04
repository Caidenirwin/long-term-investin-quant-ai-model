"""Daily decisions for forward testing and live trading.

Uses the same rule functions as the backtest (src/swing.py, src/indicators.py, select_core_targets),
so live decisions match what was tested. Timing also matches the backtest's execution_lag_days = 1:
signals come from the last completed close, and orders fill at the NEXT session's close
(market-on-close orders at the broker).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import select_core_targets
from src.data import is_usd_listed
from src.indicators import calculate_dividend_features, score_dividend_features
from src.live_book import Book
from src.swing import (
    entry_signals, exit_reason, market_allows_entries, swing_indicators, swing_slots, swing_universe,
)


@dataclass
class Market:
    """Prices (CAD) up to the last completed close, plus precomputed swing indicators."""
    prices: pd.DataFrame
    signal_prices: pd.DataFrame
    dividends: pd.DataFrame
    ind: dict[str, np.ndarray]
    col: dict[str, int]
    ticker_to_sector: dict[str, str]
    withholding: float

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.prices.index

    @property
    def last_date(self) -> pd.Timestamp:
        return self.prices.index[-1]

    def close(self, ticker: str, date: pd.Timestamp | None = None) -> float:
        series = self.prices[ticker]
        value = series.loc[:date].dropna() if date is not None else series.dropna()
        return float(value.iloc[-1]) if len(value) else float("nan")

    def next_session(self, date: pd.Timestamp) -> pd.Timestamp | None:
        later = self.dates[self.dates > date]
        return later[0] if len(later) else None


def build_market(config: dict[str, Any], prices: pd.DataFrame, dividends: pd.DataFrame, signals: pd.DataFrame) -> Market:
    prices = prices.sort_index().ffill()
    signals = signals.reindex(index=prices.index, columns=prices.columns).ffill()
    dividends = dividends.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    ticker_to_sector = {t: s for s, tickers in config["sectors"].items() for t in tickers if t in prices.columns}
    return Market(
        prices=prices,
        signal_prices=signals,
        dividends=dividends,
        ind=swing_indicators(signals, config),
        col={t: j for j, t in enumerate(prices.columns)},
        ticker_to_sector=ticker_to_sector,
        withholding=float(config.get("dividends", {}).get("us_withholding_tax", 0.0)),
    )


# ---------------------------------------------------------------------------- costs


def costs(config: dict[str, Any], sleeve: str, ticker: str) -> tuple[float, float]:
    """(flat fee, slippage fraction incl. FX) - the same cost model as the backtest."""
    fee = float(config.get("transaction_fee", 0.0))
    slippage = float(config.get("slippage_bps", 0.0)) / 10000
    if sleeve == "swing":
        swing = config.get("swing", {})
        fee = float(swing.get("transaction_fee", fee))
        slippage = float(swing.get("slippage_bps", config.get("slippage_bps", 0.0))) / 10000
    if is_usd_listed(ticker):
        slippage += float(config.get("fx_fee_bps", 0.0)) / 10000
    return fee, slippage


# ---------------------------------------------------------------------------- valuation


def valuation(book: Book, market: Market) -> dict[str, float]:
    core = sum(q * market.close(t) for t, q in book.core_shares.items() if q > 0)
    swing = sum(p["shares"] * market.close(t) for t, p in book.swing_positions.items())
    return {
        "core_value": book.core_cash + core,
        "swing_value": book.swing_cash + swing,
        "equity": book.core_cash + core + book.swing_cash + swing,
    }


# ---------------------------------------------------------------------------- fills, dividends, sweep


def apply_fill(config: dict[str, Any], book: Book, order: dict[str, Any], quantity: float, close_cad: float,
               fill_date: pd.Timestamp, cash_limit: float | None = None) -> tuple[dict[str, Any] | None, float]:
    """Update the book for a fill at the closing price.

    Returns (closed swing trade record or None, quantity actually filled). With cash_limit (forward test),
    a buy is trimmed to that much cash, i.e. what the sleeve had before the day's sales, as in the backtest.
    """
    ticker, sleeve = order["ticker"], order["sleeve"]
    fee, slippage = costs(config, sleeve, ticker)
    if quantity <= 0:
        return None, 0.0

    if order["action"] == "BUY":
        price = close_cad * (1 + slippage)
        if cash_limit is not None:
            cash = min(cash_limit, book.core_cash if sleeve == "core" else book.swing_cash)
            quantity = min(quantity, max(0.0, math.floor((cash - fee) / price + 1e-9)))
            if quantity <= 0:
                return None, 0.0
        spent = quantity * price + fee
        if sleeve == "core":
            book.core_cash -= spent
            book.core_shares[ticker] = book.core_shares.get(ticker, 0.0) + quantity
        else:
            book.swing_cash -= spent
            book.swing_positions[ticker] = {
                "shares": quantity, "entry_price": price, "entry_date": str(fill_date.date()),
                "cost": spent, "setup": order.get("setup", "breakout"), "dividends": 0.0,
            }
        return None, quantity

    price = close_cad * (1 - slippage)
    proceeds = quantity * price - fee
    if sleeve == "core":
        book.core_shares[ticker] = max(0.0, book.core_shares.get(ticker, 0.0) - quantity)
        if book.core_shares[ticker] < 1e-9:
            book.core_shares.pop(ticker)
        book.core_cash += proceeds
        return None, quantity

    position = book.swing_positions.pop(ticker)
    book.swing_cash += proceeds
    pnl = proceeds + position["dividends"] - position["cost"]
    # The profit sweep runs once after all of the day's exits (so losses net against gains), in live_trader.py.
    return {
        "ticker": ticker, "setup": position["setup"], "entry_date": position["entry_date"],
        "exit_date": str(fill_date.date()), "entry_price": position["entry_price"], "exit_price": price,
        "shares": quantity, "pnl": pnl, "return": pnl / position["cost"], "exit_reason": order["reason"],
    }, quantity


def sweep_profits(config: dict[str, Any], book: Book) -> float:
    """Same rule as the backtest: realized net swing profit above the sleeve's base moves to the core."""
    fraction = float(config.get("profit_sweep", {}).get("fraction", 1.0))
    open_cost = sum(p["cost"] for p in book.swing_positions.values())
    amount = min((book.swing_cash + open_cost - book.swing_base) * fraction, book.swing_cash)
    if amount <= 1:
        return 0.0
    book.swing_cash -= amount
    book.core_cash += amount
    book.cumulative_swept += amount
    book.swing_base = book.swing_cash + open_cost
    return amount


def collect_dividends(config: dict[str, Any], book: Book, market: Market) -> float:
    """Forward-test only: credit dividends with ex-dates since the last run (DRIP into the core stock)."""
    since = pd.Timestamp(book.last_dividend_date) if book.last_dividend_date else market.last_date
    window = market.dividends.loc[(market.dividends.index > since) & (market.dividends.index <= market.last_date)]
    mode = config.get("dividends", {}).get("mode", "drip")
    total = 0.0
    for date, row in window.iterrows():
        for ticker, per_share in row[row > 0].items():
            net = per_share * (1 - (market.withholding if is_usd_listed(ticker) else 0.0))
            core_qty = book.core_shares.get(ticker, 0.0)
            if core_qty > 0:
                income = core_qty * net
                if mode == "drip":
                    book.core_shares[ticker] += income / market.close(ticker, date)
                elif mode == "smart":
                    book.core_cash += income
                total += income
            if ticker in book.swing_positions:
                income = book.swing_positions[ticker]["shares"] * net
                book.swing_positions[ticker]["dividends"] += income
                book.swing_cash += income
                total += income
    book.cumulative_dividends += total
    book.last_dividend_date = str(market.last_date.date())
    return total


# ---------------------------------------------------------------------------- planning


def plan_orders(config: dict[str, Any], book: Book, market: Market, rebalance_due: bool) -> list[dict[str, Any]]:
    """Orders to execute at the next close, whole shares only.

    Both sleeves may trade the same stock (as in the backtest), but never in opposite directions in one run:
    a broker only reports the net change, so a buy and a sell together couldn't be matched back to each sleeve.
    """
    orders: list[dict[str, Any]] = []
    signal_date = market.last_date
    k = len(market.dates) - 1
    swing = config.get("swing", {})
    actions: dict[str, str] = {}

    def add(ticker: str, sleeve: str, action: str, quantity: float, reason: str, setup: str | None = None) -> bool:
        """Queue an order; returns False (and queues nothing) if it would conflict with another order."""
        quantity = math.floor(quantity + 1e-9) if action == "BUY" else quantity
        if quantity <= 0 or actions.get(ticker, action) != action:
            return False
        for existing in orders:
            if existing["ticker"] == ticker and existing["sleeve"] == sleeve:
                existing["quantity"] += quantity  # e.g. a rebalance buy plus a deploy buy: one order at the broker
                return True
        actions[ticker] = action
        orders.append({
            "ticker": ticker, "sleeve": sleeve, "action": action, "quantity": quantity, "reason": reason,
            "setup": setup, "signal_date": str(signal_date.date()), "est_price_cad": market.close(ticker),
        })
        return True

    # Swing exits (sell everything in the position). Buys only use cash on hand now, not today's sale
    # proceeds (unknown until the close, and unsettled until the next day). The backtest does the same.
    swing_cash_after = book.swing_cash
    if swing.get("enabled", False):
        for ticker, position in book.swing_positions.items():
            if ticker not in market.col:
                continue
            j = market.col[ticker]
            entry_k = int(np.searchsorted(market.dates, pd.Timestamp(position["entry_date"])))
            if k < entry_k:
                continue
            held = (k + 1) - entry_k  # trading days held when the order fills at the next close
            reason = exit_reason(market.ind, k, j, position["setup"], position["entry_price"],
                                 market.close(ticker), held, swing)
            if reason:
                add(ticker, "swing", "SELL", position["shares"], reason)

        # Swing entries (a small sleeve holds fewer positions; see swing_slots).
        swing_value = valuation(book, market)["swing_value"]
        total_slots = swing_slots(swing_value, swing)
        held_after = len(book.swing_positions) - sum(o["action"] == "SELL" and o["sleeve"] == "swing" for o in orders)
        slots = total_slots - held_after
        market_col = market.col.get(swing.get("market_filter_ticker", ""))
        if slots > 0 and market_allows_entries(market.ind, k, market_col):
            universe = [market.col[t] for t in swing_universe(config) if t in market.col]
            open_cols = [j for j in universe if market.prices.columns[j] not in book.swing_positions]
            slice_value = swing_value / total_slots
            # Walk down the ranked signals until the slots are filled, skipping unaffordable ones (as the backtest).
            for j, setup, _ in entry_signals(market.ind, k, open_cols, swing):
                if slots <= 0:
                    break
                ticker = market.prices.columns[j]
                fee, slip = costs(config, "swing", ticker)
                value = min(slice_value, swing_cash_after - fee)
                if value < float(swing.get("min_trade_value", 500)):
                    break
                price = market.close(ticker) * (1 + slip)
                quantity = math.floor(value / price)
                if add(ticker, "swing", "BUY", quantity, setup, setup):
                    swing_cash_after -= quantity * price + fee
                    slots -= 1

    # Dividend core.
    core = config.get("core", {})
    values = valuation(book, market)
    core_value = values["core_value"]
    core_cash_after = book.core_cash

    if rebalance_due or not book.core_targets:
        features = score_dividend_features(
            calculate_dividend_features(market.prices, market.signal_prices, market.dividends, market.last_date,
                                        market.ticker_to_sector, config),
            config,
        )
        held = {t for t, q in book.core_shares.items() if q > 0}
        book.core_targets = select_core_targets(features, config, held, core_value=core_value)
        band = float(core.get("rebalance_band", 0.04))
        for ticker in sorted(set(book.core_targets) | held):
            target = book.core_targets.get(ticker, 0.0)
            price = market.close(ticker)
            current = book.core_shares.get(ticker, 0.0) * price / core_value if core_value else 0.0
            drift = target - current
            if target == 0 and current > 0:
                cut = ticker in features.index and bool(features.at[ticker, "dividend_cut"])
                add(ticker, "core", "SELL", book.core_shares[ticker], "dividend_cut" if cut else "exit")
            elif target > 0 and drift < -band:
                add(ticker, "core", "SELL", math.floor(-drift * core_value / price), "rebalance")
        # Same order and rule as the backtest (alphabetical; buy when below target by more than the band), so
        # when cash runs short both buy the same stocks. Smaller gaps are filled by the deploy step below.
        for ticker, target in sorted(book.core_targets.items()):
            price = market.close(ticker)
            current = book.core_shares.get(ticker, 0.0) * price / core_value if core_value else 0.0
            if target - current > band:
                fee, slip = costs(config, "core", ticker)
                value = min((target - current) * core_value, core_cash_after - fee)
                quantity = math.floor(value / (price * (1 + slip))) if value > 0 else 0
                if add(ticker, "core", "BUY", quantity, "rebalance"):
                    core_cash_after -= quantity * price * (1 + slip) + fee

    # Every day (rebalance days too, as in the backtest): invest deposits, swept profits and pooled dividends
    # in the most underweight holdings, counting shares already ordered above.
    planned = {o["ticker"]: (o["quantity"] if o["action"] == "BUY" else -o["quantity"])
               for o in orders if o["sleeve"] == "core"}
    if book.core_targets:
        investable = core_cash_after - float(core.get("min_cash_buffer", 0.01)) * core_value
        min_order = float(core.get("min_trade_value", 50))
        if investable >= float(core.get("min_deploy_cash", 1000)):
            shortfall = {
                t: w * core_value - (book.core_shares.get(t, 0.0) + planned.get(t, 0.0)) * market.close(t)
                for t, w in book.core_targets.items()
            }
            # Biggest shortfall first; ties go alphabetically (same as the backtest).
            for ticker, gap in sorted(shortfall.items(), key=lambda item: (-round(item[1], 6), item[0]))[:3]:
                if gap <= 0 or investable < min_order:
                    break
                fee, slip = costs(config, "core", ticker)
                amount = min(investable, max(gap, min_order), core_cash_after - fee)  # same cap as the backtest
                quantity = math.floor(amount / (market.close(ticker) * (1 + slip)))
                if add(ticker, "core", "BUY", quantity, "deploy"):
                    spent = quantity * market.close(ticker) * (1 + slip) + fee
                    investable -= spent
                    core_cash_after -= spent

    return orders
