from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data import is_usd_listed
from src.indicators import calculate_dividend_features, score_dividend_features
from src.reporting import calculate_performance_summary, performance_metrics
from src.swing import (
    entry_signals, exit_reason, market_allows_entries, swing_indicators, swing_slots, swing_universe,
)


@dataclass
class BacktestResults:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    swing_trades: pd.DataFrame
    positions: pd.DataFrame
    sector_weights: pd.DataFrame
    dividend_income: pd.DataFrame
    performance_summary: pd.DataFrame


@dataclass
class _SwingPosition:
    shares: float
    entry_price: float
    entry_index: int
    cost: float
    setup: str
    dividends: float = 0.0


def run_backtest(
    config: dict[str, Any],
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    signal_prices: pd.DataFrame | None = None,
    feature_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> BacktestResults:
    """Simulate a swing sleeve whose realized profits fund a long-term dividend core.

    feature_cache lets the optimizer reuse dividend features across runs (they do not depend on parameters).
    """
    engine = _Engine(config, prices, dividends, signal_prices, feature_cache)
    engine.run()
    return engine.results()


class _Engine:
    def __init__(
        self,
        config: dict[str, Any],
        prices: pd.DataFrame,
        dividends: pd.DataFrame,
        signal_prices: pd.DataFrame | None,
        feature_cache: dict[pd.Timestamp, pd.DataFrame] | None,
    ) -> None:
        self.config = config
        self.core_cfg = config.get("core", {})
        self.swing_cfg = config.get("swing", {})
        self.fee = float(config.get("transaction_fee", 0.0))
        self.slippage = float(config.get("slippage_bps", 0.0)) / 10000
        # Liquid ETFs usually trade cheaper than single stocks, so the swing sleeve can override costs.
        self.costs = {
            "core": (self.fee, self.slippage),
            "swing": (
                float(self.swing_cfg.get("transaction_fee", self.fee)),
                float(self.swing_cfg.get("slippage_bps", config.get("slippage_bps", 0.0))) / 10000,
            ),
        }
        self.fx_fee = float(config.get("fx_fee_bps", 0.0)) / 10000
        self.cash_interest_rate = float(config.get("cash_interest_rate", 0.0))
        self.dividend_mode = config.get("dividends", {}).get("mode", "drip")
        self.feature_cache = feature_cache if feature_cache is not None else {}

        prices = prices.sort_index().ffill()
        if config.get("end_date"):
            prices = prices.loc[: pd.Timestamp(config["end_date"])]
        self.prices = prices
        self.signal_prices = (signal_prices if signal_prices is not None else prices).reindex(
            index=prices.index, columns=prices.columns
        ).ffill()
        self.dividends = dividends.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

        self.dates = prices.index
        self.tickers = list(prices.columns)
        self.col = {ticker: j for j, ticker in enumerate(self.tickers)}
        self.P = prices.to_numpy(float)
        self.S = self.signal_prices.to_numpy(float)
        withholding = float(config.get("dividends", {}).get("us_withholding_tax", 0.0))
        self.withholding = np.array([withholding if is_usd_listed(ticker) else 0.0 for ticker in self.tickers])
        self.D = self.dividends.to_numpy(float) * (1 - self.withholding)
        # Converting CAD to USD and back costs extra on every US-listed trade (e.g. ~1.5% at some brokers).
        self.fx_cost = np.array([self.fx_fee if is_usd_listed(ticker) else 0.0 for ticker in self.tickers])
        self.first = int(np.searchsorted(self.dates, pd.Timestamp(config["start_date"])))
        if self.first >= len(self.dates):
            raise ValueError("No price data on or after start_date.")

        self.ticker_to_sector = {
            ticker: sector
            for sector, tickers in config["sectors"].items()
            for ticker in tickers
            if ticker in self.col
        }
        self.rebalance_indices = _rebalance_indices(
            self.dates, self.first, self.core_cfg.get("rebalance_frequency", config.get("rebalance_frequency", "Q"))
        )

        starting_cash = float(config["starting_cash"])
        self.swing_enabled = bool(self.swing_cfg.get("enabled", False))
        swing_fraction = float(self.swing_cfg.get("capital_fraction", 0.0)) if self.swing_enabled else 0.0
        self.swing_cash = starting_cash * swing_fraction
        self.swing_base = self.swing_cash  # realized profits above this are swept to the core
        self.core_cash = starting_cash - self.swing_cash
        self.whole_shares = bool(config.get("whole_shares", True))

        # Monthly deposits, on the first trading day of each month after the start.
        contributions = config.get("contributions", {})
        self.monthly_deposit = float(contributions.get("monthly", 0.0))
        self.deposit_swing_fraction = (
            float(contributions.get("swing_fraction", swing_fraction)) if self.swing_enabled else 0.0
        )
        active = self.dates[self.first:]
        month_keys = active.year * 100 + active.month
        self.deposit_indices = {self.first + int(o) for o in np.flatnonzero(np.r_[False, month_keys[1:] != month_keys[:-1]])}
        self.cumulative_deposits = starting_cash
        self.swing_deposits = 0.0  # deposits into the swing sleeve after the start
        self.deposit_totals = {"core": 0.0, "swing": 0.0}
        self.previous_equity = starting_cash
        self.deposit_today = 0.0
        self.twr_index = 1.0  # time-weighted return index: growth with deposits stripped out
        benchmark = config.get("benchmark_ticker", "XIC.TO")
        self.benchmark_col = self.col.get(benchmark)
        self.benchmark_units = 0.0  # the same deposits invested in the benchmark
        n = len(self.tickers)
        self.core_shares = np.zeros(n)
        self.core_targets = np.zeros(n)
        self.swing_positions: dict[int, _SwingPosition] = {}
        self.spendable = {"core": self.core_cash, "swing": self.swing_cash}
        self.planning_cash = dict(self.spendable)
        self.bought_today: set[tuple[str, int]] = set()
        self.exited_today: set[int] = set()

        # Idle swing cash can be parked in an index ETF (while the market filter is on) instead of sitting in cash.
        self.park_col = self.col.get(self.swing_cfg.get("park_ticker", "")) if self.swing_enabled else None
        self.park_shares = 0.0
        self.park_cost = 0.0
        self.park_realized = 0.0  # gains realized on the park ETF plus its dividends

        if self.swing_enabled:
            self.ind = swing_indicators(self.signal_prices, config)
            self.swing_universe = [self.col[t] for t in swing_universe(config) if t in self.col]
            market = self.swing_cfg.get("market_filter_ticker", "")
            self.market_col = self.col.get(market) if market else None

        self.trade_rows: list[dict[str, Any]] = []
        self.swing_trade_rows: list[dict[str, Any]] = []
        self.dividend_rows: list[dict[str, Any]] = []
        self.equity_rows: list[dict[str, Any]] = []
        self.core_share_history: list[np.ndarray] = []
        self.cumulative_dividends = 0.0
        self.withdrawn_dividends = 0.0
        self.cumulative_swept = 0.0
        self.swept_today = 0.0

    # ------------------------------------------------------------------ main loop

    def run(self) -> None:
        for i in range(self.first, len(self.dates)):
            px = self.P[i]
            self.swept_today = 0.0
            self._accrue_cash_interest(i)
            self.deposit_today = self._deposit(i)
            # Orders are sized with the cash known when they're placed (before today's dividends and sells);
            # at the fill, buys can use cash on hand before today's sells (sale proceeds settle the next day).
            self.planning_cash = {"core": self.core_cash, "swing": self.swing_cash}
            self.bought_today: set[tuple[str, int]] = set()  # a 2nd buy of the same stock is the same order: no 2nd fee
            dividend_income = self._collect_dividends(i, px)
            self.spendable = {"core": self.core_cash, "swing": self.swing_cash}
            # Orders are sized from the previous close (a market-on-close order is placed before the close is
            # known) and filled at today's close, exactly as live_trader.py does it.
            decision_px = self._decision_prices(i)
            if i in self.rebalance_indices:
                self._rebalance_core(i, decision_px)
            if self.swing_enabled:
                # Position sizes come from the sleeve as it stood before today's exits and profit sweep.
                swing_decision_value = self._swing_value(decision_px)
                self._swing_exits(i, px)
                self._swing_entries(i, px, decision_px, swing_decision_value)
                self._manage_park(i, px)
            self._deploy_core_cash(i, decision_px)
            self._record(i, px, dividend_income)

    def _decision_prices(self, i: int) -> np.ndarray:
        previous = self.P[i - 1] if i > 0 else self.P[i]
        return np.where(previous > 0, previous, self.P[i])

    # ------------------------------------------------------------------ cash and dividends

    def _deposit(self, i: int) -> float:
        """Add the monthly deposit, split between the sleeves like the starting cash."""
        benchmark_price = self.S[i, self.benchmark_col] if self.benchmark_col is not None else np.nan
        amount = self.monthly_deposit if i in self.deposit_indices else 0.0
        if i == self.first and benchmark_price > 0:
            self.benchmark_units = float(self.config["starting_cash"]) / benchmark_price
        if amount <= 0:
            return 0.0
        to_swing = amount * self.deposit_swing_fraction
        self.swing_cash += to_swing
        self.swing_base += to_swing  # deposits aren't profit, so they are never swept
        self.swing_deposits += to_swing
        self.core_cash += amount - to_swing
        self.deposit_totals["swing"] += to_swing
        self.deposit_totals["core"] += amount - to_swing
        self.cumulative_deposits += amount
        if benchmark_price > 0:
            self.benchmark_units += amount / benchmark_price
        return amount

    def _accrue_cash_interest(self, i: int) -> None:
        """Idle cash earns interest, like holding a high-interest savings ETF between trades."""
        if self.cash_interest_rate <= 0 or i <= self.first:
            return
        days = (self.dates[i] - self.dates[i - 1]).days
        growth = (1 + self.cash_interest_rate) ** (days / 365) - 1
        self.swing_cash += max(self.swing_cash, 0.0) * growth
        self.core_cash += max(self.core_cash, 0.0) * growth

    def _collect_dividends(self, i: int, px: np.ndarray) -> float:
        per_share = np.nan_to_num(self.D[i])
        if not per_share.any():
            return 0.0

        core_income = self.core_shares * per_share
        swing_income = 0.0
        for j, position in self.swing_positions.items():
            amount = position.shares * per_share[j]
            position.dividends += amount
            swing_income += amount
        if self.park_col is not None and self.park_shares > 0:
            park_income = self.park_shares * per_share[self.park_col]
            self.park_realized += park_income
            swing_income += park_income
        self.swing_cash += swing_income

        for j in np.flatnonzero(core_income > 0):
            self.dividend_rows.append({"date": self.dates[i], "ticker": self.tickers[j], "sleeve": "core", "amount": core_income[j]})
        if swing_income > 0:
            self.dividend_rows.append({"date": self.dates[i], "ticker": "", "sleeve": "swing", "amount": swing_income})

        total_core = float(core_income.sum())
        if self.dividend_mode == "drip":
            # Broker DRIPs reinvest into the paying stock with no commission.
            payable = (core_income > 0) & (px > 0)
            self.core_shares[payable] += core_income[payable] / px[payable]
        elif self.dividend_mode == "smart":
            self.core_cash += total_core  # pooled and invested in the most underweight holdings
        else:
            self.withdrawn_dividends += total_core

        total = total_core + swing_income
        self.cumulative_dividends += total
        return total

    # ------------------------------------------------------------------ dividend core

    def _features(self, i: int) -> pd.DataFrame:
        as_of = self.dates[i - 1] if i > 0 else self.dates[i]
        if as_of not in self.feature_cache:
            self.feature_cache[as_of] = calculate_dividend_features(
                self.prices, self.signal_prices, self.dividends, as_of, self.ticker_to_sector, self.config
            )
        return self.feature_cache[as_of]

    def _rebalance_core(self, i: int, px: np.ndarray) -> None:
        features = score_dividend_features(self._features(i), self.config)
        held = {self.tickers[j] for j in np.flatnonzero(self.core_shares > 0)}
        targets = select_core_targets(features, self.config, held, core_value=self._core_value(px))
        self.core_targets = np.zeros(len(self.tickers))
        for ticker, weight in targets.items():
            self.core_targets[self.col[ticker]] = weight

        prices = np.nan_to_num(px)
        value = self._core_value(px)
        if value <= 0:
            return
        current_weights = self.core_shares * prices / value
        band = float(self.core_cfg.get("rebalance_band", 0.02))
        drift = self.core_targets - current_weights
        exiting = (self.core_targets == 0) & (self.core_shares > 0)
        needs_trade = exiting | ((self.core_targets > 0) & (np.abs(drift) > band))

        for j in np.flatnonzero(needs_trade & (drift < 0)):
            quantity = self.core_shares[j] if exiting[j] else -drift[j] * value / prices[j]
            reason = "dividend_cut" if exiting[j] and self._was_cut(features, j) else ("exit" if exiting[j] else "rebalance")
            self._sell("core", i, j, quantity, reason)
        for j in np.flatnonzero(needs_trade & (drift > 0)):
            self._buy("core", i, j, drift[j] * value, "rebalance", size_price=prices[j])

    def _was_cut(self, features: pd.DataFrame, j: int) -> bool:
        ticker = self.tickers[j]
        return ticker in features.index and bool(features.at[ticker, "dividend_cut"])

    def _deploy_core_cash(self, i: int, px: np.ndarray) -> None:
        """Invest swept swing profits and pooled dividends into the most underweight holdings."""
        if not self.core_targets.any():
            return
        prices = np.nan_to_num(px)
        value = self._core_value(px)
        available = min(self.core_cash, self.spendable["core"], self.planning_cash["core"])
        investable = available - float(self.core_cfg.get("min_cash_buffer", 0.01)) * value
        if investable < float(self.core_cfg.get("min_deploy_cash", 1000)):
            return
        # Each buy is at least min_trade_value so small deposits aren't eaten by fees.
        min_order = float(self.core_cfg.get("min_trade_value", 50))
        shortfall = np.where(prices > 0, self.core_targets * value - self.core_shares * prices, 0.0)
        # Biggest shortfall first; ties (common when holdings are equal) go alphabetically, same as live trading.
        candidates = sorted(np.flatnonzero(self.core_targets > 0), key=lambda j: (-round(shortfall[j], 6), self.tickers[j]))
        for j in candidates[:3]:
            if shortfall[j] <= 0 or investable < min_order:
                break
            cash_before = self.core_cash
            self._buy("core", i, j, min(investable, max(shortfall[j], min_order)), "deploy", size_price=prices[j])
            investable -= cash_before - self.core_cash

    # ------------------------------------------------------------------ swing sleeve

    def _swing_exits(self, i: int, px: np.ndarray) -> None:
        k = i - int(self.swing_cfg.get("execution_lag_days", 1))
        exited = False
        self.exited_today: set[int] = set()  # not re-bought the same day (selling and re-buying is pure cost)

        for j, position in list(self.swing_positions.items()):
            if i <= position.entry_index or k < position.entry_index or not px[j] > 0:
                continue
            reason = exit_reason(
                self.ind, k, j, position.setup, position.entry_price, self.P[k, j], i - position.entry_index, self.swing_cfg
            )
            if reason is None:
                continue

            proceeds = self._sell("swing", i, j, position.shares, reason)
            pnl = proceeds + position.dividends - position.cost
            self.swing_trade_rows.append(
                {
                    "ticker": self.tickers[j],
                    "setup": position.setup,
                    "entry_date": self.dates[position.entry_index],
                    "exit_date": self.dates[i],
                    "entry_price": position.entry_price,
                    "exit_price": px[j] * (1 - self._slippage("swing", j)),
                    "hold_days": i - position.entry_index,
                    "pnl": pnl,
                    "return": pnl / position.cost,
                    "exit_reason": reason,
                }
            )
            del self.swing_positions[j]
            self.exited_today.add(j)
            exited = True

        if exited:
            self._sweep_profits(i)

    def _swing_entries(self, i: int, px: np.ndarray, decision_px: np.ndarray, swing_value: float) -> None:
        k = i - int(self.swing_cfg.get("execution_lag_days", 1))
        min_trade = float(self.swing_cfg.get("min_trade_value", 500))
        total_slots = swing_slots(swing_value, self.swing_cfg)
        slots = total_slots - len(self.swing_positions)
        if k < 0 or slots <= 0:
            return
        if not market_allows_entries(self.ind, k, self.market_col):
            return

        open_columns = [j for j in self.swing_universe
                        if j not in self.swing_positions and j not in self.exited_today and px[j] > 0]
        candidates = entry_signals(self.ind, k, open_columns, self.swing_cfg)

        slice_value = swing_value / total_slots
        # Walk down the ranked signals until the open slots are filled, skipping stocks whose single share
        # costs more than the position (matters for small accounts).
        for j, setup, _ in candidates:
            if slots <= 0:
                break
            if self.swing_cash - self.costs["swing"][0] < slice_value:
                self._unpark(i, slice_value - self.swing_cash + 2 * self.costs["swing"][0], "unpark")
            available = min(self.swing_cash, self.spendable["swing"], self.planning_cash["swing"])
            value = min(slice_value, available - self.costs["swing"][0])
            if value < min_trade:
                break
            cash_before = self.swing_cash
            shares = self._buy("swing", i, j, value, setup, size_price=decision_px[j])
            if shares <= 0:
                continue
            self.swing_positions[j] = _SwingPosition(
                shares=shares,
                entry_price=px[j] * (1 + self._slippage("swing", j)),
                entry_index=i,
                cost=cash_before - self.swing_cash,
                setup=setup,
            )
            slots -= 1

    def _manage_park(self, i: int, px: np.ndarray) -> None:
        """Keep idle swing cash in the park ETF while the market filter is on; go to cash when it turns off."""
        if self.park_col is None:
            return
        k = i - int(self.swing_cfg.get("execution_lag_days", 1))
        if k < 0 or not market_allows_entries(self.ind, k, self.market_col):
            self._unpark(i, float("inf"), "unpark_downtrend")
            return
        fee = self.costs["swing"][0]
        idle = self.swing_cash - fee
        # Only top up in meaningful chunks so the ETF isn't traded every day.
        if idle >= max(float(self.swing_cfg.get("min_trade_value", 500)), 0.10 * self._swing_value(px)):
            cash_before = self.swing_cash
            shares = self._buy("swing", i, self.park_col, idle, "park")
            self.park_shares += shares
            self.park_cost += cash_before - self.swing_cash

    def _unpark(self, i: int, amount: float, reason: str) -> None:
        """Sell enough of the park ETF to raise `amount` of cash (inf = sell it all)."""
        if self.park_col is None or self.park_shares <= 0:
            return
        price = self.P[i, self.park_col]
        if not price > 0:
            return
        fraction = 1.0 if amount == float("inf") else min(1.0, amount / (self.park_shares * price))
        quantity = self.park_shares * fraction
        if self.whole_shares and fraction < 1.0:
            quantity = min(float(math.ceil(quantity)), self.park_shares)  # round up so enough cash is raised
            fraction = quantity / self.park_shares
        proceeds = self._sell("swing", i, self.park_col, quantity, reason)
        self.spendable["swing"] += proceeds  # parked cash exists to fund same-day entries
        self.planning_cash["swing"] += proceeds
        cost = self.park_cost * fraction
        self.park_realized += proceeds - cost
        self.park_shares -= quantity
        self.park_cost -= cost
        if self.park_shares < 1e-9:
            self.park_shares, self.park_cost = 0.0, 0.0

    def _sweep_profits(self, i: int) -> None:
        """Move realized net swing profit above the sleeve's starting capital into the dividend core."""
        fraction = float(self.config.get("profit_sweep", {}).get("fraction", 1.0))
        open_cost = sum(position.cost for position in self.swing_positions.values()) + self.park_cost
        realized_profit = self.swing_cash + open_cost - self.swing_base
        amount = min(realized_profit * fraction, self.swing_cash)
        if amount <= 1:
            return
        self.swing_cash -= amount
        self.core_cash += amount
        self.swept_today += amount
        self.cumulative_swept += amount
        # Profit kept in the sleeve (fraction < 1) becomes part of its new base so it is not swept later.
        self.swing_base = self.swing_cash + open_cost

    # ------------------------------------------------------------------ execution

    def _slippage(self, sleeve: str, j: int) -> float:
        return self.costs[sleeve][1] + self.fx_cost[j]

    def _held(self, sleeve: str, j: int) -> float:
        if sleeve == "core":
            return float(self.core_shares[j])
        if j == self.park_col:
            return self.park_shares
        return self.swing_positions[j].shares if j in self.swing_positions else 0.0

    def _buy(self, sleeve: str, i: int, j: int, value: float, reason: str, size_price: float | None = None) -> float:
        """Buy about `value` of ticker j at today's close. With size_price, the share count is set from that
        (earlier) price, as a real order must be, then filled at today's close if the cash covers it."""
        fee, slippage = self.costs[sleeve][0], self._slippage(sleeve, j)
        if (sleeve, j) in self.bought_today:
            fee = 0.0
        price = self.P[i, j]
        cash = self.core_cash if sleeve == "core" else self.swing_cash
        limit = min(cash, self.spendable[sleeve])
        value = min(value, limit - fee, self.planning_cash[sleeve] - fee)
        if not price > 0 or value <= 1:
            return 0.0
        execution_price = price * (1 + slippage)
        sizing_price = (size_price if size_price and size_price > 0 else price) * (1 + slippage)
        quantity = min(value / sizing_price, (limit - fee) / execution_price)
        if self.whole_shares:
            quantity = math.floor(quantity + 1e-9)
        if quantity <= 0:
            return 0.0  # can't afford one share
        self.planning_cash[sleeve] -= quantity * sizing_price + fee
        self.bought_today.add((sleeve, j))
        value = quantity * execution_price
        cash -= value + fee
        self.spendable[sleeve] -= value + fee
        if sleeve == "core":
            self.core_cash = cash
            self.core_shares[j] += quantity
        else:
            self.swing_cash = cash
        self._log_trade(i, j, sleeve, "BUY", quantity, execution_price, value, reason, cash, fee)
        return quantity

    def _sell(self, sleeve: str, i: int, j: int, quantity: float, reason: str) -> float:
        fee, slippage = self.costs[sleeve][0], self._slippage(sleeve, j)
        price = self.P[i, j]
        if self.whole_shares and quantity < self._held(sleeve, j) - 1e-9:
            quantity = math.floor(quantity + 1e-9)  # partial sells in whole shares; full exits sell everything
        if not price > 0 or quantity <= 0:
            return 0.0
        execution_price = price * (1 - slippage)
        gross = quantity * execution_price
        proceeds = gross - fee
        if sleeve == "core":
            self.core_shares[j] = max(0.0, self.core_shares[j] - quantity)
            self.core_cash += proceeds
            cash = self.core_cash
        else:
            self.swing_cash += proceeds
            cash = self.swing_cash
        self._log_trade(i, j, sleeve, "SELL", quantity, execution_price, gross, reason, cash, fee)
        return proceeds

    def _log_trade(
        self, i: int, j: int, sleeve: str, action: str, quantity: float, price: float, gross: float, reason: str,
        cash: float, fee: float,
    ) -> None:
        slippage = self._slippage(sleeve, j)
        self.trade_rows.append(
            {
                "date": self.dates[i],
                "sleeve": sleeve,
                "ticker": self.tickers[j],
                "action": action,
                "reason": reason,
                "quantity": quantity,
                "price": price,
                "gross_value": gross,
                "fee": fee,
                "fees_and_slippage": fee + quantity * self.P[i, j] * slippage,
                "cash_after_trade": cash,
            }
        )

    # ------------------------------------------------------------------ bookkeeping

    def _core_value(self, px: np.ndarray) -> float:
        return float(self.core_cash + (self.core_shares * np.nan_to_num(px)).sum())

    def _swing_value(self, px: np.ndarray) -> float:
        park = self.park_shares * np.nan_to_num(px[self.park_col]) if self.park_col is not None else 0.0
        return float(self.swing_cash + park + sum(p.shares * np.nan_to_num(px[j]) for j, p in self.swing_positions.items()))

    def _record(self, i: int, px: np.ndarray, dividend_income: float) -> None:
        core_value = self._core_value(px)
        swing_value = self._swing_value(px)
        equity = core_value + swing_value
        # Time-weighted return: today's growth excluding today's deposit, chained day to day.
        if self.previous_equity > 0:
            self.twr_index *= (equity - self.deposit_today) / self.previous_equity
        self.previous_equity = equity
        benchmark_price = self.S[i, self.benchmark_col] if self.benchmark_col is not None else np.nan
        self.equity_rows.append(
            {
                "date": self.dates[i],
                "equity": equity,
                "deposit": self.deposit_today,
                "cumulative_deposits": self.cumulative_deposits,
                "twr_index": self.twr_index,
                "benchmark_same_deposits": self.benchmark_units * benchmark_price if benchmark_price > 0 else np.nan,
                "core_value": core_value,
                "swing_value": swing_value,
                "core_cash": self.core_cash,
                "swing_cash": self.swing_cash,
                "dividend_income": dividend_income,
                "cumulative_dividends": self.cumulative_dividends,
                "withdrawn_dividends": self.withdrawn_dividends,
                "swept_to_core": self.swept_today,
                "cumulative_swept": self.cumulative_swept,
                # What the swing engine has produced in total: what's left in the sleeve plus everything it paid out,
                # minus money deposited into it after the start.
                "swing_contribution": swing_value + self.cumulative_swept - self.swing_deposits,
            }
        )
        self.core_share_history.append(self.core_shares.copy())

    def results(self) -> BacktestResults:
        equity_curve = pd.DataFrame(self.equity_rows)
        equity_curve["daily_return"] = equity_curve["equity"].pct_change().fillna(0.0)
        trades = pd.DataFrame(self.trade_rows)
        swing_trades = pd.DataFrame(self.swing_trade_rows)
        last_prices = np.nan_to_num(self.P[-1])

        features = calculate_dividend_features(
            self.prices, self.signal_prices, self.dividends, self.dates[-1], self.ticker_to_sector, self.config
        )
        annual_dividend = (
            features["annual_dividend"].reindex(self.tickers).fillna(0.0).to_numpy() if not features.empty else 0.0
        )
        forward_income = float((self.core_shares * annual_dividend * (1 - self.withholding)).sum())
        ending_core = equity_curve["core_value"].iloc[-1]

        start = self.dates[self.first]
        benchmark = self.config.get("benchmark_ticker", "XIC.TO")
        benchmark_stats: dict[str, Any] = {}
        if benchmark in self.signal_prices.columns:
            stats = performance_metrics(self.signal_prices.loc[start:, benchmark], self.config.get("risk_free_rate", 0.0))
            benchmark_stats = {"benchmark_CAGR": stats.get("CAGR"), "benchmark_max_drawdown": stats.get("max_drawdown")}

        summary = calculate_performance_summary(
            equity_curve,
            trades,
            swing_trades,
            float(self.config["starting_cash"]),
            float(self.config.get("risk_free_rate", 0.0)),
            extra={
                "swing_park_realized": self.park_realized,
                "total_deposited": self.cumulative_deposits,
                "deposits_core": self.deposit_totals["core"],
                "deposits_swing": self.deposit_totals["swing"],
                "gain": float(equity_curve["equity"].iloc[-1]) - self.cumulative_deposits,
                "benchmark_same_deposits": float(equity_curve["benchmark_same_deposits"].iloc[-1]),
                "forward_annual_dividends": forward_income,
                "forward_dividend_yield": forward_income / ending_core if ending_core else np.nan,
                **benchmark_stats,
            },
        )

        return BacktestResults(
            equity_curve=equity_curve,
            trades=trades,
            swing_trades=swing_trades,
            positions=self._positions(last_prices, features),
            sector_weights=self._sector_weights(equity_curve),
            dividend_income=self._dividend_income_by_year(),
            performance_summary=summary,
        )

    def _positions(self, last_prices: np.ndarray, features: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for j in np.flatnonzero(self.core_shares > 0):
            ticker = self.tickers[j]
            annual = features.at[ticker, "annual_dividend"] if ticker in features.index else 0.0
            rows.append(
                {
                    "ticker": ticker,
                    "sleeve": "core",
                    "sector": self.ticker_to_sector.get(ticker, "Unknown"),
                    "quantity": self.core_shares[j],
                    "last_price": last_prices[j],
                    "market_value": self.core_shares[j] * last_prices[j],
                    "target_weight": self.core_targets[j],
                    "dividend_yield": annual / last_prices[j] if last_prices[j] else 0.0,
                    "forward_annual_income": self.core_shares[j] * annual * (1 - self.withholding[j]),
                }
            )
        for j, position in self.swing_positions.items():
            rows.append(
                {
                    "ticker": self.tickers[j],
                    "sleeve": "swing",
                    "sector": "Swing",
                    "quantity": position.shares,
                    "last_price": last_prices[j],
                    "market_value": position.shares * last_prices[j],
                    "cost_basis": position.cost,
                    "dividends_received": position.dividends,
                }
            )
        if self.park_col is not None and self.park_shares > 0:
            rows.append(
                {
                    "ticker": self.tickers[self.park_col],
                    "sleeve": "swing",
                    "sector": "Swing (parked cash)",
                    "quantity": self.park_shares,
                    "last_price": last_prices[self.park_col],
                    "market_value": self.park_shares * last_prices[self.park_col],
                    "cost_basis": self.park_cost,
                    "dividends_received": 0.0,  # park dividends are already in park_realized
                }
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["sleeve", "market_value"], ascending=[True, False])

    def _sector_weights(self, equity_curve: pd.DataFrame) -> pd.DataFrame:
        values = np.array(self.core_share_history) * np.nan_to_num(self.P[self.first :])
        frame = pd.DataFrame(values, columns=self.tickers)
        sectors = frame.T.groupby(pd.Series(self.ticker_to_sector)).sum().T
        total = equity_curve["equity"].to_numpy()
        weights = sectors.div(total, axis=0)
        weights.insert(0, "Swing sleeve", equity_curve["swing_value"].to_numpy() / total)
        weights.insert(0, "Core cash", equity_curve["core_cash"].to_numpy() / total)
        weights.insert(0, "date", equity_curve["date"].to_numpy())
        return weights

    def _dividend_income_by_year(self) -> pd.DataFrame:
        if not self.dividend_rows:
            return pd.DataFrame(columns=["year", "core", "swing", "total"])
        log = pd.DataFrame(self.dividend_rows)
        table = log.pivot_table(index=log["date"].dt.year, columns="sleeve", values="amount", aggfunc="sum", fill_value=0.0)
        table = table.reindex(columns=["core", "swing"], fill_value=0.0)
        table["total"] = table.sum(axis=1)
        return table.rename_axis("year").reset_index()


def select_core_targets(
    features: pd.DataFrame, config: dict[str, Any], held: set[str], core_value: float | None = None
) -> dict[str, float]:
    """Pick the best dividend names per sector, keeping current holdings unless they slip well down the ranking.

    core_value lets a small account hold fewer stocks (see core.min_position_value).
    """
    if features.empty:
        return {}
    core = config.get("core", {})
    count = int(core.get("stocks_per_sector", 3))
    hold_buffer = int(core.get("hold_buffer", 2))
    min_yield = float(core.get("min_yield", 0.01))
    max_position = float(core.get("max_position_size", 1.0))

    picks: dict[str, list[str]] = {}
    for sector in config["target_allocation"]:
        eligible = features[
            (features["sector"] == sector) & ~features["dividend_cut"] & (features["dividend_yield"] >= min_yield)
        ].sort_values("score", ascending=False)
        if eligible.empty:
            continue
        ranked = eligible.index.tolist()
        keep = [ticker for ticker in ranked[: count + hold_buffer] if ticker in held]
        picks[sector] = (keep + [ticker for ticker in ranked if ticker not in keep])[:count]

    # Sectors with no eligible names hand their weight to the others.
    sector_weights = {sector: config["target_allocation"][sector] for sector in picks}

    # Optional momentum tilt: overweight industries whose stocks have risen most over 12 months, underweight
    # the laggards. tilt = 0.5 means the strongest sector gets 1.5x its base weight and the weakest 0.5x.
    tilt = float(core.get("sector_momentum_tilt", 0.0))
    if tilt > 0 and len(sector_weights) > 1:
        momentum = features.groupby("sector")["momentum_12m"].median().reindex(list(sector_weights)).fillna(0.0)
        ranks = momentum.rank()
        centered = (ranks - (len(ranks) + 1) / 2) / ((len(ranks) - 1) / 2)  # -1 (weakest) .. +1 (strongest)
        sector_weights = {s: w * max(0.0, 1 + tilt * float(centered[s])) for s, w in sector_weights.items()}
    total = sum(sector_weights.values())
    if total <= 0:
        return {}
    invested = 1.0 - float(core.get("min_cash_buffer", 0.01))
    targets = {
        ticker: min(max_position, invested * weight / total / len(picks[sector]))
        for sector, weight in sector_weights.items()
        for ticker in picks[sector]
    }

    # Small accounts: hold only as many stocks as can each reach min_position_value, spread across industries.
    min_position = float(core.get("min_position_value", 0.0))
    if core_value is not None and min_position > 0:
        slots = max(1, int(core_value * invested // min_position))
        if slots < len(targets):
            targets = _small_account_targets(features, picks, sector_weights, held, slots, invested,
                                             core_value / slots)
    return targets


def _small_account_targets(
    features: pd.DataFrame, picks: dict[str, list[str]], sector_weights: dict[str, float], held: set[str],
    slots: int, invested: float, position_value: float,
) -> dict[str, float]:
    """Pick `slots` stocks: one per industry first (biggest industry weight first), then second picks, and so on.
    Current holdings are kept first, and stocks whose single share costs more than a position are skipped."""
    affordable = {
        sector: [t for t in tickers if t in held or float(features.at[t, "price"]) <= position_value]
        for sector, tickers in picks.items()
    }
    chosen = [t for tickers in affordable.values() for t in tickers if t in held][:slots]
    order = sorted(affordable, key=lambda sector: -sector_weights[sector])
    for rank in range(max((len(v) for v in affordable.values()), default=0)):
        for sector in order:
            if len(chosen) >= slots:
                break
            if rank < len(affordable[sector]) and affordable[sector][rank] not in chosen:
                chosen.append(affordable[sector][rank])
    return {ticker: invested / len(chosen) for ticker in chosen} if chosen else {}


def _rebalance_indices(dates: pd.DatetimeIndex, first: int, frequency: str) -> set[int]:
    freq = str(frequency).upper()
    active = dates[first:]
    if freq.startswith("M"):
        keys = active.year * 100 + active.month
    elif freq.startswith("Q"):
        keys = active.year * 10 + active.quarter
    elif freq.startswith("S"):
        keys = active.year * 10 + (active.month > 6)
    elif freq.startswith("A") or freq.startswith("Y"):
        keys = active.year
    else:
        raise ValueError("core.rebalance_frequency must be M, Q, S, or A")
    is_new_period = np.r_[True, keys[1:] != keys[:-1]]
    return {first + int(offset) for offset in np.flatnonzero(is_new_period)}
