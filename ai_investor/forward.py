"""AI forward test: a second fake-money account where the AI picks the stocks.

    python -m ai_investor.forward             # run once per trading day (after the close)
    python -m ai_investor.forward --status    # what the AI account holds
    python -m ai_investor.forward --dry-run   # show today's AI orders without saving anything

Everything except the stock picking is identical to the rules forward test (live_trader.py): same money
($1,000 start + monthly deposits), same 25/75 split, fees, whole shares, profit sweep, 15% stop-loss and
market-downtrend pause. So comparing the two accounts on the dashboard shows whether the AI picks better
stocks than the tested rules.

  Swing:     buy the AI's top-ranked large companies (S&P 500, TSX 60, your lists) and hold each for the AI's
             horizon (~1 month), unless the stop-loss hits first.
  Dividends: the AI's best dividend payer in each industry (half AI one-year view, half dividend quality),
             as many as the account can hold, re-checked every quarter. Dividends are reinvested.

Files: live/book_ai.json, journal_ai.csv, orders_ai.csv, swing_trades_ai.csv.
"""
from __future__ import annotations

import argparse
import math
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import live_trader
from ai_investor.advisor import _completed_closes, _load_model, rank_dividend_payers
from ai_investor.dataset import FEATURES, MARKET_TICKERS, build_dataset, load_ai_config, load_market_data, universe
from src.brokers import SimBroker
from src.config import load_config
from src.live_book import Book
from src.live_strategy import Market, build_market, costs, valuation
from src.swing import market_allows_entries, swing_slots, swing_universe
from src.universe import load_index_constituents

MODE = "ai"
EASTERN = ZoneInfo("America/Toronto")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI's fake-money forward test for today.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    paths = {name: live_trader.LIVE_DIR / f"{name}_{MODE}.{ext}" for name, ext in
             [("book", "json"), ("journal", "csv"), ("orders", "csv"), ("swing_trades", "csv")]}
    if args.status:
        live_trader.show_status(paths)
        return

    settings = load_config(Path("config/settings.toml"))
    ai = load_ai_config()
    models = {name: _load_model(name) for name in ("swing", "longterm")}
    now = datetime.now(EASTERN)
    print(f"[AI forward test] {now:%Y-%m-%d %H:%M} Toronto time. Loading market data...")
    data = _completed_closes(load_market_data(start=(pd.Timestamp.today() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")))
    market = build_market(settings, data["close"], data["dividends"], data["adjusted"])
    print(f"Last completed close: {market.last_date.date()}")

    _, sectors, _ = universe()
    snapshot = build_dataset(data, 1, {}, sectors, dates=pd.DatetimeIndex([market.last_date]))
    for name, bundle in models.items():
        snapshot[f"ai_{name}"] = bundle["model"].predict(snapshot[FEATURES])
        snapshot[f"ai_{name}_pct"] = snapshot[f"ai_{name}"].rank(pct=True)
    snapshot["price"] = snapshot["ticker"].map(lambda t: market.close(t))

    book = Book.load(paths["book"]) if paths["book"].exists() else None
    if book is None:
        live = settings.get("live", {})
        capital = float(live.get("capital", settings["starting_cash"]))
        fraction = float(settings["swing"].get("capital_fraction", 0.0)) if settings["swing"].get("enabled") else 0.0
        book = Book.new(MODE, capital, fraction, str(now.date()))
        book.last_dividend_date = str(market.last_date.date())
        book.last_deposit_month = str((market.last_date + pd.offsets.BDay(1)).date())[:7]
        print(f"Started the AI's account with C${capital:,.0f} (dividends C${book.core_cash:,.0f}, swing C${book.swing_cash:,.0f}).")

    planner = make_planner(snapshot, settings, ai, int(models["swing"]["horizon"]))
    run_args = Namespace(dry_run=args.dry_run, resync=False, deposit=None)
    live_trader.run_day(settings, settings.get("live", {}), MODE, book, SimBroker(), market, paths, now, run_args,
                        planner=planner)


def make_planner(snapshot: pd.DataFrame, settings: dict[str, Any], ai: dict[str, Any], horizon: int):
    """Order planner with the AI picking stocks; money handling and risk limits match the rules account."""
    swing_cfg, core = settings["swing"], settings.get("core", {})
    tested = set(load_index_constituents(ai["plan"].get("swing_indexes", ["sp500", "tsx60"]))["ticker"])
    tested |= set(swing_universe(settings, include_all=True)) | {t for ts in settings["sectors"].values() for t in ts}
    swing_ranked = snapshot[snapshot["ticker"].isin(tested) & ~snapshot["ticker"].isin(MARKET_TICKERS)
                            & (snapshot["sector"] != "Broad Market ETF")].sort_values("ai_swing", ascending=False)
    # Dividend picks also come from large companies: where the AI's one-year skill is strongest, and the same
    # kind of stocks the rules account holds, so the comparison is about picking skill.
    payers = rank_dividend_payers(snapshot[snapshot["ticker"].isin(tested)], settings, ai)
    payers = payers.sort_values("combined", ascending=False)

    def planner(config: dict[str, Any], book: Book, market: Market, rebalance_due: bool) -> list[dict[str, Any]]:
        orders: list[dict[str, Any]] = []
        k = len(market.dates) - 1

        def add(ticker: str, sleeve: str, action: str, quantity: float, reason: str) -> bool:
            quantity = math.floor(quantity + 1e-9) if action == "BUY" else quantity
            if quantity <= 0 or any(o["ticker"] == ticker and o["action"] != action for o in orders):
                return False
            for existing in orders:
                if existing["ticker"] == ticker and existing["sleeve"] == sleeve:
                    existing["quantity"] += quantity
                    return True
            orders.append({"ticker": ticker, "sleeve": sleeve, "action": action, "quantity": quantity,
                           "reason": reason, "setup": "ai" if sleeve == "swing" else None,
                           "signal_date": str(market.last_date.date()), "est_price_cad": market.close(ticker)})
            return True

        # --- swing exits: held for the AI's horizon, or the same stop-loss as the rules account.
        stop = float(swing_cfg.get("stop_loss", 0.15))
        for ticker, position in book.swing_positions.items():
            entry_k = int(np.searchsorted(market.dates, pd.Timestamp(position["entry_date"])))
            held = (k + 1) - entry_k
            price = market.close(ticker)
            if stop > 0 and price <= position["entry_price"] * (1 - stop):
                add(ticker, "swing", "SELL", position["shares"], "stop_loss")
            elif held >= horizon:
                add(ticker, "swing", "SELL", position["shares"], "ai_time_exit")

        # --- swing entries: the AI's top-ranked large companies (paused in market downtrends, like the rules).
        swing_value = valuation(book, market)["swing_value"]
        total_slots = swing_slots(swing_value, swing_cfg)
        selling = {o["ticker"] for o in orders if o["action"] == "SELL"}
        slots = total_slots - (len(book.swing_positions) - len(selling))
        market_col = market.col.get(swing_cfg.get("market_filter_ticker", ""))
        if slots > 0 and market_allows_entries(market.ind, k, market_col):
            slice_value = swing_value / total_slots
            cash = book.swing_cash  # only cash on hand; today's sale proceeds arrive tomorrow
            for row in swing_ranked.itertuples():
                if slots <= 0:
                    break
                if row.ticker in book.swing_positions or row.ticker in selling or row.ticker not in market.col:
                    continue
                fee, slip = costs(config, "swing", row.ticker)
                value = min(slice_value, cash - fee)
                if value < float(swing_cfg.get("min_trade_value", 500)):
                    break
                price = market.close(row.ticker) * (1 + slip)
                if price > 0 and add(row.ticker, "swing", "BUY", math.floor(value / price), "ai_pick"):
                    cash -= math.floor(value / price) * price + fee
                    slots -= 1

        # --- dividend account: the AI's best payer per industry, re-checked each quarter.
        values = valuation(book, market)
        core_value, core_cash = values["core_value"], book.core_cash
        invested = 1.0 - float(core.get("min_cash_buffer", 0.01))
        if rebalance_due or not book.core_targets:
            min_position = float(core.get("min_position_value", 200))
            holdings = max(1, int(core_value * invested // min_position)) if min_position > 0 else 24
            holdings = min(holdings, 3 * payers["sector"].nunique())
            position_value = core_value * invested / holdings
            held = {t for t, q in book.core_shares.items() if q > 0}
            still_good = set(payers.groupby("sector").head(2)["ticker"])  # keep holdings still near the top
            chosen = [t for t in sorted(held) if t in still_good][:holdings]
            for rank in range(3):  # one per industry first, then second picks, then third
                for row in payers.groupby("sector").nth(rank).sort_values("combined", ascending=False).itertuples():
                    if len(chosen) >= holdings:
                        break
                    if row.ticker not in chosen and row.ticker in market.col and row.price <= position_value:
                        chosen.append(row.ticker)
            book.core_targets = {t: invested / len(chosen) for t in chosen} if chosen else {}
            band = float(core.get("rebalance_band", 0.04))
            for ticker in sorted(held - set(book.core_targets)):
                add(ticker, "core", "SELL", book.core_shares[ticker], "ai_exit")
            for ticker, target in sorted(book.core_targets.items()):
                price = market.close(ticker)
                current = book.core_shares.get(ticker, 0.0) * price / core_value if core_value else 0.0
                if target - current > band:
                    fee, slip = costs(config, "core", ticker)
                    amount = min((target - current) * core_value, core_cash - fee)
                    quantity = math.floor(amount / (price * (1 + slip))) if amount > 0 else 0
                    if add(ticker, "core", "BUY", quantity, "ai_rebalance"):
                        core_cash -= quantity * price * (1 + slip) + fee

        # Every day: invest deposits and pooled cash in the most underweight AI holdings (same as the rules).
        if book.core_targets:
            planned = {o["ticker"]: o["quantity"] for o in orders if o["sleeve"] == "core" and o["action"] == "BUY"}
            investable = core_cash - float(core.get("min_cash_buffer", 0.01)) * core_value
            min_order = float(core.get("min_trade_value", 50))
            if investable >= float(core.get("min_deploy_cash", 1000)):
                shortfall = {t: w * core_value - (book.core_shares.get(t, 0.0) + planned.get(t, 0)) * market.close(t)
                             for t, w in book.core_targets.items()}
                for ticker, gap in sorted(shortfall.items(), key=lambda item: (-round(item[1], 6), item[0]))[:3]:
                    if gap <= 0 or investable < min_order:
                        break
                    fee, slip = costs(config, "core", ticker)
                    amount = min(investable, max(gap, min_order), core_cash - fee)
                    quantity = math.floor(amount / (market.close(ticker) * (1 + slip)))
                    if add(ticker, "core", "BUY", quantity, "deploy"):
                        spent = quantity * market.close(ticker) * (1 + slip) + fee
                        investable -= spent
                        core_cash -= spent
        return orders

    return planner


if __name__ == "__main__":
    main()
