"""Run the strategy forward, once per trading day.

    python live_trader.py                      # forward test (simulated fills at real closing prices)
    python live_trader.py --mode ibkr-paper    # Interactive Brokers PAPER account (IB Gateway must be running)
    python live_trader.py --mode ibkr-live --live   # real money (also needs allow_live = true in settings)
    python live_trader.py --dry-run            # show what it would do without changing anything
    python live_trader.py --status             # show the book without trading

Each run: settle yesterday's orders -> collect dividends -> sweep swing profits -> plan today's orders
-> safety checks -> send market-on-close orders (or simulate them).
Create the file live/PAUSE to stop all new orders instantly.
"""
from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.brokers import IBKRBroker, SimBroker
from src.config import load_config
from src.data import download_price_data
from src.live_book import Book
from src.live_strategy import (
    apply_fill, build_market, collect_dividends, plan_orders, sweep_profits, valuation,
)

LIVE_DIR = Path("live")
PAUSE_FILE = LIVE_DIR / "PAUSE"
EASTERN = ZoneInfo("America/Toronto")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-test or live-trade the strategy (run once per trading day).")
    parser.add_argument("--mode", choices=["sim", "ibkr-paper", "ibkr-live"])
    parser.add_argument("--live", action="store_true", help="Required (with allow_live = true) to trade real money.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; send nothing and save nothing.")
    parser.add_argument("--status", action="store_true", help="Show the current book and exit.")
    parser.add_argument("--resync", action="store_true", help="IBKR: adopt the broker's share counts into the dividend core.")
    parser.add_argument("--deposit", type=float, help="Record money you just added to the account, e.g. --deposit 200.")
    args = parser.parse_args()

    config = load_config(Path("config/settings.toml"))
    live = config.get("live", {})
    mode = args.mode or live.get("mode", "sim")
    paths = {name: LIVE_DIR / f"{name}_{mode}.{ext}" for name, ext in
             [("book", "json"), ("journal", "csv"), ("orders", "csv"), ("swing_trades", "csv")]}

    if mode == "ibkr-live" and not (args.live and live.get("allow_live", False)):
        sys.exit("Refusing to trade real money: pass --live AND set allow_live = true under [live] in settings.")

    if args.status:
        show_status(paths)
        return

    now = datetime.now(EASTERN)
    print(f"[{mode}] {now:%Y-%m-%d %H:%M} Toronto time. Downloading market data...")
    market = load_market(config, now)
    last = market.last_date
    print(f"Last completed close: {last.date()}")

    book = Book.load(paths["book"]) if paths["book"].exists() else None
    if book is None:
        capital = float(live.get("capital", config["starting_cash"]))
        swing_fraction = float(config["swing"].get("capital_fraction", 0.0)) if config["swing"].get("enabled") else 0.0
        book = Book.new(mode, capital, swing_fraction, str(now.date()))
        book.last_dividend_date = str(last.date())
        # The first monthly deposit comes the month after the first trades fill (as in the backtest).
        book.last_deposit_month = str((last + pd.offsets.BDay(1)).date())[:7]
        print(f"Started a new {mode} book with C${capital:,.0f} "
              f"(dividend core C${book.core_cash:,.0f}, swing C${book.swing_cash:,.0f}).")

    broker = SimBroker() if mode == "sim" else IBKRBroker(live, paper=(mode == "ibkr-paper"))
    try:
        run_day(config, live, mode, book, broker, market, paths, now, args)
    finally:
        broker.close()


def load_market(config: dict[str, Any], now: datetime):
    data_config = copy.deepcopy(config)
    data_config["start_date"] = (pd.Timestamp(now.date()) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    data_config["end_date"] = None
    data_config["warmup_years"] = 2
    prices, dividends, signals = download_price_data(data_config, save=False)
    # During the trading day Yahoo returns a partial bar for today; only completed closes are used.
    if prices.index[-1].date() == now.date() and now.time() < time(16, 30):
        prices, dividends, signals = prices.iloc[:-1], dividends.iloc[:-1], signals.iloc[:-1]
    return build_market(config, prices, dividends, signals)


def run_day(config, live, mode, book: Book, broker, market, paths, now: datetime, args, planner=plan_orders) -> None:
    """One trading day. planner decides the orders (the tested rules by default; the AI forward test passes its own)."""
    last = market.last_date
    order_log: list[dict[str, Any]] = []

    # 0. Deposits first: never lost on an early exit, and recorded before a broker's cash sync (so not counted twice).
    handle_deposits(config, book, market, broker, getattr(args, "deposit", None))

    # 1. Forward test: credit dividends first. They're paid on shares held at the start of the day, before
    #    the day's fills, and are available to that day's buys (same order as the backtest).
    observed = broker.positions()
    if observed is None:
        income = collect_dividends(config, book, market)
        if income:
            print(f"  dividends received: C${income:,.2f}")

    # 2. Settle orders from the previous run (they filled at the close of the session after their signal).
    still_working = []
    closed_trades = []
    # Sells settle first (and sweep swing profits), but buys may only use the cash each sleeve had before
    # the day's sales, exactly as in the backtest.
    spendable = {"core": book.core_cash, "swing": book.swing_cash}
    for order in sorted(book.pending, key=lambda o: o["action"] != "SELL"):
        fill_date = market.next_session(pd.Timestamp(order["signal_date"]))
        if fill_date is None:
            still_working.append(order)
            continue
        if observed is None:
            quantity = order["quantity"]
        else:
            delta = observed.get(order["ticker"], 0.0) - book.total_shares(order["ticker"])
            moved = delta if order["action"] == "BUY" else -delta
            quantity = min(max(moved, 0.0), order["quantity"])
        cash_before = book.core_cash if order["sleeve"] == "core" else book.swing_cash
        trade, quantity = apply_fill(config, book, order, quantity, market.close(order["ticker"], fill_date),
                                     fill_date, cash_limit=spendable[order["sleeve"]] if observed is None else None)
        if order["action"] == "BUY":
            spendable[order["sleeve"]] -= cash_before - (book.core_cash if order["sleeve"] == "core" else book.swing_cash)
        if trade:
            closed_trades.append(trade)
        status = "filled" if quantity >= order["quantity"] - 1e-9 else ("partial" if quantity > 0 else "not filled")
        order_log.append({**order, "status": status, "filled_quantity": quantity, "fill_date": str(fill_date.date())})
        print(f"  {status:10s} {order['action']} {quantity:g} {order['ticker']} ({order['sleeve']}, {order['reason']})")
    book.pending = still_working
    if closed_trades:
        sweep_profits(config, book)  # once, after all of the day's exits: net profit, as in the backtest
        append_csv(paths["swing_trades"], pd.DataFrame(closed_trades))
        for trade in closed_trades:
            print(f"  closed swing trade {trade['ticker']}: {trade['return']:+.2%} (C${trade['pnl']:,.0f})")

    if still_working:
        print("Orders from the last run are still working; they fill at the next close. Nothing new to do.")
        finish(book, market, paths, order_log, args.dry_run)
        return

    # 3. Broker accounts: trust the broker's cash and share counts (dividends, real commissions, DRIP shares).
    blocked = []
    if observed is not None:
        blocked = reconcile_with_broker(book, broker, observed, config, args.resync)

    # 3. Plan today's orders (once per completed close).
    values = valuation(book, market)
    print(f"Portfolio C${values['equity']:,.0f}  (dividend core C${values['core_value']:,.0f}, "
          f"swing C${values['swing_value']:,.0f}, swept so far C${book.cumulative_swept:,.0f})")

    if book.last_planned_date == str(last.date()):
        print("Already planned for this close. Run again after the next close.")
        finish(book, market, paths, order_log, args.dry_run)
        return
    if PAUSE_FILE.exists():
        print(f"PAUSED ({PAUSE_FILE} exists): no new orders. Delete the file to resume.")
        finish(book, market, paths, order_log, args.dry_run)
        return
    if blocked:
        print("Book and broker disagree, so no orders were placed:\n  " + "\n  ".join(blocked) +
              "\nFix it in the account, or run with --resync to adopt the broker's share counts.")
        finish(book, market, paths, order_log, args.dry_run)
        return

    quarter = f"{now.year}Q{(now.month - 1) // 3 + 1}"
    rebalance_due = book.last_rebalance_key != quarter
    targets_before = dict(book.core_targets)
    orders = planner(config, book, market, rebalance_due)
    orders, blocked_all = safety_checks(orders, live, values["equity"], broker, now)
    if blocked_all:
        # Leave the day un-planned so the next run (at the right time, or after fixing limits) retries it.
        book.core_targets = targets_before
        finish(book, market, paths, order_log, args.dry_run)
        return

    if args.dry_run:
        print_orders(orders, "WOULD SEND (dry run)")
        return

    for order in orders:
        order["id"] = book.next_order_id
        book.next_order_id += 1
    broker.submit(orders)
    accepted = [o for o in orders if not str(o.get("status", "")).startswith(("REJECTED", "skipped"))]
    book.pending = accepted
    book.last_planned_date = str(last.date())
    if rebalance_due:
        book.last_rebalance_key = quarter
    order_log += [{**o, "created": now.isoformat(timespec="seconds")} for o in orders]
    print_orders(orders, "SENT (market-on-close)" if broker.is_real else "SIMULATED (fill at next close)")
    finish(book, market, paths, order_log, dry_run=False)


def handle_deposits(config, book: Book, market, broker, manual: float | None) -> None:
    """Monthly deposits: automatic in the forward test; at a real broker you add the money and record it."""
    contributions = config.get("contributions", {})
    swing_on = config.get("swing", {}).get("enabled", False)
    fraction = float(contributions.get("swing_fraction", config["swing"].get("capital_fraction", 0.0))) if swing_on else 0.0
    if manual:
        book.deposit(manual, fraction)
        print(f"  recorded deposit C${manual:,.2f}")
    monthly = float(contributions.get("monthly", 0.0))
    # Orders from this run fill on the next session, so the deposit month follows that day (as in the backtest).
    month = str((market.last_date + pd.offsets.BDay(1)).date())[:7]
    if monthly <= 0 or book.last_deposit_month == month:
        return
    if broker.is_real:
        if not manual:
            print(f"  New month: add C${monthly:,.0f} at your broker, then run again with --deposit {monthly:g}")
            return
    else:
        book.deposit(monthly, fraction)
        print(f"  monthly deposit C${monthly:,.2f} added")
    book.last_deposit_month = month


def reconcile_with_broker(book: Book, broker, observed: dict[str, float], config, resync: bool) -> list[str]:
    """Match the book to the broker. Small extra core shares are reinvested dividends (DRIP)."""
    cash = broker.cash_cad()
    if cash is not None:
        difference = cash - (book.core_cash + book.swing_cash)
        if abs(difference) > 1:
            book.core_cash += difference  # dividends, interest, actual commissions land in the core
            print(f"  cash synced with broker: {difference:+,.2f} (dividends, interest, real commissions)")

    # Only tickers the strategy can trade are compared; anything else in the account is left alone.
    strategy_tickers = (
        {t for tickers in config["sectors"].values() for t in tickers}
        | set(config["swing"].get("universe", []))
        | set(config["swing"].get("stock_universe", []))
        | book.tickers()
    )
    problems = []
    for ticker in sorted(strategy_tickers):
        held = observed.get(ticker, 0.0)
        expected = book.total_shares(ticker)
        difference = held - expected
        if abs(difference) < 1e-6:
            continue
        core = book.core_shares.get(ticker, 0.0)
        if resync:
            swing = book.swing_positions.get(ticker, {}).get("shares", 0.0)
            book.core_shares[ticker] = max(held - swing, 0.0)
            print(f"  resynced {ticker}: core now {book.core_shares[ticker]:g} shares")
        elif difference > 0 and core > 0 and difference <= 0.05 * core:
            book.core_shares[ticker] = core + difference
            print(f"  {ticker}: +{difference:g} shares from dividend reinvestment")
        else:
            problems.append(f"{ticker}: broker has {held:g}, book expects {expected:g}")
    return problems


def safety_checks(
    orders: list[dict[str, Any]], live: dict[str, Any], equity: float, broker, now: datetime
) -> tuple[list[dict[str, Any]], bool]:
    """Returns (orders allowed, whether the whole run was blocked)."""
    max_order = float(live.get("max_order_value", 15000))
    max_orders = int(live.get("max_orders_per_run", 40))
    max_turnover = float(live.get("max_run_turnover", 1.0))
    safe = []
    for order in orders:
        value = order["quantity"] * order["est_price_cad"]
        if value > max_order:
            print(f"  BLOCKED {order['action']} {order['ticker']}: C${value:,.0f} is over max_order_value C${max_order:,.0f}")
            continue
        safe.append(order)
    total = sum(o["quantity"] * o["est_price_cad"] for o in safe)
    if len(safe) > max_orders or total > max_turnover * equity:
        print(f"  BLOCKED ALL: {len(safe)} orders worth C${total:,.0f} breaks max_orders_per_run={max_orders} "
              f"or max_run_turnover={max_turnover:.0%} of equity. Nothing sent.")
        return [], True
    if broker.is_real and safe:
        # MOC orders must reach the exchange before the closing cutoff (TSX 3:40 pm, NYSE 3:50 pm Eastern).
        weekday = now.weekday() < 5
        if not (weekday and time(9, 30) <= now.time() <= time(15, 35)):
            print("  BLOCKED ALL: market-on-close orders must be sent on a trading day between 9:30 am and "
                  "3:35 pm Toronto time. Schedule this script for about 3:00 pm.")
            return [], True
    return safe, False


def print_orders(orders: list[dict[str, Any]], heading: str) -> None:
    if not orders:
        print("No new orders today.")
        return
    print(f"\n{heading}:")
    for o in orders:
        value = o["quantity"] * o["est_price_cad"]
        status = f"  [{o['status']}]" if o.get("status") else ""
        print(f"  {o['action']:4s} {o['quantity']:>6g} {o['ticker']:10s} ~C${value:>9,.0f}  "
              f"{o['sleeve']:5s} {o['reason']}{status}")


def finish(book: Book, market, paths: dict[str, Path], order_log: list[dict[str, Any]], dry_run: bool) -> None:
    if dry_run:
        return
    book.save(paths["book"])
    if order_log:
        append_csv(paths["orders"], pd.DataFrame(order_log))
    values = valuation(book, market)
    benchmark = market.signal_prices["XIC.TO"].dropna() if "XIC.TO" in market.signal_prices else None
    row = pd.DataFrame([{
        "date": str(market.last_date.date()), **{k: round(v, 2) for k, v in values.items()},
        "core_cash": round(book.core_cash, 2), "swing_cash": round(book.swing_cash, 2),
        "cumulative_dividends": round(book.cumulative_dividends, 2), "cumulative_swept": round(book.cumulative_swept, 2),
        "cumulative_deposits": round(book.cumulative_deposits, 2),
        "benchmark_close": round(float(benchmark.iloc[-1]), 4) if benchmark is not None and len(benchmark) else None,
    }])
    journal = pd.read_csv(paths["journal"]) if paths["journal"].exists() else pd.DataFrame()
    if not journal.empty:
        journal = journal[journal["date"] != row.at[0, "date"]]
    pd.concat([journal, row], ignore_index=True).to_csv(paths["journal"], index=False)


def append_csv(path: Path, frame: pd.DataFrame) -> None:
    """Append rows, keeping one consistent header even when new rows have extra columns (e.g. fill details)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)


def show_status(paths: dict[str, Path]) -> None:
    if not paths["book"].exists():
        print("No book yet. Run live_trader.py once to start.")
        return
    book = Book.load(paths["book"])
    print(f"{book.mode} book started {book.created} with C${book.capital:,.0f}")
    print(f"Cash: core C${book.core_cash:,.2f}, swing C${book.swing_cash:,.2f}; swept C${book.cumulative_swept:,.2f}; "
          f"dividends C${book.cumulative_dividends:,.2f}")
    print("Core:", ", ".join(f"{t} {q:g}" for t, q in sorted(book.core_shares.items())) or "none")
    print("Swing:", ", ".join(f"{t} {p['shares']:g} ({p['setup']}, since {p['entry_date']})"
                             for t, p in book.swing_positions.items()) or "none")
    print("Pending:", ", ".join(f"{o['action']} {o['quantity']:g} {o['ticker']}" for o in book.pending) or "none")
    if paths["journal"].exists():
        print(pd.read_csv(paths["journal"]).tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
