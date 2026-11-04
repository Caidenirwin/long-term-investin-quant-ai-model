"""Check that the backtest is trustworthy.

    python audit_backtest.py

1. Ledger:       cash and share counts rebuilt from the trade log must match the engine's.
2. Look-ahead:   cutting the data off early must not change any decision made before the cut.
3. Data:         flag impossible price jumps and broken adjustments that could fake profits.
4. Swing edge:   breakout trades vs random trades of the same length in the same stocks.
5. Robustness:   other start years, higher costs, slower execution, currency-conversion fees.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.config import load_config, research_config
from src.data import download_price_data, load_cached_price_data
from src.swing import swing_universe

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str) -> None:
    RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def main() -> None:
    plan = load_config(Path("config/settings.toml"))
    base = research_config(plan)
    try:
        data = load_cached_price_data(base)
    except FileNotFoundError:
        data = download_price_data(base)

    print("\n1. LEDGER: does every dollar and share add up?")
    ledger_check(base, data)

    print("\n2. LOOK-AHEAD: do decisions change when future data is removed?")
    for cutoff in ["2016-06-30", "2022-03-31"]:
        lookahead_check(base, data, cutoff)

    print("\n3. DATA QUALITY")
    data_check(base, data)

    print("\n4. SWING EDGE: breakout trades vs random trades")
    random_entry_check(base, data)

    print("\n5. ROBUSTNESS")
    robustness_table(base, data)

    print("\n6. LIVE ENGINE: does live_trader.py make the same decisions as the backtest?")
    live_matches_backtest(base, data, label="C$100k research account")
    live_matches_backtest(plan, data, label=f"your plan: C${plan['starting_cash']:,.0f} + "
                                            f"C${plan.get('contributions', {}).get('monthly', 0):,.0f}/month")

    failed = [name for name, passed, _ in RESULTS if not passed]
    print("\n" + "=" * 80)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed" + (f". Failed: {failed}" if failed else "."))


# ---------------------------------------------------------------------------- 1. ledger


def ledger_check(base: dict[str, Any], data: tuple) -> None:
    # Interest off and dividends paid to cash, so every cash movement is in a log we can add up.
    config = copy.deepcopy(base)
    config["cash_interest_rate"] = 0.0
    config["dividends"]["mode"] = "smart"
    config["contributions"]["monthly"] = 500.0  # exercise the deposit bookkeeping too
    results = run_backtest(config, *data)
    trades, curve, summary = results.trades, results.equity_curve, results.performance_summary.iloc[0]
    dividends = results.dividend_income

    starting = float(config["starting_cash"])
    swing_start = starting * float(config["swing"]["capital_fraction"])
    swept = float(summary["swept_to_core"])

    for sleeve, start, sweep_sign in [("core", starting - swing_start, 1), ("swing", swing_start, -1)]:
        sleeve_trades = trades[trades["sleeve"] == sleeve]
        buys = sleeve_trades[sleeve_trades["action"] == "BUY"]
        sells = sleeve_trades[sleeve_trades["action"] == "SELL"]
        expected = (
            start
            + float(summary[f"deposits_{sleeve}"])
            - (buys["gross_value"] + buys["fee"]).sum()
            + (sells["gross_value"] - sells["fee"]).sum()
            + float(dividends[sleeve].sum())
            + sweep_sign * swept
        )
        actual = float(curve[f"{sleeve}_cash"].iloc[-1])
        check(f"{sleeve} cash reconciles", abs(expected - actual) < 1.0,
              f"rebuilt C${expected:,.2f} vs engine C${actual:,.2f}")

    core = trades[trades["sleeve"] == "core"]
    signed = np.where(core["action"] == "BUY", core["quantity"], -core["quantity"])
    rebuilt = pd.Series(signed, index=core["ticker"]).groupby(level=0).sum()
    held = results.positions[results.positions["sleeve"] == "core"].set_index("ticker")["quantity"]
    diff = (rebuilt.reindex(held.index.union(rebuilt.index), fill_value=0)
            - held.reindex(held.index.union(rebuilt.index), fill_value=0)).abs()
    check("core share counts reconcile", float(diff.max()) < 1e-6,
          f"largest mismatch {diff.max():.2e} shares across {len(held)} holdings")

    last_prices = data[0].ffill().iloc[-1]
    positions_value = (results.positions["quantity"] * results.positions["ticker"].map(last_prices)).sum()
    cash = curve["core_cash"].iloc[-1] + curve["swing_cash"].iloc[-1]
    check("ending equity = cash + holdings", abs(cash + positions_value - curve["equity"].iloc[-1]) < 1.0,
          f"C${cash + positions_value:,.2f} vs C${curve['equity'].iloc[-1]:,.2f}")

    lowest = min(curve["core_cash"].min(), curve["swing_cash"].min())
    check("cash never goes negative (no hidden leverage)", lowest > -1.0, f"lowest cash balance C${lowest:,.2f}")

    # What the swing sleeve made (value now + everything swept out - starting capital) must equal
    # closed-trade P&L plus the unrealized P&L of trades still open.
    swing_positions = results.positions[results.positions["sleeve"] == "swing"]
    open_pnl = float((swing_positions["market_value"] - swing_positions["cost_basis"]
                      + swing_positions["dividends_received"]).sum()) if not swing_positions.empty else 0.0
    closed_pnl = float(results.swing_trades["pnl"].sum()) if not results.swing_trades.empty else 0.0
    parked = float(summary.get("swing_park_realized", 0.0))  # realized gains + dividends on parked cash
    total_made = float(curve["swing_value"].iloc[-1]) + swept - swing_start - float(summary["deposits_swing"])
    check("swing P&L is fully explained by its trades", abs(total_made - closed_pnl - open_pnl - parked) < 1.0,
          f"sleeve made C${total_made:,.2f} = closed trades C${closed_pnl:,.2f} + open C${open_pnl:,.2f}"
          + (f" + parked cash C${parked:,.2f}" if parked else ""))


# ---------------------------------------------------------------------------- 2. look-ahead


def lookahead_check(base: dict[str, Any], data: tuple, cutoff: str) -> None:
    full = run_backtest(copy.deepcopy(base), *data)
    truncated_config = copy.deepcopy(base)
    truncated_config["end_date"] = cutoff
    truncated = run_backtest(truncated_config, *data)

    key = ["date", "sleeve", "ticker", "action"]
    a = full.trades[full.trades["date"] <= pd.Timestamp(cutoff)][key + ["quantity"]].reset_index(drop=True)
    b = truncated.trades[key + ["quantity"]].reset_index(drop=True)
    same_trades = len(a) == len(b) and (a[key].astype(str) == b[key].astype(str)).all().all() and \
        np.allclose(a["quantity"], b["quantity"], rtol=1e-9)
    check(f"identical trades up to {cutoff}", same_trades, f"{len(a)} trades with full data vs {len(b)} with data cut off")

    ea = full.equity_curve.set_index("date")["equity"].loc[:cutoff]
    eb = truncated.equity_curve.set_index("date")["equity"]
    check(f"identical equity curve up to {cutoff}", len(ea) == len(eb) and np.allclose(ea.values, eb.values),
          f"max difference C${(ea - eb).abs().max():,.6f}")


# ---------------------------------------------------------------------------- 3. data


def data_check(base: dict[str, Any], data: tuple) -> None:
    prices, dividends, signals = data
    start = pd.Timestamp(base["start_date"])
    raw = prices.loc[start:]
    returns = raw.pct_change(fill_method=None)
    jumps = returns.abs().stack()
    big = jumps[jumps > 0.40].sort_values(ascending=False)
    detail = ", ".join(f"{t} {d.date()} {r:+.0%}" for (d, t), r in big.head(6).items()) or "none"
    check("no impossible one-day price jumps (>40%)", len(big) <= 3, f"{len(big)} found: {detail}")

    # If Close and Adj Close disagree suddenly, a split or dividend was mis-adjusted.
    ratio = (signals / prices).loc[start:]
    ratio_jump = ratio.pct_change(fill_method=None).abs().stack()
    broken = ratio_jump[ratio_jump > 0.15]
    detail = ", ".join(f"{t} {d.date()}" for (d, t) in broken.index[:6]) or "none"
    check("price adjustments consistent (no split errors)", len(broken) == 0, f"{len(broken)} suspicious: {detail}")

    yields = []
    for ticker in dividends.columns:
        yearly = dividends[ticker].loc[start:].resample("YE").sum()
        avg_price = prices[ticker].loc[start:].resample("YE").mean()
        y = (yearly / avg_price).replace([np.inf, -np.inf], np.nan).dropna()
        if len(y) and y.max() > 0.25:
            yields.append(f"{ticker} {y.idxmax().year} {y.max():.0%}")
    check("no absurd dividend yields (>25% in a year)", not yields, ", ".join(yields[:6]) or "none")

    last = prices.ffill().iloc[-1]
    stale = [t for t in prices.columns if prices[t].dropna().index[-1] < prices.index[-1] - pd.Timedelta(days=10)]
    check("all tickers have current prices", not stale, f"stale: {stale}" if stale else f"{len(last)} tickers current")


# ---------------------------------------------------------------------------- 4. random entries


def random_entry_check(base: dict[str, Any], data: tuple, simulations: int = 500) -> None:
    results = run_backtest(copy.deepcopy(base), *data)
    trades = results.swing_trades
    if trades.empty:
        check("swing beats random entries", False, "no swing trades")
        return
    signals = data[2].ffill()
    universe = [t for t in swing_universe(base) if t in signals.columns]
    dates = signals.index
    index_of = {d: i for i, d in enumerate(dates)}

    # Strategy return per trade, measured the same way as the random trades (adjusted prices, before costs).
    actual = np.array([
        signals.at[row.exit_date, row.ticker] / signals.at[row.entry_date, row.ticker] - 1 for row in trades.itertuples()
    ])
    holds = np.array([index_of[row.exit_date] - index_of[row.entry_date] for row in trades.itertuples()])

    market = base["swing"].get("market_filter_ticker", "SPY")
    uptrend = (signals[market] > signals[market].rolling(200).mean()).to_numpy() if market in signals else None
    S = signals[universe].to_numpy()
    first = int(np.searchsorted(dates, pd.Timestamp(base["start_date"])))
    rng = np.random.default_rng(7)
    candidates = np.arange(first, len(dates) - holds.max() - 1)
    if uptrend is not None:
        candidates = candidates[uptrend[candidates]]  # same market regime the strategy trades in

    means = []
    for _ in range(simulations):
        entries = rng.choice(candidates, size=len(holds))
        cols = rng.integers(0, len(universe), size=len(holds))
        r = S[entries + holds, cols] / S[entries, cols] - 1
        means.append(np.nanmean(r))
    means = np.array(means)
    beat = float((means >= actual.mean()).mean())
    check(
        "breakout trades beat random trades",
        beat < 0.05,
        f"strategy avg {actual.mean():+.2%}/trade vs random {means.mean():+.2%} "
        f"(random did as well {beat:.1%} of {simulations} tries; under 5% = real edge)",
    )


# ---------------------------------------------------------------------------- 5. robustness


def robustness_table(base: dict[str, Any], data: tuple) -> None:
    def variant(**changes: Any) -> dict[str, Any]:
        config = copy.deepcopy(base)
        for key, value in changes.items():
            section, _, name = key.partition("__")
            if name:
                config[section][name] = value
            else:
                config[section] = value
        return config

    canadian_swing = {
        "swing__universe": [t for t in base["swing"]["universe"] if t.endswith(".TO")],
        "swing__stock_universe": [t for t in base["swing"]["stock_universe"] if t.endswith(".TO")],
    }
    scenarios = [
        ("Current settings", variant()),
        ("Start in 2015", variant(start_date="2015-01-01")),
        ("Start in 2018", variant(start_date="2018-01-01")),
        ("Double trading costs", variant(transaction_fee=9.90, slippage_bps=20, swing__slippage_bps=6)),
        ("Trade 2 days after signal", variant(swing__execution_lag_days=2)),
        ("1.5% FX fee on US trades", variant(fx_fee_bps=150)),
        ("1.5% FX fee, Canadian-only swing", variant(fx_fee_bps=150, **canadian_swing)),
        ("No swing (dividends only)", variant(swing__enabled=False)),
    ]
    rows = []
    for label, config in scenarios:
        summary = run_backtest(config, *data).performance_summary.iloc[0]
        rows.append({
            "scenario": label,
            "CAGR": f"{summary['CAGR']:.2%}",
            "benchmark": f"{summary['benchmark_CAGR']:.2%}",
            "beats_TSX_by": f"{summary['CAGR'] - summary['benchmark_CAGR']:+.2%}",
            "max_DD": f"{summary['max_drawdown']:.1%}",
            "swing_pnl": f"C${summary['swing_total_pnl']:,.0f}",
            "swing_avg_trade": f"{summary['swing_avg_return']:+.2%}" if pd.notna(summary["swing_avg_return"]) else "-",
            "_beat": summary["CAGR"] > summary["benchmark_CAGR"],
        })
    table = pd.DataFrame(rows)
    print(table.drop(columns="_beat").to_string(index=False))
    check("beats the TSX in every scenario", bool(table["_beat"].all()),
          f"{int(table['_beat'].sum())}/{len(table)} scenarios beat the benchmark")


# ---------------------------------------------------------------------------- 6. live vs backtest


def live_matches_backtest(base: dict[str, Any], data: tuple, label: str = "",
                          start: str = "2023-01-03", end: str = "2024-12-31") -> None:
    """Replay live_trader.py one day at a time (seeing only data up to each day) and compare with the backtest."""
    import contextlib
    import io
    import tempfile
    from argparse import Namespace
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import live_trader
    from src.brokers import SimBroker
    from src.live_book import Book
    from src.live_strategy import build_market

    config = copy.deepcopy(base)
    config["cash_interest_rate"] = 0.0  # the live book doesn't earn modelled interest
    config["start_date"], config["end_date"] = start, end
    prices, dividends, signals = data
    days = prices.loc[start:end].index
    first_signal_day = prices.index[prices.index < pd.Timestamp(start)][-1]

    book = Book.new("replay", float(config["starting_cash"]),
                    float(config["swing"]["capital_fraction"]), str(first_signal_day.date()))
    book.last_dividend_date = str(first_signal_day.date())
    book.last_deposit_month = str(days[0].date())[:7]  # first deposit the month after the start, as in the backtest
    args = Namespace(dry_run=False, resync=False, deposit=None)
    with tempfile.TemporaryDirectory() as folder:
        paths = {name: Path(folder) / f"{name}.csv" for name in ["journal", "orders", "swing_trades"]}
        paths["book"] = Path(folder) / "book.json"
        for signal_day in [first_signal_day, *days]:
            window = slice(signal_day - pd.Timedelta(days=1100), signal_day)
            market = build_market(config, prices.loc[window], dividends.loc[window], signals.loc[window])
            fill_day = prices.index[prices.index > signal_day]
            when = fill_day[0] if len(fill_day) else signal_day
            now = datetime(when.year, when.month, when.day, 12, tzinfo=ZoneInfo("America/Toronto"))
            with contextlib.redirect_stdout(io.StringIO()):
                live_trader.run_day(config, config.get("live", {}), "replay", book, SimBroker(), market, paths, now, args)
        live_journal = pd.read_csv(paths["journal"])
        live_trades = pd.read_csv(paths["swing_trades"]) if paths["swing_trades"].exists() else pd.DataFrame()

    backtest = run_backtest(config, *data)
    bt_equity = float(backtest.equity_curve["equity"].iloc[-1])
    last_row = live_journal.set_index("date").loc[str(days[-1].date())]
    # Live records a month's deposit the evening before that month's first trading day (so it can be invested
    # then); the backtest records it on the day. Compare after removing deposit-timing differences.
    deposit_offset = float(last_row.get("cumulative_deposits", 0.0)) - float(backtest.equity_curve["cumulative_deposits"].iloc[-1])
    live_equity = float(last_row["equity"]) - deposit_offset
    gap = live_equity / bt_equity - 1
    check(f"live engine ends at the same value as the backtest ({label})", abs(gap) < 0.02,
          f"{start} to {end}: live replay C${live_equity:,.0f} vs backtest C${bt_equity:,.0f} ({gap:+.2%})")

    bt_keys = {(r.ticker, str(pd.Timestamp(r.entry_date).date())) for r in backtest.swing_trades.itertuples()
               if pd.Timestamp(r.exit_date) <= days[-1]}
    live_keys = {(r.ticker, r.entry_date) for r in live_trades.itertuples()} if not live_trades.empty else set()
    overlap = len(bt_keys & live_keys) / max(len(bt_keys | live_keys), 1)
    check(f"live engine takes the same swing trades ({label})", overlap >= 0.8,
          f"{len(bt_keys & live_keys)} of {len(bt_keys | live_keys)} trades identical (ticker + entry day), {overlap:.0%}")


if __name__ == "__main__":
    main()
