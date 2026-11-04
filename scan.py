"""Daily research scanner for quick swing trades.

Uses exactly the same setup rules as the backtest (src/swing.py), on fresh data, in each stock's own currency.

    python scan.py                                # scan the configured universe
    python scan.py --tickers SHOP.TO NVDA AAPL    # research specific tickers
    python scan.py --positions my_swing_positions.csv   # also check exits on trades you hold
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime
from datetime import time as dt_time
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import load_config
from src.data import download_with_retry
from src.universe import company_info, load_index_constituents
from src.swing import (
    entry_setup,
    exit_reason,
    market_allows_entries,
    setup_track_record,
    swing_indicators,
    swing_universe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find today's swing-trade setups.")
    parser.add_argument("--tickers", nargs="+", help="Research these tickers instead of the configured universe.")
    parser.add_argument("--positions", default="my_swing_positions.csv",
                        help="CSV of open trades (ticker,entry_date,entry_price,setup) to check for exits.")
    parser.add_argument("--no-earnings", action="store_true", help="Skip the (slower) earnings-date lookup.")
    parser.add_argument("--no-discover", action="store_true",
                        help="Only scan your configured list, not the whole S&P 500 / TSX 60 (much faster).")
    parser.add_argument("--industry", help='Break down one industry in detail, e.g. --industry Financials or "health"')
    args = parser.parse_args()
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # hide per-ticker 404 noise

    config = load_config(Path("config/settings.toml"))
    swing = config.get("swing", {})
    scanner = config.get("scanner", {})
    positions = _load_positions(Path(args.positions))

    your_list = swing_universe(config, include_all=True) + list(scanner.get("extra_tickers", []))
    tickers = args.tickers or your_list
    indexes = list(scanner.get("discover_indexes", ["sp500", "tsx60"]))
    info = company_info(indexes)  # names and industries, also used when discovery is off
    discover = not args.tickers and not args.no_discover and scanner.get("discover", True)
    if discover:
        constituents = load_index_constituents(indexes)
        tickers = list(dict.fromkeys(tickers + constituents["ticker"].tolist()))
        print(f"Discovery on: scanning {len(tickers)} stocks ({' + '.join(indexes)} plus your list).")

    market = swing.get("market_filter_ticker", "")
    download_list = sorted(set(tickers) | set(positions["ticker"]) | ({market} if market else set()))
    print(f"Downloading {len(download_list)} tickers (a big scan takes a few minutes)...")
    close, adjusted, volume = _download(download_list, int(scanner.get("history_years", 5)))
    # During market hours Yahoo includes today's unfinished bar. Signals use completed closes only,
    # the same as live_trader.py and the backtest, so the scanner and the trader always agree.
    now = datetime.now(ZoneInfo("America/Toronto"))
    if close.index[-1].date() == now.date() and now.time() < dt_time(16, 30):
        close, adjusted, volume = close.iloc[:-1], adjusted.iloc[:-1], volume.iloc[:-1]
        print("Market is open: using yesterday's completed close (today's bar is unfinished).")

    ind = swing_indicators(adjusted, config)
    columns = list(adjusted.columns)
    col = {ticker: j for j, ticker in enumerate(columns)}
    k = len(adjusted) - 1
    as_of = adjusted.index[-1].date()
    market_ok = market_allows_entries(ind, k, col.get(market))

    print(f"\nData as of {as_of}.  Setups: {', '.join(swing.get('setups', ['pullback']))}")
    if market:
        state = "ABOVE" if market_ok else "BELOW"
        print(f"Market filter: {market} is {state} its {swing.get('trend_ma_days', 200)}-day average"
              + ("" if market_ok else "  -> the backtest takes NO new swing trades in this regime"))

    scanned = [ticker for ticker in tickers if ticker in col]
    sector_of = {ticker: info.get(ticker, {}).get("sector") or "Other / Unknown" for ticker in scanned}
    if args.industry:
        wanted = args.industry.lower()
        scanned = [t for t in scanned if wanted in sector_of[t].lower()]
        matched = sorted({sector_of[t] for t in scanned})
        if not scanned:
            sys.exit(f"No stocks in an industry matching '{args.industry}'. "
                     f"Industries: {', '.join(sorted(set(sector_of.values())))}")
        print(f"Industry filter: {', '.join(matched)} ({len(scanned)} stocks)")
        sector_of = {t: sector_of[t] for t in scanned}
    rows = [_research(ticker, col[ticker], ind, k, close, adjusted, volume, config) for ticker in scanned]
    ideas = pd.DataFrame([row for row in rows if row is not None])

    # Snapshot of every stock scanned (not just the ones signalling), for the dashboard's industry explorer.
    status_of = {r["ticker"]: ("Buy" if r["status"] == "1 BUY" else "Watch") for r in rows if r is not None}
    setup_of = {r["ticker"]: r["setup"] for r in rows if r is not None}
    snapshot = pd.DataFrame([_snapshot_row(t, col[t], ind, k, close, volume, info, sector_of, status_of, setup_of)
                             for t in scanned])
    Path("outputs").mkdir(exist_ok=True)
    if not args.industry:
        snapshot.to_csv(Path(f"outputs/scan_{as_of}_all.csv"), index=False)
    else:
        _print_industry_detail(snapshot)

    if ideas.empty:
        print("\nNo buy signals or near-signals today.")
    else:
        ideas["new"] = ~ideas["ticker"].isin(your_list) if discover else False
        ideas["name"] = ideas["ticker"].map(lambda t: info.get(t, {}).get("name", ""))
        ideas["sector"] = ideas["ticker"].map(sector_of)
        buys = ideas[ideas["status"] == "1 BUY"].sort_values("rank_score", ascending=False)
        watch = ideas[ideas["status"] == "2 WATCH"].sort_values("priority", ascending=False)

        breakdown = _industry_breakdown(sector_of, buys, watch)
        _print_breakdown(breakdown)

        # Spread the shortlist across industries: at most max_per_sector picks from any one sector.
        max_buys = int(scanner.get("max_buy_ideas", 15))
        max_per_sector = int(scanner.get("max_per_sector", 3))
        shortlist = buys[buys.groupby("sector").cumcount() < max_per_sector].head(max_buys)
        if len(buys) > len(shortlist):
            print(f"\n{len(buys)} stocks triggered today; showing the top {len(shortlist)}, at most {max_per_sector} "
                  "per industry (all of them are in the saved CSV).")
        shown = pd.concat([shortlist, watch.head(int(scanner.get("max_watchlist", 10)))])

        if not args.no_earnings:
            shown = _add_earnings(shown, set(swing.get("universe", [])), int(scanner.get("earnings_buffer_days", 5)))
        _print_ideas(shown.drop(columns=["priority", "rank_score"]), market_ok)

        if args.industry:
            print("\n(Industry view only: today's saved scan results were left unchanged.)")
            return
        # Save every buy signal (not only the shortlist), marking which ones were shown.
        ideas = pd.concat([shown.assign(shortlisted=True), buys[~buys.index.isin(shown.index)].assign(shortlisted=False)])
        output = Path(f"outputs/scan_{as_of}.csv")
        ideas.drop(columns=["priority", "rank_score"]).to_csv(output, index=False)
        breakdown.to_csv(Path(f"outputs/scan_{as_of}_industries.csv"), index=False)
        meta = {
            "as_of": str(as_of),
            "market": market,
            "market_ok": bool(market_ok),
            "scanned": len(scanned),
            "buy_signals": len(buys),
            "discover": bool(discover),
            "setups": swing.get("setups", ["pullback"]),
        }
        Path(f"outputs/scan_{as_of}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"\nSaved {output} and outputs/scan_{as_of}_industries.csv")

    if not positions.empty:
        _check_positions(positions, ind, k, close, adjusted, col, swing)


# ---------------------------------------------------------------------------- research


def _research(
    ticker: str,
    j: int,
    ind: dict[str, np.ndarray],
    k: int,
    close: pd.DataFrame,
    adjusted: pd.DataFrame,
    volume: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    swing = config.get("swing", {})
    scanner = config.get("scanner", {})
    setups = swing.get("setups", ["pullback"])
    last_close = close[ticker].iloc[-1]
    if not last_close > 0 or np.isnan(ind["trend_ma"][k, j]):
        return None

    signal = entry_setup(ind, k, j, swing)
    if signal:
        status, setup, priority = "1 BUY", signal[0], signal[1]
    else:
        status, setup, priority = "2 WATCH", *_near_signal(ind, k, j, swing, setups)
        if setup is None:
            return None

    dollar_volume = float((close[ticker] * volume[ticker]).tail(20).mean())
    if dollar_volume < float(scanner.get("min_dollar_volume", 0)):
        return None

    # Indicators are on adjusted prices; convert levels back to today's quoted price.
    to_quote = last_close / ind["price"][k, j]
    stop_loss = float(swing.get("stop_loss", 0.0))
    if setup == "breakout":
        exit_level = ind["trail_low"][k, j] * to_quote
        exit_rule = f"sell on close < {exit_level:.2f} ({swing.get('trail_days', 10)}-day low, rises as it trails)"
    else:
        exit_level = ind["exit_ma"][k, j] * to_quote
        exit_rule = f"sell on close > {exit_level:.2f} ({swing.get('exit_ma_days', 10)}-day avg)"

    history = setup_track_record(ind, j, swing)
    history = history[history["setup"] == setup]
    position_value = float(scanner.get("account_size", 25000)) / int(swing.get("max_positions", 3))

    # Rank buys by signal strength plus the setup's past results on this stock,
    # trusting the track record more the more trades it has (full weight at 10+).
    track_edge = history["return"].mean() * min(len(history), 10) / 10 if len(history) else 0.0

    return {
        "status": status,
        "ticker": ticker,
        "setup": setup,
        "priority": priority,
        "rank_score": priority + 5 * track_edge,
        "close": round(last_close, 2),
        "rsi": round(ind["rsi"][k, j], 1),
        "pct_above_200d": round(ind["price"][k, j] / ind["trend_ma"][k, j] - 1, 3),
        "momentum_3m": round(ind["momentum"][k, j], 3),
        "stop_price": round(last_close * (1 - stop_loss), 2) if stop_loss > 0 else None,
        "exit_rule": exit_rule,
        "shares": math.floor(position_value / last_close),
        "hist_trades": len(history),
        "hist_win_rate": round((history["return"] > 0).mean(), 2) if len(history) else None,
        "hist_avg_return": round(history["return"].mean(), 4) if len(history) else None,
        "hist_avg_hold_days": round(history["hold_days"].mean(), 1) if len(history) else None,
        "dollar_volume_m": round(dollar_volume / 1e6, 1),
        "warning": "",
    }


def _snapshot_row(
    ticker: str, j: int, ind: dict[str, np.ndarray], k: int, close: pd.DataFrame, volume: pd.DataFrame,
    info: dict[str, dict[str, str]], sector_of: dict[str, str], status_of: dict[str, str], setup_of: dict[str, str],
) -> dict[str, Any]:
    price, trend = ind["price"][k, j], ind["trend_ma"][k, j]
    return {
        "ticker": ticker,
        "name": info.get(ticker, {}).get("name", ""),
        "sector": sector_of[ticker],
        "status": status_of.get(ticker, ""),
        "setup": setup_of.get(ticker, ""),
        "close": round(float(close[ticker].iloc[-1]), 2),
        "momentum_3m": round(float(ind["momentum"][k, j]), 4) if np.isfinite(ind["momentum"][k, j]) else None,
        "pct_above_200d": round(float(price / trend - 1), 4) if np.isfinite(trend) and trend > 0 else None,
        "uptrend": bool(price > trend and ind["fast_ma"][k, j] > trend) if np.isfinite(trend) else False,
        "rsi": round(float(ind["rsi"][k, j]), 1) if np.isfinite(ind["rsi"][k, j]) else None,
        "dollar_volume_m": round(float((close[ticker] * volume[ticker]).tail(20).mean()) / 1e6, 1),
    }


def _print_industry_detail(snapshot: pd.DataFrame) -> None:
    """Terminal version of the dashboard's industry drill-down."""
    for sector, group in snapshot.groupby("sector"):
        group = group.sort_values("momentum_3m", ascending=False)
        uptrend = group["uptrend"].mean()
        print(f"\n{sector.upper()}: {len(group)} stocks · {int((group['status'] == 'Buy').sum())} buy signals · "
              f"{uptrend:.0%} in an uptrend · median 3-month move {group['momentum_3m'].median():+.1%}")
        print(f"  {'Ticker':9s} {'Company':32s} {'Signal':7s} {'3m':>7s} {'vs 200d':>8s} {'RSI':>5s}")
        for _, row in group.iterrows():
            momentum = f"{row['momentum_3m']:+.1%}" if pd.notna(row["momentum_3m"]) else "-"
            above = f"{row['pct_above_200d']:+.1%}" if pd.notna(row["pct_above_200d"]) else "-"
            rsi = f"{row['rsi']:.0f}" if pd.notna(row["rsi"]) else "-"
            print(f"  {row['ticker']:9s} {str(row['name'])[:32]:32s} {row['status'] or '-':7s} {momentum:>7s} {above:>8s} {rsi:>5s}")


def _industry_breakdown(sector_of: dict[str, str], buys: pd.DataFrame, watch: pd.DataFrame) -> pd.DataFrame:
    """Per industry: how many stocks were scanned, how many are signalling, and the best idea.

    "% signalling" is industry breadth: a high number means the whole industry is moving, not one stock.
    """
    scanned = pd.Series(sector_of).value_counts()
    rows = []
    for sector, count in scanned.items():
        sector_buys = buys[buys["sector"] == sector]
        top = sector_buys.iloc[0] if len(sector_buys) else None
        rows.append(
            {
                "industry": sector,
                "scanned": int(count),
                "buy_signals": len(sector_buys),
                "pct_signalling": len(sector_buys) / count,
                "watchlist": int((watch["sector"] == sector).sum()),
                "avg_3m_momentum": sector_buys["momentum_3m"].mean() if len(sector_buys) else np.nan,
                "top_pick": f"{top['ticker']} ({top['name'] or top['setup']})" if top is not None else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["pct_signalling", "buy_signals"], ascending=False)


def _print_breakdown(breakdown: pd.DataFrame) -> None:
    print("\nINDUSTRY BREAKDOWN (hottest first: share of the industry's stocks with a buy signal)")
    print(f"  {'Industry':24s} {'Scanned':>7s} {'Buys':>5s} {'% buy':>6s} {'':10s} {'Watch':>5s} {'Avg 3m':>7s}  Top pick")
    for _, row in breakdown.iterrows():
        momentum = f"{row['avg_3m_momentum']:+.1%}" if pd.notna(row["avg_3m_momentum"]) else "-"
        bar = "#" * round(row["pct_signalling"] * 10)
        print(
            f"  {row['industry'][:24]:24s} {row['scanned']:7d} {row['buy_signals']:5d} {row['pct_signalling']:6.0%} "
            f"{bar:10s} {row['watchlist']:5d} {momentum:>7s}  {row['top_pick'][:40]}"
        )


def _add_earnings(shown: pd.DataFrame, etfs: set[str], buffer_days: int) -> pd.DataFrame:
    shown = shown.copy()
    shown["next_earnings"] = [None if ticker in etfs else _next_earnings(ticker) for ticker in shown["ticker"]]
    days_to = shown["next_earnings"].map(lambda d: (d - date.today()).days if isinstance(d, date) else None)
    shown["warning"] = np.where(days_to.notna() & (days_to <= buffer_days), "EARNINGS SOON", shown["warning"])
    return shown


def _near_signal(
    ind: dict[str, np.ndarray], k: int, j: int, swing: dict[str, Any], setups: list[str]
) -> tuple[str | None, float]:
    """Close to triggering: worth watching for tomorrow. Returns (setup, priority) or (None, 0)."""
    price = ind["price"][k, j]
    if not price > ind["trend_ma"][k, j]:
        return None, 0.0
    entry_rsi = float(swing.get("entry_rsi", 10))
    if "pullback" in setups and ind["rsi"][k, j] < entry_rsi + 15:
        return "pullback", 1 - (ind["rsi"][k, j] - entry_rsi) / 15
    high = ind["breakout_high"][k, j]
    if "breakout" in setups and ind["fast_ma"][k, j] > ind["trend_ma"][k, j] and price >= 0.98 * high:
        return "breakout", float(price / high)
    return None, 0.0


def _check_positions(
    positions: pd.DataFrame,
    ind: dict[str, np.ndarray],
    k: int,
    close: pd.DataFrame,
    adjusted: pd.DataFrame,
    col: dict[str, int],
    swing: dict[str, Any],
) -> None:
    print("\nYour open swing trades:")
    for _, position in positions.iterrows():
        ticker = position["ticker"]
        if ticker not in col:
            print(f"  {ticker:10s} no data")
            continue
        j = col[ticker]
        held_days = int((adjusted.index > pd.Timestamp(position["entry_date"])).sum())
        last = close[ticker].iloc[-1]
        setup = position.get("setup") or "pullback"
        reason = exit_reason(ind, k, j, setup, float(position["entry_price"]), last, held_days, swing)
        change = last / float(position["entry_price"]) - 1
        action = f"SELL ({reason})" if reason else "HOLD"
        print(f"  {ticker:10s} {setup:9s} held {held_days:3d}d  {change:+.1%}  -> {action}")


# ---------------------------------------------------------------------------- data


def _download(tickers: list[str], years: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = download_with_retry(tickers, start, end, threads=True)
    close = data["Close"].reindex(columns=tickers)
    keep = [ticker for ticker in tickers if close[ticker].notna().sum() > 0]
    missing = sorted(set(tickers) - set(keep))
    if missing:
        print(f"Warning: no data for {missing}")
    close = close[keep].ffill()
    adjusted = data["Adj Close"].reindex(columns=keep).ffill()
    volume = data["Volume"].reindex(columns=keep).fillna(0.0)
    return close, adjusted, volume


def _next_earnings(ticker: str) -> date | None:
    try:
        calendar = yf.Ticker(ticker).calendar
    except Exception:
        return None
    dates = calendar.get("Earnings Date") if isinstance(calendar, dict) else None
    upcoming = [d.date() if isinstance(d, datetime) else d for d in (dates or [])]
    upcoming = [d for d in upcoming if isinstance(d, date) and d >= date.today()]
    return min(upcoming) if upcoming else None


def _load_positions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "entry_date", "entry_price", "setup"])
    positions = pd.read_csv(path)
    missing = {"ticker", "entry_date", "entry_price"} - set(positions.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return positions


def _print_ideas(ideas: pd.DataFrame, market_ok: bool) -> None:
    pd.set_option("display.width", 200)
    labels = [
        ("1 BUY", "BUY SIGNALS (ranked by signal strength + past results of this setup on the stock)"),
        ("2 WATCH", "WATCHLIST (close to triggering)"),
    ]
    for status, label in labels:
        group = ideas[ideas["status"] == status]
        if group.empty:
            continue
        if status == "1 BUY" and not market_ok:
            label += "  [market filter is OFF: backtest would skip these]"
        print(f"\n{label}")
        for _, row in group.iterrows():
            history = (
                f"{row['hist_trades']} past trades, {row['hist_win_rate']:.0%} wins, avg {row['hist_avg_return']:+.2%}"
                if row["hist_trades"] else "no past trades in history"
            )
            earnings = f"  earnings {row['next_earnings']}" if pd.notna(row.get("next_earnings")) else ""
            warning = f"  !! {row['warning']}" if row["warning"] else ""
            stop = f"stop {row['stop_price']}" if pd.notna(row["stop_price"]) else "no stop"
            tag = "NEW " if row.get("new") else ""
            company = f"{row['name']} ({row['sector']})" if row.get("name") else ""
            if tag or company:
                print(f"  {tag}{company}")
            print(
                f"  {row['ticker']:10s} {row['setup']:9s} close {row['close']:>9.2f}  RSI {row['rsi']:5.1f}  "
                f"3m {row['momentum_3m']:+.1%}  {stop}  ~{row['shares']} sh\n"
                f"  {'':10s} {row['exit_rule']}\n"
                f"  {'':10s} {history}  |  ${row['dollar_volume_m']}M/day{earnings}{warning}"
            )


if __name__ == "__main__":
    main()
