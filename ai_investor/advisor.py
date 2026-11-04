"""Your AI's daily research report and plan.

    python -m ai_investor.advisor              # full report (downloads recent prices, reads news)
    python -m ai_investor.advisor --no-news    # skip the news check (faster)

What it does:
  1. Market check: is the market trending up, how many stocks are in uptrends, how volatile it is.
  2. Swing plan: the tested breakout rules find today's setups; the AI ranks them by how likely each is to beat
     other stocks over the next month. Sized for your swing account.
  3. Long-term plan: dividend payers ranked by the AI's one-year model within each industry, so the plan stays
     spread across industries. Sized for your dividend account.
  4. News check: recent headlines for the top ideas and your holdings, scored locally, with red flags.
  5. Writes everything to ai_investor/reports/plan_<date>.md (and .json).

It advises; it doesn't place trades. live_trader.py keeps trading the tested rules.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from ai_investor import news
from ai_investor.dataset import FEATURES, MARKET_TICKERS, ROOT, build_dataset, load_ai_config, load_market_data, universe
from ai_investor.train import MODELS_DIR, REPORTS_DIR
from src.config import load_config
from src.live_book import Book
from src.swing import entry_setup, market_allows_entries, swing_indicators, swing_slots, swing_universe
from src.universe import load_index_constituents


def main() -> None:
    parser = argparse.ArgumentParser(description="AI research report and plan.")
    parser.add_argument("--no-news", action="store_true")
    args = parser.parse_args()

    models = {name: _load_model(name) for name in ("swing", "longterm")}
    ai = load_ai_config()
    settings = load_config(Path("config/settings.toml"))
    _, sectors, names = universe()

    start = (pd.Timestamp.today() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    data = load_market_data(start=start)
    data = _completed_closes(data)
    as_of = data["adjusted"].index[-1]
    print(f"Research as of the {as_of.date()} close.")

    snapshot = build_dataset(data, 1, {}, sectors, dates=pd.DatetimeIndex([as_of]))
    for name, bundle in models.items():
        snapshot[f"ai_{name}"] = bundle["model"].predict(snapshot[FEATURES])
        snapshot[f"ai_{name}_pct"] = snapshot[f"ai_{name}"].rank(pct=True).clip(upper=0.99)  # never "better than 100%"
    snapshot["name"] = snapshot["ticker"].map(lambda t: names.get(t, t))
    snapshot["price"] = snapshot["ticker"].map(lambda t: float(data["close"][t].dropna().iloc[-1]))

    account = _account(settings, data)
    market = _market_check(data, snapshot, settings)
    swing = _swing_plan(snapshot, data, settings, ai, account, models["swing"])
    if not market["swing_allowed"] and not swing["picks"].empty:
        swing["picks"]["action"] = "WAIT (market filter off)"  # the tested rules don't open trades in downtrends
    longterm = _longterm_plan(snapshot, settings, ai, account, models["longterm"])

    news_table = pd.DataFrame()
    if not args.no_news:
        tickers = list(dict.fromkeys(
            swing["picks"]["ticker"].tolist() + longterm["buy_list"]["ticker"].tolist() + account["holdings"]
        ))[: int(ai["plan"]["news_candidates"]) + len(account["holdings"])]
        print(f"Reading news for {len(tickers)} stocks...")
        news_table = news.summarize(news.score_headlines(news.fetch_headlines(tickers, names=names)))
        for plan in (swing["picks"], longterm["buy_list"]):
            _apply_news(plan, news_table)

    report = _write_report(as_of, market, swing, longterm, news_table, account, models)
    _print_summary(market, swing, longterm, report)


# ---------------------------------------------------------------------------- inputs


def _load_model(name: str) -> dict[str, Any]:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise SystemExit(f"No trained {name} model yet. Run:  python -m ai_investor.train")
    return joblib.load(path)


def _completed_closes(data: dict[str, Any]) -> dict[str, Any]:
    """During market hours, today's bar is unfinished; use yesterday's close like the rest of the system."""
    now = datetime.now(ZoneInfo("America/Toronto"))
    if data["adjusted"].index[-1].date() == now.date() and now.time() < time(16, 30):
        return {k: (v.iloc[:-1] if isinstance(v, pd.DataFrame) else v) for k, v in data.items()}
    return data


def _account(settings: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Your account from the forward-test/live book if there is one, otherwise the plan in settings.toml."""
    mode = settings.get("live", {}).get("mode", "sim")
    path = Path("live") / f"book_{mode}.json"
    close = data["close"]

    def price(ticker: str) -> float:
        return float(close[ticker].dropna().iloc[-1]) if ticker in close else 0.0

    if path.exists():
        book = Book.load(path)
        core = book.core_cash + sum(q * price(t) for t, q in book.core_shares.items())
        swing = book.swing_cash + sum(p["shares"] * price(t) for t, p in book.swing_positions.items())
        # Orders still waiting to fill count as owned (and their cash as spent), so ideas don't double up.
        pending_buys = [o for o in book.pending if o["action"] == "BUY"]
        committed = {s: sum(o["quantity"] * o["est_price_cad"] for o in pending_buys if o["sleeve"] == s)
                     for s in ("core", "swing")}
        core_owned = {t for t, q in book.core_shares.items() if q > 0} | {o["ticker"] for o in pending_buys if o["sleeve"] == "core"}
        swing_owned = set(book.swing_positions) | {o["ticker"] for o in pending_buys if o["sleeve"] == "swing"}
        return {"source": f"your {mode} book" + (f" ({len(book.pending)} orders waiting to fill)" if book.pending else ""),
                "core_value": core, "swing_value": swing,
                "core_cash": book.core_cash - committed["core"], "swing_cash": book.swing_cash - committed["swing"],
                "holdings": sorted(core_owned | swing_owned),
                "core_holdings": sorted(core_owned), "swing_holdings": sorted(swing_owned)}
    start = float(settings["starting_cash"])
    fraction = float(settings["swing"].get("capital_fraction", 0.25))
    return {"source": "settings.toml (no book yet)", "core_value": start * (1 - fraction), "swing_value": start * fraction,
            "core_cash": start * (1 - fraction), "swing_cash": start * fraction, "holdings": [],
            "core_holdings": [], "swing_holdings": []}


# ---------------------------------------------------------------------------- analysis


def _market_check(data: dict[str, Any], snapshot: pd.DataFrame, settings: dict[str, Any]) -> dict[str, Any]:
    adjusted = data["adjusted"]
    rows = {}
    for ticker in MARKET_TICKERS:
        if ticker in adjusted:
            series = adjusted[ticker].dropna()
            ma200 = series.rolling(200).mean().iloc[-1]
            rows[ticker] = {"above_200d": bool(series.iloc[-1] > ma200), "vs_200d": float(series.iloc[-1] / ma200 - 1),
                            "ret_63": float(series.iloc[-1] / series.iloc[-64] - 1)}
    stocks = snapshot[~snapshot["ticker"].isin(MARKET_TICKERS)]
    breadth = float((stocks["dist_ma200"] > 0).mean())
    volatility = float(snapshot["mkt_vol_21"].iloc[0])
    filter_ticker = settings["swing"].get("market_filter_ticker", "SPY")
    swing_on = rows.get(filter_ticker, {}).get("above_200d", True)
    mood = ("healthy uptrend" if swing_on and breadth > 0.55 else
            "mixed" if swing_on else "downtrend: the swing rules pause new trades")
    return {"indexes": rows, "breadth": breadth, "volatility": volatility, "swing_allowed": swing_on, "mood": mood}


def _swing_plan(snapshot: pd.DataFrame, data: dict[str, Any], settings: dict[str, Any], ai: dict[str, Any],
                account: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    swing = settings["swing"]
    adjusted = data["adjusted"]
    ind = swing_indicators(adjusted, settings)
    col = {t: j for j, t in enumerate(adjusted.columns)}
    k = len(adjusted) - 1
    rule = {t: entry_setup(ind, k, col[t], swing) for t in snapshot["ticker"] if t in col}
    snapshot = snapshot.assign(rule_signal=snapshot["ticker"].map(lambda t: rule.get(t)[0] if rule.get(t) else ""),
                               rule_priority=snapshot["ticker"].map(lambda t: rule.get(t)[1] if rule.get(t) else 0.0))
    held = set(account["swing_holdings"])
    # Only stocks like the ones the breakout rules were tested on: large companies and your own lists.
    tested = set(load_index_constituents(ai["plan"].get("swing_indexes", ["sp500", "tsx60"]))["ticker"])
    tested |= set(swing_universe(settings, include_all=True)) | {t for ts in settings["sectors"].values() for t in ts}
    snapshot = snapshot[snapshot["ticker"].isin(tested)]

    slots = swing_slots(account["swing_value"], swing)
    free = max(0, slots - len(held))
    position = account["swing_value"] / slots if slots else 0.0
    # Ordered by the tested rules' own priority (as the backtest and live trader do). The AI score is shown for
    # information: on the breakout trades the rules made, it hasn't reliably picked the better ones.
    signals = snapshot[(snapshot["rule_signal"] != "") & ~snapshot["ticker"].isin(held)].sort_values("rule_priority", ascending=False)
    signals = signals[signals["price"] <= position] if position > 0 else signals.iloc[0:0]
    picks = signals.head(int(ai["plan"]["swing_picks"])).copy()
    picks["shares"] = picks["price"].map(lambda p: math.floor(position / p) if p > 0 else 0)
    picks["stop"] = picks["price"] * (1 - float(swing.get("stop_loss", 0.15)))
    picks["reasons"] = picks.apply(lambda r: _reasons(r, snapshot, model), axis=1)
    picks["action"] = ["BUY (fills a free slot)" if i < free else "backup" for i in range(len(picks))]

    # Stocks the AI likes for the next month that haven't triggered the rules yet: a watchlist.
    watch = snapshot[(snapshot["rule_signal"] == "") & (snapshot["dist_ma200"] > 0)].nlargest(5, "ai_swing").copy()
    watch["reasons"] = watch.apply(lambda r: _reasons(r, snapshot, model), axis=1)
    return {"picks": picks, "watch": watch, "slots": slots, "free_slots": free, "position_value": position,
            "signals_total": int((snapshot["rule_signal"] != "").sum()), "allowed": True}


def rank_dividend_payers(snapshot: pd.DataFrame, settings: dict[str, Any], ai: dict[str, Any]) -> pd.DataFrame:
    """Dividend payers that pass your yield / dividend-cut / volatility limits, ranked for a long-term income
    portfolio. Shared by the daily report and the AI forward test (ai_investor/forward.py)."""
    core = settings.get("core", {})
    min_yield, max_yield = float(core.get("min_yield", 0.01)), float(core.get("max_yield", 0.10))
    cut = float(core.get("cut_threshold", 0.15))
    payers = snapshot[
        (snapshot["div_yield"] >= min_yield) & (snapshot["div_yield"] <= max_yield)
        & ~(snapshot["div_growth"] < -cut) & ~snapshot["ticker"].isin(MARKET_TICKERS)
        & (snapshot["sector"] != "Broad Market ETF")
    ].copy()
    # A dividend portfolio is for steady income, so very volatile stocks are left out, and the AI's one-year view
    # is blended 50/50 with dividend quality (yield, dividend growth, steadiness, each ranked within the industry).
    payers = payers[payers["vol_63"] <= float(ai["plan"].get("longterm_max_volatility", 0.40))]
    by_sector = payers.groupby("sector")
    payers["quality"] = (by_sector["div_yield"].rank(pct=True) + by_sector["div_growth"].rank(pct=True).fillna(0.5)
                         + by_sector["vol_63"].rank(pct=True, ascending=False)) / 3
    payers["combined"] = 0.5 * payers["ai_longterm_pct"] + 0.5 * payers["quality"]
    return payers


def _longterm_plan(snapshot: pd.DataFrame, settings: dict[str, Any], ai: dict[str, Any], account: dict[str, Any],
                   model: dict[str, Any]) -> dict[str, Any]:
    core = settings.get("core", {})
    payers = rank_dividend_payers(snapshot, settings, ai)
    payers["you_own"] = payers["ticker"].isin(account["core_holdings"])
    per_industry = int(ai["plan"]["longterm_picks_per_industry"])
    by_industry = (payers.sort_values("combined", ascending=False).groupby("sector").head(per_industry)
                   .sort_values(["sector", "combined"], ascending=[True, False]))
    by_industry["reasons"] = by_industry.apply(lambda r: _reasons(r, snapshot, model), axis=1)

    # What to buy next with the dividend account: one stock per industry, best industries first, as many as the
    # account can hold at the minimum position size (same rule as the backtest's small-account mode).
    min_position = float(core.get("min_position_value", 200))
    holdings_possible = max(1, int(account["core_value"] // min_position)) if min_position > 0 else 24
    best_per_industry = by_industry.groupby("sector").head(1).sort_values("combined", ascending=False)
    target = min(holdings_possible, len(best_per_industry))
    buy_list = best_per_industry[~best_per_industry["you_own"]].head(max(0, target - len(account["core_holdings"])))
    buy_list = buy_list[buy_list["price"] <= max(min_position, account["core_cash"])].copy()
    budget = min_position if len(buy_list) else 0
    buy_list["shares"] = buy_list["price"].map(lambda p: max(1, math.floor(budget / p)) if p > 0 else 0)
    weak = payers[payers["you_own"] & (payers["ai_longterm_pct"] < 0.2)]
    return {"by_industry": by_industry, "buy_list": buy_list, "holdings_possible": holdings_possible,
            "weak_holdings": weak, "payers": len(payers)}


# Plain-English descriptions of what made the AI like or dislike a stock.
DESCRIBE = {
    "mom_12_1": lambda v, p: f"{v:+.0%} over the past year (stronger than {p:.0%} of stocks)",
    "ret_252": lambda v, p: f"{v:+.0%} over 12 months",
    "ret_126": lambda v, p: f"{v:+.0%} over 6 months",
    "ret_21": lambda v, p: f"{v:+.0%} over the last month",
    "dist_ma200": lambda v, p: f"{v:+.0%} vs its 200-day average",
    "ma50_vs_ma200": lambda v, p: "medium-term trend pointing up" if v > 0 else "medium-term trend pointing down",
    "drawdown_252": lambda v, p: "at its 52-week high" if v > -0.02 else f"{-v:.0%} below its 52-week high",
    "breakout_20": lambda v, p: "at a new 20-day high" if v >= 0 else f"{-v:.0%} below its 20-day high",
    "vol_63": lambda v, p: f"{'calm' if p < 0.5 else 'volatile'}: {v:.0%} annual volatility",
    "vol_21": lambda v, p: f"recent volatility {v:.0%}",
    "beta_252": lambda v, p: f"{'low' if v < 0.8 else 'high' if v > 1.2 else 'average'} market sensitivity (beta {v:.1f})",
    "div_yield": lambda v, p: f"yields {v:.1%}",
    "div_growth": lambda v, p: ("dividend unchanged over the year" if abs(v) < 0.01 else
                                f"dividend {'grew' if v > 0 else 'shrank'} {abs(v):.0%} in a year"),
    "rsi_14": lambda v, p: f"RSI {v:.0f} ({'strong buying' if v > 65 else 'oversold' if v < 35 else 'neutral'})",
    "rsi_3": lambda v, p: f"short-term RSI {v:.0f}",
}


def _reasons(row: pd.Series, snapshot: pd.DataFrame, model: dict[str, Any]) -> list[str]:
    """The stock's most unusual values among the inputs the model relies on most."""
    reasons = []
    for feature in model.get("importance", {}) or FEATURES:
        source = feature.replace("rank_", "")
        if source not in DESCRIBE or source in [r[0] for r in reasons]:
            continue
        value = row.get(source)
        if value is None or pd.isna(value):
            continue
        percentile = float((snapshot[source] < value).mean())
        if percentile > 0.8 or percentile < 0.2:
            reasons.append((source, DESCRIBE[source](float(value), percentile)))
        if len(reasons) == 3:
            break
    return [text for _, text in reasons]


def _apply_news(plan: pd.DataFrame, news_table: pd.DataFrame) -> None:
    if plan.empty:
        return
    lookup = news_table.set_index("ticker") if not news_table.empty else pd.DataFrame()
    def note(ticker: str) -> str:
        if lookup.empty or ticker not in lookup.index:
            return "no recent headlines"
        row = lookup.loc[ticker]
        mood = "positive" if row["sentiment"] > 0.15 else "negative" if row["sentiment"] < -0.15 else "neutral"
        flags = f"; RED FLAG: {', '.join(row['flags'])}" if row["flags"] else ""
        return f"{mood} ({row['headlines']} headlines){flags}"
    plan["news"] = plan["ticker"].map(note)
    plan["caution"] = plan["news"].str.contains("RED FLAG|negative")


# ---------------------------------------------------------------------------- output


def _money(value: float) -> str:
    return f"C${value:,.0f}"


def _write_report(as_of, market, swing, longterm, news_table, account, models) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    lines = [f"# AI research report: {as_of.date()}", "",
             f"Account from {account['source']}: dividend account {_money(account['core_value'])} "
             f"(cash {_money(account['core_cash'])}), swing account {_money(account['swing_value'])} "
             f"(cash {_money(account['swing_cash'])}).", ""]

    lines += ["## 1. Market check", "", f"**Mood: {market['mood']}.**", ""]
    for ticker, info in market["indexes"].items():
        lines.append(f"- {ticker}: {'above' if info['above_200d'] else 'BELOW'} its 200-day average "
                     f"({info['vs_200d']:+.1%}), {info['ret_63']:+.1%} over 3 months")
    lines += [f"- {market['breadth']:.0%} of stocks are above their 200-day average (breadth)",
              f"- Market volatility: {market['volatility']:.0%} annualized", ""]

    lines += ["## 2. Swing plan (next ~month)", "",
              f"{swing['signals_total']} stocks triggered the tested breakout rule. Your swing account "
              f"({_money(account['swing_value'])}) has {swing['slots']} slot(s) of about {_money(swing['position_value'])}; "
              f"{swing['free_slots']} free.", ""]
    if not market["swing_allowed"]:
        lines += ["**The market filter is off: the rules take no new swing trades right now.**", ""]
    if swing["picks"].empty:
        lines += ["No affordable breakout signals today.", ""]
    for row in swing["picks"].itertuples():
        caution = " ⚠ check the news first" if getattr(row, "caution", False) else ""
        lines += [f"**{row.ticker}** ({row.name}, {row.sector}): {row.action}{caution}",
                  f"- AI swing score (information only): better than {row.ai_swing_pct:.0%} of stocks",
                  f"- ~{row.shares} shares at about {_money(row.price)} each, stop near {_money(row.stop)}",
                  f"- Why: {'; '.join(row.reasons) or 'no single standout factor'}"]
        if hasattr(row, "news"):
            lines.append(f"- News: {row.news}")
        lines.append("")
    if not swing["watch"].empty:
        lines += ["AI watchlist (liked by the AI, but no breakout signal yet):", ""]
        lines += [f"- {r.ticker} ({r.name}): better than {r.ai_swing_pct:.0%} of stocks. {'; '.join(r.reasons)}"
                  for r in swing["watch"].itertuples()]
        lines.append("")

    lines += ["## 3. Long-term dividend plan (next ~year)", "",
              f"Your dividend account can hold about {longterm['holdings_possible']} stock(s) at the minimum position "
              f"size; you hold {len(account['core_holdings'])}.", ""]
    if not longterm["buy_list"].empty:
        lines += ["**Next to buy** (best in industries you don't own yet):", ""]
        for row in longterm["buy_list"].itertuples():
            caution = " ⚠ check the news first" if getattr(row, "caution", False) else ""
            lines += [f"- **{row.ticker}** ({row.name}, {row.sector}): ~{row.shares} shares at {_money(row.price)}, "
                      f"yield {row.div_yield:.1%}, dividend quality {row.quality:.0%} in its industry, AI one-year "
                      f"score better than {row.ai_longterm_pct:.0%} of stocks. "
                      f"{'; '.join(row.reasons)}{caution}" + (f" News: {row.news}." if hasattr(row, "news") else "")]
        lines.append("")
    if not longterm["weak_holdings"].empty:
        lines += ["**Holdings the AI rates poorly** (bottom 20% for the next year). Not an automatic sell; the "
                  "rules decide that at the quarterly rebalance:", ""]
        lines += [f"- {r.ticker}: better than only {r.ai_longterm_pct:.0%} of stocks" for r in longterm["weak_holdings"].itertuples()]
        lines.append("")
    lines += ["Top dividend payers by industry (half AI one-year view, half dividend quality; volatile stocks excluded):", "",
              "| Industry | Stock | Yield | Dividend growth | Volatility | Quality | AI score | You own |",
              "|---|---|---|---|---|---|---|---|"]
    for row in longterm["by_industry"].itertuples():
        growth = f"{row.div_growth:+.0%}" if pd.notna(row.div_growth) else "-"
        lines.append(f"| {row.sector} | {row.ticker} ({row.name}) | {row.div_yield:.1%} | {growth} | {row.vol_63:.0%} | "
                     f"{row.quality:.0%} | {row.ai_longterm_pct:.0%} | {'yes' if row.you_own else ''} |")
    lines.append("")

    if not news_table.empty:
        lines += ["## News check", "", "| Stock | Headlines | Mood | Red flags | Most notable headline |",
                  "|---|---|---|---|---|"]
        for row in news_table.sort_values("sentiment").itertuples():
            lines.append(f"| {row.ticker} | {row.headlines} | {row.sentiment:+.2f} | {', '.join(row.flags) or '-'} | "
                         f"{str(row.headline)[:90].replace('|', '/')} |")
        lines += ["", f"Scored by: {'your trained news model' if news.MODEL_PATH.exists() else 'finance word list'}. "
                  "Headlines are saved so your AI can learn from them (python -m ai_investor.news --train).", ""]

    lines += ["## How much to trust this", ""]
    for name, bundle in models.items():
        yearly = pd.DataFrame(bundle["walk_forward"])
        if yearly.empty:
            continue
        ai_ic, base_ic = yearly["ic_ai"].mean(), yearly["ic_momentum"].mean()
        verdict = ("beat the simple momentum rule" if ai_ic > base_ic else "did NOT beat the simple momentum rule; "
                   "treat its ranking as a tie-breaker only")
        lines.append(f"- **{name} model** (trained {bundle['trained_on']}, data through {bundle['data_through']}): "
                     f"average out-of-sample score {ai_ic:+.3f} vs momentum {base_ic:+.3f} over {len(yearly)} unseen "
                     f"years; it {verdict}.")
        fair = bundle.get("portfolio_fair") or {}
        if "AI picks" in fair:
            lines.append(f"  - On companies already large in 2012 (less hindsight): AI picks "
                         f"{fair['AI picks']['annual_return']:+.1%}/yr vs all of them equally weighted "
                         f"{fair['All stocks (equal weight)']['annual_return']:+.1%}/yr and XIC "
                         f"{fair['XIC (TSX)']['annual_return']:+.1%}/yr.")
        breakout = bundle.get("breakout_check") or {}
        if breakout:
            high, low = breakout["ai_high"], breakout["ai_low"]
            # Only call it help if the gap is meaningful AND the AI's favourites didn't win less often.
            helped = high["avg_return"] > low["avg_return"] + 0.005 and high["win_rate"] >= low["win_rate"]
            verdict = ("helped pick between breakout signals" if helped else
                       "made no clear difference between breakout signals, so the order of swing picks is a weak "
                       "tie-breaker, not an edge")
            lines.append(f"  - On the breakout trades your rules made: AI top half averaged {high['avg_return']:+.2%} "
                         f"({high['win_rate']:.0%} winners) vs bottom half {low['avg_return']:+.2%} "
                         f"({low['win_rate']:.0%}). The AI ranking {verdict}.")
    lines += ["- The AI ranks stocks; it can't see the future. Most of any stock's move is luck no model predicts.",
              "- News scores read headlines only, and can't be checked against history yet.",
              "- This is research to support your decisions, not financial advice. live_trader.py keeps trading the "
              "tested rules.", ""]

    path = REPORTS_DIR / f"plan_{as_of.date()}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "as_of": str(as_of.date()), "market": market,
        "swing_picks": swing["picks"].drop(columns=FEATURES, errors="ignore").to_dict("records"),
        "longterm_buy": longterm["buy_list"].drop(columns=FEATURES, errors="ignore").to_dict("records"),
    }
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _print_summary(market, swing, longterm, report: Path) -> None:
    print(f"\nMarket: {market['mood']} ({market['breadth']:.0%} of stocks in uptrends)")
    print("Swing picks (tested rule order; AI score for information):")
    for row in swing["picks"].itertuples():
        flag = "  ⚠ news" if getattr(row, "caution", False) else ""
        print(f"  {row.action:24s} {row.ticker:9s} ~{row.shares} sh  AI {row.ai_swing_pct:.0%}  {'; '.join(row.reasons)[:80]}{flag}")
    if swing["picks"].empty:
        print("  none today")
    print("Long-term next buys:")
    for row in longterm["buy_list"].itertuples():
        flag = "  ⚠ news" if getattr(row, "caution", False) else ""
        print(f"  {row.ticker:9s} {row.sector:24s} yield {row.div_yield:.1%}  AI {row.ai_longterm_pct:.0%}{flag}")
    if longterm["buy_list"].empty:
        print("  nothing new: your dividend account is full for its size")
    print(f"\nFull report: {report}")


if __name__ == "__main__":
    main()
