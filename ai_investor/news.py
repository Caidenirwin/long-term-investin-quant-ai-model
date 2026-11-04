"""News research, all local: free Yahoo Finance headlines, scored on your computer.

    python -m ai_investor.news --tickers CNQ.TO BCE.TO     # read and score recent headlines
    python -m ai_investor.news --train                     # train the news model on your saved headlines

How it learns: every headline the advisor reads is saved to data/news_archive.csv. Once enough of them have
a known outcome (how the stock moved over the next 5 trading days vs the market), --train teaches a model
which words actually mattered for prices. Until then, a finance word list is used. Free news sources only
offer recent headlines, so this archive is how your AI gets news history: it grows every day you run it.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from ai_investor.dataset import DATA_DIR, ROOT

ARCHIVE = DATA_DIR / "news_archive.csv"
# Optional data key (not an AI service). Read from the environment so it never ends up in a file.
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY", "")
MODEL_PATH = ROOT / "models" / "news.joblib"
MIN_TRAINING_HEADLINES = 1500

# Finance-specific words (in the spirit of the Loughran-McDonald word lists). General-purpose sentiment
# lists misread finance: "liability" or "tax" aren't bad news, "beat" is good news.
POSITIVE = {
    "beat", "beats", "tops", "exceeds", "surpass", "surpasses", "record", "raises", "raised", "hike", "hikes",
    "upgrade", "upgraded", "upgrades", "outperform", "overweight", "growth", "grows", "strong", "stronger",
    "surge", "surges", "soar", "soars", "jump", "jumps", "rally", "rallies", "gain", "gains", "rises", "climbs",
    "profit", "profitable", "boost", "boosts", "expands", "expansion", "approval", "approved", "wins", "win",
    "buyback", "repurchase", "increase", "increases", "accelerate", "accelerates", "optimistic", "upbeat",
    "milestone", "breakthrough", "partnership", "contract", "awarded", "bullish",
}
NEGATIVE = {
    "miss", "misses", "missed", "falls", "fell", "drop", "drops", "plunge", "plunges", "slump", "slumps", "tumble",
    "tumbles", "sink", "sinks", "downgrade", "downgraded", "downgrades", "underperform", "underweight",
    "cut", "cuts", "slash", "slashes", "lowers", "lowered", "weak", "weaker", "weakness", "loss", "losses",
    "lawsuit", "sued", "probe", "investigation", "fraud", "scandal", "recall", "recalls", "warning", "warns",
    "layoffs", "layoff", "bankruptcy", "default", "delay", "delays", "halt", "halts", "decline", "declines",
    "concern", "concerns", "risk", "risks", "bearish", "pessimistic", "fined", "penalty", "suspends",
    "resigns", "resignation", "shortfall", "downturn", "slowdown", "investigates", "breach",
}
NEGATORS = {"not", "no", "never", "without", "fails", "failed"}
# Phrases that matter a lot for a dividend or swing investor, flagged in the report.
RED_FLAGS = {
    "dividend cut": "dividend cut", "cuts dividend": "dividend cut", "suspends dividend": "dividend suspended",
    "slashes dividend": "dividend cut", "guidance cut": "guidance cut", "lowers guidance": "guidance cut",
    "downgrade": "analyst downgrade", "investigation": "investigation", "lawsuit": "lawsuit", "fraud": "fraud",
    "bankruptcy": "bankruptcy", "recall": "recall", "ceo resigns": "CEO leaving", "restatement": "accounting restatement",
}


def fetch_headlines(tickers: list[str], per_ticker: int = 10, names: dict[str, str] | None = None) -> pd.DataFrame:
    """Recent headlines from free sources, no key needed. Tries Yahoo Finance, then Yahoo's RSS feed, then
    Google News (searched by company name), so one source being down doesn't stop the research.
    Saved to the archive so the AI can learn from them."""
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    names = names or {}
    rows = []
    for ticker in tickers:
        found: list[dict[str, Any]] = []
        if FINNHUB_KEY:
            today = datetime.now(timezone.utc).date()
            found = _finnhub(ticker, today - pd.Timedelta(days=7), today)
        if not found:
            try:
                found = [p for p in (_parse_item(item) for item in (yf.Ticker(ticker).news or [])[:per_ticker]) if p]
            except Exception:
                pass
        if not found:
            found = _rss(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(ticker)}&region=US&lang=en-US")
        if not found:
            query = quote(f'"{names.get(ticker) or ticker.removesuffix(".TO")}" stock')
            found = _rss(f"https://news.google.com/rss/search?q={query}&hl=en-CA&gl=CA&ceid=CA:en")
        about = _mentions(ticker, names.get(ticker, ""))
        rows += [{"ticker": ticker, **item, "relevant": bool(about.search(item["title"]))} for item in found[:per_ticker]]
    headlines = pd.DataFrame(rows, columns=["ticker", "published", "title", "publisher", "url", "relevant"])
    if not headlines.empty:
        _append_archive(headlines)
    return headlines


GENERIC_WORDS = {"inc", "inc.", "corp", "corporation", "company", "the", "group", "holdings", "limited", "ltd", "co",
                 "plc", "trust", "partners", "l.p.", "international", "financial", "energy", "resources"}


def _mentions(ticker: str, name: str) -> re.Pattern[str]:
    """Matches headlines that are actually about this company (feeds mix in general market news): the ticker, or
    the distinctive start of the company name ("Sun Life", "Canadian Natural", "Enbridge"), as whole words."""
    root = ticker.removesuffix(".TO").split("-")[0]
    words = [w.strip(",.()") for w in name.split()]
    distinctive = [w for w in words if len(w) > 2 and w.lower() not in GENERIC_WORDS][:2]
    options = [re.escape(root)]
    if distinctive:
        options.append(r"\s+".join(re.escape(w) for w in distinctive))
    return re.compile(r"\b(" + "|".join(options) + r")\b", re.IGNORECASE)


def _finnhub(ticker: str, start, end) -> list[dict[str, Any]]:
    """Company headlines from Finnhub (free key at finnhub.io; data only, not an AI service)."""
    try:
        response = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": ticker, "from": str(start), "to": str(end), "token": FINNHUB_KEY}, timeout=20,
        )
        if response.status_code == 429:  # free tier: 60 requests a minute
            time.sleep(5)
            return []
        items = response.json() if response.ok else []
    except Exception:
        return []
    return [
        {"published": datetime.fromtimestamp(item["datetime"], tz=timezone.utc).isoformat(),
         "title": str(item.get("headline", "")).strip(), "publisher": item.get("source", ""), "url": item.get("url", "")}
        for item in items if isinstance(item, dict) and item.get("headline") and item.get("datetime")
    ]


def backfill(tickers: list[str], days: int = 365) -> int:
    """Download past headlines (needs FINNHUB_API_KEY) so the news model can train without waiting months."""
    if not FINNHUB_KEY:
        raise SystemExit("Set a free Finnhub key first (finnhub.io), e.g. in PowerShell:  setx FINNHUB_API_KEY yourkey\n"
                         "then open a new terminal.")
    today = datetime.now(timezone.utc).date()
    total = 0
    for n, ticker in enumerate(tickers, start=1):
        rows = []
        for months_back in range(0, days, 30):  # month by month: each request returns a limited number of items
            end = today - pd.Timedelta(days=months_back)
            rows += _finnhub(ticker, end - pd.Timedelta(days=30), end)
            time.sleep(1.1)  # stay under the free tier's 60 requests per minute
        if rows:
            _append_archive(pd.DataFrame([{"ticker": ticker, **row} for row in rows]))
            total += len(rows)
        print(f"  {n}/{len(tickers)} {ticker}: {len(rows)} headlines")
    return total


def _rss(url: str) -> list[dict[str, Any]]:
    """Headlines from an RSS feed (title, date, source, link)."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (personal research script)"}, timeout=15)
        items = ElementTree.fromstring(response.content).findall(".//item")
    except Exception:
        return []
    parsed = []
    for item in items:
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "").isoformat()
        except (TypeError, ValueError):
            published = datetime.now(timezone.utc).isoformat()
        source = item.findtext("source") or ""
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]  # Google News appends " - Publisher"
        parsed.append({"published": published, "title": title, "publisher": source, "url": item.findtext("link") or ""})
    return parsed


def _parse_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """yfinance has used two formats; handle both."""
    content = item.get("content", item)
    title = content.get("title")
    if not title:
        return None
    published = content.get("pubDate") or content.get("displayTime")
    if published is None and item.get("providerPublishTime"):
        published = datetime.fromtimestamp(item["providerPublishTime"], tz=timezone.utc).isoformat()
    provider = content.get("provider") or {}
    url = (content.get("canonicalUrl") or {}).get("url") or item.get("link", "")
    return {
        "published": str(published or datetime.now(timezone.utc).isoformat()),
        "title": str(title).strip(),
        "publisher": provider.get("displayName", item.get("publisher", "")),
        "url": url,
    }


def _append_archive(headlines: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([pd.read_csv(ARCHIVE), headlines], ignore_index=True) if ARCHIVE.exists() else headlines
    combined.drop_duplicates(subset=["ticker", "title"], keep="first").to_csv(ARCHIVE, index=False)


# ---------------------------------------------------------------------------- scoring


def lexicon_score(title: str) -> float:
    """-1 (clearly negative) .. +1 (clearly positive), from finance words, with simple negation handling."""
    words = re.findall(r"[a-z']+", title.lower())
    score, hits = 0.0, 0
    for i, word in enumerate(words):
        sign = 1 if word in POSITIVE else -1 if word in NEGATIVE else 0
        if sign == 0:
            continue
        if any(w in NEGATORS for w in words[max(0, i - 2):i]):
            sign = -sign
        score += sign
        hits += 1
    return float(np.clip(score / (hits + 1), -1, 1))


def red_flags(title: str) -> list[str]:
    text = title.lower()
    return sorted({label for phrase, label in RED_FLAGS.items() if phrase in text})


def score_headlines(headlines: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment (your trained news model if it passed its test, else the word list) and red flags."""
    if headlines.empty:
        return headlines.assign(sentiment=[], flags=[], scored_by=[])
    learned = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    scored = headlines.copy()
    if learned:
        raw = learned["model"].predict(learned["vectorizer"].transform(scored["title"]))
        scored["sentiment"] = np.clip(raw / learned["scale"], -1, 1)
        scored["scored_by"] = "your trained news model"
    else:
        scored["sentiment"] = scored["title"].map(lexicon_score)
        scored["scored_by"] = "finance word list"
    scored["flags"] = scored["title"].map(red_flags)
    return scored


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker: average sentiment, number of headlines, red flags, most notable headline."""
    if scored.empty:
        return pd.DataFrame(columns=["ticker", "headlines", "sentiment", "flags", "headline"])
    rows = []
    for ticker, group in scored.groupby("ticker"):
        # Only headlines that mention the company count; the rest is general market news from the feed.
        about = group[group["relevant"]] if "relevant" in group else group
        if about.empty:
            rows.append({"ticker": ticker, "headlines": 0, "sentiment": 0.0, "flags": [],
                         "headline": f"(only general market news; {len(group)} headlines)"})
            continue
        notable = about.loc[about["sentiment"].abs().idxmax()]
        flags = sorted({flag for flags in about["flags"] for flag in flags})
        rows.append({"ticker": ticker, "headlines": len(about), "sentiment": float(about["sentiment"].mean()),
                     "flags": flags, "headline": notable["title"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- learning from your archive


def train_news_model(adjusted: pd.DataFrame, market: str = "XIC.TO") -> str:
    """Learn which words moved prices, from your saved headlines. Only kept if it beats chance on held-out news."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    if not ARCHIVE.exists():
        return "No saved headlines yet. Run the advisor daily; it saves every headline it reads."
    archive = pd.read_csv(ARCHIVE)
    archive["date"] = pd.to_datetime(archive["published"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    archive = archive.dropna(subset=["date"])

    # Outcome: the stock's next-5-trading-day return minus the market's, starting the day after the headline.
    outcomes = []
    for row in archive.itertuples():
        if row.ticker not in adjusted or market not in adjusted:
            outcomes.append(np.nan)
            continue
        after = adjusted.index[adjusted.index > row.date]
        if len(after) < 6:
            outcomes.append(np.nan)
            continue
        start, end = after[0], after[5]
        stock = adjusted.at[end, row.ticker] / adjusted.at[start, row.ticker] - 1
        index = adjusted.at[end, market] / adjusted.at[start, market] - 1
        outcomes.append(stock - index)
    archive["outcome"] = outcomes
    labeled = archive.dropna(subset=["outcome"]).sort_values("date")
    if len(labeled) < MIN_TRAINING_HEADLINES:
        return (f"{len(labeled)} headlines have a known outcome so far; {MIN_TRAINING_HEADLINES} are needed to train. "
                "Keep running the advisor daily (about 1-2 months).")

    split = int(len(labeled) * 0.8)  # train on older headlines, test on the newest ones
    train, test = labeled.iloc[:split], labeled.iloc[split:]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True)
    model = Ridge(alpha=10.0).fit(vectorizer.fit_transform(train["title"]), train["outcome"].clip(-0.3, 0.3))
    predicted = model.predict(vectorizer.transform(test["title"]))
    skill = float(pd.Series(predicted).corr(test["outcome"].reset_index(drop=True), method="spearman"))
    baseline = float(test["title"].map(lexicon_score).reset_index(drop=True).corr(
        test["outcome"].reset_index(drop=True), method="spearman"))
    if not skill > max(baseline, 0.02):
        return (f"Trained a news model, but on the newest {len(test)} headlines it scored {skill:+.3f} vs the word "
                f"list's {baseline:+.3f}, so it was NOT adopted. The word list stays in use.")
    full = Ridge(alpha=10.0).fit(vectorizer.fit_transform(labeled["title"]), labeled["outcome"].clip(-0.3, 0.3))
    raw = full.predict(vectorizer.transform(labeled["title"]))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": full, "vectorizer": vectorizer, "scale": float(np.percentile(np.abs(raw), 95)) or 1.0,
                 "trained_on": datetime.now().strftime("%Y-%m-%d"), "headlines": len(labeled), "test_skill": skill},
                MODEL_PATH)
    return (f"News model trained on {len(labeled)} headlines and adopted: held-out skill {skill:+.3f} "
            f"vs word list {baseline:+.3f}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local news research.")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--train", action="store_true", help="Train the news model on your saved headlines.")
    parser.add_argument("--backfill", action="store_true",
                        help="Download ~1 year of past headlines for your stocks (needs FINNHUB_API_KEY), then train.")
    args = parser.parse_args()
    if args.backfill:
        from src.config import load_config
        settings = load_config(Path("config/settings.toml"))
        tickers = args.tickers or sorted({t for ts in settings["sectors"].values() for t in ts}
                                         | set(settings["swing"].get("stock_universe", [])))
        print(f"Backfilling a year of headlines for {len(tickers)} stocks (about {len(tickers) * 13 // 60 + 1} minutes)...")
        print(f"Saved {backfill(tickers)} headlines to {ARCHIVE}")
        args.train = True
    if args.train:
        from ai_investor.dataset import load_market_data
        print(train_news_model(load_market_data(start=(pd.Timestamp.today() - pd.DateOffset(years=3)).strftime("%Y-%m-%d"))["adjusted"]))
        return
    scored = score_headlines(fetch_headlines(args.tickers or ["XIU.TO"]))
    for row in scored.sort_values(["ticker", "published"], ascending=[True, False]).itertuples():
        flag = f"  !! {', '.join(row.flags)}" if row.flags else ""
        print(f"{row.ticker:8s} {row.sentiment:+.2f}  {row.title[:100]}{flag}")


if __name__ == "__main__":
    main()
