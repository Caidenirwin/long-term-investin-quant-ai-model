"""Index constituent lists for market-wide discovery in scan.py.

Lists come from Wikipedia and are cached in data/ for a week, so a scan still works offline
or if Wikipedia is unreachable (it falls back to the last saved copy).
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


INDEXES = {
    "sp500": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "columns": ("Symbol", "Security", "GICS Sector"),
        "suffix": "",
    },
    "tsx60": {
        "url": "https://en.wikipedia.org/wiki/S%26P/TSX_60",
        "columns": ("Symbol", "Company", "Sector"),
        "suffix": ".TO",
    },
    "sp400": {  # US mid-caps
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "columns": ("Symbol", "Security", "GICS Sector"),
        "suffix": "",
    },
    "sp600": {  # US small-caps
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        "columns": ("Symbol", "Security", "GICS Sector"),
        "suffix": "",
    },
    "tsx_composite": {  # ~220 Canadian companies, large to small
        "url": "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index",
        "columns": ("Ticker", "Company", "Sector"),
        "suffix": ".TO",
    },
}
CACHE_MAX_AGE_DAYS = 7

# The TSX list uses Morningstar-style sector names; map them to the S&P's GICS names so industries line up.
SECTOR_ALIASES = {
    "Basic Materials": "Materials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Technology": "Information Technology",
}

# ETFs in the swing universe, labelled with the industry they track.
ETF_INFO = {
    "SPY": ("SPDR S&P 500 ETF", "Broad Market ETF"),
    "QQQ": ("Invesco QQQ (Nasdaq-100)", "Broad Market ETF"),
    "IWM": ("iShares Russell 2000", "Broad Market ETF"),
    "DIA": ("SPDR Dow Jones Industrial Average", "Broad Market ETF"),
    "XIU.TO": ("iShares S&P/TSX 60", "Broad Market ETF"),
    "XLE": ("Energy Select Sector SPDR", "Energy"),
    "XEG.TO": ("iShares S&P/TSX Capped Energy", "Energy"),
    "XLF": ("Financial Select Sector SPDR", "Financials"),
    "ZEB.TO": ("BMO Equal Weight Banks", "Financials"),
    "XLK": ("Technology Select Sector SPDR", "Information Technology"),
    "XLV": ("Health Care Select Sector SPDR", "Health Care"),
    "XLI": ("Industrial Select Sector SPDR", "Industrials"),
    "XLU": ("Utilities Select Sector SPDR", "Utilities"),
    "XLP": ("Consumer Staples Select Sector SPDR", "Consumer Staples"),
    "GLD": ("SPDR Gold Shares", "Materials"),
}


def load_index_constituents(names: list[str]) -> pd.DataFrame:
    """Return ticker, name, sector, index for every requested index (Yahoo-style tickers)."""
    frames = []
    for name in names:
        if name not in INDEXES:
            raise ValueError(f"Unknown index '{name}'. Choose from: {sorted(INDEXES)}")
        frames.append(_cached_constituents(name).assign(index=name))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", "name", "sector", "index"])
    combined["sector"] = combined["sector"].replace(SECTOR_ALIASES)
    return combined.drop_duplicates("ticker")


def company_info(names: list[str]) -> dict[str, dict[str, str]]:
    """Ticker -> {name, sector} for index members and known ETFs. Missing lists are skipped quietly."""
    info = {ticker: {"name": name, "sector": sector} for ticker, (name, sector) in ETF_INFO.items()}
    try:
        constituents = load_index_constituents(names)
        info.update(constituents.set_index("ticker")[["name", "sector"]].to_dict("index"))
    except RuntimeError:
        pass
    return info


def _cached_constituents(name: str) -> pd.DataFrame:
    path = Path(f"data/universe_{name}.csv")
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < CACHE_MAX_AGE_DAYS * 86400
    if fresh:
        return pd.read_csv(path)
    try:
        table = _fetch_constituents(name)
        path.parent.mkdir(exist_ok=True)
        table.to_csv(path, index=False)
        return table
    except Exception as error:
        if path.exists():
            print(f"Could not refresh the {name} list ({error}); using the saved copy.")
            return pd.read_csv(path)
        raise RuntimeError(f"Could not download the {name} constituent list: {error}") from error


def _fetch_constituents(name: str) -> pd.DataFrame:
    spec = INDEXES[name]
    response = requests.get(spec["url"], headers={"User-Agent": "Mozilla/5.0 (long-term-quant research script)"}, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    symbol_col, name_col, sector_col = spec["columns"]

    for table in soup.find_all("table", class_="wikitable"):
        rows = [[cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])] for row in table.find_all("tr")]
        # Header cells can carry footnote marks, e.g. "Sector [ 10 ]".
        header = [re.sub(r"\[.*?\]", "", cell).replace(" ", "") for cell in rows[0]] if rows else []
        wanted = [column.replace(" ", "") for column in (symbol_col, name_col, sector_col)]
        if not all(column in header for column in wanted):
            continue
        positions = [header.index(column) for column in wanted]
        records = [
            {
                "ticker": _to_yahoo(row[positions[0]], spec["suffix"]),
                "name": row[positions[1]],
                "sector": row[positions[2]],
            }
            for row in rows[1:]
            if len(row) > max(positions) and row[positions[0]]
        ]
        if records:
            return pd.DataFrame(records)
    raise RuntimeError(f"No constituent table found at {spec['url']}")


def _to_yahoo(symbol: str, suffix: str) -> str:
    # Share classes use a dot on exchanges (BRK.B, RCI.B) and a dash on Yahoo (BRK-B, RCI-B.TO).
    return symbol.strip().upper().replace(".", "-") + suffix
