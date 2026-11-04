"""Local dashboard for the swing + dividend portfolio.

    python dashboard.py            # opens http://127.0.0.1:8050 in your browser

Reads the CSVs in outputs/ (from main.py and scan.py) and lets you rerun both from buttons.
It only listens on your own computer (127.0.0.1); nothing is shared or uploaded.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from src.universe import ETF_INFO, SECTOR_ALIASES

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
PORT = 8050
COMMANDS = {
    "scan": [sys.executable, "scan.py"],
    "backtest": [sys.executable, "main.py"],
}


class Job:
    """One background run of scan.py or main.py, with the tail of its output for error messages."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.state = "idle"
        self.started: float | None = None
        self.finished: float | None = None
        self.log: deque[str] = deque(maxlen=40)
        self.lock = threading.Lock()

    def start(self) -> bool:
        with self.lock:
            if self.state == "running":
                return False
            self.state, self.started, self.finished = "running", time.time(), None
            self.log.clear()
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self) -> None:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        try:
            process = subprocess.Popen(
                COMMANDS[self.name], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env=env,
            )
            for line in process.stdout:
                self.log.append(line.rstrip())
            code = process.wait()
            self.state = "done" if code == 0 else "failed"
        except Exception as error:
            self.log.append(str(error))
            self.state = "failed"
        self.finished = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "started": self.started,
            "finished": self.finished,
            "log": list(self.log)[-8:],
        }


JOBS = {name: Job(name) for name in COMMANDS}


# ---------------------------------------------------------------------------- data


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _read(path: Path, **kwargs: Any) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, **kwargs) if path.exists() else None
    except Exception:
        return None


# The dividend core's sector names (settings.toml) -> the scanner's standard (GICS) names, so one industry
# view can show both today's scan and your dividend holdings.
CORE_TO_STANDARD = {
    "Energy / Pipelines": "Energy",
    "Healthcare": "Health Care",
    "Industrials / Transport": "Industrials",
    "Telecom": "Communication Services",
}


def _swing_by_industry(trades: pd.DataFrame, info: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Backtest swing results grouped by the industry of each traded stock."""
    trades = trades.assign(sector=trades["ticker"].map(lambda t: info.get(t, {}).get("sector") or "Other / Unknown"))
    result = {}
    for sector, group in trades.groupby("sector"):
        by_ticker = group.groupby("ticker").agg(trades=("pnl", "size"), pnl=("pnl", "sum"),
                                                win_rate=("pnl", lambda x: (x > 0).mean()))
        result[sector] = {
            "trades": len(group),
            "win_rate": float((group["pnl"] > 0).mean()),
            "avg_return": float(group["return"].mean()),
            "pnl": float(group["pnl"].sum()),
            "avg_hold": float(group["hold_days"].mean()),
            "best": float(group["return"].max()),
            "worst": float(group["return"].min()),
            "by_ticker": _records(by_ticker.sort_values("pnl", ascending=False).reset_index()),
        }
    return result


def _company_names() -> dict[str, dict[str, str]]:
    """Names and industries from the cached index lists (no network calls, so the page loads fast)."""
    info = {ticker: {"name": name, "sector": sector} for ticker, (name, sector) in ETF_INFO.items()}
    for path in (ROOT / "data").glob("universe_*.csv"):
        table = _read(path)
        if table is not None:
            table["sector"] = table["sector"].replace(SECTOR_ALIASES)
            info.update(table.set_index("ticker")[["name", "sector"]].to_dict("index"))
    return info


def _latest_scan() -> dict[str, Any] | None:
    # Newest file written, not the latest date in its name (a scan's name is the date of its last completed close).
    scans = sorted((p for p in OUTPUTS.glob("scan_*.csv") if p.stem.count("_") == 1), key=lambda p: p.stat().st_mtime)
    if not scans:
        return None
    latest = scans[-1]
    as_of = latest.stem.split("_", 1)[1]
    ideas = _read(latest)
    industries = _read(OUTPUTS / f"scan_{as_of}_industries.csv")
    every_stock = _read(OUTPUTS / f"scan_{as_of}_all.csv")
    meta_path = OUTPUTS / f"scan_{as_of}_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"as_of": as_of}
    if ideas is not None and "next_earnings" in ideas.columns:
        ideas["next_earnings"] = ideas["next_earnings"].astype(str).replace({"nan": None, "None": None})
    return {
        "meta": meta,
        "ideas": _records(ideas) if ideas is not None else [],
        "industries": _records(industries) if industries is not None else [],
        "all": _records(every_stock) if every_stock is not None else [],
        "updated": latest.stat().st_mtime,
    }


def _forward_test(mode: str) -> dict[str, Any] | None:
    """The live/forward-test journal and book written by live_trader.py."""
    folder = ROOT / "live"
    book_path = folder / f"book_{mode}.json"
    if not book_path.exists():
        return None
    book = json.loads(book_path.read_text(encoding="utf-8"))
    journal = _read(folder / f"journal_{mode}.csv")
    trades = _read(folder / f"swing_trades_{mode}.csv")
    orders = _read(folder / f"orders_{mode}.csv")
    if orders is not None and "id" in orders.columns:
        # Each order is logged when placed and again when filled; show only its latest status.
        orders = orders.drop_duplicates("id", keep="last")
    return {
        "mode": mode,
        "created": book["created"],
        "capital": book["capital"],
        "pending": book["pending"],
        "core_holdings": len([q for q in book["core_shares"].values() if q > 0]),
        "swing_positions": [{"ticker": t, **p} for t, p in book["swing_positions"].items()],
        "cumulative_swept": book["cumulative_swept"],
        "cumulative_dividends": book["cumulative_dividends"],
        "cumulative_deposits": book.get("cumulative_deposits", book["capital"]),
        "paused": (folder / "PAUSE").exists(),
        "journal": _records(journal) if journal is not None else [],
        "closed_trades": _records(trades.tail(10)) if trades is not None else [],
        "recent_orders": _records(orders.tail(12)) if orders is not None else [],
    }


def build_payload() -> dict[str, Any]:
    config = load_config(ROOT / "config/settings.toml")
    names = _company_names()
    payload: dict[str, Any] = {
        "starting_cash": config["starting_cash"],
        "start_date": config["start_date"],
        "benchmark_ticker": config.get("benchmark_ticker", "XIC.TO"),
        "swing_capital_fraction": config.get("swing", {}).get("capital_fraction", 0),
        "monthly_deposit": config.get("contributions", {}).get("monthly", 0),
        "names": {ticker: value["name"] for ticker, value in names.items()},
    }

    equity = _read(OUTPUTS / "equity_curve.csv", parse_dates=["date"])
    if equity is not None:
        columns = ["date", "equity", "core_value", "swing_value", "twr_index", "cumulative_deposits", "benchmark_same_deposits"]
        series = equity[[c for c in columns if c in equity.columns]].copy()
        signals = _read(ROOT / "data/signal_prices_cad.csv", index_col=0, parse_dates=True)
        benchmark = payload["benchmark_ticker"]
        if signals is not None and benchmark in signals.columns:
            series["benchmark"] = signals[benchmark].reindex(series["date"]).ffill().to_numpy()
        series["date"] = series["date"].dt.strftime("%Y-%m-%d")
        series = series.round(2).astype(object)
        payload["equity"] = series.where(series.notna(), None).values.tolist()
        payload["equity_columns"] = list(series.columns)
        payload["backtest_updated"] = (OUTPUTS / "equity_curve.csv").stat().st_mtime

    summary = _read(OUTPUTS / "performance_summary.csv")
    if summary is not None and not summary.empty:
        payload["summary"] = _records(summary)[0]

    positions = _read(OUTPUTS / "positions.csv")
    if positions is not None:
        payload["positions"] = _records(positions)

    dividends = _read(OUTPUTS / "dividend_income_by_year.csv")
    if dividends is not None:
        payload["dividends_by_year"] = _records(dividends)

    trades = _read(OUTPUTS / "swing_trades.csv", parse_dates=["entry_date", "exit_date"])
    payload["core_to_standard"] = CORE_TO_STANDARD
    payload["sectors"] = {ticker: value["sector"] for ticker, value in names.items()}
    if trades is not None and not trades.empty:
        payload["swing_by_industry"] = _swing_by_industry(trades, names)
        recent = trades.sort_values("exit_date", ascending=False).head(12).copy()
        for column in ("entry_date", "exit_date"):
            recent[column] = recent[column].dt.strftime("%Y-%m-%d")
        payload["swing_trades"] = _records(recent)

    payload["scan"] = _latest_scan()
    payload["forward"] = _forward_test(config.get("live", {}).get("mode", "sim"))
    payload["forward_ai"] = _forward_test("ai")  # the AI's fake-money account (ai_investor/forward.py)
    payload["jobs"] = {name: job.to_dict() for name, job in JOBS.items()}
    return payload


# ---------------------------------------------------------------------------- server


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, (ROOT / "dashboard/index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/data":
            self._json(build_payload())
        elif self.path == "/api/jobs":
            self._json({name: job.to_dict() for name, job in JOBS.items()})
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self) -> None:
        name = self.path.removeprefix("/api/run/")
        if name not in JOBS:
            self._send(404, b"Not found", "text/plain")
            return
        started = JOBS[name].start()
        self._json({"started": started, "job": JOBS[name].to_dict()})

    def _json(self, data: Any) -> None:
        self._send(200, json.dumps(data, allow_nan=False, default=str).encode("utf-8"), "application/json")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:  # keep the terminal quiet
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the portfolio dashboard.")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true", help="Don't open a browser tab automatically.")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Dashboard running at {url}  (press Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
