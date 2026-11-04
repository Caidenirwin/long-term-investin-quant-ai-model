"""Walk-forward test: would re-tuning the settings each year (using ONLY past years) have beaten the TSX?

    python walk_forward.py

For every year from --first-year on:
  1. score each candidate setting on all years BEFORE it (excess return over the TSX, minus a drawdown penalty),
  2. pick the best one,
  3. record how that pick did in the year itself, which it had never seen.
Chaining those years gives an honest, out-of-sample track record. The same is reported for keeping your
current settings fixed, and for the "best in hindsight" setting (what a naive optimizer would show).

Note: chaining yearly returns ignores the cost of switching holdings when the pick changes.
"""
from __future__ import annotations

import argparse
import copy
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.config import load_config, research_config
from src.data import download_price_data, load_cached_price_data

TREND_HEAVY = {"dividend_yield": 0.25, "dividend_growth": 0.15, "low_volatility": 0.10, "trend": 0.50}
TSX_LIKE = {
    "Financials": 0.32, "Energy / Pipelines": 0.20, "Industrials / Transport": 0.14, "Utilities": 0.06,
    "Telecom": 0.06, "Consumer Staples": 0.08, "Healthcare": 0.06, "Real Estate": 0.08,
}

# Each choice is (label, changes). Edit to test other ideas; keep the list short to limit luck.
CHOICES: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "swing size": [("25%", {"swing.capital_fraction": 0.25}), ("40%", {"swing.capital_fraction": 0.40})],
    "stocks/sector": [("3", {"core.stocks_per_sector": 3}), ("2", {"core.stocks_per_sector": 2})],
    "sector tilt": [("off", {"core.sector_momentum_tilt": 0.0}), ("0.5", {"core.sector_momentum_tilt": 0.5})],
    "scoring": [("balanced", {}), ("trend-heavy", {"core.score_weights": TREND_HEAVY})],
    "industry weights": [("equal", {}), ("TSX-like", {"target_allocation": TSX_LIKE})],
    "park cash": [("no", {"swing.park_ticker": ""}), ("XIU", {"swing.park_ticker": "XIU.TO"})],
}

_worker: dict[str, Any] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward test against the benchmark.")
    parser.add_argument("--first-year", type=int, default=2015, help="First out-of-sample year (needs 3+ years before it).")
    parser.add_argument("--drawdown-penalty", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    base = research_config(load_config(Path("config/settings.toml")))
    try:
        data = load_cached_price_data(base)
    except FileNotFoundError:
        data = download_price_data(base)

    candidates = build_candidates(base)
    print(f"Running {len(candidates)} candidate settings...")
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(data,)) as pool:
        curves = dict(zip([label for label, _ in candidates], pool.map(_equity, [c for _, c in candidates])))

    benchmark_ticker = base.get("benchmark_ticker", "XIC.TO")
    any_curve = next(iter(curves.values()))
    benchmark = data[2][benchmark_ticker].reindex(any_curve.index).ffill()
    yearly = pd.DataFrame({label: _yearly(curve) for label, curve in curves.items()})
    bench_yearly = _yearly(benchmark)
    current_label = candidates[0][0]

    rows = []
    for year in [y for y in yearly.index if y >= args.first_year]:
        train_end = pd.Timestamp(f"{year - 1}-12-31")
        scores = {
            label: _train_score(curve.loc[:train_end], benchmark.loc[:train_end], args.drawdown_penalty)
            for label, curve in curves.items()
        }
        pick = max(scores, key=scores.get)
        rows.append({
            "year": year, "picked": pick,
            "walk_forward": yearly.at[year, pick], "current": yearly.at[year, current_label],
            "TSX": bench_yearly.get(year, np.nan),
        })
    table = pd.DataFrame(rows).set_index("year")

    hindsight = max(curves, key=lambda label: _train_score(curves[label].loc[f"{args.first_year}":],
                                                            benchmark.loc[f"{args.first_year}":], args.drawdown_penalty))
    table["best_in_hindsight"] = yearly.loc[table.index, hindsight]

    years = _years_spanned(any_curve, args.first_year)
    summary = {name: (1 + table[name]).prod() ** (1 / years) - 1
               for name in ["walk_forward", "current", "best_in_hindsight", "TSX"]}

    show = table.copy()
    for column in ["walk_forward", "current", "best_in_hindsight", "TSX"]:
        show[column] = show[column].map(lambda v: f"{v:+.1%}")
    print("\nYear by year (each 'walk_forward' pick used only data from before that year):")
    print(show.to_string())
    print(f"\nAnnualized {args.first_year}-{table.index[-1]} (last year is year-to-date):")
    print(f"  Walk-forward (honest)     {summary['walk_forward']:.2%}   vs TSX {summary['walk_forward'] - summary['TSX']:+.2%}")
    print(f"  Current settings, fixed   {summary['current']:.2%}   vs TSX {summary['current'] - summary['TSX']:+.2%}")
    print(f"  Best in hindsight         {summary['best_in_hindsight']:.2%}   vs TSX "
          f"{summary['best_in_hindsight'] - summary['TSX']:+.2%}   <- what a naive optimizer would claim")
    print(f"  TSX ({benchmark_ticker})            {summary['TSX']:.2%}")
    beat = (table["walk_forward"] > table["TSX"]).sum()
    print(f"\nWalk-forward beat the TSX in {beat} of {len(table)} years. Picks changed "
          f"{(table['picked'] != table['picked'].shift()).sum() - 1} times.")
    print(f"Hindsight-best setting: {hindsight}")

    Path("outputs").mkdir(exist_ok=True)
    table.to_csv("outputs/walk_forward.csv")
    print("Saved outputs/walk_forward.csv")


def build_candidates(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every combination of CHOICES. The first candidate is always the current settings."""
    candidates = [("current settings", copy.deepcopy(base))]
    for combo in product(*CHOICES.values()):
        config = copy.deepcopy(base)
        for _, changes in combo:
            for key, value in changes.items():
                if "." in key:
                    section, name = key.split(".")
                    config.setdefault(section, {})[name] = value
                else:
                    config[key] = value
        label = " | ".join(f"{name}={choice}" for name, (choice, _) in zip(CHOICES, combo))
        candidates.append((label, config))
    return candidates


def _init(data: tuple) -> None:
    _worker["data"] = data
    _worker["cache"] = {}


def _equity(config: dict[str, Any]) -> pd.Series:
    results = run_backtest(config, *_worker["data"], _worker["cache"])
    return results.equity_curve.set_index("date")["twr_index"]  # growth excluding any deposits


def _yearly(series: pd.Series) -> pd.Series:
    year_end = series.resample("YE").last()
    returns = year_end.pct_change()
    returns.iloc[0] = year_end.iloc[0] / series.iloc[0] - 1
    returns.index = returns.index.year
    return returns


def _years_spanned(curve: pd.Series, first_year: int) -> float:
    start = pd.Timestamp(f"{first_year - 1}-12-31")
    return (curve.index[-1] - max(start, curve.index[0])).days / 365.25


def _train_score(curve: pd.Series, benchmark: pd.Series, penalty: float) -> float:
    curve, benchmark = curve.dropna(), benchmark.dropna()
    if len(curve) < 250:
        return -np.inf
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1
    bench = (benchmark.iloc[-1] / benchmark.iloc[0]) ** (1 / years) - 1
    drawdown = float((curve / curve.cummax() - 1).min())
    return cagr - bench - penalty * abs(drawdown)


if __name__ == "__main__":
    main()
