from __future__ import annotations

import argparse
import copy
import os
import random
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest import run_backtest
from src.config import load_config, research_config
from src.data import download_price_data, load_cached_price_data
from src.reporting import performance_metrics


# Each entry is "section.key": candidate values. Edit these to change what gets searched.
# swing.setups values are "+"-joined so they can be grouped in the results CSV.
SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "swing": {
        "swing.setups": ["pullback", "breakout", "pullback+breakout"],
        "swing.use_stocks": [False, True],
        "swing.max_positions": [3, 5, 8],
        "swing.stop_loss": [0.0, 0.06, 0.10, 0.15],
        "swing.rsi_days": [2, 3],
        "swing.entry_rsi": [5, 10, 20],
        "swing.exit_ma_days": [5, 10],
        "swing.max_hold_days": [10, 20],
        "swing.breakout_days": [20, 55, 100],
        "swing.trail_days": [10, 20, 40],
        "swing.breakout_max_hold_days": [40, 80, 160],
    },
    "core": {
        "swing.capital_fraction": [0.15, 0.25, 0.35],
        "profit_sweep.fraction": [0.5, 0.75, 1.0],
        "core.stocks_per_sector": [2, 3, 4],
        "core.rebalance_frequency": ["Q", "S", "A"],
        "core.rebalance_band": [0.01, 0.02, 0.04],
        "dividends.mode": ["drip", "smart"],
    },
}

_worker_data: dict[str, Any] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize settings with an out-of-sample check.")
    parser.add_argument("--focus", choices=["swing", "core"], default="swing",
                        help="swing: tune the swing engine, scored on its own profits. core: tune allocation and dividend core.")
    parser.add_argument("--max-runs", type=int, default=400, help="Random parameter sets to test.")
    parser.add_argument("--refresh-data", action="store_true", help="Download fresh yfinance data first.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    base_config = research_config(load_config(Path("config/settings.toml")))
    if args.refresh_data:
        data = download_price_data(base_config)
    else:
        try:
            data = load_cached_price_data(base_config)
        except FileNotFoundError:
            data = download_price_data(base_config)

    Path("outputs").mkdir(exist_ok=True)
    oos_start = pd.Timestamp(base_config.get("out_of_sample_start", "2021-01-01"))
    space = SEARCH_SPACES[args.focus]

    jobs = [("current settings", base_config)]
    if args.focus == "core":
        no_swing = copy.deepcopy(base_config)
        no_swing["swing"]["enabled"] = False
        jobs.append(("no swing (100% dividend core)", no_swing))
    jobs += [(f"run {n}", config) for n, config in enumerate(generate_configs(base_config, args.max_runs, space), start=1)]

    rows: list[dict[str, Any]] = []
    initargs = (data, oos_start, space, args.focus)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=initargs) as pool:
        for index, row in enumerate(pool.map(_evaluate, jobs, chunksize=4), start=1):
            rows.append(row)
            if index % 50 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} tested")

    output = pd.DataFrame(rows).sort_values("in_sample_score", ascending=False)
    output.to_csv(f"outputs/optimization_results_{args.focus}.csv", index=False)

    searched = output[output["label"].str.startswith("run ")]
    baselines = output[~output["label"].str.startswith("run ")]
    correlation = searched["in_sample_score"].corr(searched["out_of_sample_score"], method="spearman")
    robust = robust_choices(searched, space)
    write_best_config_summary(
        searched.iloc[0], baselines, robust, correlation, oos_start, space, args.focus,
        Path(f"outputs/best_optimization_summary_{args.focus}.txt"),
    )

    columns = ["label", "in_sample_score", "is_CAGR", "is_max_drawdown", "oos_CAGR", "oos_max_drawdown", "swing_total_pnl"]
    scored_on = "swing engine (sleeve + profits swept out)" if args.focus == "swing" else "total portfolio"
    print(f"\nScores and CAGR/drawdown below are for the {scored_on}.")
    print("\nBaselines:")
    print(baselines[columns].to_string(index=False))
    print("\nTop 10 by in-sample score (out-of-sample columns were not used to pick them):")
    print(searched[columns].head(10).to_string(index=False))
    print(f"\nIn-sample vs out-of-sample rank correlation: {correlation:.2f}  (near 0 = top runs are mostly luck)")
    print("\nRobust choices (best average in BOTH periods):")
    for key, value in robust.items():
        print(f"  {key} = {value}")
    print(f"\nDetails: outputs/best_optimization_summary_{args.focus}.txt")


def _init_worker(data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], oos_start: pd.Timestamp,
                 space: dict[str, list[Any]], focus: str) -> None:
    _worker_data.update(data=data, oos_start=oos_start, space=space, focus=focus, feature_cache={})


def _evaluate(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    label, config = job
    prices, dividends, signal_prices = _worker_data["data"]
    oos_start = _worker_data["oos_start"]
    results = run_backtest(config, prices, dividends, signal_prices, _worker_data["feature_cache"])

    curve = results.equity_curve.set_index("date")
    if _worker_data["focus"] == "swing" and config["swing"].get("enabled", False):
        series = curve["swing_contribution"]
        starting_value = config["starting_cash"] * float(config["swing"].get("capital_fraction", 0.0))
    else:
        series = curve["equity"]
        starting_value = config["starting_cash"]

    risk_free = float(config.get("risk_free_rate", 0.0))
    in_sample = performance_metrics(series.loc[: oos_start - pd.Timedelta(days=1)], risk_free, starting_value)
    # Out-of-sample starts from the last in-sample value so the first OOS day's return is included.
    out_of_sample = performance_metrics(series.loc[series.index[series.index < oos_start][-1] :], risk_free)
    summary = results.performance_summary.iloc[0]

    row: dict[str, Any] = {"label": label, **extract_parameters(config, _worker_data["space"])}
    row["in_sample_score"] = optimization_score(in_sample)
    row["out_of_sample_score"] = optimization_score(out_of_sample)
    row.update({f"is_{key}": value for key, value in in_sample.items()})
    row.update({f"oos_{key}": value for key, value in out_of_sample.items()})
    row["portfolio_CAGR"] = summary["CAGR"]
    row["portfolio_max_drawdown"] = summary["max_drawdown"]
    for key in ["swing_trades", "swing_win_rate", "swing_avg_return", "swing_total_pnl", "swept_to_core",
                "cumulative_dividends", "forward_annual_dividends", "ending_equity"]:
        row[key] = summary[key]
    return row


def generate_configs(base_config: dict[str, Any], max_runs: int, space: dict[str, list[Any]]):
    """Random sample of the search space (deterministic seed so runs are repeatable)."""
    keys = list(space)
    grid = list(product(*space.values()))
    random.Random(42).shuffle(grid)

    for values in grid[:max_runs]:
        config = copy.deepcopy(base_config)
        config["swing"]["enabled"] = True
        for key, value in zip(keys, values):
            section, name = key.split(".")
            config.setdefault(section, {})[name] = value.split("+") if key == "swing.setups" else value
        yield config


def extract_parameters(config: dict[str, Any], space: dict[str, list[Any]]) -> dict[str, Any]:
    params = {}
    for key in space:
        section, name = key.split(".")
        value = config.get(section, {}).get(name)
        params[key] = "+".join(value) if isinstance(value, list) else value
    return params


def optimization_score(metrics: dict[str, float]) -> float:
    """Growth, penalized for drawdown and volatility. Costs are already inside CAGR."""
    if not metrics:
        return float("nan")
    return metrics["CAGR"] - 0.35 * abs(metrics["max_drawdown"]) - 0.15 * metrics["annual_volatility"]


def robust_choices(searched: pd.DataFrame, space: dict[str, list[Any]]) -> dict[str, Any]:
    """For each parameter, the value with the best average score across both periods.

    Averaging over hundreds of runs is much less luck-driven than taking the single top run.
    """
    choices = {}
    for key in space:
        means = searched.groupby(key)[["in_sample_score", "out_of_sample_score"]].mean()
        # Rank in each period separately and add the ranks, so a value must do well in both.
        combined = means.rank(ascending=False).sum(axis=1) + 0.001 * means.rank(ascending=False)["in_sample_score"]
        choices[key] = combined.idxmin()
    return choices


def _render(key: str, value: Any) -> str:
    if key == "swing.setups":
        return "[" + ", ".join(f'"{name}"' for name in str(value).split("+")) + "]"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (bool,)) or str(value) in {"True", "False"}:
        return str(value).lower()
    return str(value)


def write_best_config_summary(
    best: pd.Series,
    baselines: pd.DataFrame,
    robust: dict[str, Any],
    correlation: float,
    oos_start: pd.Timestamp,
    space: dict[str, list[Any]],
    focus: str,
    path: Path,
) -> None:
    def describe(row: pd.Series) -> str:
        return (
            f"in-sample CAGR {row['is_CAGR']:.2%} / DD {row['is_max_drawdown']:.2%}   "
            f"out-of-sample CAGR {row['oos_CAGR']:.2%} / DD {row['oos_max_drawdown']:.2%}   "
            f"(whole portfolio CAGR {row['portfolio_CAGR']:.2%})"
        )

    scored_on = "the swing engine (sleeve value + profits swept out)" if focus == "swing" else "the total portfolio"
    lines = [
        f"Optimization focus: {focus}. CAGR/drawdown are for {scored_on}.",
        f"Settings were chosen on data before {oos_start.date()} only.",
        "",
        f"Top run:  {describe(best)}",
    ]
    lines += [f"{row['label']}:  {describe(row)}" for _, row in baselines.iterrows()]
    lines += [
        "",
        f"In-sample vs out-of-sample rank correlation across all runs: {correlation:.2f}",
        "",
        "RECOMMENDED: robust choices (best average in both periods). Paste under the matching [section]:",
    ]
    lines += _toml_sections(robust)
    lines += ["Top single run (more likely to be luck):"]
    lines += _toml_sections({key: best[key] for key in space})
    path.write_text("\n".join(lines), encoding="utf-8")


def _toml_sections(values: dict[str, Any]) -> list[str]:
    sections: dict[str, list[str]] = {}
    for key, value in values.items():
        section, name = key.split(".")
        sections.setdefault(section, []).append(f"{name} = {_render(key, value)}")
    lines = []
    for section, entries in sections.items():
        lines += [f"[{section}]", *entries, ""]
    return lines


if __name__ == "__main__":
    main()
