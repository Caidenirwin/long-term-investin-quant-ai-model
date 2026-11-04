"""Train your AI.

    python -m ai_investor.train                 # use cached data (downloads the first time)
    python -m ai_investor.train --refresh-data  # re-download everything first

Two models (gradient-boosted decision trees, trained locally with scikit-learn):
  swing     - which stocks will beat the others over the next ~month
  longterm  - which stocks will beat the others over the next ~year

Walk-forward test: for every year from first_test_year, a model trained ONLY on earlier data predicts that
year. Those predictions are graded against what really happened and compared with a simple momentum rule,
so you can see whether the AI adds anything. Then the final models are trained on all the data and saved.
"""
from __future__ import annotations

import argparse
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from ai_investor.dataset import (
    CATEGORICAL, FEATURES, ROOT, build_dataset, index_groups, load_ai_config, load_market_data, universe,
)
from src.backtest import run_backtest
from src.config import load_config, research_config
from src.data import load_cached_price_data

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
BASELINE = "rank_mom_12_1"  # the simple rule the AI has to beat: buy last year's winners
TOP_FRACTION = 0.10


def make_model() -> HistGradientBoostingRegressor:
    # Conservative settings: big leaves and strong regularisation, because stock returns are mostly noise and
    # a flexible model would happily memorise it.
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=250, max_leaf_nodes=15, min_samples_leaf=1000, l2_regularization=5.0,
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL], random_state=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AI investor.")
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()

    ai = load_ai_config()
    horizons = {"swing": int(ai["models"]["swing_horizon_days"]), "longterm": int(ai["models"]["longterm_horizon_days"])}
    _, sectors, names = universe()
    groups = index_groups()
    data = load_market_data(refresh=args.refresh_data)
    started = time.time()
    table = build_dataset(data, int(ai["data"]["sample_every_days"]), horizons, sectors)
    print(f"Dataset: {len(table):,} stock-weeks, {table['ticker'].nunique()} stocks, "
          f"{table['date'].min().date()} to {table['date'].max().date()} ({time.time() - started:.0f}s)")

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    sections = []
    for name, horizon in horizons.items():
        print(f"\n=== {name} model ({horizon}-day horizon) ===")
        yearly, predictions, importance, yearly_models = walk_forward(
            table, name, horizon, int(ai["models"]["first_test_year"]))
        groups_table = group_breakdown(predictions, name, groups)
        portfolio = portfolio_test(predictions, data["adjusted"], name, horizon)
        # Same test on companies that were already large in 2012 (your stock lists): much less survivorship bias.
        fair = fair_universe()
        print(f"  Same test on the {len(fair)} companies that were already large in 2012 (less hindsight):")
        portfolio_fair = portfolio_test(predictions[predictions["ticker"].isin(fair)], data["adjusted"], name, horizon,
                                        picks=5 if name == "swing" else 10)
        breakout = breakout_check(yearly_models, data, sectors) if name == "swing" else {}
        final = train_final(table, name)
        joblib.dump({
            "model": final, "features": FEATURES, "horizon": horizon, "name": name,
            "trained_on": datetime.now().strftime("%Y-%m-%d"),
            "data_through": str(table["date"].max().date()),
            "sector_names": sorted(set(sectors.values()) | {"Other / Unknown"}),
            "walk_forward": yearly.to_dict("records"), "portfolio": portfolio, "portfolio_fair": portfolio_fair,
            "breakout_check": breakout, "importance": importance, "groups": groups_table.to_dict("records"),
        }, MODELS_DIR / f"{name}.joblib")
        print(f"Saved {MODELS_DIR / f'{name}.joblib'}")
        sections.append(report_section(name, horizon, yearly, portfolio, importance, portfolio_fair, breakout,
                                       groups_table))

    report = REPORTS_DIR / "training_report.md"
    report.write_text(report_header(table) + "\n".join(sections), encoding="utf-8")
    print(f"\nTraining report: {report}")


def walk_forward(table: pd.DataFrame, name: str, horizon: int, first_year: int):
    label = f"label_{name}"
    purge = pd.Timedelta(days=int(horizon * 1.5) + 7)  # training labels must end before the test year starts
    rows, predictions, importance = [], [], {}
    models: dict[int, HistGradientBoostingRegressor] = {}
    last_year = int(table["date"].dt.year.max())
    for year in range(first_year, last_year + 1):
        test = table[(table["date"].dt.year == year) & table[label].notna()]
        train = table[(table["date"] < pd.Timestamp(f"{year}-01-01") - purge) & table[label].notna()]
        if test.empty or len(train) < 20000:
            continue
        model = make_model().fit(train[FEATURES], train[label])
        models[year] = model
        scored = test[["date", "ticker", "sector", label, f"fwd_{name}", BASELINE]].copy()
        scored["prediction"] = model.predict(test[FEATURES])
        predictions.append(scored)

        # Never-seen companies: a second model learns only from half the companies, then is graded on the other
        # half. If the AI had just memorised specific stocks, it would fail here.
        half = test["ticker"].map(_half) == 1
        unseen_model = make_model().fit(train.loc[train["ticker"].map(_half) == 0, FEATURES],
                                        train.loc[train["ticker"].map(_half) == 0, label])
        other_half = scored[half.to_numpy()].copy()
        other_half["unseen_prediction"] = unseen_model.predict(test.loc[half, FEATURES])
        ic_unseen = _mean_ic(other_half, "unseen_prediction", label)
        ic_seen = _mean_ic(other_half, "prediction", label)

        ic_model = _mean_ic(scored, "prediction", label)
        ic_base = _mean_ic(scored, BASELINE, label)
        spread_model = _top_spread(scored, "prediction", f"fwd_{name}")
        spread_base = _top_spread(scored, BASELINE, f"fwd_{name}")
        rows.append({"year": year, "ic_ai": ic_model, "ic_momentum": ic_base,
                     "top10_vs_all_ai": spread_model, "top10_vs_all_momentum": spread_base, "test_rows": len(test),
                     "ic_unseen_companies": ic_unseen, "ic_same_companies_full_model": ic_seen})
        print(f"  {year}: AI score {ic_model:+.3f} (momentum rule {ic_base:+.3f}) | never-seen companies "
              f"{ic_unseen:+.3f} | top 10% beat the average stock by {spread_model:+.2%} (momentum {spread_base:+.2%})")
        if year == last_year or (name == "longterm" and year == last_year - 1):
            importance = _importance(model, test, label)
    yearly = pd.DataFrame(rows)
    if not yearly.empty:
        print(f"  Average: AI score {yearly['ic_ai'].mean():+.3f} vs momentum {yearly['ic_momentum'].mean():+.3f}; "
              f"AI better in {(yearly['ic_ai'] > yearly['ic_momentum']).sum()} of {len(yearly)} years")
        print(f"  Never-seen companies: {yearly['ic_unseen_companies'].mean():+.3f} (vs {yearly['ic_same_companies_full_model'].mean():+.3f} "
              "for the full model on the same stocks)")
    return yearly, pd.concat(predictions) if predictions else pd.DataFrame(), importance, models


def _half(ticker: str) -> int:
    """Stable 50/50 split of companies (the same every run)."""
    return zlib.crc32(ticker.encode()) % 2


def group_breakdown(predictions: pd.DataFrame, name: str, groups: dict[str, str]) -> pd.DataFrame:
    """Does the AI work on every kind of stock? Skill measured within each group (large/mid/small, US/Canada)."""
    label, forward = f"label_{name}", f"fwd_{name}"
    scored = predictions.assign(group=predictions["ticker"].map(lambda t: groups.get(t, "Your own lists / ETFs")))
    rows = []
    for group, part in scored.groupby("group"):
        if part["ticker"].nunique() < 20:
            continue
        # Re-rank inside the group so each group is judged against its own members.
        part = part.assign(group_label=part.groupby("date")[forward].rank(pct=True))
        rows.append({
            "group": group, "stocks": int(part["ticker"].nunique()),
            "ic_ai": _mean_ic(part, "prediction", "group_label"),
            "ic_momentum": _mean_ic(part, BASELINE, "group_label"),
            "top10_vs_all_ai": _top_spread(part, "prediction", forward),
        })
    table = pd.DataFrame(rows).sort_values("stocks", ascending=False)
    print("  By kind of stock (skill within each group, years the model never saw):")
    for row in table.itertuples():
        print(f"    {row.group:34s} {row.stocks:4d} stocks  AI {row.ic_ai:+.3f}  momentum {row.ic_momentum:+.3f}  "
              f"top 10% vs average {row.top10_vs_all_ai:+.2%}")
    return table


def _mean_ic(scored: pd.DataFrame, column: str, label: str) -> float:
    """Average per-date rank correlation between a score and what actually happened (0 = useless, 1 = perfect)."""
    ics = scored.groupby("date").apply(lambda g: g[column].corr(g[label], method="spearman"), include_groups=False)
    return float(ics.mean())


def _top_spread(scored: pd.DataFrame, column: str, forward: str) -> float:
    """How much the top 10% by score beat the average stock, per period."""
    def spread(group: pd.DataFrame) -> float:
        cutoff = group[column].quantile(1 - TOP_FRACTION)
        return group.loc[group[column] >= cutoff, forward].mean() - group[forward].mean()
    return float(scored.groupby("date").apply(spread, include_groups=False).mean())


def _importance(model: HistGradientBoostingRegressor, test: pd.DataFrame, label: str) -> dict[str, float]:
    sample = test.sample(min(len(test), 15000), random_state=0)
    result = permutation_importance(model, sample[FEATURES], sample[label], n_repeats=3, random_state=0, n_jobs=1)
    pairs = sorted(zip(FEATURES, result.importances_mean), key=lambda p: -p[1])
    return {name: float(value) for name, value in pairs}


def portfolio_test(predictions: pd.DataFrame, adjusted: pd.DataFrame, name: str, horizon: int,
                   picks: int | None = None) -> dict[str, Any]:
    """Invest in the AI's top picks using only out-of-sample predictions, with trading costs, vs alternatives."""
    if predictions.empty:
        return {}
    dates = sorted(predictions["date"].unique())
    step = max(1, round(horizon / 5))  # samples are weekly; hold each portfolio for the full horizon
    rebalance = dates[::step]
    picks = picks or (10 if name == "swing" else 20)
    cost_per_turnover = 0.002  # 0.1% each way for fees and spreads
    benchmark = adjusted["XIC.TO"] if "XIC.TO" in adjusted else None

    results: dict[str, list[float]] = {"AI picks": [], "Momentum rule": [], "All stocks (equal weight)": [], "XIC (TSX)": []}
    previous: dict[str, set[str]] = {"AI picks": set(), "Momentum rule": set()}
    for date in rebalance:
        group = predictions[predictions["date"] == date].dropna(subset=[f"fwd_{name}"])
        if len(group) < picks * 3:
            continue
        for label, column in [("AI picks", "prediction"), ("Momentum rule", BASELINE)]:
            chosen = group.nlargest(picks, column)
            turnover = 1 - len(previous[label] & set(chosen["ticker"])) / picks if previous[label] else 1.0
            results[label].append(chosen[f"fwd_{name}"].mean() - turnover * cost_per_turnover)
            previous[label] = set(chosen["ticker"])
        results["All stocks (equal weight)"].append(group[f"fwd_{name}"].mean())
        if benchmark is not None:
            future = benchmark.shift(-horizon)
            results["XIC (TSX)"].append(float(future.loc[date] / benchmark.loc[date] - 1))

    periods_per_year = 252 / horizon
    summary = {}
    for label, returns in results.items():
        if not returns:
            continue
        r = np.array(returns, dtype=float)
        r = r[~np.isnan(r)]
        growth = np.prod(1 + r)
        years = len(r) / periods_per_year
        summary[label] = {
            "annual_return": float(growth ** (1 / years) - 1) if years > 0 else float("nan"),
            "periods": int(len(r)),
            "win_rate_vs_all": None,
        }
    ai, everything = np.array(results["AI picks"]), np.array(results["All stocks (equal weight)"])
    if len(ai) and len(ai) == len(everything):
        summary["AI picks"]["win_rate_vs_all"] = float((ai > everything).mean())
    summary["first_date"], summary["last_date"] = str(pd.Timestamp(rebalance[0]).date()), str(pd.Timestamp(rebalance[-1]).date())
    print(f"  Out-of-sample investing test ({summary['first_date']} to {summary['last_date']}, top {picks}, after costs):")
    for label in results:
        if label in summary:
            print(f"    {label:28s} {summary[label]['annual_return']:+.1%} per year")
    return summary


def fair_universe() -> set[str]:
    """Companies chosen for being large in 2012 (settings.toml stock lists + dividend list), not for later success."""
    settings = load_config(Path("config/settings.toml"))
    return set(settings["swing"].get("stock_universe", [])) | {t for ts in settings["sectors"].values() for t in ts}


def breakout_check(models: dict[int, HistGradientBoostingRegressor], data: dict[str, Any],
                   sectors: dict[str, str]) -> dict[str, Any]:
    """The test that matches how the advisor uses the AI: of the breakout trades your tested rules actually made,
    did the ones the AI ranked highly (using a model that never saw that year) do better than the rest?"""
    settings = research_config(load_config(Path("config/settings.toml")))
    trades = run_backtest(settings, *load_cached_price_data(settings)).swing_trades
    if trades.empty:
        return {}
    index = data["adjusted"].index
    trades = trades.assign(signal_date=[index[index < pd.Timestamp(d)][-1] if (index < pd.Timestamp(d)).any() else pd.NaT
                                        for d in trades["entry_date"]])
    trades = trades[trades["signal_date"].dt.year.isin(models.keys()) & trades["ticker"].isin(data["adjusted"].columns)]
    if trades.empty:
        return {}
    dates = pd.DatetimeIndex(sorted(trades["signal_date"].unique()))
    snapshot = build_dataset(data, 1, {}, sectors, dates=dates)
    snapshot["prediction"] = np.nan
    for year, model in models.items():
        rows = snapshot["date"].dt.year == year
        if rows.any():
            snapshot.loc[rows, "prediction"] = model.predict(snapshot.loc[rows, FEATURES])
    snapshot["ai_pct"] = snapshot.groupby("date")["prediction"].rank(pct=True)
    merged = trades.merge(snapshot[["date", "ticker", "ai_pct"]], left_on=["signal_date", "ticker"],
                          right_on=["date", "ticker"], how="inner")
    if len(merged) < 30:
        return {}
    high, low = merged[merged["ai_pct"] >= 0.5], merged[merged["ai_pct"] < 0.5]
    result = {
        "trades": int(len(merged)),
        "ai_high": {"trades": int(len(high)), "avg_return": float(high["return"].mean()), "win_rate": float((high["pnl"] > 0).mean())},
        "ai_low": {"trades": int(len(low)), "avg_return": float(low["return"].mean()), "win_rate": float((low["pnl"] > 0).mean())},
    }
    print(f"  Breakout trades your rules made ({len(merged)} since {models and min(models)}):")
    print(f"    AI rated in the top half:    {len(high):3d} trades, avg {result['ai_high']['avg_return']:+.2%}, "
          f"{result['ai_high']['win_rate']:.0%} winners")
    print(f"    AI rated in the bottom half: {len(low):3d} trades, avg {result['ai_low']['avg_return']:+.2%}, "
          f"{result['ai_low']['win_rate']:.0%} winners")
    return result


def train_final(table: pd.DataFrame, name: str) -> HistGradientBoostingRegressor:
    label = f"label_{name}"
    usable = table[table[label].notna()]
    print(f"  Training final {name} model on {len(usable):,} rows through {usable['date'].max().date()}...")
    return make_model().fit(usable[FEATURES], usable[label])


# ---------------------------------------------------------------------------- report


def report_header(table: pd.DataFrame) -> str:
    return (
        f"# AI training report\n\nTrained {datetime.now():%Y-%m-%d %H:%M} on {len(table):,} stock-weeks "
        f"({table['ticker'].nunique()} stocks, {table['date'].min().date()} to {table['date'].max().date()}).\n\n"
        "**How to read this.** *AI score* is the average rank correlation between the model's prediction and what "
        "really happened, graded only on years the model never saw. 0 means no skill; in stock picking, 0.02-0.05 "
        "is useful and above 0.10 is suspicious. *Momentum rule* is a simple, well-known strategy (buy last "
        "year's winners) the AI has to beat to be worth using.\n\n"
        "**Caveat.** The stock list is today's S&P 500 and TSX 60, so companies that failed or were removed are "
        "missing. That flatters *absolute* returns, so compare the AI with *All stocks* and *Momentum rule*, which "
        "share the same bias, rather than with XIC.\n\n"
    )


def report_section(name: str, horizon: int, yearly: pd.DataFrame, portfolio: dict[str, Any],
                   importance: dict[str, float], portfolio_fair: dict[str, Any] | None = None,
                   breakout: dict[str, Any] | None = None, groups_table: pd.DataFrame | None = None) -> str:
    lines = [f"## {name.title()} model ({horizon} trading days ahead)\n"]
    if not yearly.empty and "ic_unseen_companies" in yearly:
        lines += [f"**Never-seen companies:** a model trained on only half the companies scored "
                  f"{yearly['ic_unseen_companies'].mean():+.3f} on the other half, vs "
                  f"{yearly['ic_same_companies_full_model'].mean():+.3f} for the model that had seen them. Similar "
                  "numbers mean the AI learned general patterns, not specific stocks.\n"]
    if groups_table is not None and not groups_table.empty:
        lines += ["**By kind of stock** (skill within each group):\n",
                  "| Group | Stocks | AI score | Momentum rule | AI top 10% vs group average |", "|---|---|---|---|---|"]
        lines += [f"| {r.group} | {r.stocks} | {r.ic_ai:+.3f} | {r.ic_momentum:+.3f} | {r.top10_vs_all_ai:+.2%} |"
                  for r in groups_table.itertuples()]
        lines.append("")
    if not yearly.empty:
        lines += ["| Year | AI score | Momentum rule | AI top 10% vs average | Momentum top 10% vs average |",
                  "|---|---|---|---|---|"]
        for row in yearly.itertuples():
            lines.append(f"| {row.year} | {row.ic_ai:+.3f} | {row.ic_momentum:+.3f} | "
                         f"{row.top10_vs_all_ai:+.2%} | {row.top10_vs_all_momentum:+.2%} |")
        lines.append(f"| **Average** | **{yearly['ic_ai'].mean():+.3f}** | **{yearly['ic_momentum'].mean():+.3f}** | "
                     f"**{yearly['top10_vs_all_ai'].mean():+.2%}** | **{yearly['top10_vs_all_momentum'].mean():+.2%}** |\n")
    for title, result in [("today's S&P 500 + TSX 60 (strong survivorship bias: absolute returns are NOT achievable)", portfolio),
                          ("companies already large in 2012 (much less hindsight)", portfolio_fair or {})]:
        if not result:
            continue
        lines += [f"**Investing test on {title}** ({result['first_date']} to {result['last_date']}, out-of-sample, "
                  "after costs):\n", "| Strategy | Per year |", "|---|---|"]
        for label, stats in result.items():
            if isinstance(stats, dict):
                lines.append(f"| {label} | {stats['annual_return']:+.1%} |")
        lines.append("")
    if breakout:
        high, low = breakout["ai_high"], breakout["ai_low"]
        lines += ["**Does the AI improve your breakout trades?** Of the trades the tested rules actually made "
                  f"({breakout['trades']}, all graded by models that never saw that year):\n",
                  "| AI rating | Trades | Avg trade | Winners |", "|---|---|---|---|",
                  f"| Top half | {high['trades']} | {high['avg_return']:+.2%} | {high['win_rate']:.0%} |",
                  f"| Bottom half | {low['trades']} | {low['avg_return']:+.2%} | {low['win_rate']:.0%} |", ""]
    if importance:
        lines += ["**What the model pays most attention to** (how much its accuracy drops without each input):\n"]
        lines += [f"{i}. `{feature}`" for i, feature in enumerate(list(importance)[:8], start=1)]
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
