from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def performance_metrics(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
    starting_value: float | None = None,
) -> dict[str, float]:
    """Return, risk, and calendar-year stats for a dated equity series."""
    equity = equity.dropna()
    if len(equity) < 2:
        return {}
    start_value = float(starting_value if starting_value is not None else equity.iloc[0])
    ending_value = float(equity.iloc[-1])
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    cagr = (ending_value / start_value) ** (1 / years) - 1
    returns = equity.pct_change().dropna()
    annual_volatility = returns.std() * np.sqrt(252) if len(returns) else 0.0
    drawdowns = equity / equity.cummax() - 1
    yearly = pd.concat([pd.Series([start_value], index=[equity.index[0]]), equity.resample("YE").last()])
    annual_returns = yearly.pct_change().dropna()

    return {
        "total_return": ending_value / start_value - 1,
        "CAGR": cagr,
        "max_drawdown": float(drawdowns.min()),
        "annual_volatility": annual_volatility,
        "sharpe_ratio": (cagr - risk_free_rate) / annual_volatility if annual_volatility else np.nan,
        "best_year": annual_returns.max() if len(annual_returns) else np.nan,
        "worst_year": annual_returns.min() if len(annual_returns) else np.nan,
    }


def calculate_performance_summary(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    swing_trades: pd.DataFrame,
    starting_cash: float,
    risk_free_rate: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    equity = equity_curve.set_index(pd.to_datetime(equity_curve["date"]))
    # Returns come from the time-weighted index so monthly deposits don't count as performance.
    if "twr_index" in equity.columns:
        metrics = performance_metrics(equity["twr_index"], risk_free_rate, 1.0)
    else:
        metrics = performance_metrics(equity["equity"], risk_free_rate, starting_cash)
    last = equity.iloc[-1]

    if not swing_trades.empty:
        wins = swing_trades["pnl"] > 0
        swing_stats = {
            "swing_trades": len(swing_trades),
            "swing_win_rate": float(wins.mean()),
            "swing_avg_return": float(swing_trades["return"].mean()),
            "swing_total_pnl": float(swing_trades["pnl"].sum()),
        }
    else:
        swing_stats = {"swing_trades": 0, "swing_win_rate": np.nan, "swing_avg_return": np.nan, "swing_total_pnl": 0.0}

    summary = {
        **metrics,
        "number_of_trades": len(trades),
        "turnover": _calculate_turnover(trades, equity["equity"].mean()),
        "cumulative_dividends": last["cumulative_dividends"],
        "withdrawn_dividends": last["withdrawn_dividends"],
        "swept_to_core": last["cumulative_swept"],
        **swing_stats,
        "ending_core_value": last["core_value"],
        "ending_swing_value": last["swing_value"],
        "ending_equity": last["equity"],
        **(extra or {}),
    }
    return pd.DataFrame([summary])


def save_outputs(results: Any, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    results.equity_curve.to_csv(output_dir / "equity_curve.csv", index=False)
    results.trades.to_csv(output_dir / "trades.csv", index=False)
    results.swing_trades.to_csv(output_dir / "swing_trades.csv", index=False)
    results.positions.to_csv(output_dir / "positions.csv", index=False)
    results.performance_summary.to_csv(output_dir / "performance_summary.csv", index=False)
    results.sector_weights.to_csv(output_dir / "sector_weights.csv", index=False)
    results.dividend_income.to_csv(output_dir / "dividend_income_by_year.csv", index=False)


def print_summary(summary: pd.DataFrame) -> None:
    row = summary.iloc[0]
    pct = lambda key: f"{row[key]:.2%}" if pd.notna(row.get(key)) else "n/a"
    money = lambda key: f"C${row[key]:,.0f}" if pd.notna(row.get(key)) else "n/a"
    lines = []
    if row.get("deposits_core", 0) + row.get("deposits_swing", 0) > 0:
        lines += [
            f"You deposited       {money('total_deposited')}  (start + monthly deposits)",
            f"Worth now           {money('ending_equity')}  -> gain {money('gain')}",
            f"Same deposits in TSX index (XIC): {money('benchmark_same_deposits')}",
            "",
        ]
    lines += [
        f"CAGR                {pct('CAGR')}   (benchmark {pct('benchmark_CAGR')})  time-weighted, deposits excluded",
        f"Max drawdown        {pct('max_drawdown')}   (benchmark {pct('benchmark_max_drawdown')})",
        f"Volatility          {pct('annual_volatility')}",
        f"Sharpe              {row['sharpe_ratio']:.2f}",
        f"Ending equity       {money('ending_equity')}  (core {money('ending_core_value')}, swing {money('ending_swing_value')})",
        f"Swing trades        {int(row['swing_trades'])}  win rate {pct('swing_win_rate')}  avg {pct('swing_avg_return')}",
        f"Swing P&L           {money('swing_total_pnl')}  swept to dividend core {money('swept_to_core')}",
        f"Dividends received  {money('cumulative_dividends')}",
        f"Forward income      {money('forward_annual_dividends')}/yr  (yield {pct('forward_dividend_yield')})",
        f"Trades / turnover   {int(row['number_of_trades'])} / {row['turnover']:.1f}x",
    ]
    print("\n".join(lines))


def print_industry_breakdown(positions: pd.DataFrame) -> None:
    """Dividend core by industry: weight, holdings, yield, and how much income each industry produces."""
    core = positions[positions["sleeve"] == "core"] if not positions.empty else positions
    if core.empty:
        return
    by_industry = core.groupby("sector").agg(
        value=("market_value", "sum"),
        income=("forward_annual_income", "sum"),
        holdings=("ticker", lambda tickers: ", ".join(tickers)),
    )
    total_value, total_income = by_industry["value"].sum(), by_industry["income"].sum()
    by_industry = by_industry.sort_values("value", ascending=False)

    print("\nDIVIDEND CORE BY INDUSTRY")
    print(f"  {'Industry':24s} {'Weight':>6s} {'':12s} {'Yield':>5s} {'Income/yr':>10s} {'% income':>8s}  Holdings")
    for industry, row in by_industry.iterrows():
        weight = row["value"] / total_value
        print(
            f"  {industry[:24]:24s} {weight:6.1%} {'#' * round(weight * 50):12s} {row['income'] / row['value']:5.1%} "
            f"C${row['income']:>8,.0f} {row['income'] / total_income:8.0%}  {row['holdings']}"
        )


def _calculate_turnover(trades: pd.DataFrame, average_equity: float) -> float:
    if trades.empty or average_equity <= 0:
        return 0.0
    traded_value = trades["gross_value"].abs().sum()
    return traded_value / average_equity
