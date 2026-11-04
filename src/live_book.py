"""The live/forward-test book: what each sleeve owns, saved between daily runs.

A broker only knows total positions and cash. The strategy also needs to know which shares belong to
the dividend core vs the swing sleeve, each swing trade's entry, and orders still waiting to fill.
That lives here, in live/book_<mode>.json.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Book:
    mode: str
    capital: float
    created: str
    core_cash: float
    swing_cash: float
    swing_base: float
    core_shares: dict[str, float] = field(default_factory=dict)
    swing_positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    core_targets: dict[str, float] = field(default_factory=dict)
    pending: list[dict[str, Any]] = field(default_factory=list)
    last_rebalance_key: str | None = None
    last_planned_date: str | None = None
    last_dividend_date: str | None = None
    cumulative_swept: float = 0.0
    cumulative_dividends: float = 0.0
    cumulative_deposits: float = 0.0
    last_deposit_month: str | None = None
    next_order_id: int = 1

    @classmethod
    def new(cls, mode: str, capital: float, swing_fraction: float, today: str) -> "Book":
        swing_cash = capital * swing_fraction
        return cls(
            mode=mode,
            capital=capital,
            created=today,
            core_cash=capital - swing_cash,
            swing_cash=swing_cash,
            swing_base=swing_cash,
            cumulative_deposits=capital,
            last_deposit_month=today[:7],
        )

    def deposit(self, amount: float, swing_fraction: float) -> None:
        """Add money, split between the sleeves. Deposits raise the swing base so they're never swept as profit."""
        to_swing = amount * swing_fraction
        self.swing_cash += to_swing
        self.swing_base += to_swing
        self.core_cash += amount - to_swing
        self.cumulative_deposits += amount

    @classmethod
    def load(cls, path: Path) -> "Book":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        # Write to a temp file first so a crash mid-write can never corrupt the book.
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        os.replace(temp, path)

    def total_shares(self, ticker: str) -> float:
        swing = self.swing_positions.get(ticker, {}).get("shares", 0.0)
        return self.core_shares.get(ticker, 0.0) + swing

    def tickers(self) -> set[str]:
        return {t for t, q in self.core_shares.items() if q > 0} | set(self.swing_positions)
