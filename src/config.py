import copy
from pathlib import Path
import tomllib
from typing import Any


def research_config(config: dict[str, Any]) -> dict[str, Any]:
    """The settings used to judge the RULES (audit, optimizer, walk-forward, new-company test).

    Those tools use a fixed C$100k account with no deposits so results stay comparable over time and
    aren't dominated by account size or deposit timing. main.py and the dashboard use your real plan.
    """
    research = copy.deepcopy(config)
    research["starting_cash"] = float(config.get("research_starting_cash", 100000))
    research.setdefault("contributions", {})["monthly"] = 0.0
    return research


def load_config(path: Path) -> dict[str, Any]:
    """Load the editable TOML settings file."""
    with path.open("rb") as file:
        config = tomllib.load(file)

    required = [
        "start_date",
        "starting_cash",
        "target_allocation",
        "sectors",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    if not config.get("end_date"):
        config["end_date"] = None

    target_total = sum(config["target_allocation"].values())
    if abs(target_total - 1.0) > 0.001:
        raise ValueError("target_allocation must sum to 1.0")

    unknown = set(config["target_allocation"]) - set(config["sectors"])
    if unknown:
        raise ValueError(f"target_allocation has sectors with no tickers: {sorted(unknown)}")

    swing_fraction = float(config.get("swing", {}).get("capital_fraction", 0.0))
    if not 0.0 <= swing_fraction <= 1.0:
        raise ValueError("swing.capital_fraction must be between 0 and 1")

    mode = config.get("dividends", {}).get("mode", "drip")
    if mode not in {"drip", "smart", "cash"}:
        raise ValueError("dividends.mode must be drip, smart, or cash")

    return config
