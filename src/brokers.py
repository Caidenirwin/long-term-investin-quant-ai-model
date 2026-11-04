"""Brokers for live_trader.py.

SimBroker   - forward test: no real orders; fills are simulated at the actual closing price.
IBKRBroker  - Interactive Brokers through IB Gateway / TWS, using market-on-close (MOC) orders,
              which fill at the official closing price, matching the backtest's assumption.
"""
from __future__ import annotations

import math
from typing import Any


class SimBroker:
    name = "sim"
    is_real = False

    def positions(self) -> dict[str, float] | None:
        return None  # the book is the only record in a forward test

    def cash_cad(self) -> float | None:
        return None

    def submit(self, orders: list[dict[str, Any]]) -> None:
        for order in orders:
            order["status"] = "simulated"

    def close(self) -> None:
        pass


def ib_contract_spec(ticker: str) -> tuple[str, str, str | None]:
    """Yahoo ticker -> (IB symbol, currency, primary exchange).  RCI-B.TO -> RCI.B on TSE; BRK-B -> BRK B."""
    if ticker.endswith(".TO"):
        return ticker[:-3].replace("-", "."), "CAD", "TSE"
    return ticker.replace("-", " "), "USD", None


class IBKRBroker:
    """Talks to a running IB Gateway or TWS. Paper and live use different ports and account types."""

    name = "ibkr"
    is_real = True

    def __init__(self, live_config: dict[str, Any], paper: bool) -> None:
        try:
            import ib_async
        except ImportError as error:
            raise RuntimeError("IBKR trading needs the ib_async package:  python -m pip install ib_async") from error
        self._ib_async = ib_async
        self.paper = paper
        port = int(live_config.get("ib_port_paper" if paper else "ib_port_live", 4002 if paper else 4001))
        self.ib = ib_async.IB()
        self.ib.connect(live_config.get("ib_host", "127.0.0.1"), port,
                        clientId=int(live_config.get("ib_client_id", 17)), timeout=20)
        accounts = self.ib.managedAccounts()
        self.account = live_config.get("ib_account") or (accounts[0] if accounts else "")
        if self.account not in accounts:
            raise RuntimeError(f"Account {self.account!r} not found. IB Gateway reports: {accounts}")
        # IBKR paper accounts start with "D" (e.g. DU1234567). Refuse to mix paper and live up.
        is_paper_account = self.account.startswith("D")
        if paper and not is_paper_account:
            raise RuntimeError(f"Paper mode but connected to LIVE account {self.account}. Refusing to trade.")
        if not paper and is_paper_account:
            raise RuntimeError(f"Live mode but connected to paper account {self.account}. Check the port.")
        self._contracts: dict[str, Any] = {}

    def _contract(self, ticker: str) -> Any:
        if ticker not in self._contracts:
            symbol, currency, primary = ib_contract_spec(ticker)
            stock = self._ib_async.Stock(symbol, "SMART", currency, primaryExchange=primary or "")
            qualified = self.ib.qualifyContracts(stock)
            if not qualified:
                raise RuntimeError(f"IBKR does not recognise {ticker} ({symbol} {currency})")
            self._contracts[ticker] = qualified[0]
        return self._contracts[ticker]

    def positions(self) -> dict[str, float]:
        """Positions keyed by our tickers. Holdings outside the strategy are reported as IB symbols."""
        result: dict[str, float] = {}
        for position in self.ib.positions(self.account):
            contract = position.contract
            if contract.secType != "STK":
                continue
            if contract.currency == "CAD":
                ticker = contract.symbol.replace(".", "-") + ".TO"
            else:
                ticker = contract.symbol.replace(" ", "-")
            result[ticker] = result.get(ticker, 0.0) + float(position.position)
        return result

    def cash_cad(self) -> float | None:
        values = self.ib.accountSummary(self.account)
        for currency in ("BASE", "CAD"):
            for value in values:
                if value.tag == "TotalCashValue" and value.currency == currency:
                    return float(value.value)
        return None

    def working_orders(self) -> list[str]:
        return [f"{t.order.action} {t.order.totalQuantity} {t.contract.symbol}" for t in self.ib.openTrades()]

    def submit(self, orders: list[dict[str, Any]]) -> None:
        for order in orders:
            contract = self._contract(order["ticker"])
            ib_order = self._ib_async.Order(
                action=order["action"], totalQuantity=int(math.floor(order["quantity"])),
                orderType="MOC", tif="DAY", account=self.account,
            )
            if ib_order.totalQuantity <= 0:
                order["status"] = "skipped (fractional quantity)"
                continue
            trade = self.ib.placeOrder(contract, ib_order)
            self.ib.sleep(1.0)
            order["broker_id"] = trade.order.orderId
            order["status"] = trade.orderStatus.status
            if trade.orderStatus.status in ("Cancelled", "ApiCancelled", "Inactive"):
                messages = "; ".join(entry.message for entry in trade.log if entry.message)
                order["status"] = f"REJECTED: {messages or trade.orderStatus.status}"

    def close(self) -> None:
        self.ib.disconnect()
