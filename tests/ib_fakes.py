"""An `ib_async`-shaped fake: a paper gateway whose every order call is recorded.

Shared by the smoke, kill-switch, hold-book, drill and command tests. It fakes
the SDK surface this package touches and nothing else, so a new SDK call in the
source fails here with `AttributeError` instead of passing silently.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from crucible.runner import RunContext

#: Tuesday 2026-09-08, 10:00 ET: a regular NYSE session (Labor Day was 09-07).
T0 = dt.datetime(2026, 9, 8, 14, 0, tzinfo=dt.UTC)
DAY = dt.date(2026, 9, 8)
RUN_ID = "01JG0000000000000000000000"
KILL_SWITCH_REF = "crucible-kill-switch"


class FakeClock:
    def __init__(self, start: dt.datetime = T0) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += dt.timedelta(seconds=seconds)


def _entry(time: dt.datetime, status: str) -> SimpleNamespace:
    return SimpleNamespace(time=time, status=status)


class FakeIB:
    def __init__(
        self,
        *,
        clock: FakeClock | None = None,
        accounts: tuple[str, ...] = ("DU1",),
        positions: dict[str, int] | None = None,
        cash: float = 1_000.0,
        fills: bool = True,
        connect_error: BaseException | None = None,
    ) -> None:
        self.clock = clock or FakeClock()
        self.accounts = list(accounts)
        self.book = dict(positions or {})
        self.cash = cash
        self.fills = fills
        self.connect_error = connect_error
        self.connected_with: dict | None = None
        self.disconnected = 0
        self.order_calls: list[tuple] = []
        self.cancel_calls = 0
        self.foreign_on_sleep: list[str] = []
        self._trades: list[SimpleNamespace] = []
        self._next_id = 1

    # connection
    def connect(self, host, port, clientId, timeout, readonly):  # noqa: N803 - SDK spelling
        if self.connect_error is not None:
            raise self.connect_error
        self.connected_with = {
            "host": host,
            "port": port,
            "clientId": clientId,
            "timeout": timeout,
            "readonly": readonly,
        }

    def disconnect(self):
        self.disconnected += 1

    # reads
    def managedAccounts(self):  # noqa: N802 - SDK spelling
        return list(self.accounts)

    def positions(self):
        return [
            SimpleNamespace(
                account=self.accounts[0],
                contract=SimpleNamespace(symbol=symbol, secType="STK"),
                position=float(shares),
            )
            for symbol, shares in self.book.items()
        ]

    def accountSummary(self):  # noqa: N802 - SDK spelling
        return [
            SimpleNamespace(
                account=self.accounts[0], tag="TotalCashValue", currency="USD", value=str(self.cash)
            )
        ]

    def openTrades(self):  # noqa: N802 - SDK spelling
        return [t for t in self._trades if t.orderStatus.status in ("PreSubmitted", "Submitted")]

    def trades(self):
        return list(self._trades)

    # order-capable calls
    def reqGlobalCancel(self):  # noqa: N802 - SDK spelling
        self.cancel_calls += 1
        for trade in self.openTrades():
            trade.orderStatus.status = "Cancelled"
            trade.log.append(_entry(self.clock(), "Cancelled"))

    def placeOrder(self, contract, order):  # noqa: N802 - SDK spelling
        self.order_calls.append((contract, order))
        return self._add(contract, order, "Submitted", self.clock())

    def sleep(self, seconds):
        self.clock.advance(seconds)
        if self.fills:
            for trade in self._trades:
                if (
                    trade.orderStatus.status == "Submitted"
                    and trade.order.orderRef == KILL_SWITCH_REF
                ):
                    sign = 1 if trade.order.action == "BUY" else -1
                    symbol = trade.contract.symbol
                    self.book[symbol] = self.book.get(symbol, 0) + sign * int(
                        trade.order.totalQuantity
                    )
                    if self.book[symbol] == 0:
                        del self.book[symbol]
                    trade.orderStatus.status = "Filled"
                    trade.log.append(_entry(self.clock(), "Filled"))
        for symbol in self.foreign_on_sleep:
            self.working_order(symbol, ref="rebalance")
        self.foreign_on_sleep = []

    # test helpers
    def working_order(self, symbol="ZZZ", *, ref="", at=None, status="Submitted", qty=1):
        order = SimpleNamespace(action="BUY", totalQuantity=qty, orderRef=ref, tif="DAY")
        return self._add(SimpleNamespace(symbol=symbol), order, status, at or self.clock())

    def _add(self, contract, order, status, at):
        order.orderId = self._next_id
        self._next_id += 1
        trade = SimpleNamespace(
            contract=contract,
            order=order,
            orderStatus=SimpleNamespace(status=status),
            log=[_entry(at, status)],
        )
        self._trades.append(trade)
        return trade


class FakeSdk:
    def __init__(self, ib: FakeIB) -> None:
        self.ib = ib
        self.loaded = 0

    def __call__(self) -> FakeSdk:  # usable directly as an `sdk_loader`
        self.loaded += 1
        return self

    def IB(self):  # noqa: N802 - SDK spelling
        return self.ib

    @staticmethod
    def Stock(symbol, exchange, currency):  # noqa: N802 - SDK spelling
        return SimpleNamespace(symbol=symbol, exchange=exchange, currency=currency)

    @staticmethod
    def MarketOrder(action, quantity):  # noqa: N802 - SDK spelling
        return SimpleNamespace(action=action, totalQuantity=quantity, orderRef="", tif="")


def ctx_for(store, job: str, *, day: dt.date = DAY, started: dt.datetime = T0) -> RunContext:
    return RunContext(
        run_id=RUN_ID,
        job=job,
        trading_day=day,
        calendar_date=day,
        store=store,
        seed=0,
        started=started,
    )


def fake_run_job(job, fn, *, store, discriminator=None, **_ignored):
    """Stands in for `crucible.runner.run_job` ONLY where the job name is not yet
    admitted by the harness contract (`trader.kill_switch`, `trader.fire_drill`);
    the real runner is exercised separately by each module's contract-dependency test."""
    ctx = ctx_for(store, job)
    if callable(discriminator):
        discriminator(ctx)
    fn(ctx)
    return ctx


def environ_for(store_path, **extra) -> dict[str, str]:
    values = {
        "CRUCIBLE_TRADER_STORE_URI": f"file://{store_path}",
        "CRUCIBLE_TRADER_IB_HOST": "127.0.0.1",
        "CRUCIBLE_TRADER_IB_PORT": "4002",
        "CRUCIBLE_TRADER_IB_CLIENT_ID": "7",
    }
    values.update(extra)
    return values
