"""The order router: the trader's target book in, IB PAPER orders out, execution records back.

`alpha-engine-config-I11545` ruling 3 (2026-09-24): build the OrderRouter against
IB paper. It implements :class:`crucible_trader.session.OrderRouter` -- the
`batches(book)` / `send(batch)` seam `session_cycle` already drives, re-checking
the kill switch between batches -- and returns what the execution-shortfall
artifact (`crucible_trader.execution_shortfall`) and reconciliation need.

**OFF BY DEFAULT.** :func:`build_router` returns a :class:`ShadowRouter` -- which
yields no batch, so no order can be sent -- unless :data:`ROUTING_VAR` is set to
exactly :data:`ROUTING_ENABLED_VALUE`. Unset (or ``off``) is shadow. Any other
value is a misconfiguration and raises; a typo never reads as "on", and never as
a silent "off" either. Nothing in this repository sets the variable.

**PAPER ONLY, checked three times, each failing closed:**

1. :class:`RouterConfig` refuses an account id that is not ``DU…`` (IB paper)
   and a port that is not :data:`PAPER_GATEWAY_PORT` -- at construction, before
   any socket is opened. There is no live port and no live account in this
   module except in the refusals' own messages.
2. :func:`crucible_trader.broker_session.connect` refuses a gateway managing any
   non-``DU`` account.
3. :class:`PaperOrderRouter` refuses a session whose managed accounts are not
   exactly the configured paper account, and every order names that account.

**The kill switch is checked before EVERY order**, not only per batch:
`kill_switch.assert_trading_permitted` is called inside :meth:`PaperOrderRouter.send`
immediately before each `placeOrder`, on top of `session_cycle`'s per-batch check.

**Risk caps are checked on the WHOLE plan before any batch is returned**: a
per-order notional cap, a per-session notional cap (both required, no defaults),
the book's own gross (sum of tradable weights <= 1, long only), and a refusal to
touch any position the book does not know about -- on an account shared with
another trader, a delta against the whole account would liquidate the other
trader's names. A breach raises and sends nothing.

**Idempotent per trading day.** Every order carries an `orderRef` whose key is
``crucible-trader|{trading_day}|{symbol}``; the rest of the ref records the
sizing decision (side, shares, decision price, decision instant). Before sending,
the router reads every ref the broker knows (this connection's trades, all open
orders, completed API orders, and executions). A key already present is ADOPTED
-- never re-sent -- whatever its status, so a retried session cannot
double-submit, and the adopted order's original decision price is recovered from
the broker's own record for the shortfall row.

**The broker plumbing is v1's, ported, not invented** (`crucible-executor`
`executor/ibkr.py`): `Stock(symbol, "SMART", "USD")` qualified before use; a
`MarketOrder` with `tif="DAY"`, `outsideRth=False`, `transmit=True` (the fields
v1 set explicitly after the paper preset's Error 10349 cancelled bare market
orders); fills polled to a terminal status for up to :data:`FILL_TIMEOUT_S`;
fill price the share-weighted mean of the executions; decision prices from
`reqMarketDataType(3)` + `reqMktData`, polled up to :data:`QUOTE_MAX_WAIT_S`
(v1's fix for a cold data farm that nan'd a whole book), subscription released
in a `finally`.

**Declared scope:** an order still working when the fill wait ends is left
working (a DAY order), and its record carries the broker status and what filled
so far. An adopted order's `submitted_at` is its decision instant, because the
broker's completed-order record does not carry a submission time.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from crucible.store import Store

from crucible_trader.broker_session import GatewayAddress, connect, load_sdk
from crucible_trader.broker_statement import PAPER_ACCOUNT_PREFIX, IbPaperStatementSource
from crucible_trader.construction import TargetBook
from crucible_trader.execution_shortfall import (
    Fill,
    SizingDecision,
    build_no_orders_document,
    build_shortfall_document,
    decide,
)
from crucible_trader.kill_switch import assert_trading_permitted, utcnow

__all__ = [
    "ExecutionRecord",
    "PaperOrderRouter",
    "PlannedOrder",
    "RouterConfig",
    "ShadowRouter",
    "build_router",
    "open_paper_router",
    "plan_orders",
]

#: The one switch. Unset or ``off``: shadow, nothing is sent.
ROUTING_VAR = "CRUCIBLE_TRADER_ORDER_ROUTING"
ROUTING_ENABLED_VALUE = "ib_paper"
ROUTING_DISABLED_VALUES: frozenset[str] = frozenset({"", "off"})

#: The paper account the router trades. Required when routing is on.
ACCOUNT_VAR = "CRUCIBLE_TRADER_IB_ACCOUNT"
MAX_ORDER_NOTIONAL_VAR = "CRUCIBLE_TRADER_MAX_ORDER_NOTIONAL_USD"
MAX_SESSION_NOTIONAL_VAR = "CRUCIBLE_TRADER_MAX_SESSION_NOTIONAL_USD"

#: IB Gateway's PAPER API port (v1 `config/risk.yaml.example`: "4002 = paper
#: trading"). The only port this router will connect an order session to.
PAPER_GATEWAY_PORT = 4002

ORDER_REF_PREFIX = "crucible-trader"
_REF_SEP = "|"
_STAMP = "%Y-%m-%dT%H:%M:%S.%fZ"

#: v1's fill wait and poll (`IBKRClient.place_market_order`).
FILL_TIMEOUT_S = 30.0
POLL_INTERVAL_S = 0.5
#: v1's quote wait (`IBKRClient.get_current_price`).
QUOTE_MAX_WAIT_S = 6.0
#: IB market-data type 3: delayed when the account has no live subscription,
#: live when it has one (v1: avoids Error 10089 on paper accounts).
MARKET_DATA_TYPE = 3

TERMINAL_STATUSES: frozenset[str] = frozenset({"Filled", "Cancelled", "ApiCancelled", "Inactive"})

#: A tolerance on the long-only budget: the engine's weights sum to 1 in float.
_GROSS_TOLERANCE = 1e-6


class RoutingRefusedError(RuntimeError):
    """The router will not send. Every refusal below is one of these."""


class RouterSettingsError(RoutingRefusedError):
    """The routing configuration is missing or malformed. Page; never default."""


class NotPaperError(RoutingRefusedError):
    """An account or port that is not IB PAPER. There is no live path."""


class RoutingDisabledError(RoutingRefusedError):
    """An order-capable router was asked for with routing switched off."""


class RiskCapExceededError(RoutingRefusedError):
    """The plan breaches a cap. Raised before any order is sent."""


class PriceUnavailableError(RoutingRefusedError):
    """No usable decision price for a name the plan must trade."""


class ForeignPositionError(RoutingRefusedError):
    """The account holds a name the book does not know. Never traded away."""


class OrderRefError(RoutingRefusedError):
    """A broker order carries this router's key and a ref it cannot read."""


def assert_paper(account: str | None, port: int) -> None:
    """Refuse anything but an IB paper account on the paper gateway port."""
    if not account or not account.startswith(PAPER_ACCOUNT_PREFIX):
        raise NotPaperError(
            f"account {account!r} is not an IB paper account ({PAPER_ACCOUNT_PREFIX}…). The "
            "order router has no live path; the paper -> live crossing is a reserved ruling "
            "(plan §9.5)."
        )
    if port != PAPER_GATEWAY_PORT:
        raise NotPaperError(
            f"gateway port {port} is not the IB Gateway paper port {PAPER_GATEWAY_PORT}. The "
            "order router connects to the paper port only."
        )


def _positive_float(source: Mapping[str, str], var: str) -> float:
    raw = (source.get(var) or "").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise RouterSettingsError(f"{var}={raw!r} is not a number") from exc
    if not (math.isfinite(value) and value > 0):
        raise RouterSettingsError(f"{var}={raw!r} must be a positive, finite dollar amount")
    return value


@dataclasses.dataclass(frozen=True)
class RouterConfig:
    """Whether orders are routed, and to what. Disabled unless explicitly enabled."""

    enabled: bool
    account: str | None = None
    address: GatewayAddress | None = None
    max_order_notional_usd: float | None = None
    max_session_notional_usd: float | None = None

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.address is None:
            raise RouterSettingsError("routing is enabled with no gateway address")
        assert_paper(self.account, self.address.port)
        for name in ("max_order_notional_usd", "max_session_notional_usd"):
            value = getattr(self, name)
            if value is None or not (math.isfinite(value) and value > 0):
                raise RouterSettingsError(f"routing is enabled with {name}={value!r}")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RouterConfig:
        """Resolve from the environment. Unset :data:`ROUTING_VAR` is shadow."""
        source = os.environ if env is None else env
        switch = (source.get(ROUTING_VAR) or "").strip()
        if switch in ROUTING_DISABLED_VALUES:
            return cls(enabled=False)
        if switch != ROUTING_ENABLED_VALUE:
            raise RouterSettingsError(
                f"{ROUTING_VAR}={switch!r} is not a routing mode. Unset or 'off' is shadow "
                f"(no orders); {ROUTING_ENABLED_VALUE!r} routes to IB paper. Nothing else is "
                "accepted, so a typo can neither switch routing on nor hide as off."
            )
        account = (source.get(ACCOUNT_VAR) or "").strip()
        if not account:
            raise RouterSettingsError(f"{ACCOUNT_VAR} is unset; routing names its paper account")
        return cls(
            enabled=True,
            account=account,
            address=GatewayAddress.from_env(dict(source)),
            max_order_notional_usd=_positive_float(source, MAX_ORDER_NOTIONAL_VAR),
            max_session_notional_usd=_positive_float(source, MAX_SESSION_NOTIONAL_VAR),
        )


def order_key(trading_day: str, symbol: str) -> str:
    """The idempotency key: one order per name per trading day."""
    return _REF_SEP.join((ORDER_REF_PREFIX, trading_day, symbol))


@dataclasses.dataclass(frozen=True)
class PlannedOrder:
    """One order to send, built from a frozen sizing decision."""

    trading_day: str
    decision: SizingDecision

    @property
    def key(self) -> str:
        return order_key(self.trading_day, self.decision.symbol)

    @property
    def action(self) -> str:
        return "BUY" if self.decision.side == "buy" else "SELL"

    @property
    def order_ref(self) -> str:
        d = self.decision
        stamp = d.decided_at.astimezone(dt.UTC).strftime(_STAMP)
        return _REF_SEP.join(
            (self.key, d.side, f"{int(d.quantity)}", repr(float(d.decision_price)), stamp)
        )


def parse_order_ref(ref: str) -> PlannedOrder:
    """The decision an order was sent under, read back from its `orderRef`."""
    parts = ref.split(_REF_SEP)
    try:
        prefix, day, symbol, side, quantity, price, stamp = parts
        if prefix != ORDER_REF_PREFIX:
            raise ValueError(f"prefix {prefix!r}")
        decided_at = dt.datetime.strptime(stamp, _STAMP).replace(tzinfo=dt.UTC)
        decision = decide(symbol, side, int(quantity), price=float(price), now=decided_at)  # type: ignore[arg-type]
    except ValueError as exc:
        raise OrderRefError(
            f"broker order ref {ref!r} carries this router's key and cannot be read ({exc}); "
            "refusing to guess whether it is this session's order"
        ) from exc
    return PlannedOrder(trading_day=day, decision=decision)


QuoteSource = Callable[[str], float | None]


def _tradable(book: TargetBook) -> dict[str, float]:
    tickers = book.universe.tickers
    sentinels = {tickers[book.universe.benchmark_idx], tickers[book.universe.cash_idx]}
    return {t: w for t, w in book.weights.items() if t not in sentinels}


def plan_orders(
    book: TargetBook,
    *,
    positions: Mapping[str, int],
    quote: QuoteSource,
    clock: Callable[[], dt.datetime],
    max_order_notional_usd: float,
    max_session_notional_usd: float,
) -> tuple[tuple[PlannedOrder, ...], tuple[PlannedOrder, ...]]:
    """(sells, buys) moving ``positions`` to ``book``, or a refusal. Sends nothing.

    Target shares are ``floor(weight * book notional / decision price)``; every
    cap is checked over the whole plan before it is returned.
    """
    weights = _tradable(book)
    foreign = sorted(set(positions) - set(weights))
    if foreign:
        raise ForeignPositionError(
            f"the account holds {foreign}, which the book for {book.trading_day} does not "
            "contain. The router trades only the names its own book declares; a delta against "
            "positions it does not own would liquidate them."
        )
    short = sorted(t for t, w in weights.items() if w < 0)
    if short:
        raise RiskCapExceededError(f"negative target weight(s) {short}: the router is long only")
    gross = sum(weights.values())
    if gross > 1.0 + _GROSS_TOLERANCE:
        raise RiskCapExceededError(
            f"the book's tradable weights sum to {gross:.6f} > 1: gross exceeds the book notional"
        )

    sells: list[PlannedOrder] = []
    buys: list[PlannedOrder] = []
    session_notional = 0.0
    for symbol in sorted(weights):
        weight, held = weights[symbol], positions.get(symbol, 0)
        if weight == 0.0 and held == 0:
            continue
        price = quote(symbol)
        if price is None or not (math.isfinite(price) and price > 0):
            raise PriceUnavailableError(
                f"no usable decision price for {symbol} ({price!r}); the book is not traded "
                "on a partial price set"
            )
        target = math.floor(weight * book.portfolio_notional_usd / price)
        delta = target - held
        if delta == 0:
            continue
        notional = abs(delta) * price
        if notional > max_order_notional_usd:
            raise RiskCapExceededError(
                f"{symbol}: {abs(delta)} shares at {price} is ${notional:,.2f}, over the "
                f"per-order cap ${max_order_notional_usd:,.2f}"
            )
        session_notional += notional
        side = "buy" if delta > 0 else "sell"
        order = PlannedOrder(
            trading_day=book.trading_day,
            decision=decide(symbol, side, abs(delta), price=price, now=clock()),
        )
        (buys if side == "buy" else sells).append(order)
    if session_notional > max_session_notional_usd:
        raise RiskCapExceededError(
            f"the plan trades ${session_notional:,.2f}, over the per-session cap "
            f"${max_session_notional_usd:,.2f}"
        )
    return tuple(sells), tuple(buys)


@dataclasses.dataclass(frozen=True)
class ExecutionRecord:
    """What became of one planned order at the broker."""

    order: PlannedOrder
    disposition: str  # "placed" | "adopted"
    broker_status: str
    submitted_at: dt.datetime
    filled_quantity: float
    fill_price: float | None

    @property
    def fill(self) -> Fill:
        """The `execution_shortfall.Fill` for this order."""
        return Fill(
            order_id=self.order.key,
            decision=self.order.decision,
            submitted_at=self.submitted_at,
            filled_quantity=self.filled_quantity,
            fill_price=self.fill_price,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "order_ref": self.order.order_ref,
            "disposition": self.disposition,
            "broker_status": self.broker_status,
            **self.fill.record(),
        }


class IbQuoteSource:
    """Decision prices from a connected IB session (v1 `get_current_price`)."""

    def __init__(self, client: Any, sdk: Any, *, max_wait: float = QUOTE_MAX_WAIT_S) -> None:
        self._client = client
        self._sdk = sdk
        self._max_wait = max_wait

    def __call__(self, symbol: str) -> float | None:
        contract = self._sdk.Stock(symbol, "SMART", "USD")
        if not _qualified(self._client, contract):
            return None
        self._client.reqMarketDataType(MARKET_DATA_TYPE)
        ticker = self._client.reqMktData(contract, "", False, False)
        price: float | None = None
        waited = 0.0
        try:
            while price is None and waited < self._max_wait:
                self._client.sleep(POLL_INTERVAL_S)
                waited += POLL_INTERVAL_S
                price = _valid(ticker.last) or _valid(ticker.close)
        finally:
            self._client.cancelMktData(contract)
        return price


def _qualified(client: Any, contract: Any) -> bool:
    """`ib_async` returns ``[None]`` (not ``[]``) for a contract it cannot qualify."""
    result = client.qualifyContracts(contract)
    return bool(result) and result[0] is not None


def _valid(value: Any) -> float | None:
    return float(value) if value is not None and math.isfinite(value) and value > 0 else None


class ShadowRouter:
    """Routing switched off: yields no batch, so nothing can be sent."""

    enabled = False
    records: tuple[ExecutionRecord, ...] = ()

    def batches(self, book: TargetBook) -> Iterable[Sequence[Any]]:
        return ()

    def send(self, batch: Sequence[Any]) -> None:
        raise RoutingDisabledError("a shadow session sends no order")

    def shortfall_document(self, *, champion: str, trading_day: str, now: dt.datetime) -> dict:
        return build_no_orders_document(
            trading_day=trading_day,
            champion=champion,
            reason=f"shadow session: order routing is off ({ROUTING_VAR} unset)",
            now=now,
        )

    def close(self) -> None:
        return None


class PaperOrderRouter:
    """:class:`~crucible_trader.session.OrderRouter` over a connected IB PAPER session."""

    enabled = True

    def __init__(
        self,
        config: RouterConfig,
        *,
        client: Any,
        sdk: Any,
        store: Store,
        trading_day: str,
        clock: Callable[[], dt.datetime] = utcnow,
        quote: QuoteSource | None = None,
        fill_timeout_s: float = FILL_TIMEOUT_S,
    ) -> None:
        if not config.enabled:
            raise RoutingDisabledError(f"order routing is off ({ROUTING_VAR} unset)")
        assert config.address is not None  # RouterConfig refuses enabled without one
        assert_paper(config.account, config.address.port)
        managed = list(client.managedAccounts())
        if managed != [config.account]:
            raise NotPaperError(
                f"the gateway session manages {managed}; the router trades exactly the "
                f"configured paper account {config.account!r} and nothing else"
            )
        self._config = config
        self._client = client
        self._sdk = sdk
        self._store = store
        self._day = trading_day
        self._clock = clock
        self._quote = quote or IbQuoteSource(client, sdk)
        self._fill_timeout_s = fill_timeout_s
        self._planned: tuple[PlannedOrder, ...] = ()
        self._records: list[ExecutionRecord] = []

    @property
    def records(self) -> tuple[ExecutionRecord, ...]:
        return tuple(self._records)

    def batches(self, book: TargetBook) -> Iterable[Sequence[Any]]:
        if book.trading_day != self._day:
            raise RoutingRefusedError(
                f"the book is for {book.trading_day}; this router was opened for {self._day}"
            )
        statement = IbPaperStatementSource(self._client).read_statement(self._day)
        sells, buys = plan_orders(
            book,
            positions=statement.positions,
            quote=self._quote,
            clock=self._clock,
            max_order_notional_usd=self._config.max_order_notional_usd,  # type: ignore[arg-type]
            max_session_notional_usd=self._config.max_session_notional_usd,  # type: ignore[arg-type]
        )
        self._planned = sells + buys
        # Sells first, so their proceeds are cash before the buys go out.
        return [batch for batch in (sells, buys) if batch]

    def _known_refs(self) -> dict[str, str]:
        """Every order ref the broker knows under this router's prefix, by key."""
        refs: list[str] = []
        for source in (
            self._client.trades(),
            self._client.reqAllOpenOrders(),
            self._client.reqCompletedOrders(True),
        ):
            refs.extend(trade.order.orderRef or "" for trade in source)
        refs.extend(record.execution.orderRef or "" for record in self._client.fills())
        known: dict[str, str] = {}
        for ref in refs:
            if ref.startswith(ORDER_REF_PREFIX + _REF_SEP):
                known.setdefault(parse_order_ref(ref).key, ref)
        return known

    def send(self, batch: Sequence[Any]) -> None:
        known = self._known_refs()
        placed: list[tuple[PlannedOrder, Any, dt.datetime]] = []
        try:
            for order in batch:
                # The kill switch, re-read before EVERY order.
                assert_trading_permitted(self._store, self._day)
                if order.key in known:
                    prior = parse_order_ref(known[order.key])
                    self._records.append(self._record(prior, "adopted", "Adopted", None))
                    continue
                placed.append((order, self._place(order), self._clock()))
        except BaseException:
            # Whatever stopped the batch, the orders already at the broker are recorded.
            self._record_placed(placed)
            raise
        waited = 0.0
        while placed and waited < self._fill_timeout_s:
            if all(t.orderStatus.status in TERMINAL_STATUSES for _, t, _ in placed):
                break
            self._client.sleep(POLL_INTERVAL_S)
            waited += POLL_INTERVAL_S
        self._record_placed(placed)

    def _place(self, order: PlannedOrder) -> Any:
        contract = self._sdk.Stock(order.decision.symbol, "SMART", "USD")
        if not _qualified(self._client, contract):
            raise RoutingRefusedError(f"IB could not qualify {order.decision.symbol}")
        ib_order = self._sdk.MarketOrder(order.action, int(order.decision.quantity))
        ib_order.orderRef = order.order_ref
        ib_order.account = self._config.account
        ib_order.tif = "DAY"
        ib_order.outsideRth = False
        ib_order.transmit = True
        return self._client.placeOrder(contract, ib_order)

    def _record_placed(self, placed: list[tuple[PlannedOrder, Any, dt.datetime]]) -> None:
        for order, trade, submitted_at in placed:
            self._records.append(
                self._record(order, "placed", trade.orderStatus.status, submitted_at)
            )

    def _record(
        self,
        order: PlannedOrder,
        disposition: str,
        status: str,
        submitted_at: dt.datetime | None,
    ) -> ExecutionRecord:
        executions = [
            f.execution
            for f in self._client.fills()
            if (f.execution.orderRef or "").startswith(order.key + _REF_SEP)
        ]
        shares = sum(float(e.shares) for e in executions)
        value = sum(float(e.shares) * float(e.price) for e in executions)
        price = value / shares if shares else None
        return ExecutionRecord(
            order=order,
            disposition=disposition,
            broker_status=status,
            submitted_at=submitted_at or order.decision.decided_at,
            filled_quantity=shares,
            fill_price=price,
        )

    def shortfall_document(self, *, champion: str, trading_day: str, now: dt.datetime) -> dict:
        """The session's `execution_shortfall.v1` document (validated, not written)."""
        if not self._records:
            reason = (
                f"{len(self._planned)} order(s) planned and none routed: the session stopped "
                "before the first order"
                if self._planned
                else "the target book matched the held book; no order was needed"
            )
            return build_no_orders_document(
                trading_day=trading_day, champion=champion, reason=reason, now=now
            )
        return build_shortfall_document(
            trading_day=trading_day,
            champion=champion,
            fills=[r.fill for r in self._records],
            now=now,
        )

    def close(self) -> None:
        self._client.disconnect()


def open_paper_router(
    config: RouterConfig,
    *,
    store: Store,
    trading_day: str,
    sdk_loader: Callable[[], Any] = load_sdk,
    clock: Callable[[], dt.datetime] = utcnow,
) -> PaperOrderRouter:
    """Connect an ORDER-capable paper session and wrap it. Refuses before connecting."""
    if not config.enabled:
        raise RoutingDisabledError(f"order routing is off ({ROUTING_VAR} unset)")
    assert config.address is not None
    assert_paper(config.account, config.address.port)
    sdk = sdk_loader()
    client = connect(config.address, readonly=False, sdk_loader=lambda: sdk)
    try:
        return PaperOrderRouter(
            config, client=client, sdk=sdk, store=store, trading_day=trading_day, clock=clock
        )
    except BaseException:
        client.disconnect()
        raise


def build_router(
    config: RouterConfig,
    *,
    store: Store,
    trading_day: str,
    sdk_loader: Callable[[], Any] = load_sdk,
    clock: Callable[[], dt.datetime] = utcnow,
) -> ShadowRouter | PaperOrderRouter:
    """The router a session should use: shadow unless routing is explicitly on."""
    if not config.enabled:
        return ShadowRouter()
    return open_paper_router(
        config, store=store, trading_day=trading_day, sdk_loader=sdk_loader, clock=clock
    )
