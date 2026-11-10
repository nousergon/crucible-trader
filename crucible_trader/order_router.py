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

**A durable intent is written before every submit** (`alpha-engine-config-I12558`,
:mod:`crucible_trader.order_intent`). Immediately before `placeOrder`, after the
contract qualifies, the router creates ONE immutable intent document per (day,
symbol) by conditional create. A failed write sends nothing. An intent that
already exists for an order the broker does NOT know is an UNKNOWN outcome --
the earlier submit may have reached IB -- so that order is never re-sent: the
rest of the batch still goes, every order already at the broker is recorded,
and the batch then raises :class:`UnknownOrderStateError` naming each unknown
key, which stops the session loud. The broker's refs are still read first, so an
order the broker knows is adopted exactly as before.

**The broker plumbing is v1's, ported, not invented** (`crucible-executor`
`executor/ibkr.py`): `Stock(symbol, "SMART", "USD")` qualified before use; a
`MarketOrder` with `tif="DAY"`, `outsideRth=False`, `transmit=True` (the fields
v1 set explicitly after the paper preset's Error 10349 cancelled bare market
orders); fills polled to a terminal status for up to :data:`FILL_TIMEOUT_S`;
fill price the share-weighted mean of the executions; decision prices from
`reqMarketDataType(3)` + `reqMktData`, polled up to :data:`QUOTE_MAX_WAIT_S`
(v1's fix for a cold data farm that nan'd a whole book), subscription released
in a `finally`.

**The book is sized on the account's live NAV** (2026-11-09 v1/v2 parity
audit, row 5). :meth:`PaperOrderRouter.session_sizing` reads the broker's
``NetLiquidation`` once at session start (:func:`read_net_liquidation`) and the
book is constructed on ``min(book_notional_usd, NAV)``: the declared notional
is the ceiling. An unreadable NAV refuses the session, and :meth:`batches`
refuses a book sized past the NAV it read, so a session can never size past
the money the account has. This read is for SIZING only; the ledger and the
reconciler still never read ``NetLiquidation`` (`broker_statement.CASH_TAG`).

**Every held long carries a standing GTC protective stop** (parity audit row
1). After the session's orders, :meth:`PaperOrderRouter.protect_holdings`
re-asserts exactly one ``STP`` SELL, ``tif="GTC"``, per held name at
:attr:`RouterConfig.stop_distance_pct` (default 20%, v1's bracket distance)
below the session's reference price. A stop already standing for today at the
right size is kept, so a rerun places nothing; a stale one (another day, or a
size the holdings no longer match) is cancelled -- confirmed -- before its
replacement goes. The size is the shares held LESS every other working sell
on the name, so stops never exceed the holding. Before a session SELL the
name's own stops are cancelled, so a sell and a stop can never both fill.

**With timed orders the stops are placed as the day goes**, not only at its
end: every name held at the session start before any order is released, and
the names again right after each timed batch returns (:meth:`_protect_in_session`),
so a bought name gets its stop and a trimmed name's stop is re-sized. A name
whose sell is still waiting keeps its stop while it waits. A sell goes only after
its name's stops are confirmed cancelled, and is clipped to the shares still
held then, so a stop that fired while the sell waited never leaves the book
short. A buy never released holds nothing and gets no stop. A stop that cannot be
placed mid-day is logged and left to the final :meth:`protect_holdings`, which
fails the run; the waiting orders are not stopped for it. A name the watch
streams takes its stop reference from the stream, never a fresh snapshot that
would cancel the stream's subscription.

**Every order waits for its trigger** (`crucible_trader.entry_timing`; buys
Brian 2026-11-09, sells Brian 2026-11-10). With :attr:`RouterConfig.entry_timing`
set -- the default -- the buys are held for v1's pullback / VWAP discount /
support bounce / graduated / expiry triggers and the sells for their mirror (a
2% rally off the day's low, within 1% of the sized price from 15:30, and
unconditionally at 15:55 ET), on streaming IB quotes, so the session runs until
the last order is released or the market closes. No sell can be carried
overnight: its expiry needs no price. Each released order is its own batch, so
`session_cycle` re-checks the kill switch before it and
:meth:`PaperOrderRouter.send` before its order; the watch also re-reads the
switch on every poll. The idempotency key is unchanged -- one order per name per
trading day -- and an order this day's broker already knows is adopted at once
rather than watched. :data:`ENTRY_TIMING_VAR` ``immediate`` restores the old
behaviour (every order at the open, sells first).

**A buy is never sized on cash a pending sell has not freed.** The plan is sized
on the book's notional assuming every sell's proceeds, so once sells wait too, a
buy could otherwise trigger at 10:00 against proceeds of a sell that goes at
15:55. :class:`_CashLedger` holds what the timed buys may spend: the book's cash
at the start (:func:`book_cash`) plus what this run's sells have actually FILLED,
less what its buys have committed. Each poll judges and sends the sells first,
then the buys against the refreshed ledger. A triggered buy the ledger cannot pay
for in full keeps waiting while any sell is still waiting (one key per name per
day means a short buy can never be topped up); with no sell left, or at the
expiry -- when every sell has just gone first -- it is clipped to the cash.

**A held name that gapped down is sold whole at the session start**
(`crucible_trader.gap_stop`, v1's catastrophic gap stop): priced live, never from
the close fallback, against the book day's close; at
:attr:`RouterConfig.gap_stop_pct` (v1's 0.15) or more below it, the name's planned
sell becomes a full exit and any buy of it is dropped. Hard risk outranks timing:
with timed orders the gap exits are the session's first batch, sent at once and
never held for an exit trigger; immediately, they lead the opening sells.

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
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from crucible.slots.strategy import CASH_TICKER
from crucible.store import Store

from crucible_trader.broker_session import GatewayAddress, connect, load_sdk
from crucible_trader.broker_statement import PAPER_ACCOUNT_PREFIX, IbPaperStatementSource
from crucible_trader.construction import SessionSizing, TargetBook
from crucible_trader.entry_timing import (
    NYSE_ZONE,
    EntryTimingParams,
    EntryWatch,
    IbIntradayQuotes,
    WatchedEntry,
    entry_trigger,
    exit_trigger,
    panel_support_levels,
)
from crucible_trader.execution_shortfall import (
    Fill,
    SizingDecision,
    build_no_orders_document,
    build_shortfall_document,
    decide,
)
from crucible_trader.gap_stop import (
    DEFAULT_GAP_STOP_PCT,
    GAP_STOP_EXIT,
    apply_gap_exits,
    check_gaps,
    exit_decision,
    gap_stop_document,
    write_gap_stop_document,
)
from crucible_trader.inherited_positions import inherited_for
from crucible_trader.kill_switch import assert_trading_permitted, utcnow
from crucible_trader.order_intent import (
    OrderIntentConflictError,
    bought_by_this_trader,
    claim_order_intent,
    order_intent_document,
    order_intent_key,
    read_order_intent,
)

__all__ = [
    "DEFAULT_STOP_DISTANCE_PCT",
    "ExecutionRecord",
    "NavUnavailableError",
    "PaperOrderRouter",
    "PlannedOrder",
    "ProtectiveStopError",
    "RouterConfig",
    "ShadowRouter",
    "StagingRoutingRefusedError",
    "UnknownOrderStateError",
    "assert_staging_never_routes",
    "book_cash",
    "build_router",
    "held_weights",
    "is_staging",
    "open_paper_router",
    "plan_orders",
    "read_net_liquidation",
    "stop_price_for",
]

#: The one switch. Unset or ``off``: shadow, nothing is sent.
ROUTING_VAR = "CRUCIBLE_TRADER_ORDER_ROUTING"
ROUTING_ENABLED_VALUE = "ib_paper"
ROUTING_DISABLED_VALUES: frozenset[str] = frozenset({"", "off"})

#: The fleet's environment selector (`nousergon_lib.env.ENV_VAR`). Read here
#: for ONE purpose: staging never routes (`alpha-engine-config-I12507`).
ENVIRONMENT_VAR = "NE_ENV"
STAGING_ENVIRONMENT = "staging"

#: The paper account the router trades. Required when routing is on.
ACCOUNT_VAR = "CRUCIBLE_TRADER_IB_ACCOUNT"
MAX_ORDER_NOTIONAL_VAR = "CRUCIBLE_TRADER_MAX_ORDER_NOTIONAL_USD"
MAX_SESSION_NOTIONAL_VAR = "CRUCIBLE_TRADER_MAX_SESSION_NOTIONAL_USD"

#: When the orders go out. Unset or ``v1_triggers``: buys held for v1's entry
#: triggers and sells for their mirror (`crucible_trader.entry_timing`).
#: ``immediate``: every order at the session start, sells first, the behaviour
#: before 2026-11-09. Anything else is refused.
ENTRY_TIMING_VAR = "CRUCIBLE_TRADER_ENTRY_TIMING"

#: The catastrophic gap stop (`crucible_trader.gap_stop`). Unset: v1's 0.15.
#: A fraction in (0, 1) overrides it; ``off`` disables it.
GAP_STOP_VAR = "CRUCIBLE_TRADER_GAP_STOP_PCT"
ENTRY_TIMING_TRIGGERED = "v1_triggers"
ENTRY_TIMING_IMMEDIATE = "immediate"

#: IB Gateway's PAPER API port (v1 `config/risk.yaml.example`: "4002 = paper
#: trading"). The only port this router will connect an order session to.
PAPER_GATEWAY_PORT = 4002

ORDER_REF_PREFIX = "crucible-trader"
#: Every protective stop's `orderRef` starts with this. It is NOT
#: ``ORDER_REF_PREFIX + "|"``, so the session orders' readers (adoption here,
#: the intent reconciliation) never take a stop for a session order.
STOP_REF_PREFIX = "crucible-trader-stop"

#: How far below the reference price the standing stop sits: v1's bracket STP
#: (`catastrophic_gap_stop_pct` 0.15 + `bracket_backstop_extra_pct` 0.05).
DEFAULT_STOP_DISTANCE_PCT = 0.20
STOP_DISTANCE_VAR = "CRUCIBLE_TRADER_STOP_DISTANCE_PCT"

#: The account-summary tag the SIZING reads. Read only here, never by the ledger.
NAV_TAG = "NetLiquidation"
_REF_SEP = "|"
_STAMP = "%Y-%m-%dT%H:%M:%S.%fZ"

#: v1's fill wait and poll (`IBKRClient.place_market_order`).
FILL_TIMEOUT_S = 30.0
POLL_INTERVAL_S = 0.5
#: v1's quote wait (`IBKRClient.get_current_price`).
QUOTE_MAX_WAIT_S = 6.0
#: A second poll, this many times longer, for a name the first one missed: a
#: cold data farm answers late, and one refused name holds the whole book.
QUOTE_RETRY_FACTOR = 2.0
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


class StagingRoutingRefusedError(RoutingRefusedError):
    """The routing switch is present in staging. Staging never routes."""


class NavUnavailableError(RoutingRefusedError):
    """The broker's NetLiquidation could not be read. A live session never sizes blind."""


class ProtectiveStopError(RoutingRefusedError):
    """A held name could not be given its protective stop. Page; the name is unprotected."""


class UnknownOrderStateError(RoutingRefusedError):
    """An order whose intent exists but which the broker does not know. Never re-sent."""

    def __init__(self, unknown: Sequence[OrderIntentConflictError]) -> None:
        self.unknown = tuple(unknown)
        super().__init__(
            f"{len(self.unknown)} order(s) have a durable intent and no broker record, so "
            "their outcome is unknown and they were NOT re-sent: "
            + "; ".join(str(e) for e in self.unknown)
        )


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


def is_staging(source: Mapping[str, str]) -> bool:
    """Whether :data:`ENVIRONMENT_VAR` names staging (case and whitespace ignored)."""
    return (source.get(ENVIRONMENT_VAR) or "").strip().lower() == STAGING_ENVIRONMENT


def assert_staging_never_routes(source: Mapping[str, str]) -> None:
    """Refuse :data:`ROUTING_VAR` in staging, whatever its value, before anything opens.

    Present at all is refused -- ``off`` and empty included -- because staging's
    env declares the switch absent, and an env file that names it is one edit
    from ``ib_paper``. Outside staging this returns without reading anything else.
    """
    if is_staging(source) and ROUTING_VAR in source:
        raise StagingRoutingRefusedError(
            f"{ROUTING_VAR}={source[ROUTING_VAR]!r} is set under "
            f"{ENVIRONMENT_VAR}={STAGING_ENVIRONMENT}. Staging never routes: its orders are "
            "simulated (crucible_trader.simulated_broker) and the IB paper account is shared "
            "with the v1 executor, so a staging order would land on the v1 executor's book."
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


def _gap_stop_pct(source: Mapping[str, str]) -> float | None:
    raw = (source.get(GAP_STOP_VAR) or "").strip()
    if not raw:
        return DEFAULT_GAP_STOP_PCT
    if raw == "off":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise RouterSettingsError(f"{GAP_STOP_VAR}={raw!r} is not a number or 'off'") from exc
    if not (math.isfinite(value) and 0.0 < value < 1.0):
        raise RouterSettingsError(f"{GAP_STOP_VAR}={raw!r} must be a fraction in (0, 1)")
    return value


@dataclasses.dataclass(frozen=True)
class RouterConfig:
    """Whether orders are routed, and to what. Disabled unless explicitly enabled."""

    enabled: bool
    account: str | None = None
    address: GatewayAddress | None = None
    max_order_notional_usd: float | None = None
    max_session_notional_usd: float | None = None
    stop_distance_pct: float = DEFAULT_STOP_DISTANCE_PCT
    #: The intraday triggers, or None to send every order at the session start.
    entry_timing: EntryTimingParams | None = EntryTimingParams()
    #: The opening batch's gap-stop threshold (`crucible_trader.gap_stop`), or None.
    gap_stop_pct: float | None = DEFAULT_GAP_STOP_PCT

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
        if not (math.isfinite(self.stop_distance_pct) and 0 < self.stop_distance_pct < 1):
            raise RouterSettingsError(
                f"stop_distance_pct={self.stop_distance_pct!r} must be a fraction in (0, 1)"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RouterConfig:
        """Resolve from the environment. Unset :data:`ROUTING_VAR` is shadow."""
        source = os.environ if env is None else env
        assert_staging_never_routes(source)
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
        timing = (source.get(ENTRY_TIMING_VAR) or "").strip()
        if timing not in ("", ENTRY_TIMING_TRIGGERED, ENTRY_TIMING_IMMEDIATE):
            raise RouterSettingsError(
                f"{ENTRY_TIMING_VAR}={timing!r} is not an entry timing. Unset or "
                f"{ENTRY_TIMING_TRIGGERED!r} holds orders for their triggers; "
                f"{ENTRY_TIMING_IMMEDIATE!r} sends them at the session start."
            )
        return cls(
            enabled=True,
            account=account,
            address=GatewayAddress.from_env(dict(source)),
            max_order_notional_usd=_positive_float(source, MAX_ORDER_NOTIONAL_VAR),
            max_session_notional_usd=_positive_float(source, MAX_SESSION_NOTIONAL_VAR),
            stop_distance_pct=(
                _positive_float(source, STOP_DISTANCE_VAR)
                if (source.get(STOP_DISTANCE_VAR) or "").strip()
                else DEFAULT_STOP_DISTANCE_PCT
            ),
            entry_timing=None if timing == ENTRY_TIMING_IMMEDIATE else EntryTimingParams(),
            gap_stop_pct=_gap_stop_pct(source),
        )


def read_net_liquidation(client: Any, account: str) -> float:  # pyright: ignore[reportExplicitAny]  # an ib_async IB
    """The account's ``NetLiquidation`` in USD, for SIZING the session's book.

    Exactly one finite, positive USD row for ``account``, or
    :class:`NavUnavailableError`: a NAV that cannot be read is not the
    declared notional, and a live session refuses rather than size on a guess.
    """
    rows = [
        item
        for item in client.accountSummary()
        if item.account == account and item.tag == NAV_TAG and item.currency == "USD"
    ]
    if len(rows) != 1:
        raise NavUnavailableError(
            f"expected exactly one USD {NAV_TAG} row for {account}, found {len(rows)}; the "
            "session is not sized on the declared notional in its place"
        )
    try:
        value = float(rows[0].value)
    except (TypeError, ValueError) as exc:
        raise NavUnavailableError(f"{NAV_TAG} {rows[0].value!r} is not a number") from exc
    if not (math.isfinite(value) and value > 0):
        raise NavUnavailableError(f"{NAV_TAG} {value!r} is not a positive, finite amount")
    return value


def stop_price_for(reference: float, distance: float) -> float:
    """The stop price ``distance`` below ``reference``, floored to the cent."""
    price = math.floor(reference * (1.0 - distance) * 100.0) / 100.0
    if not price > 0:
        raise PriceUnavailableError(
            f"a stop {distance:.0%} below {reference!r} rounds to {price!r}; no stop is placed "
            "at a price that is not positive"
        )
    return price


def stop_ref(trading_day: str, symbol: str, quantity: int, stop_price: float) -> str:
    """A protective stop's `orderRef`: the day it was asserted, its size and price."""
    return _REF_SEP.join((STOP_REF_PREFIX, trading_day, symbol, str(quantity), f"{stop_price:.2f}"))


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


#: Float slack in the target-share floor (see `plan_orders`).
_SHARE_EPSILON = 1e-6


def held_weights(
    positions: Mapping[str, int], *, quote: QuoteSource, portfolio_notional_usd: float
) -> dict[str, float]:
    """The book held into the session, as weights of the book notional.

    Each held name is valued at the price the router will plan against, so a
    name whose weight the engine leaves unchanged plans back to exactly the
    shares held. Cash is the remainder; a book worth more than the notional is
    scaled to it, all cash spent, since the engine takes weights that sum to one.

    2026-11-05: every live session constructed from all cash, so the engine's
    turnover budget measured each day against an empty book and never bound
    (97% one-way on 11-03 and 11-04): PLTR was bought on 11-04 and sold whole
    on 11-05. Raises `PriceUnavailableError` for a held name with no price: a
    book valued on a partial price set is not the book held.
    """
    if portfolio_notional_usd <= 0:
        raise RiskCapExceededError(f"book notional {portfolio_notional_usd!r} is not positive")
    values: dict[str, float] = {}
    for symbol, shares in sorted(positions.items()):
        if shares == 0:
            continue
        if shares < 0:
            raise RiskCapExceededError(f"the account is short {symbol}: the router is long only")
        price = quote(symbol)
        if price is None or not (math.isfinite(price) and price > 0):
            raise PriceUnavailableError(
                f"no usable price for held {symbol} ({price!r}); the book held into the "
                "session cannot be valued on a partial price set"
            )
        values[symbol] = shares * price
    gross = sum(values.values())
    scale = 1.0 if gross <= portfolio_notional_usd else portfolio_notional_usd / gross
    weights = {s: v * scale / portfolio_notional_usd for s, v in values.items()}
    weights[CASH_TICKER] = max(0.0, 1.0 - sum(weights.values()))
    return weights


def plan_orders(
    book: TargetBook,
    *,
    positions: Mapping[str, int],
    quote: QuoteSource,
    clock: Callable[[], dt.datetime],
    max_order_notional_usd: float,
    max_session_notional_usd: float,
    inherited: frozenset[str] = frozenset(),
    owned: frozenset[str] = frozenset(),
) -> tuple[tuple[PlannedOrder, ...], tuple[PlannedOrder, ...]]:
    """(sells, buys) moving ``positions`` to ``book``, or a refusal. Sends nothing.

    Target shares are ``floor(weight * book notional / decision price)``; every
    cap is checked over the whole plan before it is returned. A held name in
    ``inherited`` (`crucible_trader.inherited_positions`) or in ``owned`` (a
    name this trader bought itself,
    `crucible_trader.order_intent.bought_by_this_trader`) that the book does
    not contain is planned to zero rather than refused.
    """
    weights = _tradable(book)
    foreign = sorted(set(positions) - set(weights) - inherited - owned)
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

    planned = dict(weights)
    for symbol in set(positions) - set(weights):
        planned[symbol] = 0.0  # an inherited name, sold whole

    sells: list[PlannedOrder] = []
    buys: list[PlannedOrder] = []
    session_notional = 0.0
    for symbol in sorted(planned):
        weight, held = planned[symbol], positions.get(symbol, 0)
        if weight == 0.0 and held == 0:
            continue
        price = quote(symbol)
        if price is None or not (math.isfinite(price) and price > 0):
            detail = getattr(quote, "last_miss", None)
            raise PriceUnavailableError(
                f"no usable decision price for {symbol} ({price!r}); the book is not traded "
                "on a partial price set" + (f" [{detail}]" if detail else "")
            )
        # The epsilon keeps a weight carried from the held shares at this same
        # price (`held_weights`) mapping back to exactly those shares, not one
        # fewer through float rounding.
        target = math.floor(weight * book.portfolio_notional_usd / price + _SHARE_EPSILON)
        delta = target - held
        if delta == 0:
            continue
        notional = abs(delta) * price
        if notional > max_order_notional_usd:
            # Clipped to the cap, not refused: the cap bounds one order, and an
            # order cut to it trades less than the book asks, never more. The
            # 2026-11-04 rerun refused the whole book over PLTR at $287k.
            clipped = math.floor(max_order_notional_usd / price)
            print(
                f"order_router: {symbol} {abs(delta)} shares at {price} is ${notional:,.2f}, "
                f"over the per-order cap ${max_order_notional_usd:,.2f}; clipped to {clipped}",
                file=sys.stderr,
            )
            if clipped == 0:
                continue
            delta = clipped if delta > 0 else -clipped
            notional = clipped * price
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


def book_cash(book: TargetBook, positions: Mapping[str, int], *, quote: QuoteSource) -> float:
    """The book's cash at the session start: its notional less the held book's value.

    The same arithmetic as :func:`held_weights`' cash weight, in dollars, at the
    prices the plan was sized on. A held book worth more than the notional has no
    cash (never negative). The paper account is shared with the v1 executor, so
    the account's own cash balance is not this trader's to spend; the book's
    notional is the budget the plan was sized against.
    """
    held = 0.0
    for symbol, shares in positions.items():
        if shares > 0:
            price = _valid(quote(symbol))
            if price is None:
                raise PriceUnavailableError(f"no usable price for held {symbol}; no cash figure")
            held += shares * price
    return max(0.0, book.portfolio_notional_usd - held)


class _MemoQuote:
    """One price per name per plan: a second ask of a name returns the first answer."""

    def __init__(self, source: QuoteSource) -> None:
        self._source = source
        self._seen: dict[str, float | None] = {}

    @property
    def last_miss(self) -> str | None:
        return getattr(self._source, "last_miss", None)

    def __call__(self, symbol: str) -> float | None:
        if symbol not in self._seen:
            self._seen[symbol] = self._source(symbol)
        return self._seen[symbol]


class _CashLedger:
    """What a timed session's buys may spend: the book's cash plus what its sells FILLED.

    ``cash = book cash at the start + proceeds of this run's filled sells
    - what this run's buys have committed``. A sell counts only once the broker
    reports its executions, so no buy is ever sized on proceeds a still-pending
    sell has not freed. A buy is charged at the larger of its release-time
    notional and its filled value, so slippage can only make the ledger stricter.
    An ADOPTED sell (a rerun's) adds nothing: what it filled before the rerun is
    already in the positions the book cash was measured from, and what it fills
    after is left uncounted rather than risk counting it twice. An adopted buy
    is charged its re-planned (still-unbought) quantity at its sized price.

    It also keeps the per-session notional released so far, for the cap.
    """

    def __init__(self, start: float) -> None:
        self.start = start
        self._sells: set[str] = set()
        self._buys: dict[str, float] = {}
        self._adopted = 0.0
        self.released_notional = 0.0

    def adopt(self, entry: WatchedEntry) -> None:
        notional = entry.quantity * entry.reference_price
        self.released_notional += notional
        if entry.side == "buy":
            self._adopted += notional

    def release_sell(self, entry: WatchedEntry, reason: str, now: dt.datetime) -> None:
        price = entry.last if entry.last is not None else entry.reference_price
        entry.status, entry.trigger = "triggered", reason
        entry.released_at, entry.release_price = now, price
        entry.released_quantity = entry.quantity
        self.released_notional += entry.quantity * price
        self._sells.add(entry.order_key)

    def release_buy(self, entry: WatchedEntry, order: PlannedOrder) -> float:
        notional = float(order.decision.quantity) * (entry.release_price or 0.0)
        self.released_notional += notional
        self._buys[entry.order_key] = notional
        return notional

    def session_room(self, cap: float, waiting: Iterable[WatchedEntry]) -> float:
        """The session cap less what has gone, and less the sells still owed their room."""
        owed = sum(e.quantity * e.reference_price for e in waiting if e.side == "sell")
        return cap - self.released_notional - owed

    def cash(self, executions: Iterable[tuple[str, float, float]]) -> float:
        """The cash left, from the broker's executions as ``(order ref, shares, price)``."""
        filled: dict[str, float] = {}
        for ref, shares, price in executions:
            key = _REF_SEP.join(ref.split(_REF_SEP)[:3])
            filled[key] = filled.get(key, 0.0) + shares * price
        freed = sum(filled.get(key, 0.0) for key in self._sells)
        spent = sum(max(charged, filled.get(key, 0.0)) for key, charged in self._buys.items())
        return self.start + freed - spent - self._adopted


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
    """Decision prices from a connected IB session (v1 `get_current_price`).

    A name that yields no price is polled for :data:`QUOTE_MAX_WAIT_S`, then
    once more for :data:`QUOTE_RETRY_FACTOR` times as long (a cold data farm), before it is
    a miss. Every miss leaves :attr:`last_miss`: whether the contract
    qualified, the ticker's fields and every IB error code the session
    reported while it waited, which the router's refusal carries. The first
    live session (2026-11-04) refused on ANF with nothing else to go on.
    """

    def __init__(self, client: Any, sdk: Any, *, max_wait: float = QUOTE_MAX_WAIT_S) -> None:
        self._client = client
        self._sdk = sdk
        self._max_wait = max_wait
        self._retry_wait = max_wait * QUOTE_RETRY_FACTOR
        self.last_miss: str | None = None

    def __call__(self, symbol: str) -> float | None:
        self.last_miss = None
        contract = self._sdk.Stock(symbol, "SMART", "USD")
        errors: list[str] = []

        def on_error(_req_id: object, code: object, message: object = "", *_rest: object) -> None:
            errors.append(f"{code} {message}".strip())

        error_event = getattr(self._client, "errorEvent", None)
        if error_event is not None:
            error_event.connect(on_error)
        try:
            if not _qualified(self._client, contract):
                self.last_miss = _miss(symbol, "contract did not qualify", None, errors)
                return None
            price, ticker = None, None
            for wait in (self._max_wait, self._retry_wait):
                price, ticker = self._poll(contract, wait)
                if price is not None:
                    return price
            self.last_miss = _miss(
                symbol, f"no tick within {self._max_wait + self._retry_wait:g}s", ticker, errors
            )
            return None
        finally:
            if error_event is not None:
                error_event.disconnect(on_error)

    def _poll(self, contract: object, max_wait: float) -> tuple[float | None, object]:
        self._client.reqMarketDataType(MARKET_DATA_TYPE)
        ticker = self._client.reqMktData(contract, "", False, False)
        price: float | None = None
        waited = 0.0
        try:
            while price is None and waited < max_wait:
                self._client.sleep(POLL_INTERVAL_S)
                waited += POLL_INTERVAL_S
                price = _valid(ticker.last) or _valid(ticker.close)
        finally:
            self._client.cancelMktData(contract)
        return price, ticker


class CloseFallbackQuote:
    """IB first; when IB has no price for a name, the book day's close from the panel.

    The orders are market orders, so the decision price only sizes them and
    checks the caps; the book day's close (the data the book was scored on) is a
    sound stand-in. Without this one silent name refused the whole book: the
    first live session (2026-11-04) placed nothing over ANF. Every name priced
    this way is kept in :attr:`fell_back` and logged. A name neither source
    prices is still a miss, so the book is still refused on a partial price set.
    """

    def __init__(self, primary: QuoteSource, fallback: QuoteSource) -> None:
        self._primary = primary
        self._fallback = fallback
        self.fell_back: dict[str, float] = {}
        self.last_miss: str | None = None

    def __call__(self, symbol: str) -> float | None:
        self.last_miss = None
        price = _valid(self._primary(symbol))
        if price is not None:
            return price
        miss = getattr(self._primary, "last_miss", None)
        try:
            close = _valid(self._fallback(symbol))
        except PriceUnavailableError as exc:
            close, miss = None, f"{miss or symbol}; close fallback: {exc}"
        if close is None:
            self.last_miss = miss
            return None
        self.fell_back[symbol] = close
        print(
            f"order_router: {symbol} priced at the book-day close {close} "
            f"({miss or 'IB: no price'})",
            file=sys.stderr,
        )
        return close


def _miss(symbol: str, why: str, ticker: object, errors: list[str]) -> str:
    fields = (
        ""
        if ticker is None
        else " ".join(
            f"{name}={getattr(ticker, name, None)!r}"
            for name in ("marketDataType", "last", "close", "bid", "ask")
        )
    )
    seen = "; ".join(errors) if errors else "none"
    return f"{symbol}: {why}; ticker {fields or 'n/a'}; IB errors: {seen}"


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
        live_quote: QuoteSource | None = None,
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
        if quote is None:
            from crucible_trader.simulated_broker import PanelCloseSource  # imports this module

            primary = IbQuoteSource(client, sdk)
            live_quote = live_quote or primary
            quote = CloseFallbackQuote(primary, PanelCloseSource(store, trading_day))
        self._quote = quote
        # The gap stop's price: IB only, never the close fallback, whose price
        # IS the reference and would read every unpriced name as a 0% drop.
        self._live_quote = live_quote or quote
        self._fill_timeout_s = fill_timeout_s
        self._planned: tuple[PlannedOrder, ...] = ()
        self._records: list[ExecutionRecord] = []
        self._nav: float | None = None
        #: The names this session may protect: the book's, inherited and owned.
        self._managed: frozenset[str] = frozenset()
        self._watch: EntryWatch | None = None
        self._intraday: IbIntradayQuotes | None = None
        self._stop_events: list[dict[str, object]] = []

    @property
    def records(self) -> tuple[ExecutionRecord, ...]:
        return tuple(self._records)

    def _net_liquidation(self) -> float:
        """The account's NAV, read once per session (`read_net_liquidation`)."""
        if self._nav is None:
            self._nav = read_net_liquidation(self._client, self._config.account)  # pyright: ignore[reportArgumentType]  # __init__'s assert_paper refused a None account
        return self._nav

    def session_sizing(self, ceiling_usd: float) -> SessionSizing:
        """The live sizing: ``min(ceiling_usd, NetLiquidation)``. Raises when unreadable."""
        return SessionSizing(ceiling_usd=ceiling_usd, nav_usd=self._net_liquidation())

    def batches(self, book: TargetBook) -> Iterable[Sequence[Any]]:
        if book.trading_day != self._day:
            raise RoutingRefusedError(
                f"the book is for {book.trading_day}; this router was opened for {self._day}"
            )
        # The chokepoint for live sizing: whatever built the book, it is never
        # sized past the money the account holds (parity audit row 5).
        nav = self._net_liquidation()
        if book.portfolio_notional_usd > nav * (1 + _GROSS_TOLERANCE):
            raise RiskCapExceededError(
                f"the book is sized on ${book.portfolio_notional_usd:,.2f}, over the account's "
                f"{NAV_TAG} ${nav:,.2f}: a live book is sized on min(declared notional, NAV)"
            )
        statement = IbPaperStatementSource(self._client).read_statement(self._day)
        inherited = inherited_for(self._day)
        # A held name the book dropped is this trader's to sell when its own
        # filled orders still leave it shares (I13237: an intent alone is not a
        # holding); only an undeclared name it never bought is foreign (I13164).
        owned = bought_by_this_trader(
            self._store,
            set(statement.positions) - set(_tradable(book)) - inherited,
            account=self._config.account,  # pyright: ignore[reportArgumentType]  # __init__'s assert_paper refused a None account
            before=self._day,
        )
        # Memoised: the timed path values the held book at the very prices it planned on.
        quote = _MemoQuote(self._quote)
        sells, buys = plan_orders(
            book,
            positions=statement.positions,
            quote=quote,
            clock=self._clock,
            max_order_notional_usd=self._config.max_order_notional_usd,  # type: ignore[arg-type]
            max_session_notional_usd=self._config.max_session_notional_usd,  # type: ignore[arg-type]
            inherited=inherited,
            owned=owned,
        )
        gapped: frozenset[str] = frozenset()
        if self._config.gap_stop_pct is not None:
            sells, buys, gapped = self._gap_stop(statement.positions, sells, buys)
        self._planned = sells + buys
        self._managed = frozenset(_tradable(book)) | inherited | owned
        if self._config.entry_timing is None or not self._planned:
            # Sells first, so their proceeds are cash before the buys go out.
            return [batch for batch in (sells, buys) if batch]
        return self._timed_orders(
            self._config.entry_timing,
            sells,
            buys,
            book_cash=book_cash(book, statement.positions, quote=quote),
            urgent=gapped,
        )

    def _gap_stop(
        self,
        positions: Mapping[str, int],
        sells: tuple[PlannedOrder, ...],
        buys: tuple[PlannedOrder, ...],
    ) -> tuple[tuple[PlannedOrder, ...], tuple[PlannedOrder, ...], frozenset[str]]:
        """The plan with every gapped held name sold whole (`gap_stop`), and those names.

        The record is written before the orders exist, so an exit that reaches
        the broker is never off the record.
        """
        from crucible_trader.simulated_broker import PanelCloseSource  # imports this module

        threshold = self._config.gap_stop_pct
        assert threshold is not None  # the caller checked
        reference = PanelCloseSource(self._store, self._day)

        def safe_reference(symbol: str) -> float | None:
            try:
                return reference(symbol)
            except PriceUnavailableError:
                return None

        checks = check_gaps(
            positions, reference=safe_reference, live_price=self._live_quote, threshold=threshold
        )
        now = self._clock()
        write_gap_stop_document(
            self._store,
            self._day,
            gap_stop_document(
                self._day,
                checks,
                threshold=threshold,
                reference_source=reference.source_key,
                now=now,
            ),
        )
        for check in checks:
            if check.fired:
                print(
                    f"order_router: GAP STOP {check.symbol} at {check.price} is "
                    f"{check.drop:.1%} below the {self._day} close {check.reference_close}; "
                    f"selling all {check.shares} shares",
                    file=sys.stderr,
                )
        sells, buys = apply_gap_exits(
            sells,
            buys,
            checks,
            symbol_of=lambda o: o.decision.symbol,
            full_exit=lambda c: PlannedOrder(
                trading_day=self._day, decision=exit_decision(c, now=now)
            ),
        )
        return sells, buys, frozenset(c.symbol for c in checks if c.fired)

    def _timed_orders(
        self,
        params: EntryTimingParams,
        sells: Sequence[PlannedOrder],
        buys: Sequence[PlannedOrder],
        *,
        book_cash: float,
        urgent: frozenset[str] = frozenset(),
    ) -> Iterator[tuple[PlannedOrder, ...]]:
        """Every order, released as its own batch when its trigger fires.

        In each poll the sells are judged and sent first, so the cash their fills
        free is on the ledger before any buy of the same poll is sized against it.
        A generator, so each step below runs only after the batch before it was
        SENT (and its fill wait ended). The sells of the ``urgent`` names (the gap
        stop's exits) are hard risk and wait for no trigger: they are the first
        batch, sent at once, before any stop is placed. Then the names held are
        protected, and every timed batch's names again right after it returns, not
        at the session's end.
        """
        now = self._clock()
        schedule = params.schedule(now.astimezone(NYSE_ZONE).date())
        by_key = {order.key: order for order in (*sells, *buys)}
        support = panel_support_levels(
            self._store,
            self._day,
            [o.decision.symbol for o in buys],
            lookback_days=params.support_lookback_days,
        )
        watch = EntryWatch(
            self._store,
            trading_day=self._day,
            params=params,
            schedule=schedule,
            entries=[
                WatchedEntry(
                    symbol=o.decision.symbol,
                    order_key=o.key,
                    quantity=int(o.decision.quantity),
                    reference_price=float(o.decision.decision_price),
                    decided_at=o.decision.decided_at,
                    support_level=support.get(o.decision.symbol),
                    side=o.decision.side,
                )
                for o in (*sells, *buys)
            ],
        )
        self._watch = watch
        ledger = _CashLedger(book_cash)
        unknown: list[OrderIntentConflictError] = []
        quotes = IbIntradayQuotes(self._client, self._sdk)
        self._intraday = quotes
        try:
            # A rerun of the day: an order the broker already knows is adopted now,
            # and one with an intent the broker does not know is UNKNOWN -- never
            # sent, and the session fails on it once every other order has had its
            # chance.
            known = self._known_refs()
            adopted: list[WatchedEntry] = []
            for entry in watch.entries:
                if entry.order_key in known:
                    entry.status, entry.trigger = "triggered", "adopted: already at the broker"
                    adopted.append(entry)
                    ledger.adopt(entry)
                elif (
                    intent := read_order_intent(self._store, self._day, entry.symbol)
                ) is not None:
                    entry.status = "not_sent"
                    entry.trigger = "unknown: an intent exists and the broker has no order"
                    unknown.append(
                        OrderIntentConflictError(order_intent_key(self._day, entry.symbol), intent)
                    )
            # Hard risk outranks timing: a gap exit never waits for an exit trigger.
            gap_exits = [e for e in watch.waiting() if e.side == "sell" and e.symbol in urgent]
            if gap_exits:
                for entry in gap_exits:
                    ledger.release_sell(entry, GAP_STOP_EXIT, now)
                watch.write(now)
                yield tuple(by_key[e.order_key] for e in gap_exits)
                for entry in gap_exits:
                    entry.status = "sent"
            self._protect_in_session("at the session start")
            if adopted:
                watch.write(now)
                yield tuple(by_key[e.order_key] for e in adopted)
                for entry in adopted:
                    entry.status = "sent"
                self._protect_in_session("after the adopted orders")
            quotes.subscribe(e.symbol for e in watch.waiting())
            watch.write(self._clock())
            while watch.waiting():
                # The switch is re-read on every poll, not only before a batch.
                assert_trading_permitted(self._store, self._day)
                if not self._client.isConnected():
                    raise RoutingRefusedError(
                        f"the gateway connection dropped with {len(watch.waiting())} order(s) "
                        "still waiting; they were not sent"
                    )
                now = self._clock()
                watch.polled_at = now
                if now >= schedule.closes_at:
                    for entry in watch.waiting():
                        entry.status, entry.trigger = "not_sent", "market closed before a trigger"
                    break
                seen = {e.order_key: e.observe(quotes(e.symbol)) for e in watch.waiting()}
                exits: list[WatchedEntry] = []
                for entry in watch.waiting():
                    if entry.side != "sell":
                        continue
                    reason = exit_trigger(
                        reference_price=entry.reference_price,
                        quote=seen[entry.order_key],
                        now=now,
                        schedule=schedule,
                        params=params,
                    )
                    if reason is not None:
                        ledger.release_sell(entry, reason, now)
                        exits.append(entry)
                if exits:
                    watch.write(now)
                    yield tuple(by_key[e.order_key] for e in exits)
                    for entry in exits:
                        entry.status = "sent"
                    names = ", ".join(entry.symbol for entry in exits)
                    self._protect_in_session(f"after the timed sell of {names}")
                # The sells sent above have had their fill wait in `send`.
                cash = ledger.cash(
                    (
                        f.execution.orderRef or "",
                        float(f.execution.shares),
                        float(f.execution.price),
                    )
                    for f in self._client.fills()
                )
                entries: list[tuple[WatchedEntry, PlannedOrder]] = []
                for entry in watch.waiting():
                    if entry.side != "buy":
                        continue
                    reason = entry_trigger(
                        reference_price=entry.reference_price,
                        quote=seen[entry.order_key],
                        now=now,
                        schedule=schedule,
                        params=params,
                        support_level=entry.support_level,
                    )
                    if reason is None:
                        continue
                    sells_waiting = any(e.side == "sell" for e in watch.waiting())
                    order = self._release(
                        by_key[entry.order_key],
                        entry,
                        reason,
                        now,
                        session_room=ledger.session_room(
                            float(self._config.max_session_notional_usd or 0.0), watch.waiting()
                        ),
                        cash=cash,
                        may_wait=sells_waiting and now < schedule.expires_at,
                    )
                    if order is not None:
                        cash -= ledger.release_buy(entry, order)
                        entries.append((entry, order))
                watch.write(now, force=bool(entries))
                if entries:
                    yield tuple(order for _, order in entries)
                    for entry, _ in entries:
                        entry.status = "sent"
                    names = ", ".join(entry.symbol for entry, _ in entries)
                    self._protect_in_session(f"after the timed buy of {names}")
                if watch.waiting():
                    self._client.sleep(params.poll_interval_s)
            watch.state = "complete"
        finally:
            quotes.close()
            no_vwap = sorted(
                e.symbol
                for e in watch.entries
                if e.side == "buy" and e.last is not None and e.vwap is None
            )
            if no_vwap:
                # Fail-safe, not silent: the VWAP trigger could not fire for these.
                print(
                    f"order_router: no VWAP was streamed for {', '.join(no_vwap)}; "
                    "the VWAP-discount trigger could not fire for them",
                    file=sys.stderr,
                )
            if watch.state == "watching":
                watch.state = "stopped"
            watch.write(self._clock())
        if unknown:
            raise UnknownOrderStateError(unknown)

    def _release(
        self,
        order: PlannedOrder,
        entry: WatchedEntry,
        reason: str,
        now: dt.datetime,
        *,
        session_room: float,
        cash: float,
        may_wait: bool,
    ) -> PlannedOrder | None:
        """The order a fired buy trigger sends, re-held to the caps and the cash at the price now.

        The shares were sized at the session start; the price may have moved since.
        The per-order and per-session caps are re-checked at the last price seen and
        the order clipped to them -- never enlarged. The book's cash
        (:class:`_CashLedger`) bounds it too: while a sell is still waiting and the
        expiry has not come, a buy the cash cannot pay for in full stays WAITING
        rather than going out short, since one key per name per day means a clipped
        buy can never be topped up. With no sell left to free cash, or at the
        expiry, it is clipped to the cash instead. The decision (price and instant)
        is the one taken at the session start, so the shortfall measures the wait.
        """
        price = entry.last if entry.last is not None else entry.reference_price
        quantity = entry.quantity
        if quantity * price > cash and may_wait:
            note = (
                f"{reason}; held for sell proceeds: ${max(cash, 0.0):,.2f} of the book's cash "
                f"for ${quantity * price:,.2f}"
            )
            if not (entry.trigger or "").startswith(reason):
                print(f"order_router: {entry.symbol} {note}", file=sys.stderr)
            entry.trigger = note
            return None
        caps = min(float(self._config.max_order_notional_usd or 0.0), session_room)
        room = min(caps, cash)
        if quantity * price > room:
            quantity = max(0, math.floor(room / price))
        bound = "the book's cash" if cash < caps else "the caps"
        entry.trigger, entry.released_at, entry.release_price = reason, now, price
        if quantity == 0:
            entry.status = "not_sent"
            entry.trigger = (
                f"{reason}; the notional caps leave no room at {price:.2f}"
                if bound == "the caps"
                else f"{reason}; the book's cash (${max(cash, 0.0):,.2f}) buys no share at "
                f"{price:.2f}"
            )
            return None
        entry.status, entry.released_quantity = "triggered", quantity
        if quantity == entry.quantity:
            return order
        d = order.decision
        print(
            f"order_router: {d.symbol} {entry.quantity} shares at {price} exceeds {bound} at "
            f"its trigger; clipped to {quantity}",
            file=sys.stderr,
        )
        return PlannedOrder(
            trading_day=order.trading_day,
            decision=decide(d.symbol, d.side, quantity, price=d.decision_price, now=d.decided_at),
        )

    def _protect_in_session(self, moment: str) -> None:
        """:meth:`protect_holdings` mid-session; a stop it cannot place never stops the day.

        A name left unprotected here is logged and re-attempted by the session's
        final :meth:`protect_holdings`, which fails the run if it still cannot;
        stopping the waiting buys would not protect it. The kill switch and a
        broken connection still raise, as on every poll.
        """
        try:
            results = self.protect_holdings()
        except ProtectiveStopError as exc:
            self._stop_events.append({"moment": moment, "error": str(exc)})
            print(
                f"order_router: {moment}: {exc}; re-attempted at the session's end",
                file=sys.stderr,
            )
            return
        self._stop_events.extend(
            {"moment": moment, **result} for result in results if result["disposition"] != "kept"
        )

    @property
    def protective_stop_events(self) -> tuple[dict[str, object], ...]:
        """Every stop placed or cancelled, and every failure, during the all-day session."""
        return tuple(self._stop_events)

    def _stop_reference(self, symbol: str) -> float | None:
        """The price a stop is set below: the live stream's last for a watched name, else a quote.

        A name the entry watch streams is never re-quoted: a snapshot request
        and its cancel on the same contract would end the watch's subscription.
        """
        if self._intraday is not None and symbol in self._intraday:
            return self._intraday(symbol).last
        return self._quote(symbol)

    def entry_watch_payload(self, now: dt.datetime) -> tuple[str, bytes] | None:
        """The day's final entry-watch document, or None when no buy was held."""
        if self._watch is None:
            return None
        return self._watch.key, self._watch.payload(now)

    def held_weights(self, portfolio_notional_usd: float) -> dict[str, float]:
        """The account's positions as the book held into the session (`held_weights`)."""
        statement = IbPaperStatementSource(self._client).read_statement(self._day)
        return held_weights(
            statement.positions, quote=self._quote, portfolio_notional_usd=portfolio_notional_usd
        )

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
        unknown: list[OrderIntentConflictError] = []
        try:
            for order in batch:
                # The kill switch, re-read before EVERY order.
                assert_trading_permitted(self._store, self._day)
                if order.key in known:
                    prior = parse_order_ref(known[order.key])
                    self._records.append(self._record(prior, "adopted", "Adopted", None))
                    continue
                if order.action == "SELL":
                    # A sell and a standing stop must never both fill.
                    self._cancel_stops(order.decision.symbol, self._open_stops())
                    clipped = self._clip_to_held(order)
                    if clipped is None:
                        continue
                    order = clipped
                try:
                    trade = self._place(order)
                except OrderIntentConflictError as conflict:
                    # Intent exists, broker does not know the order: unknown, never re-sent.
                    unknown.append(conflict)
                    continue
                placed.append((order, trade, self._clock()))
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
        if unknown:
            raise UnknownOrderStateError(unknown)

    def _clip_to_held(self, order: PlannedOrder) -> PlannedOrder | None:
        """A sell clipped to the shares held once its name's stops are cancelled.

        A timed sell waits all day while its name's protective stop stands. If the
        stop fired in the meantime, the planned quantity is more than is held and
        sending it would leave the book short, so it is cut to what is held -- and
        not sent at all when nothing is.
        """
        d = order.decision
        statement = IbPaperStatementSource(self._client).read_statement(self._day)
        held = max(0, statement.positions.get(d.symbol, 0))
        if held >= d.quantity:
            return order
        if held == 0:
            print(
                f"order_router: {d.symbol}: no share is held once its stops are cancelled "
                f"(a protective stop filled); the sell of {int(d.quantity)} is not sent",
                file=sys.stderr,
            )
            return None
        print(
            f"order_router: {d.symbol}: {held} share(s) held once its stops are cancelled; "
            f"the sell of {int(d.quantity)} is clipped to {held}",
            file=sys.stderr,
        )
        return PlannedOrder(
            trading_day=order.trading_day,
            decision=decide(d.symbol, d.side, held, price=d.decision_price, now=d.decided_at),
        )

    def _claim_intent(self, order: PlannedOrder) -> None:
        """The durable intent, written before the submit. Raises; never sends past it."""
        d = order.decision
        claim_order_intent(
            self._store,
            order_intent_document(
                order_key=order.key,
                order_ref=order.order_ref,
                trading_day=order.trading_day,
                symbol=d.symbol,
                side=d.side,
                quantity=int(d.quantity),
                decision_price=float(d.decision_price),
                account=self._config.account,  # pyright: ignore[reportArgumentType]  # __init__'s assert_paper refused a None account
                intended_at=self._clock(),
            ),
        )

    def _place(self, order: PlannedOrder) -> Any:
        contract = self._sdk.Stock(order.decision.symbol, "SMART", "USD")
        if not _qualified(self._client, contract):
            raise RoutingRefusedError(f"IB could not qualify {order.decision.symbol}")
        # The last step before the broker: nothing below runs unless the intent is durable.
        self._claim_intent(order)
        ib_order = self._sdk.MarketOrder(order.action, int(order.decision.quantity))
        ib_order.orderRef = order.order_ref
        ib_order.account = self._config.account
        ib_order.tif = "DAY"
        ib_order.outsideRth = False
        ib_order.transmit = True
        return self._client.placeOrder(contract, ib_order)

    def _open_stops(self) -> dict[str, list[Any]]:  # pyright: ignore[reportExplicitAny]  # ib_async Trades
        """This router's working protective stops, by symbol."""
        stops: dict[str, list[Any]] = {}  # pyright: ignore[reportExplicitAny]  # ib_async Trades
        for trade in self._client.reqAllOpenOrders():
            ref = trade.order.orderRef or ""
            if ref.startswith(STOP_REF_PREFIX + _REF_SEP) and (
                trade.orderStatus.status not in TERMINAL_STATUSES
            ):
                stops.setdefault(trade.contract.symbol, []).append(trade)
        return stops

    def _cancel_stops(self, symbol: str, stops: Mapping[str, list[Any]]) -> list[str]:  # pyright: ignore[reportExplicitAny]
        """Cancel ``symbol``'s stops and wait until the broker confirms it. Raises if not."""
        trades = stops.get(symbol, [])
        for trade in trades:
            self._client.cancelOrder(trade.order)
        waited = 0.0
        while any(t.orderStatus.status not in TERMINAL_STATUSES for t in trades):
            if waited >= self._fill_timeout_s:
                raise ProtectiveStopError(
                    f"{symbol}: a stale protective stop was not confirmed cancelled within "
                    f"{self._fill_timeout_s:g}s; no further order on it is sent, so the stops "
                    "can never exceed the shares held"
                )
            self._client.sleep(POLL_INTERVAL_S)
            waited += POLL_INTERVAL_S
        return [t.order.orderRef for t in trades]

    def protect_holdings(self) -> tuple[dict[str, object], ...]:
        """Re-assert one GTC STP SELL per held name, after the session's orders.

        Sized to the shares held less every other working sell on the name, at
        :attr:`RouterConfig.stop_distance_pct` below the session's reference
        price (:attr:`_quote`). A stop already standing for today at that size
        is kept; any other stop of this router's on the name is cancelled --
        confirmed -- first. A stop on a name no longer held is cancelled. The
        kill switch is re-read before every stop. Every name is attempted; any
        that could not be protected raise :class:`ProtectiveStopError` at the end.
        """
        assert_trading_permitted(self._store, self._day)
        positions = IbPaperStatementSource(self._client).read_statement(self._day).positions
        stops = self._open_stops()
        other_sells: dict[str, float] = {}
        for trade in self._client.reqAllOpenOrders():
            ref = trade.order.orderRef or ""
            if (
                trade.order.action == "SELL"
                and not ref.startswith(STOP_REF_PREFIX + _REF_SEP)
                and trade.orderStatus.status not in TERMINAL_STATUSES
            ):
                remaining = float(trade.order.totalQuantity) - float(trade.orderStatus.filled)
                symbol = trade.contract.symbol
                other_sells[symbol] = other_sells.get(symbol, 0.0) + max(0.0, remaining)
        held_names = {s for s, q in positions.items() if q > 0 and s in self._managed}
        results: list[dict[str, object]] = []
        failures: list[str] = []
        for symbol in sorted(held_names | set(stops)):
            quantity = 0
            if symbol in held_names:
                quantity = max(0, math.floor(positions[symbol] - other_sells.get(symbol, 0.0)))
            existing = stops.get(symbol, [])
            keep_prefix = _REF_SEP.join((STOP_REF_PREFIX, self._day, symbol, str(quantity))) + (
                _REF_SEP
            )
            if (
                quantity > 0
                and len(existing) == 1
                and existing[0].order.orderRef.startswith(keep_prefix)
            ):
                results.append(
                    {
                        "symbol": symbol,
                        "disposition": "kept",
                        "order_ref": existing[0].order.orderRef,
                    }
                )
                continue
            try:
                results.append(self._reassert_stop(symbol, quantity, stops))
            except RoutingRefusedError as exc:
                failures.append(f"{symbol}: {exc}")
        if failures:
            raise ProtectiveStopError(
                f"{len(failures)} held name(s) are without a protective stop: "
                + "; ".join(failures)
            )
        return tuple(results)

    def _reassert_stop(
        self,
        symbol: str,
        quantity: int,
        stops: Mapping[str, list[Any]],  # pyright: ignore[reportExplicitAny]
    ) -> dict[str, object]:
        """Cancel ``symbol``'s stale stops, then place one for ``quantity`` (none at 0)."""
        cancelled = self._cancel_stops(symbol, stops)
        if quantity == 0:
            return {"symbol": symbol, "disposition": "cancelled", "cancelled": cancelled}
        reference = self._stop_reference(symbol)
        if reference is None or not (math.isfinite(reference) and reference > 0):
            raise PriceUnavailableError(f"no reference price for {symbol} ({reference!r})")
        price = stop_price_for(reference, self._config.stop_distance_pct)
        ref = stop_ref(self._day, symbol, quantity, price)
        # The kill switch, re-read before every stop as before every order.
        assert_trading_permitted(self._store, self._day)
        self._place_stop(symbol, quantity, price, ref)
        return {
            "symbol": symbol,
            "disposition": "placed",
            "order_ref": ref,
            "quantity": quantity,
            "stop_price": price,
            "reference_price": reference,
            "cancelled": cancelled,
        }

    def _place_stop(self, symbol: str, quantity: int, stop_price: float, ref: str) -> None:
        contract = self._sdk.Stock(symbol, "SMART", "USD")
        if not _qualified(self._client, contract):
            raise RoutingRefusedError(f"IB could not qualify {symbol}")
        ib_order = self._sdk.StopOrder("SELL", quantity, stop_price)
        ib_order.orderRef = ref
        ib_order.account = self._config.account
        ib_order.tif = "GTC"
        ib_order.outsideRth = False
        ib_order.transmit = True
        self._client.placeOrder(contract, ib_order)

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
        if self._intraday is not None:
            self._intraday.close()
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
