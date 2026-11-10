"""The pre-open plan: the day's order book, built and saved before the open.

Brian 2026-11-09/10 (`alpha-engine-config-I12024`): the Crucible v2 daily
order book must exist before the US market opens at 09:30 ET, every trading
day. Until this module it was written only inside the session
(`crucible_trader.session` calls `crucible_trader.order_book.record_order_book`),
and the session starts after the open. The inputs are ready the evening
before: `predictions/{day}.json` is published at about 20:45 ET.

The box timer (crucible-executor `alpha-engine-trader-order-book-plan.timer`)
runs :func:`main` at 08:40 ET on weekdays. It:

1. **refuses a day that is not an NYSE session**, and a start at or after
   09:30 ET (the book is then the session's to write). An early-close day is
   a trading day and is planned like any other;
2. **binds the session's own trading day**, the last closed session
   (`crucible.calendar.resolve_trading_day`), so the book is keyed exactly as
   the session that trades it will key its own;
3. runs the session's read-only guards in the session's order: the wheel the
   pin names, the two-document contract, the degenerate-feed test and the
   kill switch. Where the session would HOLD or REFUSE, no book is written:
   the plan writes no kill-switch state, so a degenerate feed is only
   reported here and the session engages the hold itself;
4. **reads the book held into the session from IB, read-only**, when order
   routing is on, exactly as a routed session does: the account's positions
   valued at the latest prices the router would plan against (IB, then the
   book day's close). It connects with its own client id,
   ``CRUCIBLE_TRADER_PLAN_IB_CLIENT_ID``, which may not be the session's, the
   reconcile's master id, or one of v1's (1-3). With routing off the session
   starts from all cash, and so does the plan, which then never connects;
5. builds the book through `crucible_trader.construction.construct_target_book`,
   the session's engine on the session's inputs, and writes it once, create
   only, at ``trader/order_books/{day}.json`` with ``stage: pre_open``, under
   COMPLIANCE retention (`crucible_trader.order_book.write_pre_open_order_book`).

**It places no order and builds no router.** The broker session is opened
with ``readonly=True``, its positions are read through
:class:`~crucible_trader.broker_session.ReadOnlyBroker`, and the only other
calls are market-data quotes.

**Not a `crucible.runner` job.** A job files a manifest under a name in
`crucible.models.JOB_VALUES`, and that list is in the harness, which reaches
the trader only through the wheel the trader pin names. Like
`crucible_trader.shadow_books_daily`, the plan's record is its own artifact:
the book carries its stage and its write time, and a failure is a non-zero
exit, which is the systemd unit's failure page. The session's
`trader_order_book_written` metric still says, on the session's manifest,
whether the pre-open book was traded as written (`crucible_trader.order_book`).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from zoneinfo import ZoneInfo

from crucible.calendar import is_trading_day, resolve_trading_day
from crucible.store import Store

from crucible_trader.broker_session import (
    CLIENT_ID_VAR,
    MASTER_CLIENT_ID_VAR,
    BrokerSettingsError,
    GatewayAddress,
    ReadOnlyBroker,
    connect,
    load_sdk,
)
from crucible_trader.broker_statement import IbPaperStatementSource
from crucible_trader.construction import TargetBook, construct_target_book
from crucible_trader.contract import ContractUnavailable, ResolvedContract, resolve
from crucible_trader.daily_session import SESSION_OPEN_ET, _held_into_session  # pyright: ignore[reportPrivateUsage]  # the session's own valuation, reused
from crucible_trader.hold_book import should_hold_book
from crucible_trader.kill_switch import TradingHaltedError, assert_trading_permitted, utcnow
from crucible_trader.order_book import write_pre_open_order_book
from crucible_trader.order_router import (
    CloseFallbackQuote,
    IbQuoteSource,
    QuoteSource,
    RouterConfig,
    held_weights,
)
from crucible_trader.paper_smoke import (
    RunningWheelMismatchError,
    installed_crucible_version,
    open_configured_store,
    running_wheel_disagreement,
)
from crucible_trader.settings import Settings
from crucible_trader.simulated_broker import PanelCloseSource, StagingSessionConfig

__all__ = [
    "PLAN_CLIENT_ID_VAR",
    "V1_CLIENT_IDS",
    "IbHoldings",
    "PlanOutcome",
    "main",
    "plan_gateway_address",
    "plan_order_book",
]

NYSE_ZONE = ZoneInfo("America/New_York")

#: The plan's own IB client id. Distinct from every other session on the
#: gateway, so the plan can never collide with the session, the reconcile or v1.
PLAN_CLIENT_ID_VAR = "CRUCIBLE_TRADER_PLAN_IB_CLIENT_ID"

#: The v1 executor's client ids on the shared paper gateway.
V1_CLIENT_IDS = frozenset({1, 2, 3})


def plan_gateway_address(env: Mapping[str, str]) -> GatewayAddress:
    """The gateway address on the plan's own client id. Refuses a shared id.

    The session's id (:data:`~crucible_trader.broker_session.CLIENT_ID_VAR`)
    must be set, so the comparison is never skipped; the master id is compared
    when it is set.
    """
    source = dict(env)
    plan = GatewayAddress.from_env(source, client_id_var=PLAN_CLIENT_ID_VAR)
    taken = {CLIENT_ID_VAR: GatewayAddress.from_env(source).client_id}
    if (source.get(MASTER_CLIENT_ID_VAR) or "").strip():
        taken[MASTER_CLIENT_ID_VAR] = GatewayAddress.from_env(
            source, client_id_var=MASTER_CLIENT_ID_VAR
        ).client_id
    for var, client_id in taken.items():
        if plan.client_id == client_id:
            raise BrokerSettingsError(
                f"{PLAN_CLIENT_ID_VAR} equals {var} ({client_id}). The plan has its own client "
                "id, so it can never take a session the session or the reconcile needs."
            )
    if plan.client_id in V1_CLIENT_IDS:
        raise BrokerSettingsError(
            f"{PLAN_CLIENT_ID_VAR}={plan.client_id} is one of the v1 executor's client ids "
            f"{sorted(V1_CLIENT_IDS)}"
        )
    return plan


class IbHoldings:
    """The account's positions, read-only, as the book held into the session.

    The same valuation a routed session makes (`order_router.held_weights`
    over `IbQuoteSource`, then the book day's close), through a session that
    has no order method.
    """

    def __init__(
        self,
        client: Any,  # pyright: ignore[reportExplicitAny]  # an ib_async IB
        sdk: Any,  # pyright: ignore[reportExplicitAny]  # the ib_async module
        *,
        store: Store,
        trading_day: str,
        quote: QuoteSource | None = None,
    ) -> None:
        self._client = client
        self._day = trading_day
        self._quote: QuoteSource = (
            quote
            if quote is not None
            else CloseFallbackQuote(
                IbQuoteSource(client, sdk), PanelCloseSource(store, trading_day)
            )
        )

    def held_weights(self, portfolio_notional_usd: float) -> dict[str, float]:
        statement = IbPaperStatementSource(ReadOnlyBroker(self._client)).read_statement(self._day)
        return held_weights(
            statement.positions, quote=self._quote, portfolio_notional_usd=portfolio_notional_usd
        )


@dataclasses.dataclass(frozen=True)
class PlanOutcome:
    """What one plan run did. ``key`` is set when a book is at the day's key."""

    trading_day: str
    outcome: str  # written | already_written | held
    reason: str
    key: str | None = None


def plan_order_book(
    store: Store,
    *,
    trading_day: str,
    installed_version: str,
    holdings: Callable[[], object] | None,
    clock: Callable[[], dt.datetime] = utcnow,
    contract_resolver: Callable[[Store, str], ResolvedContract] = resolve,
    construct: Callable[..., TargetBook] = construct_target_book,
) -> PlanOutcome:
    """Build and write the day's pre-open book. Raises where the session would.

    ``holdings`` opens the read-only holdings source (an :class:`IbHoldings`)
    when routing is on, or is None for the all-cash start of a shadow session.
    It is called only after every guard has passed, so a day that is held
    never connects to the gateway.
    """
    disagreement = running_wheel_disagreement(store, installed_version)
    if disagreement is not None:
        raise RunningWheelMismatchError(disagreement)
    try:
        contract = contract_resolver(store, trading_day)
    except ContractUnavailable as exc:
        return PlanOutcome(trading_day, "held", f"no book: {exc}")
    decision = should_hold_book(contract.feed)
    if decision.hold:
        return PlanOutcome(
            trading_day,
            "held",
            f"no book: the feed is {decision.decision}; the session engages the hold",
        )
    try:
        assert_trading_permitted(store, trading_day)
    except TradingHaltedError as exc:
        return PlanOutcome(trading_day, "held", f"no book: {exc}")
    previous = _held_into_session(store, None if holdings is None else holdings())
    book = construct(store, trading_day=trading_day, previous_weights=previous, now=clock())
    key, written = write_pre_open_order_book(store, book, previous_weights=previous, now=clock())
    if not written:
        return PlanOutcome(trading_day, "already_written", f"{key} already holds a book", key)
    start = "the account's positions" if previous is not None else "all cash"
    return PlanOutcome(trading_day, "written", f"{key} written before the open, from {start}", key)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    sdk_loader: Callable[[], Any] = load_sdk,  # pyright: ignore[reportExplicitAny]
    version_reader: Callable[[], str] = installed_crucible_version,
    clock: Callable[[], dt.datetime] = utcnow,
    printer: Callable[[str], None] = print,
) -> int:
    """Box entry point (`scripts/trader_pinned.sh order_book_plan`). Takes no arguments."""
    if argv:
        raise SystemExit(f"crucible_trader.order_book_plan takes no arguments (got {list(argv)})")
    now = clock()
    local = now.astimezone(NYSE_ZONE)
    if not is_trading_day(local.date()):
        printer(f"SKIP: {local.date()} is not an NYSE session; the plan serves trading days only")
        return 0
    if local.time() >= SESSION_OPEN_ET:
        printer(
            f"SKIP: {local:%H:%M} ET is not before the {SESSION_OPEN_ET:%H:%M} open; "
            "the session writes the day's book"
        )
        return 0
    source = None if environ is None else dict(environ)
    if StagingSessionConfig.from_env(source) is not None:
        printer("SKIP: staging has no broker and no pre-open plan")
        return 0
    env: Mapping[str, str] = source if source is not None else os.environ
    store = open_configured_store(Settings.from_env(source))
    day = resolve_trading_day(now).isoformat()
    opened: list[Any] = []  # pyright: ignore[reportExplicitAny]

    def holdings() -> IbHoldings:
        address = plan_gateway_address(env)
        sdk = sdk_loader()
        client = connect(address, readonly=True, sdk_loader=lambda: sdk)
        opened.append(client)
        return IbHoldings(client, sdk, store=store, trading_day=day)

    try:
        outcome = plan_order_book(
            store,
            trading_day=day,
            installed_version=version_reader(),
            holdings=holdings if RouterConfig.from_env(source).enabled else None,
            clock=clock,
        )
    finally:
        for client in opened:
            client.disconnect()
    printer(f"order book plan {outcome.trading_day}: {outcome.outcome}: {outcome.reason}")
    return 0
