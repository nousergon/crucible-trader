"""The daily trader session on the executor box: serve the champion, record the day.

`alpha-engine-config-I11545`, Brian's rulings of 2026-09-24:

1. (a) the v2 trader runs on the executor box, beside the IB Gateway paper
   session;
2. **a shadow session counts as a served day** -- "a shadow session resolves
   the champion, builds the book and records the session without sending
   orders. It starts the trader-week clock before order routing exists";
3. the order router is built against IB paper (`crucible_trader.order_router`),
   OFF unless `CRUCIBLE_TRADER_ORDER_ROUTING=ib_paper`.

This module is the entry point the box timer runs, once per trading day. It
runs :func:`crucible_trader.session.session_cycle` -- the one place the guards
are composed (wheel pin, contract, hold-book, kill switch, construction) --
with the router :func:`~crucible_trader.order_router.build_router` returns, and
then files the two artifacts the phase-4 gate reads, both as OUTPUTS of the
`trader.session` run manifest:

* `trader/evidence.json` -- the served day, with its SESSION MODE: `shadow`
  when routing is off (the router yields no batch), `live` when it is on;
* `trader/execution_shortfall/{day}.json` -- the router's own
  `shortfall_document()`: `no_orders` for a shadow session, measured fills
  for a routed one.

**Routing is decided by the environment and nothing here sets it.** Unset or
`off` is shadow; the box's env file leaves it unset. Flipping it is an
operator decision (I11545 ruling 3), not a deploy.

**What is and is not a served day.**

* A clean session: the day is recorded under its mode.
* No champion, feed or construction input yet
  (:class:`~crucible_trader.contract.ContractUnavailable`): HELD. Nothing is
  served, so nothing is recorded -- the evidence is not touched and no
  shortfall artifact is filed (its schema requires the champion it would
  name). The run is `ok`: absence is the harness's page to raise on its own
  deadline (`contract` module docstring), and a failed manifest every morning
  before the first promotion is how a channel gets muted.
* A refused contract, a wheel the pin does not name, a degenerate feed (the
  hold-book halt), an engaged kill switch: the run FAILS and nothing is
  recorded. A halted day is not a served day. A ROUTED session that fails
  after placing orders still files its shortfall document, so the orders that
  reached the broker are on the record, then fails.

**The hold-book safeguard is handed a READ-ONLY broker**
(:class:`ReadOnlyHoldControl`, over a `readonly` gateway session): a `hold`
reads the statement and the accepted orders and changes nothing, and the two
calls that could change the book -- `cancel_all`, `submit_close` -- are
refused, so no session can cancel the v1 executor's stops on the shared paper
account.

**The book held into the session is all cash** (``previous_weights=None``)
while every session is shadow: no order has ever left this trader, so its own
book is empty. The IB paper account may hold positions -- the v1 executor
trades it -- and those are not this trader's book. Deriving
``previous_weights`` from the trader's own routed positions is part of turning
routing on, not of this module.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from zoneinfo import ZoneInfo

from crucible.calendar import is_trading_day
from crucible.execution import EXECUTION_SHORTFALL_SCHEMA_VERSION, execution_shortfall_key
from crucible.runner import RunContext, run_job
from crucible.store import Store

from crucible_trader.broker_session import GatewayAddress, OrderRefusedError, connect, load_sdk
from crucible_trader.construction import TargetBook, construct_target_book
from crucible_trader.contract import (
    EVIDENCE_SCHEMA_VERSION,
    ContractUnavailable,
    ResolvedContract,
    record_session,
    resolve,
)
from crucible_trader.kill_switch import IbBrokerControl, assert_trading_permitted, utcnow
from crucible_trader.order_router import RouterConfig, build_router
from crucible_trader.paper_smoke import installed_crucible_version, open_configured_store
from crucible_trader.session import session_cycle
from crucible_trader.settings import Settings

__all__ = [
    "SESSION_JOB",
    "ReadOnlyHoldControl",
    "daily_session_cycle",
    "main",
    "run_daily_session",
]

#: The job every session files its manifest under
#: (`crucible.models.TRADER_JOB_VALUES`).
SESSION_JOB = "trader.session"

METRIC_MODULE = "crucible_trader.daily_session"

NYSE_ZONE = ZoneInfo("America/New_York")

#: The router surface this module relies on beyond `session.OrderRouter`.
RouterFactory = Callable[..., Any]


class ReadOnlyHoldControl(IbBrokerControl):
    """The hold-book safeguard's broker, over a READ-ONLY gateway session.

    It reads the statement and the accepted orders exactly as the live control
    does; the two calls that change the book are refused.
    """

    def cancel_all(self) -> None:
        raise OrderRefusedError("the session's hold-book control never cancels an order")

    def submit_close(self, symbol: str, shares: int) -> None:
        raise OrderRefusedError(
            f"the session's hold-book control never closes a position ({symbol})"
        )


def _metric(ctx: RunContext, *, served: bool, reason: str) -> dict[str, Any]:
    return {
        "name": "trader_session_served",
        "module": METRIC_MODULE,
        "metric_type": "operational",
        "value": 1.0 if served else 0.0,
        "unit": "flag",
        "n_floor": 1,
        "status": "OK",
        "status_reason": reason,
        "source_path": f"predictions/{ctx.trading_day.isoformat()}.json",
        "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _file_shortfall(
    ctx: RunContext, router: Any, *, champion: str, clock: Callable[[], dt.datetime]
) -> dict:
    day = ctx.trading_day.isoformat()
    document = router.shortfall_document(champion=champion, trading_day=day, now=clock())
    ctx.record_output(
        execution_shortfall_key(day),
        json.dumps(document, indent=2, sort_keys=True).encode("utf-8"),
        EXECUTION_SHORTFALL_SCHEMA_VERSION,
    )
    return document


def daily_session_cycle(
    ctx: RunContext,
    *,
    broker: IbBrokerControl,
    router: Any,
    installed_version: str,
    clock: Callable[[], dt.datetime] = utcnow,
    contract_resolver: Callable[[Store, str], ResolvedContract] = resolve,
    construct: Callable[..., TargetBook] | None = None,
) -> dict[str, Any]:
    """The job body. Returns the session document; raises on every refusal.

    ``construct`` defaults to `construct_target_book`, resolved at call time.
    """
    day = ctx.trading_day.isoformat()
    mode = "live" if router.enabled else "shadow"
    resolved: list[ResolvedContract] = []

    def capture(store: Store, trading_day: str) -> ResolvedContract:
        contract = contract_resolver(store, trading_day)
        resolved.append(contract)
        return contract

    try:
        result = session_cycle(
            ctx,
            broker=broker,
            router=router,
            installed_version=installed_version,
            previous_weights=None,
            clock=clock,
            contract_resolver=capture,
            construct=construct or construct_target_book,
        )
    except ContractUnavailable as exc:
        # Failure mode swallowed: the harness has not published a champion,
        # a feed or a construction input for this day. Why the deliverable
        # survives: nothing is served, so nothing is recorded -- the evidence
        # document is left exactly as it was, and the day does not count.
        # Recording surface: this run's `trader_session_served` metric (0,
        # naming the reason) and the returned document in its manifest; the
        # absence PAGE is the harness's, on its own deadline.
        ctx.record_metric(_metric(ctx, served=False, reason=f"held: {exc}"))
        return {"trading_day": day, "session_mode": mode, "outcome": "held"}
    except BaseException:
        # A routed session stopped after orders reached the broker: those
        # orders are filed before the run fails, so they are never off the
        # record. A shadow router holds no records and files nothing here.
        if resolved and router.records:
            _file_shortfall(ctx, router, champion=resolved[0].arm_id, clock=clock)
        raise

    # A halted day is not a served day. `session_cycle` checks the switch only
    # before each batch, and a shadow session has none.
    assert_trading_permitted(ctx.store, day)
    (contract,) = resolved

    evidence = record_session(
        ctx.store,
        contract,
        mode=mode,
        calendar_date=ctx.calendar_date.isoformat(),
        put=lambda key, payload: ctx.record_output(key, payload, EVIDENCE_SCHEMA_VERSION),
    )
    shortfall = _file_shortfall(ctx, router, champion=contract.arm_id, clock=clock)
    shadow_days = len(evidence.shadow_days())
    ctx.record_metric(
        _metric(
            ctx,
            served=True,
            reason=(
                f"{mode} session on {contract.arm_id}: {evidence.trading_days} served day(s), "
                f"{shadow_days} of them shadow"
            ),
        )
    )
    return {
        **result,
        "session_mode": mode,
        "outcome": "served",
        "shortfall_outcome": shortfall["outcome"],
        "trading_days": evidence.trading_days,
        "shadow_days": shadow_days,
    }


def run_daily_session(
    store: Store,
    *,
    connector: Callable[[], Any],
    sdk: Any,
    router_config: RouterConfig,
    installed_version: str,
    clock: Callable[[], dt.datetime] = utcnow,
    run_mode: str | None = None,
    router_factory: RouterFactory = build_router,
) -> RunContext:
    """Run one session as `trader.session`, always closing what it opened."""
    opened: list[Any] = []

    def body(ctx: RunContext) -> dict[str, Any]:
        router = router_factory(
            router_config,
            store=ctx.store,
            trading_day=ctx.trading_day.isoformat(),
            sdk_loader=lambda: sdk,
            clock=clock,
        )
        opened.append(router.close)
        if router.enabled:
            # One gateway session per client id: a routed session's hold-book
            # control reads through the router's own connection (still behind
            # the read-only wrapper) rather than opening a second one the
            # gateway would refuse as a duplicate client id.
            client = router._client  # noqa: SLF001 - the router's one connection
        else:
            client = connector()
            opened.append(client.disconnect)
        return daily_session_cycle(
            ctx,
            broker=ReadOnlyHoldControl(client, sdk),
            router=router,
            installed_version=installed_version,
            clock=clock,
        )

    try:
        return run_job(
            SESSION_JOB,
            body,
            store=store,
            now=clock(),
            run_mode=run_mode,
            # One attempt: a session is one decision per day, and a gateway
            # that is not logged in is a human action, not a transient.
            transient_retry=False,
        )
    finally:
        for close in opened:
            close()


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    sdk_loader: Callable[[], Any] = load_sdk,
    version_reader: Callable[[], str] = installed_crucible_version,
    clock: Callable[[], dt.datetime] = utcnow,
    printer: Callable[[str], None] = print,
    router_factory: RouterFactory = build_router,
) -> int:
    """Box entry point (`scripts/trader_pinned.sh daily_session`)."""
    parser = argparse.ArgumentParser(prog="crucible_trader.daily_session")
    parser.add_argument("--run-mode", choices=("live", "replay"), default="live")
    args = parser.parse_args(argv)
    today = clock().astimezone(NYSE_ZONE).date()
    if not is_trading_day(today):
        printer(f"SKIP: {today} is not an NYSE session; a session serves trading days only")
        return 0
    source = None if environ is None else dict(environ)
    settings = Settings.from_env(source)
    address = GatewayAddress.from_env(source)
    router_config = RouterConfig.from_env(source)
    sdk = sdk_loader()
    ctx = run_daily_session(
        open_configured_store(settings),
        connector=lambda: connect(address, readonly=True, sdk_loader=lambda: sdk),
        sdk=sdk,
        router_config=router_config,
        installed_version=version_reader(),
        clock=clock,
        run_mode=args.run_mode,
        router_factory=router_factory,
    )
    mode = "routed" if router_config.enabled else "shadow"
    printer(f"trader.session {ctx.trading_day} {ctx.run_id}: {mode} session filed")
    return 0
