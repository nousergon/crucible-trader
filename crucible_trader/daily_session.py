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
  when routing is off (the router yields no batch), `live` when it is on,
  `simulated` for the staging simulated router (staging only,
  `alpha-engine-config-I12685`);
* `trader/execution_shortfall/{day}.json` -- the router's own
  `shortfall_document()`: `no_orders` for a shadow session, measured fills
  for a routed one, ``fill_source: simulated_close`` fills for a simulated one.

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

**In staging there is no broker at all** (`alpha-engine-config-I12507`, S-19).
Under ``NE_ENV=staging`` :func:`main` runs :func:`run_staging_session`: no SDK
is loaded and no gateway is connected; the plan is filled by simulation at the
session close when ``CRUCIBLE_TRADER_SIMULATED_FILLS=on`` and is shadow
otherwise (`crucible_trader.simulated_broker`). The routing switch present in
staging, or simulated fills switched on outside it, raise before anything
opens. Outside staging, with the flag unset or ``off``, the path below is the
prod path unchanged.

**The hold-book safeguard is handed a READ-ONLY broker**
(:class:`ReadOnlyHoldControl`, over a `readonly` gateway session): a `hold`
reads the statement and the accepted orders and changes nothing, and the two
calls that could change the book -- `cancel_all`, `submit_close` -- are
refused, so no session can cancel the v1 executor's stops on the shared paper
account.

**A routed session lasts the trading day** (Brian 2026-11-09,
`crucible_trader.entry_timing`). Its buys wait for v1's entry triggers and its
sells for their mirror (Brian 2026-11-10), and the last order goes out by 15:55 ET
(earlier on a half day), so the run -- and its one manifest, evidence entry and shortfall document
-- ends after that rather than two minutes after the start. The final watch
document is recorded as an output of the run, failed or not.

**The book held into the session is the account's positions** when routing
is live (``_held_into_session``), valued as weights of the book notional, so
the engine's turnover budget binds against what is actually held. Until
2026-11-05 every session started from all cash, written when no order had ever
left this trader; it survived routing going live on 11-03, and the budget then
measured each day against an empty book (97% one-way turnover on 11-03 and
11-04). A shadow or simulated session still starts from all cash.

**A live session is sized on the account's NAV and leaves every held long
protected** (the 2026-11-09 v1/v2 trading-parity audit, rows 5 and 1). The
book notional is ``min(book_notional_usd, NetLiquidation)`` read at session
start -- an unreadable NAV refuses the session -- and the sizing is filed in
the run's returned document and on the order book. After the session's
orders the router re-asserts one GTC protective stop per held name; a name
it cannot protect fails the run, after the orders are on the record.

**Under the all-day session the stops do not wait for the end.** The router
protects the names held at the start once the opening sells have settled,
and each timed buy as soon as its batch returns (`order_router`), so a run
that fails at 13:00 leaves every name it holds by then protected. The
end-of-session re-assertion here is the backstop: it keeps what already
stands, places what the day could not, and is the step that fails the run.

**A routed session starts only inside the regular session**
(`alpha-engine-config-I13447`). On 2026-11-09 at 20:36 ET a routed session
started after hours: a changed timer was reloaded on a box that was still up,
and systemd fired it at once. It held only because the day's feed had not been
published yet. :func:`outside_market_hours` is checked before the router is
built or the gateway is connected. Outside 09:30 ET to the close on a trading
day (13:00 on an early-close day), the session is HELD: the run is `ok`, its
`trader_session_served` metric is 0 and names the reason, and nothing is routed
or recorded. The check is made at START only. A session that started in hours
may run past its start, and later past the close. Shadow and simulated
sessions are not checked, because they route nothing.

**Its calendar date is the New York date.** The harness stamps
`calendar_date` from the UTC date, which is the next day after 20:00 ET (19:00
in winter). So :func:`_bind_new_york_calendar_date` re-stamps it, and the
manifest and evidence of the 20:36 ET run read 11-09, not 11-10. The
`trading_day` the run binds to was already New York's: `crucible.runner`
resolves it through `crucible.calendar.resolve_trading_day`, which converts to
New York time before it asks which session last closed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from zoneinfo import ZoneInfo

from crucible.calendar import is_trading_day
from crucible.execution import EXECUTION_SHORTFALL_SCHEMA_VERSION, execution_shortfall_key
from crucible.models import TraderEvidenceBench
from crucible.portfolio import PortfolioParamsError, load_portfolio_params_from_store
from crucible.runner import RunContext, run_job
from crucible.store import Store
from krepis.trading_calendar import session_close_et

from crucible_trader import turnover_tripwire
from crucible_trader.broker_session import GatewayAddress, OrderRefusedError, connect, load_sdk
from crucible_trader.construction import (
    STRATEGY_SLOT,
    SessionSizing,
    TargetBook,
    construct_target_book,
)
from crucible_trader.contract import (
    EVIDENCE_SCHEMA_VERSION,
    ContractRefusal,
    ContractUnavailable,
    ResolvedContract,
    record_session,
    resolve,
)
from crucible_trader.entry_timing import ENTRY_WATCH_SCHEMA_VERSION
from crucible_trader.hard_risk import BrokerFacts
from crucible_trader.kill_switch import (
    BrokerControl,
    IbBrokerControl,
    assert_trading_permitted,
    utcnow,
)
from crucible_trader.order_router import RouterConfig, ShadowRouter, build_router
from crucible_trader.paper_smoke import installed_crucible_version, open_configured_store
from crucible_trader.session import session_cycle
from crucible_trader.settings import Settings
from crucible_trader.simulated_broker import (
    FILL_SOURCE,
    SIMULATED_FILLS_SCHEMA_VERSION,
    SIMULATED_POSITIONS_SCHEMA_VERSION,
    SIMULATED_SLIPPAGE_BPS,
    SimulatedFillRouter,
    StagingHoldControl,
    StagingSessionConfig,
    simulated_fills_key,
    simulated_positions_key,
)

__all__ = [
    "SESSION_JOB",
    "ReadOnlyHoldControl",
    "daily_session_cycle",
    "main",
    "outside_market_hours",
    "run_daily_session",
    "run_staging_session",
]

#: The job every session files its manifest under
#: (`crucible.models.TRADER_JOB_VALUES`).
SESSION_JOB = "trader.session"

METRIC_MODULE = "crucible_trader.daily_session"

NYSE_ZONE = ZoneInfo("America/New_York")

log = logging.getLogger(__name__)

#: The NYSE regular session's open. A routed session starts at or after it.
#: `krepis.trading_calendar` holds the same value but does not export it.
SESSION_OPEN_ET = dt.time(9, 30)

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


def outside_market_hours(now: dt.datetime) -> str | None:
    """Why a ROUTED session may not start at ``now``, or None when it may.

    A routed session may start on an NYSE trading day from 09:30 ET until the
    session's close. The close is 16:00 ET, or 13:00 ET on an early-close day
    (`krepis.trading_calendar.session_close_et`, the calendar
    `crucible.calendar` wraps). The open is inclusive and the close is
    exclusive, so a 09:35 start is in hours and a start at the close is not.

    ``now`` must be timezone-aware. A naive time could be read in either zone,
    and the incident this guards was a UTC/New York confusion.
    """
    if now.tzinfo is None:
        raise ValueError(f"outside_market_hours needs an aware time, got naive {now.isoformat()}")
    local = now.astimezone(NYSE_ZONE)
    day = local.date()
    stamp = f"{local:%Y-%m-%d %H:%M:%S} ET"
    if not is_trading_day(day):
        return f"outside market hours: {stamp} is not on an NYSE trading day"
    close = session_close_et(day)
    if not SESSION_OPEN_ET <= local.time() < close:
        return (
            f"outside market hours: {stamp} is outside the regular session "
            f"({SESSION_OPEN_ET:%H:%M}-{close:%H:%M} ET on {day.isoformat()})"
        )
    return None


def _bind_new_york_calendar_date(ctx: RunContext) -> None:
    """Re-stamp the run's `calendar_date` with the New York date it started on.

    `crucible.runner.run_job` stamps it from the UTC date. Between 20:00 ET
    (19:00 in winter) and midnight, that is the next day. The field is
    provenance only (`run_manifest.v2`), so it may be corrected. It is set on
    the context the harness writes the manifest from, before any output is
    recorded.
    """
    ctx.calendar_date = ctx.started.astimezone(NYSE_ZONE).date()


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


def _record_entry_watch(
    ctx: RunContext, router: object, *, clock: Callable[[], dt.datetime]
) -> None:
    """Record the router's final entry-watch document on the run. Never raises.

    The router has already written it to the store as the day went; this makes
    the final state an output of the manifest. A failed write must not fail a
    session whose orders are already at the broker.
    """
    payload: Callable[[dt.datetime], tuple[str, bytes] | None] | None = getattr(
        router, "entry_watch_payload", None
    )
    found = None if payload is None else payload(clock())
    if found is None:
        return
    key, body = found
    try:
        ctx.record_output(key, body, ENTRY_WATCH_SCHEMA_VERSION)
    except Exception as exc:  # noqa: BLE001 - a missing record must never fail the orders
        log.exception("entry watch output not recorded")
        print(
            f"daily_session: entry watch {key} not recorded ({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )


def _held_into_session(
    store: Store, router: object
) -> tuple[dict[str, float] | None, SessionSizing | None]:
    """The book held into the session (None: all cash) and the live sizing (None: declared).

    A live router sizes the session on the account's NAV
    (`order_router.PaperOrderRouter.session_sizing`: ``min(book_notional_usd,
    NetLiquidation)``, refusing when the NAV is unreadable) and values the
    account's positions as weights of that notional (`order_router.held_weights`),
    so the engine's turnover budget and cost model measure each day against the
    book actually held. The staging simulated router answers the held book from
    its carried simulated book (`alpha-engine-config-I13305`) on the declared
    notional. A shadow router holds nothing of its own and keeps the all-cash
    start on the declared notional.
    """
    held: Callable[[float], dict[str, float]] | None = getattr(router, "held_weights", None)
    sizer: Callable[[float], SessionSizing] | None = getattr(router, "session_sizing", None)
    if held is None and sizer is None:
        return None, None
    try:
        params = load_portfolio_params_from_store(STRATEGY_SLOT, store=store)
    except PortfolioParamsError as exc:
        raise ContractRefusal(f"the S slot's portfolio parameter set is unusable: {exc}") from exc
    ceiling = float(params.book_notional_usd)
    sizing = None if sizer is None else sizer(ceiling)
    notional = ceiling if sizing is None else sizing.notional_usd
    return (None if held is None else held(notional)), sizing


def _hard_risk_facts(router: object) -> BrokerFacts | None:
    """The broker's facts for the hard-risk layer, or None for a router with no broker.

    A live router reads average cost, marks and NAV from the account
    (`order_router.PaperOrderRouter.hard_risk_facts`); a shadow or staging
    simulated router has no account, so the layer stays off for that session.
    """
    read: Callable[[], BrokerFacts] | None = getattr(router, "hard_risk_facts", None)
    return None if read is None else read()


def daily_session_cycle(
    ctx: RunContext,
    *,
    broker: BrokerControl,
    router: Any,
    installed_version: str,
    clock: Callable[[], dt.datetime] = utcnow,
    contract_resolver: Callable[[Store, str], ResolvedContract] = resolve,
    construct: Callable[..., TargetBook] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The job body. Returns the session document; raises on every refusal.

    ``construct`` defaults to `construct_target_book`, resolved at call time.
    ``environ`` (default `os.environ`) is what `record_session` checks a
    `simulated` day against: only staging may file one.
    """
    day = ctx.trading_day.isoformat()
    # `simulated` is a class attribute of the staging simulated router alone;
    # every prod router lacks it, so prod reads exactly `live` or `shadow`.
    if getattr(router, "simulated", False) is True:
        mode = "simulated"
    else:
        mode = "live" if router.enabled else "shadow"
    resolved: list[ResolvedContract] = []
    protective_stops: tuple[dict[str, object], ...] | None = None

    def capture(store: Store, trading_day: str) -> ResolvedContract:
        contract = contract_resolver(store, trading_day)
        resolved.append(contract)
        return contract

    try:
        previous_weights, sizing = _held_into_session(ctx.store, router)
        constructor = construct or construct_target_book
        result = session_cycle(
            ctx,
            broker=broker,
            router=router,
            installed_version=installed_version,
            previous_weights=previous_weights,
            clock=clock,
            contract_resolver=capture,
            construct=constructor
            if sizing is None
            else functools.partial(constructor, sizing=sizing),
            hard_risk_facts=_hard_risk_facts(router),
        )
        # Orders still working past the per-batch fill bound are waited on
        # (bounded) so the stops below and the shortfall filed after size from
        # what filled (`PaperOrderRouter.settle`). A router without it (shadow,
        # simulated) has nothing working.
        settle: Callable[[], int] | None = getattr(router, "settle", None)
        if settle is not None:
            settle()
        # After the session's orders: one standing GTC stop per held long
        # (`PaperOrderRouter.protect_holdings`). A router without stops (shadow,
        # simulated) has nothing to protect.
        protect: Callable[[], tuple[dict[str, object], ...]] | None = getattr(
            router, "protect_holdings", None
        )
        if protect is not None:
            protective_stops = protect()
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
        _record_entry_watch(ctx, router, clock=clock)
        raise

    # A halted day is not a served day. `session_cycle` checks the switch only
    # before each batch, and a shadow session has none.
    assert_trading_permitted(ctx.store, day)
    (contract,) = resolved
    bench = result.get("alpha_bench")
    alpha_bench = None if bench is None else TraderEvidenceBench.model_validate(bench)

    evidence = record_session(
        ctx.store,
        contract,
        mode=mode,
        calendar_date=ctx.calendar_date.isoformat(),
        put=lambda key, payload: ctx.record_output(key, payload, EVIDENCE_SCHEMA_VERSION),
        alpha_bench=alpha_bench,
        environ=environ,
    )
    shortfall = _file_shortfall(ctx, router, champion=contract.arm_id, clock=clock)
    _record_entry_watch(ctx, router, clock=clock)
    _record_turnover(ctx, result)
    _check_turnover_tripwire(ctx, result)
    shadow_days = len(evidence.shadow_days())
    ctx.record_metric(
        _metric(
            ctx,
            served=True,
            reason=(
                f"{mode} session on {contract.arm_id}"
                f"{'' if alpha_bench is None else f' benched at weight 0 ({alpha_bench.key})'}: "
                f"{evidence.trading_days} served day(s), {shadow_days} of them shadow"
            ),
        )
    )
    return {
        **result,
        "session_mode": mode,
        "outcome": "served",
        "shortfall_outcome": shortfall["outcome"],
        "sizing": None if sizing is None else sizing.record(),
        "protective_stops": None if protective_stops is None else list(protective_stops),
        # What the all-day session placed, resized or could not place as it went.
        "protective_stops_in_session": list(getattr(router, "protective_stop_events", ())),
        "trading_days": evidence.trading_days,
        "shadow_days": shadow_days,
    }


#: The budget recorded when the S slot's parameter set disables the turnover
#: budget: a long-only book can trade at most all of itself one way.
UNBOUNDED_TURNOVER_BUDGET = 1.0


def _record_turnover(ctx: RunContext, result: Mapping[str, object]) -> None:
    """Put the session's turnover and its budget on the run record (I13305).

    The staging soak gate grades ``turnover.breached`` on `trader.session`
    records. The harness method is looked up rather than called directly: the
    crucible that RUNS is the release wheel `trader/release_pin` names, which
    moves only through the trader's own smoke and may predate
    `RunContext.record_turnover`. A wheel without it files no turnover — the
    gate then reads the day as unknown, never green — and the session itself
    is untouched, because a missing telemetry method must never cost a trading
    day.
    """
    record: Callable[..., None] | None = getattr(ctx, "record_turnover", None)
    value = result.get("turnover_one_way")
    if record is None or not isinstance(value, float):
        return
    budget = result.get("turnover_budget")
    record(
        value=value,
        budget=float(budget) if isinstance(budget, (int, float)) else UNBOUNDED_TURNOVER_BUDGET,
    )


def _check_turnover_tripwire(ctx: RunContext, result: Mapping[str, object]) -> None:
    """File the churn alarm on the run (v1 parity audit 2026-11-09, row 16).

    `crucible_trader.turnover_tripwire.check` never raises; a tripwire that
    could not run files an ``unmeasurable`` metric naming why, so the alarm
    being dead is itself on the record.
    """
    day = ctx.trading_day.isoformat()
    block = turnover_tripwire.check(ctx.store, trading_day=day, result=result)
    ctx.record_metric(
        turnover_tripwire.metric_record(
            block,
            trading_day=day,
            now_utc=ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            module=METRIC_MODULE,
        )
    )


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
        _bind_new_york_calendar_date(ctx)
        if router_config.enabled:
            refusal = outside_market_hours(ctx.started)
            if refusal is not None:
                # Failure mode swallowed: a routed session started outside the
                # regular session (I13447: systemd fired one at 20:36 ET). Why
                # the deliverable survives: no router is built and no gateway
                # is connected, so no order can be routed, and nothing is
                # recorded, so the day stays unserved. The timer's next firing
                # in hours is the real session. Recording surface: the
                # `trader_session_served` metric (0, naming the reason) and the
                # returned document. The run is `ok` because a HOLD is not a
                # failure, so this does not page as one.
                ctx.record_metric(_metric(ctx, served=False, reason=f"held: {refusal}"))
                return {
                    "trading_day": ctx.trading_day.isoformat(),
                    "session_mode": "live",
                    "outcome": "held",
                    "held_reason": refusal,
                }
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


def run_staging_session(
    store: Store,
    *,
    config: StagingSessionConfig,
    installed_version: str,
    clock: Callable[[], dt.datetime] = utcnow,
    run_mode: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> RunContext:
    """Run one STAGING session as `trader.session`: no broker, no SDK, no connection.

    `alpha-engine-config-I12507`. The same :func:`daily_session_cycle` prod runs,
    handed :class:`~crucible_trader.simulated_broker.StagingHoldControl` and
    either a :class:`~crucible_trader.simulated_broker.SimulatedFillRouter`
    (simulated fills on) or the shadow router. A served simulated session also
    files its fills at `simulated_fills_key(day)`.
    """

    def body(ctx: RunContext) -> dict[str, object]:
        _bind_new_york_calendar_date(ctx)
        day = ctx.trading_day.isoformat()
        router: SimulatedFillRouter | ShadowRouter = (
            SimulatedFillRouter(config, store=ctx.store, trading_day=day, clock=clock)
            if config.simulated
            else ShadowRouter()
        )
        result: dict[str, object] = daily_session_cycle(
            ctx,
            broker=StagingHoldControl(),
            router=router,
            installed_version=installed_version,
            clock=clock,
            environ=environ,
        )
        if not isinstance(router, SimulatedFillRouter) or result["outcome"] != "served":
            return result
        # The closing simulated book, which tomorrow's session opens from
        # (`alpha-engine-config-I13305`).
        ctx.record_output(
            simulated_positions_key(day),
            router.simulated_positions_payload(trading_day=day, now=clock()),
            SIMULATED_POSITIONS_SCHEMA_VERSION,
        )
        key = simulated_fills_key(day)
        ctx.record_output(
            key,
            router.simulated_fills_payload(
                champion=str(result["m_champion"]), trading_day=day, now=clock()
            ),
            SIMULATED_FILLS_SCHEMA_VERSION,
        )
        # The evidence records the day as `simulated` and the shortfall carries
        # `fill_source: simulated_close`; this metric counts the fills, and the
        # side file above is their debug copy (`alpha-engine-config-I12685`).
        ctx.record_metric(
            {
                "name": "trader_session_simulated_fills",
                "module": METRIC_MODULE,
                "metric_type": "operational",
                "value": float(len(router.records)),
                "unit": "count",
                "n_floor": 1,
                "status": "OK",
                "status_reason": (
                    f"staging simulated session: {len(router.records)} fill(s) at the session "
                    f"close ({FILL_SOURCE}, {SIMULATED_SLIPPAGE_BPS} bps declared); no broker"
                ),
                "source_path": key,
                "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        return {**result, "simulated_fills_key": key}

    return run_job(
        SESSION_JOB,
        body,
        store=store,
        now=clock(),
        run_mode=run_mode,
        transient_retry=False,
    )


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
    # `None` outside staging, so the prod path below is exactly what it was.
    # Raises -- before any SDK load -- on routing in staging or simulation in prod.
    staging = StagingSessionConfig.from_env(source)
    if staging is not None:
        ctx = run_staging_session(
            open_configured_store(Settings.from_env(source)),
            config=staging,
            installed_version=version_reader(),
            clock=clock,
            run_mode=args.run_mode,
            environ=source,
        )
        mode = "simulated" if staging.simulated else "shadow"
        printer(
            f"trader.session {ctx.trading_day} {ctx.run_id}: staging {mode} session filed "
            "(no broker)"
        )
        return 0
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
