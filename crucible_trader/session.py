"""The trader session: the PR7 guards composed around the book construction.

`alpha-engine-config-I10654` (remaining item 1). The one place the guards are
called, in the order the plan (§9.5) needs them, so no order path can reach the
broker by a route that skips one:

1. **At session start**, `paper_smoke.running_wheel_disagreement`: a trader
   running a wheel `trader/release_pin` does not name raises
   :class:`~crucible_trader.paper_smoke.RunningWheelMismatchError`, which fails
   the enclosing run's manifest -- the page (I10649 deliverable 5).
2. **On the prediction feed, before construction**,
   `hold_book.enforce_hold_book`: a degenerate batch engages the kill switch in
   `hold` mode for the day. An engaged hold is then refused immediately through
   `kill_switch.assert_trading_permitted`, so nothing is constructed on a
   forecast the safeguard has just called degenerate.
3. **Before the contract is resolved**, `v1_rules_guard.enforce_v1_rules`
   (`alpha-engine-config-I12024`): every v1 trading rule the private record
   still owes is named on the run. It never holds a session: from the
   record's ``enforce_from`` the debt fails CI instead.
4. **Between the plan and the broker**, `hard_risk.enforce_precedence`: hard
   risk > eligibility > optimizer > timing. The session's hard-risk record is
   filed before any order (`trader/hard_risk/{day}.json`); a plan that BUYs a
   name the layer forced out today is refused whole, and at the drawdown
   circuit breaker every BUY is dropped so the forced sells still go.
5. **Before every order batch**, `kill_switch.assert_trading_permitted` raises
   :class:`~crucible_trader.kill_switch.TradingHaltedError` when flatten or
   freeze is engaged or a hold covers the day. It is re-read per batch, so a
   switch fired mid-session stops the next batch.

The book is `construction.construct_target_book` -- the grade's engine on the
grade's inputs -- and its `portfolio_construction` metric record is recorded on
the run. It is handed the minimum hold's clock, each held name's consecutive
sessions in the trader's own order books
(`order_book.held_sessions_from_order_books`, `alpha-engine-config-I12024`),
and the day's order book is stamped with the clock it carries forward.
Turning the book into orders is the :class:`OrderRouter` handed in;
this module sizes and prices nothing.

`session_cycle` is a job body (like `paper_smoke.smoke_cycle`): the caller runs
it inside `crucible.runner.run_job`, so every raise here files a failed manifest.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol

from crucible.runner import RunContext
from crucible.store import Store

from crucible_trader.construction import TargetBook, construct_target_book
from crucible_trader.contract import ResolvedContract, resolve
from crucible_trader.hard_risk import (
    HARD_RISK_SCHEMA_VERSION,
    BrokerFacts,
    HardRiskInputs,
    carried_state,
    enforce_precedence,
    hard_risk_key,
)
from crucible_trader.hold_book import enforce_hold_book
from crucible_trader.kill_switch import BrokerControl, assert_trading_permitted, utcnow
from crucible_trader.order_book import held_sessions_from_order_books, record_order_book
from crucible_trader.paper_smoke import RunningWheelMismatchError, running_wheel_disagreement
from crucible_trader.v1_rules_guard import enforce_v1_rules

SESSION_SCHEMA_VERSION = "trader_session.v1"

__all__ = ["SESSION_SCHEMA_VERSION", "OrderRouter", "session_cycle"]


class OrderRouter(Protocol):
    """Turns a target book into order batches and sends one batch."""

    def batches(self, book: TargetBook) -> Iterable[Sequence[Any]]: ...
    def send(self, batch: Sequence[Any]) -> None: ...


def session_cycle(
    ctx: RunContext,
    *,
    broker: BrokerControl,
    router: OrderRouter,
    installed_version: str,
    previous_weights: Mapping[str, float] | None,
    clock: Callable[[], dt.datetime] = utcnow,
    feature_version: str | None = None,
    contract_resolver: Callable[[Store, str], ResolvedContract] = resolve,
    construct: Callable[..., TargetBook] = construct_target_book,
    rules_guard: Callable[[RunContext], object] = enforce_v1_rules,
    hard_risk_facts: BrokerFacts | None = None,
) -> dict[str, Any]:
    """Run one trading session's decision and orders. Raises on every refusal.

    ``hard_risk_facts`` (the broker's average costs, marks and NAV) switches the
    hard-risk layer on for the session; ``None`` (no broker: shadow, staging)
    leaves it off, and ``construct`` is then called exactly as before.
    """
    disagreement = running_wheel_disagreement(ctx.store, installed_version)
    if disagreement is not None:
        raise RunningWheelMismatchError(disagreement)
    rules_guard(ctx)  # reports only; never raises (no trading holds)

    day = ctx.trading_day.isoformat()
    contract = contract_resolver(ctx.store, day)
    decision, _ = enforce_hold_book(ctx, contract.feed, broker, clock=clock)
    if decision.hold:
        assert_trading_permitted(ctx.store, day)  # raises: the hold covers this day

    extra: dict[str, HardRiskInputs] = {}
    if hard_risk_facts is not None:
        extra["hard_risk"] = HardRiskInputs(
            facts=hard_risk_facts, carried=carried_state(ctx.store, day)
        )
    held_sessions = held_sessions_from_order_books(ctx.store, day)
    book = construct(
        ctx.store,
        trading_day=day,
        previous_weights=previous_weights,
        now=clock(),
        feature_version=feature_version,
        held_sessions=held_sessions,
        **extra,
    )
    ctx.record_metric(book.metric)
    cut = getattr(book, "hard_risk", None)
    if cut is not None:
        # Filed before any order: tomorrow's peak NAV and lockouts are read
        # from it, so a cut that reached the broker is never off the record.
        ctx.record_output(
            hard_risk_key(day),
            json.dumps(cut.to_document(), indent=2, sort_keys=True).encode("utf-8"),
            HARD_RISK_SCHEMA_VERSION,
        )

    def save_order_book() -> None:
        # The day's order book, saved before any order is sent so the record of
        # what was decided exists whatever the broker then does. Never raises.
        record_order_book(
            ctx,
            book,
            previous_weights=None if previous_weights is None else dict(previous_weights),
            now=clock(),
            held_sessions=held_sessions,
            share_basis=getattr(router, "share_basis", None),
            feature_version=feature_version,
        )

    # The router plans first -- reading the account and pricing each name, and
    # sending nothing -- so the book's share columns carry the shares the plan
    # sized at (`order_book.ShareBasis`). A refused plan still saves the book,
    # without shares, before the refusal fails the run. The hard-risk
    # chokepoint then holds the planned batches to the cut's precedence.
    try:
        batches = router.batches(book)
    except BaseException:
        save_order_book()
        raise
    save_order_book()

    sent = 0
    for batch in enforce_precedence(batches, cut):
        assert_trading_permitted(ctx.store, day)
        router.send(batch)
        sent += 1
    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "trading_day": day,
        "m_champion": contract.arm_id,
        # `alpha-engine-config-I12024`: the ruled M bench this session's alpha
        # was served under, or None. Carried into the served-day evidence by
        # `crucible_trader.daily_session`.
        "alpha_bench": None if book.alpha_bench is None else book.alpha_bench.model_dump(),
        "s_champion": book.s_champion,
        "hold_decision": decision.decision,
        "batches_sent": sent,
        # `alpha-engine-config-I13305`: what the book traded against the cap
        # it was constructed under, recorded on the run by the daily session.
        # Read leniently: a constructor double carries neither.
        "turnover_one_way": getattr(book, "turnover_one_way_ratio", None),
        "turnover_budget": getattr(book, "turnover_budget", None),
        "hard_risk": None
        if cut is None
        else {
            "forced": dict(cut.forced),
            "new_exits": dict(cut.new_exits),
            "buy_multiplier": cut.buy_multiplier,
            "circuit_breaker": cut.circuit_breaker,
        },
        # v1 parity row 16: why it moved that far, for the churn alarm.
        "turnover_drivers": getattr(book, "turnover_drivers", None),
    }
