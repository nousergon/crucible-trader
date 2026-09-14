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
3. **Before every order batch**, `kill_switch.assert_trading_permitted` raises
   :class:`~crucible_trader.kill_switch.TradingHaltedError` when flatten or
   freeze is engaged or a hold covers the day. It is re-read per batch, so a
   switch fired mid-session stops the next batch.

The book is `construction.construct_target_book` -- the grade's engine on the
grade's inputs -- and its `portfolio_construction` metric record is recorded on
the run. Turning the book into orders is the :class:`OrderRouter` handed in;
this module sizes and prices nothing.

`session_cycle` is a job body (like `paper_smoke.smoke_cycle`): the caller runs
it inside `crucible.runner.run_job`, so every raise here files a failed manifest.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol

from crucible.runner import RunContext
from crucible.store import Store

from crucible_trader.construction import TargetBook, construct_target_book
from crucible_trader.contract import ResolvedContract, resolve
from crucible_trader.hold_book import enforce_hold_book
from crucible_trader.kill_switch import BrokerControl, assert_trading_permitted, utcnow
from crucible_trader.paper_smoke import RunningWheelMismatchError, running_wheel_disagreement

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
) -> dict[str, Any]:
    """Run one trading session's decision and orders. Raises on every refusal."""
    disagreement = running_wheel_disagreement(ctx.store, installed_version)
    if disagreement is not None:
        raise RunningWheelMismatchError(disagreement)

    day = ctx.trading_day.isoformat()
    contract = contract_resolver(ctx.store, day)
    decision, _ = enforce_hold_book(ctx, contract.feed, broker, clock=clock)
    if decision.hold:
        assert_trading_permitted(ctx.store, day)  # raises: the hold covers this day

    book = construct(
        ctx.store,
        trading_day=day,
        previous_weights=previous_weights,
        now=clock(),
        feature_version=feature_version,
    )
    ctx.record_metric(book.metric)

    sent = 0
    for batch in router.batches(book):
        assert_trading_permitted(ctx.store, day)
        router.send(batch)
        sent += 1
    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "trading_day": day,
        "m_champion": contract.arm_id,
        "s_champion": book.s_champion,
        "hold_decision": decision.decision,
        "batches_sent": sent,
    }
