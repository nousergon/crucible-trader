"""The kill-switch and hold-book fire drill on paper: each drill leaves an artifact.

`alpha-engine-config-I10650` deliverables 2 and 5; plan §9.5 (kill-switch row,
and paper -> live entry condition 4: *"the kill-switch and hold-book fire drill
passed on paper twice, at least one of them unannounced"*).

**A drill is the real path, fired on purpose.** There is no drill mode in the
switch and no drill flag the trader session can see:

    kill_switch_flatten / kill_switch_freeze
        `crucible_trader.kill_switch.fire` -- the function the kill-switch command
        calls -- against the live paper book.
    hold_book
        `crucible_trader.hold_book.enforce_hold_book` -- the function the session
        calls on every feed -- over a PLANTED degenerate feed held in memory (never
        written to the store), which engages the same `hold` halt a real degenerate
        batch would.

Whether a drill was announced is recorded ONLY in its own artifact, written after
the fire: nothing in the trader's configuration marks the day (I10650 gotcha 2).
An UNANNOUNCED drill fires only under a sealed schedule
(`crucible_trader.drill_schedule`, `alpha-engine-config-I10761`): the drill opens
the seal before it fires, refuses if the seal does not open or it is not the
sealed instant, and its artifact reveals the instant and nonce so the harness can
verify the claim.

**The artifact** (`trader/fire_drills/{trading_day}/{run_id}.json`,
`fire_drill.v1`) carries the instant fired, the instant the book reached
flat/held, the declared bound, the book before and after, and every order the
broker accepted after the fire instant that the switch did not place. `passed`
is derived from those fields, never supplied. A drill that did not pass raises
:class:`FireDrillFailedError`, so its manifest is `failed` -- a page.

**Not done means not done, and the harness says which.** `fire-drill status`
calls `crucible.gate.kill_switch_fire_drill_reading` -- the phase-4 clause
itself -- rather than a second count here: the trader's own count read
`announced: false` at face value and would have said DONE where the gate says
UNMET (`alpha-engine-config-I10761`).

Drills run only inside regular NYSE hours (`krepis.trading_calendar.is_market_hours`)
and only against a paper account; both refusals raise before any broker call.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Iterable
from typing import Any

from crucible.keys import TRADER_FIRE_DRILLS_PREFIX
from crucible.keys import trader_fire_drill_key as fire_drill_key
from crucible.runner import RunContext
from crucible.serving import PredictionsFeed
from krepis.trading_calendar import is_market_hours

from crucible_trader.drill_schedule import SealOpening, open_seal
from crucible_trader.hold_book import enforce_hold_book
from crucible_trader.kill_switch import BrokerControl, KillSwitchOutcome, fire, utcnow

FIRE_DRILL_JOB = "trader.fire_drill"
FIRE_DRILL_SCHEMA_VERSION = "fire_drill.v1"
#: `crucible.keys` owns this prefix and the key function below
#: (`alpha-engine-config-I10649`/`-I10650`); kept as a local alias so nothing
#: here or in the test suite has to change its import.
FIRE_DRILL_PREFIX = TRADER_FIRE_DRILLS_PREFIX
METRIC_MODULE = "crucible_trader.fire_drill"

DRILL_KINDS: tuple[str, ...] = ("kill_switch_flatten", "kill_switch_freeze", "hold_book")
_KIND_MODE = {"kill_switch_flatten": "flatten", "kill_switch_freeze": "freeze"}

#: The planted feed's champion id. Never written to the store; it names itself
#: so a hold cause read back from the halt document is traceable to the drill.
PLANTED_CHAMPION = "fire_drill:planted_degenerate_batch"


class DrillRefusedError(RuntimeError):
    """The drill was not fired: outside market hours, or not a known kind."""


class FireDrillFailedError(RuntimeError):
    """The drill fired and the switch did not pass. A page, by manifest status."""


def planted_degenerate_feed(trading_day: str, symbols: Iterable[str]) -> PredictionsFeed:
    """Every name the same alpha: the batch `should_hold_book` must hold on.

    At least `hold_book.HOLD_BOOK_MIN_BATCH` names so the decision exercised is
    the degeneracy test (`hold_signal_degenerate`), not the small-batch one.
    """
    names = sorted(set(symbols))
    names += [f"DRILL{i}" for i in range(max(0, 5 - len(names)))]
    return PredictionsFeed(
        slot="m",
        trading_day=trading_day,
        champion=PLANTED_CHAMPION,
        feature_version="fire_drill",
        source_key="fire_drill:in_memory",
        predicted_alpha={name: 0.0 for name in names},
    )


def drill_cycle(
    ctx: RunContext,
    *,
    broker: BrokerControl,
    kind: str,
    announced: bool,
    seal: SealOpening | None = None,
    clock: Callable[[], dt.datetime] = utcnow,
    market_open: Callable[[dt.datetime], bool] = is_market_hours,
) -> dict[str, Any]:
    """The job body: refuse, open the seal, fire, record, and raise unless the drill passed."""
    if kind not in DRILL_KINDS:
        raise DrillRefusedError(f"drill kind {kind!r} not in {DRILL_KINDS}")
    if announced and seal is not None:
        raise DrillRefusedError(
            "an announced drill was handed a seal; a seal exists only to make an unannounced "
            "claim checkable"
        )
    if not announced and seal is None:
        raise DrillRefusedError(
            "an unannounced drill fires only under a sealed schedule (`fire-drill seal`); an "
            "unsealed `announced: false` is a claim the gate will not count"
        )
    now = clock()
    if not market_open(now):
        raise DrillRefusedError(
            f"{now.isoformat()} is outside regular NYSE hours. A fire drill measures the "
            "switch against a live paper session; outside it, market orders queue and the "
            "drill measures the queue."
        )
    reveal = None if seal is None else open_seal(ctx.store, seal, now=now)
    day = ctx.trading_day.isoformat()
    hold_decision: str | None = None
    if kind == "hold_book":
        feed = planted_degenerate_feed(day, broker.statement(day).positions)
        decision, outcome = enforce_hold_book(ctx, feed, broker, clock=clock)
        hold_decision = decision.decision
        if outcome is None:
            raise FireDrillFailedError(
                f"the hold-book gate did not hold on a planted degenerate batch "
                f"({decision.decision}); the safeguard is not working"
            )
    else:
        outcome = fire(ctx, broker, mode=_KIND_MODE[kind], cause=f"fire_drill:{kind}", clock=clock)

    document = drill_document(
        ctx, outcome, kind=kind, announced=announced, hold_decision=hold_decision, schedule=reveal
    )
    key = fire_drill_key(day, ctx.run_id)
    ctx.record_output(
        key,
        json.dumps(document, indent=2, sort_keys=True).encode("utf-8"),
        FIRE_DRILL_SCHEMA_VERSION,
    )
    ctx.record_metric(
        {
            "name": "fire_drill_passed",
            "module": METRIC_MODULE,
            "metric_type": "operational",
            "value": 1.0 if outcome.passed else 0.0,
            "unit": "flag",
            "n_floor": 1,
            "status": "OK" if outcome.passed else "FAIL",
            "status_reason": (
                f"{kind} ({'announced' if announced else 'unannounced'}) reached its target in "
                f"{outcome.settle_seconds}s, zero orders accepted after the fire"
                if outcome.passed
                else f"{kind}: {outcome.reason()}"
            ),
            "source_path": key,
            "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    if not outcome.passed:
        raise FireDrillFailedError(f"{kind}: {outcome.reason()}")
    return document


def drill_document(
    ctx: RunContext,
    outcome: KillSwitchOutcome,
    *,
    kind: str,
    announced: bool,
    hold_decision: str | None,
    schedule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = outcome.to_document()
    return {
        "schema_version": FIRE_DRILL_SCHEMA_VERSION,
        "kind": kind,
        "announced": announced,
        "run_id": ctx.run_id,
        "trading_day": outcome.trading_day,
        "account": outcome.account,
        "mode": outcome.mode,
        "fired_at": outcome.fired_at,
        "settled_at": outcome.settled_at,
        "settle_seconds": outcome.settle_seconds,
        "bound_seconds": outcome.bound_seconds,
        "state_before": body["state_before"],
        "state_after": body["state_after"],
        "orders_accepted_after_fire": body["orders_accepted_after_fire"],
        "hold_decision": hold_decision,
        "passed": outcome.passed,
        "schedule": schedule,
    }
