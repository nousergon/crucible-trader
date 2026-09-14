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

**The artifact** (`trader/fire_drills/{trading_day}/{run_id}.json`,
`fire_drill.v1`) carries the instant fired, the instant the book reached
flat/held, the declared bound, the book before and after, and every order the
broker accepted after the fire instant that the switch did not place. `passed`
is derived from those fields, never supplied. A drill that did not pass raises
:class:`FireDrillFailedError`, so its manifest is `failed` -- a page.

**Not done means not done.** :func:`evaluate_drills` reads the artifacts and
returns ``done`` only for >= 2 passed drills inside the window with >= 1 of them
unannounced. No artifacts, only failed drills, or only announced drills are all
not-done, each with its reason. The phase-4 gate clause that reads the same
artifacts belongs to `crucible.gate` (a contract change); this is the trader's
own reading of its own evidence.

Drills run only inside regular NYSE hours (`krepis.trading_calendar.is_market_hours`)
and only against a paper account; both refusals raise before any broker call.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections.abc import Callable, Iterable
from typing import Any

from crucible.keys import TRADER_FIRE_DRILLS_PREFIX
from crucible.keys import trader_fire_drill_key as fire_drill_key
from crucible.runner import RunContext
from crucible.serving import PredictionsFeed
from crucible.store import Store
from krepis.trading_calendar import is_market_hours

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

#: Plan §9.5 entry condition 4.
REQUIRED_PASSED = 2
REQUIRED_UNANNOUNCED = 1

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
    clock: Callable[[], dt.datetime] = utcnow,
    market_open: Callable[[dt.datetime], bool] = is_market_hours,
) -> dict[str, Any]:
    """The job body: refuse, fire, record, and raise unless the drill passed."""
    if kind not in DRILL_KINDS:
        raise DrillRefusedError(f"drill kind {kind!r} not in {DRILL_KINDS}")
    now = clock()
    if not market_open(now):
        raise DrillRefusedError(
            f"{now.isoformat()} is outside regular NYSE hours. A fire drill measures the "
            "switch against a live paper session; outside it, market orders queue and the "
            "drill measures the queue."
        )
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
        ctx, outcome, kind=kind, announced=announced, hold_decision=hold_decision
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
    }


@dataclasses.dataclass(frozen=True)
class DrillReading:
    done: bool
    n_passed: int
    n_unannounced_passed: int
    reason: str


_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "announced",
        "trading_day",
        "fired_at",
        "settled_at",
        "settle_seconds",
        "bound_seconds",
        "orders_accepted_after_fire",
        "passed",
    }
)


def _drill_passed(document: dict[str, Any]) -> bool:
    """Re-derived from the evidence fields, never trusted from `passed` alone."""
    settle = document["settle_seconds"]
    return (
        document["passed"] is True
        and document["settled_at"] is not None
        and isinstance(settle, (int, float))
        and settle <= document["bound_seconds"]
        and document["orders_accepted_after_fire"] == []
    )


def evaluate_drills(
    documents: Iterable[tuple[str, dict[str, Any]]], *, window_start: str, window_end: str
) -> DrillReading:
    """Whether the drill requirement is met inside ``[window_start, window_end]``."""
    in_window = []
    for key, document in documents:
        missing = _REQUIRED_FIELDS - set(document)
        if document.get("schema_version") != FIRE_DRILL_SCHEMA_VERSION or missing:
            raise ValueError(
                f"{key} is not a {FIRE_DRILL_SCHEMA_VERSION} artifact (missing {sorted(missing)})"
            )
        if window_start <= document["trading_day"] <= window_end:
            in_window.append(document)
    passed = [d for d in in_window if _drill_passed(d)]
    unannounced = [d for d in passed if d["announced"] is False]
    if not in_window:
        reason = f"no fire drill artifact between {window_start} and {window_end}: undrilled"
    else:
        reason = (
            f"{len(passed)}/{len(in_window)} drills passed ({REQUIRED_PASSED} required), "
            f"{len(unannounced)} of them unannounced ({REQUIRED_UNANNOUNCED} required)"
        )
    done = len(passed) >= REQUIRED_PASSED and len(unannounced) >= REQUIRED_UNANNOUNCED
    return DrillReading(done, len(passed), len(unannounced), reason)


def read_drills(store: Store) -> list[tuple[str, dict[str, Any]]]:
    """Every drill artifact on the store. An unreadable one raises, never skipped."""
    out = []
    for key in sorted(store.list_keys(FIRE_DRILL_PREFIX)):
        try:
            out.append((key, json.loads(store.get_bytes(key).decode("utf-8"))))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{key} is not readable JSON: {exc}") from exc
    return out
