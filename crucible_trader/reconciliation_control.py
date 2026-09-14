"""The reconciliation control arm: a planted discrepancy per class, every cycle.

`alpha-engine-config-I10413`, plan §9.5 (amended 2026-09-10) and fault 5 of
§10.7. A reconciler that has only ever been shown a clean book cannot be told
apart from one that cannot see a dirty one -- v1's split check and ledger replay
both passed their own tests while producing a false 1.000. So every cycle this
module REPLAYS the cycle's own reconciliation inputs three times, each with one
known discrepancy planted, and requires the reconciler to report exactly that
discrepancy, correctly classified:

    share_delta        a known share count on a planted instrument
    cash_delta         a known cash amount on the broker side
    corporate_action   a known split recorded for a planted instrument and NOT
                       applied on the broker side

**An undetected or misclassified plant voids the cycle's reconciliation**, the
same way a failed grader control voids that cycle's verdicts (§10.1):
:class:`ReconciliationVoidError` is raised inside the job body, so the run's
manifest is `status: failed` with a reason naming the missed class -- never a
lower score and never a warning.

**The plant can never reach a broker or a page.** It is injected into an
in-memory copy of :class:`~crucible_trader.reconciliation.ReconciliationInputs`
and handed to the pure reconciler. Nothing here reads a store, a broker or a
clock, and the planted instruments are named so they cannot collide with a real
symbol (:data:`PLANT_PREFIX`).

**What "detected" means is exact, not approximate.** Each replay's findings are
diffed against an unplanted replay of the same inputs: the plant must add
exactly one finding, with the planted class, instrument and delta, and remove
nothing but (for the cash plant) the baseline's own cash finding it displaced.
A reconciler that reports the plant AND silently drops a real finding fails.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.reconciliation import (
    CASH_INSTRUMENT,
    CorporateAction,
    CorporateActionSet,
    Discrepancy,
    ReconciliationInputs,
    ReconciliationResult,
    reconcile,
)

#: Fault 5's registered name in the §10.7 fault set.
FAULT_NAME = "reconciliation_planted_discrepancy"

PLANT_CLASSES: tuple[str, ...] = ("share_delta", "cash_delta", "corporate_action")

#: Not a valid exchange symbol (lowercase, underscores), so never a real holding.
PLANT_PREFIX = "__planted_"
PLANT_BASE_SHARES = 100
PLANTED_SHARE_DELTA = 37
PLANTED_CASH_DELTA_USD = 1234.56
PLANTED_SPLIT_RATIO = 3.0

#: Used as the replay anchor's day when the cycle itself has no anchor (a
#: declared genesis): the control still runs, over the broker's own book.
_REPLAY_ANCHOR_DAY = "0001-01-01"

Reconciler = Callable[[ReconciliationInputs], ReconciliationResult]


class ReconciliationVoidError(RuntimeError):
    """A planted discrepancy was not detected and classified. The cycle's
    reconciliation is void."""


@dataclasses.dataclass(frozen=True)
class Plant:
    klass: str
    instrument: str
    delta: float


@dataclasses.dataclass(frozen=True)
class PlantOutcome:
    plant: Plant
    detected: bool
    explanation: str

    def to_document(self) -> dict[str, Any]:
        return {
            **dataclasses.asdict(self.plant),
            "detected": self.detected,
            "explanation": self.explanation,
        }


@dataclasses.dataclass(frozen=True)
class ControlVerdict:
    outcomes: tuple[PlantOutcome, ...]

    @property
    def missed(self) -> tuple[str, ...]:
        return tuple(o.plant.klass for o in self.outcomes if not o.detected)

    @property
    def passed(self) -> bool:
        classes = tuple(o.plant.klass for o in self.outcomes)
        return classes == PLANT_CLASSES and not self.missed

    @property
    def n_detected(self) -> int:
        return sum(1 for o in self.outcomes if o.detected)

    def reason(self) -> str:
        return (
            f"reconciliation control arm ({FAULT_NAME}) missed planted "
            f"{', '.join(self.missed)}: "
            + "; ".join(o.explanation for o in self.outcomes if not o.detected)
            + ". This cycle's reconciliation is VOID -- a reconciler that cannot see a "
            "planted discrepancy cannot be trusted to have seen a real one."
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "fault": FAULT_NAME,
            "passed": self.passed,
            "n_planted": len(self.outcomes),
            "n_detected": self.n_detected,
            "missed": list(self.missed),
            "outcomes": [o.to_document() for o in self.outcomes],
        }


def replay_base(inputs: ReconciliationInputs) -> ReconciliationInputs:
    """The cycle's inputs as a replay the plants can be injected into.

    Identical to ``inputs`` when the cycle has an anchor. Without one (declared
    genesis, or the `no_anchor` finding) the replay anchors on the broker's own
    statement, so the control is exercised on day one too.
    """
    if inputs.anchor is not None:
        return inputs
    broker = inputs.broker
    return dataclasses.replace(
        inputs,
        anchor=BrokerStatement(
            account=broker.account,
            trading_day=_REPLAY_ANCHOR_DAY,
            positions=dict(broker.positions),
            cash_usd=broker.cash_usd,
        ),
        fills=(),
        cash_flows=(),
        corporate_actions=CorporateActionSet(resolved=frozenset(broker.positions)),
        previous_trading_day=None,
        genesis=False,
    )


def plant(
    base: ReconciliationInputs, klass: str, baseline: ReconciliationResult
) -> tuple[ReconciliationInputs, Plant]:
    """``base`` with one discrepancy of ``klass`` planted, and what was planted."""
    base = replay_base(base)  # idempotent; guarantees an anchor to plant on
    assert base.anchor is not None
    anchor, broker, actions = base.anchor, base.broker, base.corporate_actions
    ticker = f"{PLANT_PREFIX}{klass}"
    if klass == "cash_delta":
        planted_broker = dataclasses.replace(
            broker, cash_usd=broker.cash_usd + PLANTED_CASH_DELTA_USD
        )
        residual = baseline.cash_residual_usd or 0.0
        return (
            dataclasses.replace(base, broker=planted_broker),
            Plant(klass, CASH_INSTRUMENT, round(residual + PLANTED_CASH_DELTA_USD, 2)),
        )
    if klass not in ("share_delta", "corporate_action"):
        raise ValueError(f"no plant for class {klass!r}; the classes are {PLANT_CLASSES}")
    broker_shares = PLANT_BASE_SHARES + (PLANTED_SHARE_DELTA if klass == "share_delta" else 0)
    new_actions = actions.actions
    delta = float(PLANTED_SHARE_DELTA)
    if klass == "corporate_action":
        new_actions = (*new_actions, CorporateAction(ticker, "split", ratio=PLANTED_SPLIT_RATIO))
        delta = float(PLANT_BASE_SHARES - PLANT_BASE_SHARES * PLANTED_SPLIT_RATIO)
    return (
        dataclasses.replace(
            base,
            anchor=dataclasses.replace(
                anchor, positions={**anchor.positions, ticker: PLANT_BASE_SHARES}
            ),
            broker=dataclasses.replace(
                broker, positions={**broker.positions, ticker: broker_shares}
            ),
            corporate_actions=CorporateActionSet(
                resolved=actions.resolved | {ticker}, actions=new_actions
            ),
        ),
        Plant(klass, ticker, delta),
    )


def run_control_arm(
    inputs: ReconciliationInputs, *, reconciler: Reconciler = reconcile
) -> ControlVerdict:
    """Replay ``inputs`` once per planted class through ``reconciler``."""
    base = replay_base(inputs)
    baseline = reconciler(base)
    before = {_key(d) for d in baseline.discrepancies}
    outcomes = []
    for klass in PLANT_CLASSES:
        planted_inputs, planted = plant(base, klass, baseline)
        after = {_key(d) for d in reconciler(planted_inputs).discrepancies}
        added, removed = after - before, before - after
        if klass == "cash_delta":
            removed = {k for k in removed if not (k[0] == "cash_delta" and k[1] == CASH_INSTRUMENT)}
        hit = [
            k
            for k in added
            if k[0] == klass
            and k[1] == planted.instrument
            and k[4] is not None
            and math.isclose(k[4], planted.delta, abs_tol=0.005)
        ]
        if len(added) == 1 and hit and not removed:
            outcomes.append(
                PlantOutcome(planted, True, f"{klass} detected on {planted.instrument}")
            )
            continue
        outcomes.append(
            PlantOutcome(
                planted,
                False,
                f"planted {klass} on {planted.instrument} (delta {planted.delta:g}) produced "
                f"added={sorted(added)} removed={sorted(removed)}",
            )
        )
    return ControlVerdict(tuple(outcomes))


def _key(d: Discrepancy) -> tuple[Any, ...]:
    return (d.klass, d.instrument, d.expected, d.actual, d.delta, d.side)
