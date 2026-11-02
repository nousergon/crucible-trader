"""`exception_proposal.v1`: what the desk WOULD do about one break. Shadow only.

`alpha-engine-config-I12892` phase 1b (the back-office thin slice). Phase 0
made every reconciliation break a record (:mod:`crucible_trader.exception_record`).
This module adds the first half of the earned-autonomy loop: for each record, a
deterministic rule proposes what the break IS and what should happen next. A
reviewer's decision, and the score of proposal against decision, are later
steps and do not appear here.

A proposal names:

* ``nature`` -- one of :data:`NATURES`:

  - ``event``: the books are right and the world moved (a split the broker has
    not applied yet, a partial fill the broker closed);
  - ``defect``: something in our own pipeline or at the broker is wrong and has
    to be fixed at its cause;
  - ``detector``: the check itself could not be trusted that day (a void
    control arm, a missing anchor, an incomplete corporate-action feed);

* ``action`` -- one of :data:`ACTIONS`, closed;
* ``promotable`` -- whether this class of break may EVER earn auto-apply. Only
  events can. A defect is fixed at its cause and is never promoted, however
  often the proposal is right; a detector finding means the run cannot be acted
  on at all. A record from a void run is never promotable either: its detector
  missed a planted break that day;
* ``rule_id`` and ``ruleset_version`` -- which rule fired, so a later score can
  attribute agreement to a rule and a rule change starts a fresh record;
* ``proposal_id`` -- derived from the exception id and the ruleset version, so
  re-running the same rules over the same break gives the same id.

Nothing here writes, places, cancels or books anything. The proposals are built
FROM stored `exception.v1` day files, and the CLI only reads the store (wrapped
in `crucible.store.read_only`) and prints JSON lines:

    python -m crucible_trader.exception_proposal --store URI --since YYYY-MM-DD \\
        [--until YYYY-MM-DD] [--out PATH]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import pathlib
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from crucible.store import Store, open_store, read_only  # pyright: ignore[reportMissingTypeStubs]  # crucible ships no stubs

from crucible_trader.exception_record import (
    CLASSES_BY_SOURCE,
    CONTROL_ARM_VOID,
    EXCEPTIONS_PREFIX,
    SOURCE_BROKER_RECONCILIATION,
    SOURCE_ORDER_INTENT_RECONCILIATION,
    ExceptionRecordError,
    validate_exception_day_document,
)

__all__ = [
    "ACTIONS",
    "NATURES",
    "PROPOSAL_SCHEMA_VERSION",
    "PROPOSER",
    "RULESET_VERSION",
    "Proposal",
    "main",
    "propose",
    "propose_day",
    "proposals_from_store",
]

PROPOSAL_SCHEMA_VERSION = "exception_proposal.v1"

#: Bumped whenever any rule below changes what it proposes. A new version is a
#: new proposal id, so agreement earned under old rules never carries over.
RULESET_VERSION = "deterministic.2026-11-02b"

PROPOSER = "crucible-trader.exception_proposal"

NATURES: tuple[str, ...] = ("event", "defect", "detector")

ACTIONS: tuple[str, ...] = (
    "await_broker_corporate_action",
    "book_corporate_action_manually",
    "accept_partial_fill",
    "investigate_cash",
    "investigate_shares",
    "investigate_position_count",
    "investigate_order",
    "rerun_after_feed",
    "refresh_anchor",
    "investigate_detector",
)


@dataclasses.dataclass(frozen=True)
class _Rule:
    rule_id: str
    nature: str
    action: str
    rationale: str


def _fixed(
    rule_id: str, nature: str, action: str, rationale: str
) -> Callable[[Mapping[str, object]], _Rule]:
    rule = _Rule(rule_id, nature, action, rationale)
    return lambda _record: rule


_B = SOURCE_BROKER_RECONCILIATION
_I = SOURCE_ORDER_INTENT_RECONCILIATION

#: One rule per (source, class). Exhaustive over `CLASSES_BY_SOURCE`, asserted
#: at import: a detector that gains a class without a rule fails loudly here.
_RULES: Mapping[tuple[str, str], Callable[[Mapping[str, object]], _Rule]] = {
    # Every cash break that reaches a record is above the detector's own
    # rounding band (`reconciliation.CASH_TOLERANCE_USD`, 5 cents, recorded as
    # `cash_rounded_to_broker_usd`; Brian 2026-10-28, crucible-trader#66). So
    # none is rounding: each is a cash entry to find, never one to absorb.
    (_B, "cash_delta"): _fixed(
        "cash.unexplained",
        "defect",
        "investigate_cash",
        "cash differs by more than the 5-cent rounding band; find the missing or wrong cash entry",
    ),
    (_B, "share_delta"): _fixed(
        "shares.delta",
        "defect",
        "investigate_shares",
        "broker shares differ from the book with no corporate action to explain it",
    ),
    (_B, "position_count"): _fixed(
        "positions.count",
        "defect",
        "investigate_position_count",
        "the broker holds a different number of positions than the book",
    ),
    (_B, "corporate_action"): _fixed(
        "corporate_action.pending",
        "event",
        "await_broker_corporate_action",
        "a known split or dividend is not yet reflected at the broker; recheck next day",
    ),
    (_B, "unhandled_corporate_action"): _fixed(
        "corporate_action.unhandled",
        "event",
        "book_corporate_action_manually",
        "a resolved corporate action of a kind the reconciler cannot apply; book it by hand",
    ),
    (_B, "corporate_action_check_incomplete"): _fixed(
        "detector.feed_incomplete",
        "detector",
        "rerun_after_feed",
        "the corporate-action feed was incomplete; rerun once it is",
    ),
    (_B, "no_anchor"): _fixed(
        "detector.no_anchor",
        "detector",
        "refresh_anchor",
        "no position anchor to reconcile from; refresh it",
    ),
    (_B, "stale_anchor"): _fixed(
        "detector.stale_anchor",
        "detector",
        "refresh_anchor",
        "the position anchor is stale; refresh it",
    ),
    (_B, CONTROL_ARM_VOID): _fixed(
        "detector.void",
        "detector",
        "investigate_detector",
        "the control arm missed a planted break; nothing from this run can be acted on",
    ),
    (_I, "closed_partial"): _fixed(
        "order.closed_partial",
        "event",
        "accept_partial_fill",
        "the broker closed the order with part of the shares; book what filled",
    ),
    **{
        (_I, klass): _fixed(
            f"order.{klass}",
            "defect",
            "investigate_order",
            f"order intent and broker disagree ({klass}); find which decision reached the broker",
        )
        for klass in (
            "unknown",
            "ref_mismatch",
            "effect_mismatch",
            "corrupt_execution",
            "overfilled",
            "orphan",
        )
    },
}


def _check_exhaustive(rules: Mapping[tuple[str, str], object]) -> None:
    """Raise unless ``rules`` has exactly one entry per class each detector can
    raise. Run at import, so a detector that gains a class without a rule fails
    loudly instead of the class going unproposed."""
    expected = {(s, c) for s, classes in CLASSES_BY_SOURCE.items() for c in classes}
    if set(rules) != expected:
        raise RuntimeError(
            "exception_proposal rules out of step with exception_record: missing "
            f"{sorted(expected - set(rules))}, extra {sorted(set(rules) - expected)}"
        )


_check_exhaustive(_RULES)


@dataclasses.dataclass(frozen=True)
class Proposal:
    exception_id: str
    trading_day: str
    source: str
    klass: str
    nature: str
    action: str
    rationale: str
    rule_id: str
    promotable: bool
    held: bool
    proposed_at: str

    @property
    def proposal_id(self) -> str:
        payload = json.dumps([self.exception_id, RULESET_VERSION], separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "exception_id": self.exception_id,
            "trading_day": self.trading_day,
            "source": self.source,
            "class": self.klass,
            "nature": self.nature,
            "action": self.action,
            "rationale": self.rationale,
            "rule_id": self.rule_id,
            "ruleset_version": RULESET_VERSION,
            "promotable": self.promotable,
            "held": self.held,
            "proposer": PROPOSER,
            "proposed_at": self.proposed_at,
        }


def propose(record: Mapping[str, object], *, proposed_at: str) -> Proposal:
    """The proposal for one validated `exception.v1` record."""
    source = cast("str", record["source"])
    klass = cast("str", record["class"])
    rule = _RULES[(source, klass)](record)
    detected_by = cast("Mapping[str, object]", record["detected_by"])
    held = detected_by["void"] is True and klass != CONTROL_ARM_VOID
    rationale = rule.rationale
    if held:
        rationale += "; HELD: the run's control arm was void"
    return Proposal(
        exception_id=cast("str", record["exception_id"]),
        trading_day=cast("str", record["trading_day"]),
        source=source,
        klass=klass,
        nature=rule.nature,
        action=rule.action,
        rationale=rationale,
        rule_id=rule.rule_id,
        promotable=rule.nature == "event" and detected_by["void"] is not True,
        held=held,
        proposed_at=proposed_at,
    )


def propose_day(document: Mapping[str, object], *, proposed_at: str) -> list[Proposal]:
    """Proposals for every record in one `exception.v1` day file, in its order.
    The file is validated first; an invalid one raises."""
    validate_exception_day_document(document)
    records = cast("list[Mapping[str, object]]", document["records"])
    return [propose(r, proposed_at=proposed_at) for r in records]


def proposals_from_store(
    store: Store, *, since: str, until: str | None, proposed_at: str
) -> list[dict[str, object]]:
    """Proposal documents for every stored day file in ``[since, until]``."""
    out: list[dict[str, object]] = []
    for key in sorted(store.list_keys(EXCEPTIONS_PREFIX)):
        if not key.endswith(".json"):
            continue
        day = key.removeprefix(EXCEPTIONS_PREFIX).removesuffix(".json")
        if day < since or (until is not None and day > until):
            continue
        raw: object = json.loads(store.get_bytes(key).decode("utf-8"))
        if not isinstance(raw, dict):
            raise ExceptionRecordError(f"{key} is not a JSON object")
        try:
            proposals = propose_day(cast("dict[str, object]", raw), proposed_at=proposed_at)
        except ExceptionRecordError as exc:
            raise ExceptionRecordError(f"{key}: {exc}") from exc
        out.extend(p.to_document() for p in proposals)
    return out


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m crucible_trader.exception_proposal",
        description="Shadow proposals for stored exception.v1 records. Read-only.",
    )
    parser.add_argument("--store", default=None, help="store URI (default: $CRUCIBLE_STORE)")
    parser.add_argument("--since", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--until", default=None, type=dt.date.fromisoformat)
    parser.add_argument(
        "--out", default=None, type=pathlib.Path, help="local file (default: stdout)"
    )
    args = parser.parse_args(argv)
    store = read_only(
        open_store(cast("str | None", args.store)),
        reason="exception_proposal is shadow-only and read-only",
    )
    until = cast("dt.date | None", args.until)
    documents = proposals_from_store(
        store,
        since=cast("dt.date", args.since).isoformat(),
        until=until.isoformat() if until is not None else None,
        proposed_at=_now(),
    )
    lines = "".join(json.dumps(d, sort_keys=True) + "\n" for d in documents)
    out = cast("pathlib.Path | None", args.out)
    if out is None:
        sys.stdout.write(lines)
    else:
        out.write_text(lines, encoding="utf-8")
    sys.stderr.write(f"{len(documents)} shadow proposals\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
