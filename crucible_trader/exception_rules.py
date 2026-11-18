"""Tier-2 rules: a break the records explain is cleared without a person.

`alpha-engine-config-I13747`. Brian ruled on 2026-11-16 that a reconciliation
rule reaches tier 2 by **approve once, then audit** (nous-ergon-ops back-office
plan, "Where a person reviews a break"), and on 2026-11-17 at 15:21Z approved
the first four (option A on `alpha-engine-config-I13791`). This module is the
registry of those rules and the code that applies them.

A rule clears a queue item only when ALL of these hold:

* the rule's registry ``state`` is ``approved`` -- a ``proposed`` rule clears
  nothing, and adding or changing a rule is a PR naming its approval;
* no person has rejected one of its clears (one disagreement demotes it,
  :func:`~crucible_trader.exception_decision.demoted_rules`);
* nobody has decided the item yet, and no rule has cleared it;
* its condition holds on the stored records, which the clear then cites by key
  with the arithmetic, so anyone can redo the sum.

A clear books nothing and moves no money: it is an `exception_decision.v1`
record with verdict ``cleared`` and reviewer ``rule:<id>``
(:func:`~crucible_trader.exception_decision.rule_clear`), filed by the daily
reconcile job (step 5c'', :func:`file_rule_clears`) under
:data:`~crucible_trader.exception_decision.RULE_CLEARS_PREFIX`. The people-only
decision writer still writes every person's call, and only those. A tenth of
each rule's clears come back to a person as ``audit`` items
(:func:`~crucible_trader.exception_decision.audit_sample`).

Defects are never cleared: each rule's condition is a fact about the world
(interest IB posted, a dividend the broker paid late or Brian waived, a check
that a fresher reconcile superseded, a split the broker has not yet applied).
A break no rule explains stays in the queue as it was.

    python -m crucible_trader.exception_rules --store URI [--as-of YYYY-MM-DD]

prints, read-only, what the rules would clear and why each other open item was
not cleared (the dry run).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from crucible.calendar import is_trading_day
from crucible.keys import trader_broker_statement_key, trader_reconciliation_key
from crucible.store import Store, open_store, read_only

from crucible_trader.dividend_entitlements import load_declared_waivers
from crucible_trader.exception_decision import (
    RULE_CLEARS_SCHEMA_VERSION,
    DecisionError,
    demoted_rules,
    load_decisions,
    load_exception_records,
    load_proposals,
    load_rule_clears,
    rule_clear,
    rule_clears_key,
)
from crucible_trader.exception_record import SOURCE_BROKER_RECONCILIATION
from crucible_trader.reconciliation import DividendWaiver

__all__ = [
    "R4_LIMIT_TRADING_DAYS",
    "RULES",
    "TOLERANCE_CENTS",
    "WAIVERS_FILE",
    "Match",
    "Rule",
    "StoreRecords",
    "clears_for",
    "evaluate",
    "file_rule_clears",
    "main",
    "registry",
]

_LOG = logging.getLogger(__name__)

#: The band every cash condition is met within (5 cents, the reconciler's own
#: rounding band; aec#13791: "within 5¢").
TOLERANCE_CENTS = 5

#: R4 clears a pending split for at most this many trading days after it is
#: first seen; from the next day the item returns to a person (aec#13791).
R4_LIMIT_TRADING_DAYS = 5

#: Where R2's waivers are declared, cited as the waiver's evidence.
WAIVERS_FILE = "crucible_trader/declared_dividend_waivers.json"

#: The reconciler's wording for a split the broker has not applied
#: (`reconciliation.reconcile`, the same-day split branch).
_SPLIT_PENDING = "broker shares equal the pre-action holding"

_DAY_KEY = re.compile(r"^trader/reconciliation/(\d{4}-\d{2}-\d{2})\.json$")

_APPROVAL = {
    "state": "approved",
    "approved_by": "cipher813",
    "approved_at": "2026-11-17T15:21:31Z",
    "approval_ref": "alpha-engine-config#13791",
}


@dataclasses.dataclass(frozen=True)
class Rule:
    """One tier-2 rule as it was approved. ``condition`` is the approved text."""

    rule_id: str
    title: str
    source: str
    klass: str
    condition: str
    state: str
    approved_by: str | None
    approved_at: str | None
    approval_ref: str | None

    def to_document(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "source": self.source,
            "class": self.klass,
            "condition": self.condition,
            "state": self.state,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "approval_ref": self.approval_ref,
        }


#: The registry. A new rule starts ``proposed`` with no approval and clears
#: nothing until a PR records Brian's approval here; ``demoted`` is set here
#: when a demotion is made permanent.
RULES: tuple[Rule, ...] = (
    Rule(
        "R1",
        "interest IB posted",
        SOURCE_BROKER_RECONCILIATION,
        "cash_delta",
        "The cash gap equals the fall in IB's accrued interest plus the new accrual that "
        "day, all from IB's own statements, within 5 cents.",
        **_APPROVAL,
    ),
    Rule(
        "R2",
        "dividend paid late or waived",
        SOURCE_BROKER_RECONCILIATION,
        "corporate_action",
        "A declared dividend was paid later than its pay date, or Brian waived it in "
        "declared_dividend_waivers.json.",
        **_APPROVAL,
    ),
    Rule(
        "R3",
        "stale anchor superseded",
        SOURCE_BROKER_RECONCILIATION,
        "stale_anchor",
        "The break compared against an older broker statement, and a later reconcile on "
        "fresh statements was clean.",
        **_APPROVAL,
    ),
    Rule(
        "R4",
        "corporate action pending at the broker",
        SOURCE_BROKER_RECONCILIATION,
        "corporate_action",
        "IB has not yet processed an announced split. Rechecked daily; the item returns "
        "after 5 trading days.",
        **_APPROVAL,
    ),
)


def registry() -> list[dict[str, object]]:
    """The rules as documents, for the score and the console."""
    return [rule.to_document() for rule in RULES]


class Records(Protocol):
    """What the rules read: the stored reconciliations and broker statements."""

    def reconciliation(self, day: str) -> Mapping[str, object] | None: ...
    def statement(self, day: str) -> Mapping[str, object] | None: ...
    def reconciliation_days(self) -> list[str]: ...


class StoreRecords:
    """:class:`Records` over a store; each document is read once."""

    def __init__(self, store: Store) -> None:
        self._store = store
        self._cache: dict[str, Mapping[str, object] | None] = {}
        self._days: list[str] | None = None

    def _get(self, key: str) -> Mapping[str, object] | None:
        if key not in self._cache:
            document: object = (
                json.loads(self._store.get_bytes(key).decode("utf-8"))
                if self._store.exists(key)
                else None
            )
            self._cache[key] = (
                cast("Mapping[str, object]", document) if isinstance(document, dict) else None
            )
        return self._cache[key]

    def reconciliation(self, day: str) -> Mapping[str, object] | None:
        return self._get(trader_reconciliation_key(day))

    def statement(self, day: str) -> Mapping[str, object] | None:
        return self._get(trader_broker_statement_key(day))

    def reconciliation_days(self) -> list[str]:
        if self._days is None:
            found = (_DAY_KEY.match(k) for k in self._store.list_keys("trader/reconciliation/"))
            self._days = sorted(m.group(1) for m in found if m is not None)
        return self._days


@dataclasses.dataclass(frozen=True)
class Match:
    """A rule's condition met: what it cites, its sum, and the sentence."""

    evidence: tuple[tuple[str, str], ...]
    arithmetic: Mapping[str, object]
    note: str


def _cents(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return round(value * 100)


def _usd(cents: int) -> float:
    return round(cents / 100, 2)


def _shift(day: str, step: int) -> str:
    current = dt.date.fromisoformat(day) + dt.timedelta(days=step)
    while not is_trading_day(current):
        current += dt.timedelta(days=step)
    return current.isoformat()


def _discrepancies(document: Mapping[str, object]) -> list[Mapping[str, object]]:
    found = document.get("discrepancies")
    if not isinstance(found, list):
        return []
    return [
        cast("Mapping[str, object]", d) for d in cast("list[object]", found) if isinstance(d, dict)
    ]


def _rows(document: Mapping[str, object], field: str) -> list[Mapping[str, object]]:
    found = document.get(field)
    if not isinstance(found, list):
        return []
    return [
        cast("Mapping[str, object]", r) for r in cast("list[object]", found) if isinstance(r, dict)
    ]


def _r1(record: Mapping[str, object], records: Records, *, as_of: str) -> tuple[Match | None, str]:
    day = str(record["trading_day"])
    found = records.reconciliation(day)
    if found is None:
        return None, f"no reconciliation for {day}"
    if found.get("void") is not False or any(
        d.get("klass") == "stale_anchor" for d in _discrepancies(found)
    ):
        return None, f"the {day} reconcile was void or compared against a stale anchor"
    gap = _cents(record.get("amount"))
    if record.get("instrument") != "USD" or gap is None or gap <= 0:
        return None, "not a cash gap in the account's favour"
    before, after = _shift(day, -1), _shift(day, 1)
    if after > as_of:
        return None, f"the next statement ({after}) is not filed yet"
    statements = {d: records.statement(d) for d in (before, day, after)}
    accrued = {
        d: _cents(s.get("accrued_cash_usd")) if s is not None else None
        for d, s in statements.items()
    }
    if any(a is None for a in accrued.values()):
        return None, f"a statement of {before}, {day} or {after} is missing or has no accrual"
    a_before, a_day, a_after = (cast("int", accrued[d]) for d in (before, day, after))
    fall, rise = a_before - a_day, a_after - a_day
    if fall <= 0:
        return None, "IB's accrual did not fall, so no interest was posted"
    if rise < 0:
        return None, "IB's accrual fell again the next day, so the day's accrual is unknown"
    days_before = (dt.date.fromisoformat(day) - dt.date.fromisoformat(before)).days
    days_after = (dt.date.fromisoformat(after) - dt.date.fromisoformat(day)).days
    # The day's own accrual at the next day's daily rate, rounded half up.
    accrued_that_day = (2 * rise * days_before + days_after) // (2 * days_after)
    explained = fall + accrued_that_day
    off = gap - explained
    if abs(off) > TOLERANCE_CENTS:
        return None, (
            f"interest explains {_usd(explained)} of the {_usd(gap)} gap, "
            f"{_usd(off)} off (more than 5 cents)"
        )
    keys = tuple(("statement", trader_broker_statement_key(d)) for d in (before, day, after))
    return (
        Match(
            evidence=(("reconciliation", trader_reconciliation_key(day)), *keys),
            arithmetic={
                "accrued_before_usd": _usd(a_before),
                "accrued_on_day_usd": _usd(a_day),
                "accrued_next_day_usd": _usd(a_after),
                "accrual_fall_usd": _usd(fall),
                "accrued_that_day_usd": _usd(accrued_that_day),
                "interest_posted_usd": _usd(explained),
                "gap_usd": _usd(gap),
                "difference_usd": _usd(off),
                "tolerance_usd": _usd(TOLERANCE_CENTS),
            },
            note=(
                f"IB posted interest: its accrual fell {_usd(fall):.2f} "
                f"({_usd(a_before):.2f} on {before} to {_usd(a_day):.2f} on {day}) and "
                f"{_usd(accrued_that_day):.2f} accrued that day, so {_usd(explained):.2f} was "
                f"posted against the {_usd(gap):.2f} gap."
            ),
        ),
        "",
    )


def _owed(record: Mapping[str, object], found: Mapping[str, object]) -> tuple[str, int] | None:
    """The (pay date, cents) of the one dividend this break is about."""
    ticker, day, short = (
        record.get("instrument"),
        str(record["trading_day"]),
        _cents(record.get("amount")),
    )
    if short is None:
        return None
    candidates = {
        (str(r.get("pay_date")), cast("int", _cents(r.get("amount_usd"))))
        for r in _rows(found, "dividends_owed")
        if r.get("ticker") == ticker and _cents(r.get("amount_usd")) == -short
    } | {
        (day, cast("int", _cents(r.get("amount_usd"))))
        for r in _rows(found, "dividends_unpaid")
        if r.get("ticker") == ticker
        and r.get("pay_date") == day
        and _cents(r.get("amount_usd")) is not None
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _r2(
    record: Mapping[str, object],
    records: Records,
    *,
    as_of: str,
    waivers: Sequence[DividendWaiver],
) -> tuple[Match | None, str]:
    day, ticker = str(record["trading_day"]), str(record.get("instrument"))
    found = records.reconciliation(day)
    if found is None:
        return None, f"no reconciliation for {day}"
    owed = _owed(record, found)
    if owed is None:
        return None, "not a dividend break, or not one dividend"
    pay_date, cents = owed
    for waiver in waivers:
        if (waiver.ticker, waiver.pay_date, _cents(waiver.amount_usd)) == (ticker, pay_date, cents):
            return (
                Match(
                    evidence=(
                        ("reconciliation", trader_reconciliation_key(day)),
                        ("waiver", WAIVERS_FILE),
                    ),
                    arithmetic={
                        "ticker": ticker,
                        "pay_date": pay_date,
                        "dividend_usd": _usd(cents),
                        "waived_on": waiver.ruled_on,
                        "ruling": waiver.ruling,
                    },
                    note=(
                        f"The {ticker} dividend of {_usd(cents):.2f} paid {pay_date} was waived "
                        f"on {waiver.ruled_on}: {waiver.ruling}"
                    ),
                ),
                "",
            )
    for later in records.reconciliation_days():
        if not day < later <= as_of:
            continue
        credit = records.reconciliation(later) or {}
        for row in _rows(credit, "dividend_credits"):
            if (row.get("ticker"), row.get("pay_date"), _cents(row.get("amount_usd"))) == (
                ticker,
                pay_date,
                cents,
            ):
                return (
                    Match(
                        evidence=(
                            ("reconciliation", trader_reconciliation_key(day)),
                            ("credited", trader_reconciliation_key(later)),
                        ),
                        arithmetic={
                            "ticker": ticker,
                            "pay_date": pay_date,
                            "dividend_usd": _usd(cents),
                            "credited_on": later,
                        },
                        note=(
                            f"The {ticker} dividend of {_usd(cents):.2f} paid {pay_date} was "
                            f"credited late, on {later}."
                        ),
                    ),
                    "",
                )
    return None, f"the {ticker} dividend paid {pay_date} is neither waived nor credited yet"


def _r3(record: Mapping[str, object], records: Records, *, as_of: str) -> tuple[Match | None, str]:
    day = str(record["trading_day"])
    for later in records.reconciliation_days():
        if not day < later <= as_of:
            continue
        found = records.reconciliation(later) or {}
        if found.get("clean") is True and found.get("void") is False and not _discrepancies(found):
            return (
                Match(
                    evidence=(
                        ("reconciliation", trader_reconciliation_key(day)),
                        ("superseded_by", trader_reconciliation_key(later)),
                        ("statement", trader_broker_statement_key(later)),
                    ),
                    arithmetic={"stale_on": day, "clean_on": later},
                    note=(
                        f"The {day} check compared against an older statement; the {later} "
                        "reconcile, against fresh statements, was clean."
                    ),
                ),
                "",
            )
    return None, f"no clean reconcile after {day} yet"


def _value(record: Mapping[str, object], side: str) -> object:
    found = record.get(side)
    return cast("Mapping[str, object]", found).get("value") if isinstance(found, Mapping) else None


def _split_pending(document: Mapping[str, object] | None, ticker: str) -> bool:
    return document is not None and any(
        d.get("klass") == "corporate_action"
        and d.get("instrument") == ticker
        and str(d.get("detail", "")).startswith(_SPLIT_PENDING)
        for d in _discrepancies(document)
    )


def _r4(record: Mapping[str, object], records: Records, *, as_of: str) -> tuple[Match | None, str]:
    del as_of
    day, ticker = str(record["trading_day"]), str(record.get("instrument"))
    if not _split_pending(records.reconciliation(day), ticker):
        return None, "not a split the broker has not applied"
    first, age = day, 0
    while age <= R4_LIMIT_TRADING_DAYS and _split_pending(
        records.reconciliation(_shift(first, -1)), ticker
    ):
        first, age = _shift(first, -1), age + 1
    if age >= R4_LIMIT_TRADING_DAYS:
        return None, (
            f"the {ticker} split has been pending since {first}, {age} trading days; "
            "it returns for review"
        )
    return (
        Match(
            evidence=(("reconciliation", trader_reconciliation_key(day)),),
            arithmetic={
                "ticker": ticker,
                "first_seen": first,
                "pending_trading_days": age,
                "limit_trading_days": R4_LIMIT_TRADING_DAYS,
                "expected_shares": _value(record, "expected"),
                "broker_shares": _value(record, "actual"),
            },
            note=(
                f"IB has not yet applied the {ticker} split, pending since {first} "
                f"({age} of {R4_LIMIT_TRADING_DAYS} trading days); rechecked daily."
            ),
        ),
        "",
    )


def evaluate(
    rule: Rule,
    record: Mapping[str, object],
    records: Records,
    *,
    as_of: str,
    waivers: Sequence[DividendWaiver] = (),
) -> tuple[Match | None, str]:
    """Whether ``rule``'s condition holds for the exception ``record``, and if
    not, why. Reads only; the state of the rule is the caller's check."""
    if record.get("source") != rule.source or record.get("class") != rule.klass:
        return None, f"{rule.rule_id} covers {rule.klass}, not {record.get('class')}"
    if rule.rule_id == "R1":
        return _r1(record, records, as_of=as_of)
    if rule.rule_id == "R2":
        return _r2(record, records, as_of=as_of, waivers=waivers)
    if rule.rule_id == "R3":
        return _r3(record, records, as_of=as_of)
    if rule.rule_id == "R4":
        return _r4(record, records, as_of=as_of)
    return None, f"{rule.rule_id} has no condition in code"


def clears_for(
    proposals: Sequence[Mapping[str, object]],
    *,
    decisions: Sequence[Mapping[str, object]],
    clears: Sequence[Mapping[str, object]],
    exceptions: Mapping[str, Mapping[str, object]],
    records: Records,
    as_of: str,
    now: str,
    waivers: Sequence[DividendWaiver],
    rules: Sequence[Rule] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """The new clears for ``proposals`` up to ``as_of``, and for every other
    undecided proposal a rule covers, why it was not cleared.

    A proposal anyone has decided, or a rule has cleared, is skipped. Only an
    ``approved`` rule no person has demoted clears; a proposed or demoted rule
    that would have matched is reported, never applied.
    """
    rules = RULES if rules is None else rules
    demoted = demoted_rules([*decisions, *clears])
    touched = {d["proposal_id"] for d in (*decisions, *clears)}
    new: list[dict[str, object]] = []
    misses: list[dict[str, object]] = []
    for proposal in sorted(proposals, key=lambda p: (str(p["trading_day"]), str(p["proposal_id"]))):
        if str(proposal["trading_day"]) > as_of or proposal["proposal_id"] in touched:
            continue
        record = exceptions.get(cast("str", proposal["exception_id"]))
        covering = [
            r for r in rules if r.source == proposal["source"] and r.klass == proposal["class"]
        ]
        if not covering:
            continue
        if record is None or record.get("tenant") != proposal["tenant"]:
            misses.append({"proposal_id": proposal["proposal_id"], "why": "no exception record"})
            continue
        reasons: list[str] = []
        for rule in covering:
            found, why = evaluate(rule, record, records, as_of=as_of, waivers=waivers)
            if found is None:
                reasons.append(f"{rule.rule_id}: {why}")
                continue
            if rule.state != "approved" or rule.rule_id in demoted:
                state = "demoted" if rule.rule_id in demoted else rule.state
                reasons.append(f"{rule.rule_id}: would clear, but the rule is {state}")
                continue
            new.append(
                rule_clear(
                    proposal,
                    rule_id=rule.rule_id,
                    evidence=[{"role": role, "record": key} for role, key in found.evidence],
                    arithmetic=found.arithmetic,
                    note=found.note,
                    decided_at=now,
                )
            )
            break
        else:
            misses.append(
                {
                    "proposal_id": proposal["proposal_id"],
                    "trading_day": proposal["trading_day"],
                    "class": proposal["class"],
                    "why": "; ".join(reasons),
                }
            )
    return new, misses


def file_rule_clears(
    store: Store, *, as_of: str, now: str
) -> tuple[str, dict[str, object] | None, list[dict[str, object]]]:
    """The run day's clears file: its key, its document (``None`` when there is
    nothing to file) and the misses. Clears an earlier run of the same day
    filed are carried forward unchanged, so a re-run never re-clears and never
    changes a decision id a person's audit may already name. Writes nothing;
    the job records the document."""
    key = rule_clears_key(as_of)
    carried: list[dict[str, object]] = []
    if store.exists(key):
        carried = cast(
            "list[dict[str, object]]",
            json.loads(store.get_bytes(key).decode("utf-8")).get("clears", []),
        )
    proposals = load_proposals(store, until=as_of)
    clears = load_rule_clears(store, until=as_of)
    new, misses = clears_for(
        proposals,
        decisions=load_decisions(store),
        clears=clears,
        exceptions=load_exception_records(store, (str(p["trading_day"]) for p in proposals)),
        records=StoreRecords(store),
        as_of=as_of,
        now=now,
        waivers=load_declared_waivers(),
    )
    if not carried and not new:
        return key, None, misses
    document: dict[str, object] = {
        "schema_version": RULE_CLEARS_SCHEMA_VERSION,
        "trading_day": as_of,
        "generated_at": now,
        "rules": registry(),
        "clears": [*carried, *new],
    }
    return key, document, misses


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m crucible_trader.exception_rules",
        description="Dry run: what the tier-2 rules would clear, read-only.",
    )
    parser.add_argument("--store", default=None, help="store URI (default: $CRUCIBLE_STORE)")
    parser.add_argument("--as-of", default=None, type=dt.date.fromisoformat)
    parser.add_argument("--now", default=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    args = parser.parse_args(argv)
    store = read_only(
        open_store(cast("str | None", args.store)), reason="exception_rules dry run is read-only"
    )
    as_of_date = cast("dt.date | None", args.as_of) or dt.datetime.now(dt.UTC).date()
    while not is_trading_day(as_of_date):
        as_of_date -= dt.timedelta(days=1)
    try:
        _key, document, misses = file_rule_clears(
            store, as_of=as_of_date.isoformat(), now=cast("str", args.now)
        )
    except DecisionError as exc:
        sys.stderr.write(f"refused: {exc}\n")
        return 2
    out = {"would_file": document, "not_cleared": misses}
    sys.stdout.write(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
