"""The v1 trading-rules guard: v2 may not trade on a v1 rule nobody has carried.

`alpha-engine-config-I12024`. On 2026-11-03 v2 began routing orders without
v1's loss floor, drawdown ladder, broker-side stop, negative-alpha exit and
more (the 2026-11-09 parity audit). The v1 -> v2 carry-over ledger existed and
did not fire, for three reasons this module answers one by one:

1. **Rules were never rows.** The ledger enumerates what v1 serves as
   ARTIFACTS (arms, champions, tuned parameter documents). A trading rule is
   code plus a risk config, so the private strategy tree now carries a sibling
   record, `strategy/v1_trading_rules.yaml`, published to :data:`RULES_KEY`:
   one row per v1 rule with its threshold and its v2 disposition.
2. **`deferred` read as a decision.** The ledger's ten executor-parameter rows
   sat `deferred`, which its phase-3 clause grades MET. Here a `deferred`
   executor-parameter row is OUTSTANDING, like a `pending` rule.
3. **The clause gates a phase, not a session.** This guard runs inside
   :func:`crucible_trader.session.session_cycle`, before the contract is
   resolved, so it stands between the record and the first order.

**What is OUTSTANDING.** A `money_path` rule that is `pending`; a `carried`
rule this wheel does not register in :data:`IMPLEMENTED_RULES`, or registers
at a different location than the record names (the record and the running code
disagree, so neither proves the rule); and a `deferred` row of the ledger's
`config/executor_params.json` section. A rule `superseded` or `excluded` on the
record carries a ruling reference and is settled.

**What it does, by date.** The record carries ``enforce_from``.

* Before it: the session TRADES. The outstanding rows are named on the run's
  ``v1_trading_rules_outstanding`` metric (status ``BREACH``) and logged at
  ERROR, every session. Brian's rule is no trading holds, so the guard lands
  ahead of the fixes it waits on and only reports until they merge.
* On and after it: :class:`V1RulesUnmetError` is raised before anything is
  built. No book, no order, no evidence; `trader.session` files a failed
  manifest whose reason names every outstanding rule, which is the failure
  page (`crucible.alerts`).

**An unreadable record is UNMEASURABLE.** When the date is known (the record
parsed but the ledger did not), an unmeasurable reading on or after it
refuses, like an outstanding one. When the record itself is absent or
malformed the date is unknown, so the guard reports and trades: an absent
record is the state before the private tree first publishes it, and refusing
on it would hold every session on a deploy-order accident. The private tree's
own test refuses a malformed or missing record before it can publish.

**Adding an implementation.** A PR that implements a v1 rule adds its entry to
:data:`IMPLEMENTED_RULES` (location + the test that proves it); the private
record's row then flips `pending` -> `carried` naming the same location. The
test suite asserts every entry here imports and names a test that exists.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import re
from collections.abc import Mapping
from typing import Literal, cast

import yaml
from crucible.carryover import V1_EXECUTOR_PARAMS_KEY, parse_ledger
from crucible.keys import v1_carryover_key
from crucible.runner import RunContext
from crucible.store import Store

log = logging.getLogger(__name__)

#: The record, inside the synced strategy tree beside `v1_carryover_key()`.
RULES_KEY = "strategy/current/v1_trading_rules.yaml"
RULES_SCHEMA_VERSION = "v1_trading_rules.v1"
METRIC_NAME = "v1_trading_rules_outstanding"
METRIC_MODULE = "crucible_trader.v1_rules_guard"

DISPOSITIONS = ("carried", "superseded", "excluded", "pending")
_RULE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_REFERENCE = re.compile(r"^alpha-engine-config-I[1-9][0-9]*$")
_FIELDS = frozenset(
    {
        "rule",
        "audit_row",
        "v1_rule",
        "v1_threshold",
        "v1_source",
        "money_path",
        "disposition",
        "v2_location",
        "v2_test",
        "v2_prs",
        "reason",
        "reference",
        "ledger_params",
        "note",
    }
)


@dataclasses.dataclass(frozen=True)
class Implementation:
    """Where this wheel implements a v1 rule, and the test that proves it.

    Both are spelled as the private record spells them:
    ``<repo>:<path>.py:<symbol>`` and ``<repo>:tests/<file>.py::<node>``,
    repo ``crucible`` or ``crucible-trader``.
    """

    location: str
    test: str


#: The v1 rules this wheel implements. Keyed by the record's ``rule`` id.
IMPLEMENTED_RULES: Mapping[str, Implementation] = {
    "hold_book_safeguard": Implementation(
        location="crucible-trader:crucible_trader/hold_book.py:enforce_hold_book",
        test="crucible-trader:tests/test_hold_book.py::test_the_v1_numbers_are_kept",
    ),
    "long_only": Implementation(
        location="crucible-trader:crucible_trader/order_router.py:plan_orders",
        test="crucible-trader:tests/test_order_router.py::test_a_short_target_is_refused",
    ),
    "conviction_turnover_gate": Implementation(
        location="crucible:crucible/portfolio.py:compute_conviction_budget_multiplier",
        test=(
            "crucible:tests/test_portfolio.py::TestConvictionBudgetGate"
            "::test_high_conviction_is_not_throttled"
        ),
    ),
}


class RulesRecordError(ValueError):
    """The record is present and does not satisfy its schema."""


class V1RulesUnmetError(RuntimeError):
    """A v1 trading rule is outstanding on or after the record's enforce date.

    Raised before construction: the session trades nothing and its run fails,
    which pages. Distinct from `ContractUnavailable` (a hold that is `ok`).
    """


@dataclasses.dataclass(frozen=True)
class Rule:
    rule: str
    money_path: bool
    disposition: str
    v2_location: str | None = None
    v2_prs: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class RulesRecord:
    enforce_from: dt.date
    rules: tuple[Rule, ...]


def _text(entry: Mapping[str, object], name: str, where: str) -> str:
    value = entry.get(name)
    if not isinstance(value, str) or not value.strip():
        raise RulesRecordError(f"{where}: `{name}` must be a non-empty string, got {value!r}")
    return value.strip()


def parse_rules(raw: bytes, *, source: str = RULES_KEY) -> RulesRecord:
    """The record, or :class:`RulesRecordError` naming the first violation.

    Strict on everything the guard decides on; the descriptive fields
    (``v1_rule``, ``v1_threshold``, ...) are required text, never interpreted.
    """
    try:
        loaded: object = yaml.safe_load(raw.decode("utf-8"))
    except Exception as exc:
        raise RulesRecordError(
            f"{source} is not readable YAML: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(loaded, dict):
        raise RulesRecordError(f"{source} parsed to {type(loaded).__name__}, not a mapping")
    document = cast("dict[str, object]", loaded)
    if document.get("schema_version") != RULES_SCHEMA_VERSION:
        raise RulesRecordError(
            f"{source}: schema_version is {document.get('schema_version')!r}, "
            f"expected {RULES_SCHEMA_VERSION!r}"
        )
    try:
        enforce_from = dt.date.fromisoformat(_text(document, "enforce_from", source))
    except ValueError as exc:
        raise RulesRecordError(f"{source}: enforce_from is not an ISO date: {exc}") from exc
    listed = document.get("rules")
    if not isinstance(listed, list) or not listed:
        raise RulesRecordError(f"{source}: `rules` must be a non-empty list")
    entries = cast("list[object]", listed)
    rules: list[Rule] = []
    seen: set[str] = set()
    for index, item in enumerate(entries):
        where = f"{source} rules[{index}]"
        if not isinstance(item, dict):
            raise RulesRecordError(f"{where} is {type(item).__name__}, not a mapping")
        entry = cast("dict[str, object]", item)
        unknown = sorted(set(entry) - _FIELDS)
        if unknown:
            raise RulesRecordError(f"{where}: unknown field(s) {unknown}")
        rule = _text(entry, "rule", where)
        if not _RULE_ID.match(rule) or rule in seen:
            raise RulesRecordError(f"{where}: rule {rule!r} is malformed or has two rows")
        seen.add(rule)
        for name in ("v1_rule", "v1_threshold", "v1_source"):
            _text(entry, name, where)
        money_path = entry.get("money_path")
        if not isinstance(money_path, bool):
            raise RulesRecordError(f"{where}: `money_path` must be true or false")
        disposition = _text(entry, "disposition", where)
        if disposition not in DISPOSITIONS:
            raise RulesRecordError(f"{where}: disposition {disposition!r} not in {DISPOSITIONS}")
        location = None
        if disposition == "carried":
            location = _text(entry, "v2_location", where)
            _text(entry, "v2_test", where)
        else:
            reference = _text(entry, "reference", where)
            if not _REFERENCE.match(reference):
                raise RulesRecordError(f"{where}: reference {reference!r} is not an issue id")
            if disposition in ("superseded", "excluded"):
                _text(entry, "reason", where)
        raw_prs: object = entry.get("v2_prs") or []
        if not isinstance(raw_prs, list):
            raise RulesRecordError(f"{where}: `v2_prs` must be a list of PR references")
        listed_prs = cast("list[object]", raw_prs)
        prs = tuple(p for p in listed_prs if isinstance(p, str))
        if len(prs) != len(listed_prs):
            raise RulesRecordError(f"{where}: `v2_prs` must be a list of PR references")
        rules.append(Rule(rule, money_path, disposition, location, prs))
    return RulesRecord(enforce_from=enforce_from, rules=tuple(rules))


@dataclasses.dataclass(frozen=True)
class RulesReading:
    state: Literal["MET", "OUTSTANDING", "UNMEASURABLE"]
    enforce_from: dt.date | None
    outstanding: tuple[str, ...] = ()
    problem: str | None = None
    #: Not findings: an implementation this wheel registers that the record
    #: still calls pending — the row is ready to flip to `carried`.
    notes: tuple[str, ...] = ()

    def enforced_on(self, day: dt.date) -> bool:
        return self.enforce_from is not None and day >= self.enforce_from

    def describe(self) -> str:
        parts: list[str] = []
        if self.problem:
            parts.append(f"UNMEASURABLE: {self.problem}")
        if self.outstanding:
            parts.append(
                f"{len(self.outstanding)} v1 trading rule(s) outstanding: "
                + "; ".join(self.outstanding)
            )
        parts.extend(self.notes)
        when = (
            "enforce_from unknown"
            if self.enforce_from is None
            else f"enforce_from {self.enforce_from.isoformat()}"
        )
        return f"{' | '.join(parts) or 'every v1 trading rule settled'} ({when})"


def _read(store: Store, key: str) -> bytes | None:
    return store.get_bytes(key) if store.exists(key) else None


def read_rules(
    store: Store, *, implemented: Mapping[str, Implementation] = IMPLEMENTED_RULES
) -> RulesReading:
    """Grade the published record and ledger against this wheel. Never raises
    on a store or schema problem: that is an UNMEASURABLE reading."""
    try:
        raw = _read(store, RULES_KEY)
        if raw is None:
            return RulesReading(
                "UNMEASURABLE", None, problem=f"no v1 trading-rules record at {RULES_KEY}"
            )
        record = parse_rules(raw)
    except Exception as exc:  # store fault or schema fault: both leave the date unknown
        return RulesReading("UNMEASURABLE", None, problem=f"{type(exc).__name__}: {exc}")

    outstanding: list[str] = []
    notes: list[str] = []
    for rule in record.rules:
        have = implemented.get(rule.rule)
        if rule.disposition == "pending":
            if rule.money_path:
                prs = f" [{', '.join(rule.v2_prs)}]" if rule.v2_prs else ""
                outstanding.append(f"{rule.rule}: pending{prs}")
            if have is not None:
                notes.append(f"{rule.rule} is implemented in this wheel; its row can be carried")
        elif rule.disposition == "carried":
            if have is None:
                outstanding.append(f"{rule.rule}: carried on the record, absent from this wheel")
            elif have.location != rule.v2_location:
                outstanding.append(
                    f"{rule.rule}: the record names {rule.v2_location}, this wheel {have.location}"
                )

    problem = None
    try:
        ledger_raw = _read(store, v1_carryover_key())
        if ledger_raw is None:
            problem = f"no v1 carry-over ledger at {v1_carryover_key()}"
        else:
            for row in parse_ledger(ledger_raw, source=v1_carryover_key()):
                if (
                    row.dimension == "parameters"
                    and row.group == V1_EXECUTOR_PARAMS_KEY
                    and row.disposition == "deferred"
                ):
                    outstanding.append(f"{row.group}:{row.item}: deferred in the ledger")
    except Exception as exc:  # LedgerError, or a store fault
        problem = f"v1 carry-over ledger unreadable: {type(exc).__name__}: {exc}"

    if problem is not None:
        state: Literal["MET", "OUTSTANDING", "UNMEASURABLE"] = "UNMEASURABLE"
    else:
        state = "OUTSTANDING" if outstanding else "MET"
    return RulesReading(state, record.enforce_from, tuple(outstanding), problem, tuple(notes))


def enforce_v1_rules(
    ctx: RunContext, *, implemented: Mapping[str, Implementation] = IMPLEMENTED_RULES
) -> RulesReading:
    """Report every session; refuse the session from ``enforce_from``.

    Records the reading as a metric first, so a refused run still names what
    it refused on.
    """
    reading = read_rules(ctx.store, implemented=implemented)
    enforced = reading.state != "MET" and reading.enforced_on(ctx.trading_day)
    status = {"MET": "OK", "OUTSTANDING": "BREACH", "UNMEASURABLE": "unmeasurable"}[reading.state]
    ctx.record_metric(
        {
            "name": METRIC_NAME,
            "module": METRIC_MODULE,
            "metric_type": "control",
            "value": float(len(reading.outstanding)),
            "unit": "count",
            "n_floor": 1,
            "status": "FAIL" if enforced else status,
            "status_reason": reading.describe(),
            "source_path": RULES_KEY,
            "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    if reading.state == "MET":
        return reading
    if enforced:
        raise V1RulesUnmetError(
            f"trading refused on {ctx.trading_day.isoformat()}: {reading.describe()}. "
            "Each outstanding rule needs a merged implementation (row -> carried) or a "
            "ruling (row -> superseded/excluded) in strategy/v1_trading_rules.yaml"
        )
    log.error("v1 trading rules NOT enforced yet, trading anyway: %s", reading.describe())
    return reading


__all__ = [
    "IMPLEMENTED_RULES",
    "METRIC_NAME",
    "RULES_KEY",
    "RULES_SCHEMA_VERSION",
    "Implementation",
    "Rule",
    "RulesReading",
    "RulesRecord",
    "RulesRecordError",
    "V1RulesUnmetError",
    "enforce_v1_rules",
    "parse_rules",
    "read_rules",
]
