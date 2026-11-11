"""The v1 trading-rules guard: no v1 rule stays owed without CI going red.

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
3. **The clause gates a phase, and nothing failed on the trading path.** This
   guard runs inside :func:`crucible_trader.session.session_cycle` and names
   every owed rule on every session; :data:`OWED_RULES` with
   :data:`CI_ENFORCE_FROM` turns the same debt into a red CI from a date.

**What is OUTSTANDING.** A `money_path` rule that is `pending`; a `carried`
rule this wheel does not register in :data:`IMPLEMENTED_RULES`, or registers
at a different location than the record names (the record and the running code
disagree, so neither proves the rule); and a `deferred` row of the ledger's
`config/executor_params.json` section. A rule `superseded` or `excluded` on the
record carries a ruling reference and is settled.

**It never holds a session.** Brian's standing rule is no trading holds. On
every session, before and after any date, the outstanding rows are named on
the run's ``v1_trading_rules_outstanding`` metric (status ``BREACH``, or
``unmeasurable`` when the record or ledger cannot be read) and logged at
ERROR, and the session trades.

**What enforces it is CI, from a date.** The private record carries
``enforce_from`` (2026-11-30). From that date:

* this repository's `tests/test_v1_rules_guard.py` fails while
  :data:`OWED_RULES` is non-empty, so every trader PR and `main` go red until
  each rule is built (its id moves to :data:`IMPLEMENTED_RULES`) or ruled
  `excluded`/`superseded` in the private record (its id is dropped here);
* the private tree's `tests/test_v1_trading_rules.py` fails while any
  money-path row is `pending` or an executor parameter is `deferred`, and
  fails on a missing or unreadable record at any date.

:data:`OWED_RULES` is the public mirror of the private record's owed set —
rule ids only, never a threshold. The private tree's test cross-checks the two
wherever a sibling checkout of this repository is present; this repository's
CI cannot read the private record, so it fails on the mirror.

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
    # crucible-trader#139 (merged 2026-11-10). Its mirror for sells (#150) is a v2
    # addition with no v1 row; it rides the same module.
    "entry_timing": Implementation(
        location="crucible-trader:crucible_trader/entry_timing.py:entry_trigger",
        test=(
            "crucible-trader:tests/test_entry_timing.py"
            "::test_a_two_percent_pullback_from_the_high_releases_the_buy"
        ),
    ),
    # crucible-trader#140: standing GTC STP SELL on every held long.
    "broker_stop": Implementation(
        location="crucible-trader:crucible_trader/order_router.py:PaperOrderRouter.protect_holdings",
        test=(
            "crucible-trader:tests/test_nav_sizing_and_stops.py"
            "::test_every_held_long_gets_one_gtc_stop_after_the_sessions_orders"
        ),
    ),
    # crucible-trader#140: a live session sizes on min(declared notional, NetLiquidation).
    "nav_sizing": Implementation(
        location="crucible-trader:crucible_trader/order_router.py:read_net_liquidation",
        test=(
            "crucible-trader:tests/test_nav_sizing_and_stops.py"
            "::test_the_router_sizes_on_the_nav_with_the_declared_notional_as_ceiling"
        ),
    ),
    # crucible-trader#144. Not a money-path rule, so it was never in OWED_RULES.
    "turnover_tripwire": Implementation(
        location="crucible-trader:crucible_trader/turnover_tripwire.py:check",
        test=(
            "crucible-trader:tests/test_turnover_tripwire.py"
            "::TestTheBands::test_churn_by_small_moves_is_a_rolling_breach"
        ),
    ),
}


class RulesRecordError(ValueError):
    """The record is present and does not satisfy its schema."""


#: From this date `tests/test_v1_rules_guard.py` fails while
#: :data:`OWED_RULES` is non-empty. Mirrors the private record's
#: ``enforce_from``; moving it later weakens a guard (red line 3).
CI_ENFORCE_FROM = dt.date(2026, 11, 30)

#: The money-path v1 rules the private record still owes (`pending`, or
#: carrying a `deferred` executor parameter). A PR that implements one moves it
#: to :data:`IMPLEMENTED_RULES`; a ruling that excludes or supersedes one
#: drops it here in the same change as the record.
#:
#: ``gap_stop_intraday`` stays owed: crucible-trader#143 checks the gap once, at
#: the session start; v1 polled it every 60 s all day.
OWED_RULES: frozenset[str] = frozenset(
    {
        "gap_stop_intraday",
        "position_loss_floor",
        "drawdown_ladder",
        "negative_alpha_exit",
        "hard_risk_reentry_lockout",
        "hard_risk_precedence",
        "research_exit_signal",
        "expectancy_derisk_gate",
        "intraday_resolve",
        "stance_caps",
        "rebalance_band",
    }
)


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
    """Report every session. Never raises and never holds: Brian's rule is no
    trading holds, and the debt is enforced by CI from ``enforce_from``."""
    reading = read_rules(ctx.store, implemented=implemented)
    late = reading.state != "MET" and reading.enforced_on(ctx.trading_day)
    status = {"MET": "OK", "OUTSTANDING": "BREACH", "UNMEASURABLE": "unmeasurable"}[reading.state]
    ctx.record_metric(
        {
            "name": METRIC_NAME,
            "module": METRIC_MODULE,
            "metric_type": "control",
            "value": float(len(reading.outstanding)),
            "unit": "count",
            "n_floor": 1,
            "status": status,
            "status_reason": reading.describe(),
            "source_path": RULES_KEY,
            "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    if reading.state == "MET":
        return reading
    log.error(
        "v1 trading rules owed, trading anyway%s: %s",
        " (PAST enforce_from: CI is red until each is built or ruled)" if late else "",
        reading.describe(),
    )
    return reading


__all__ = [
    "CI_ENFORCE_FROM",
    "IMPLEMENTED_RULES",
    "OWED_RULES",
    "METRIC_NAME",
    "RULES_KEY",
    "RULES_SCHEMA_VERSION",
    "Implementation",
    "Rule",
    "RulesReading",
    "RulesRecord",
    "RulesRecordError",
    "enforce_v1_rules",
    "parse_rules",
    "read_rules",
]
