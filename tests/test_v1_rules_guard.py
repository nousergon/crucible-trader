"""The v1 trading-rules guard (`alpha-engine-config-I12024`).

The mutations are the point: each plants one way a v1 rule can be owed into a
settled world and asserts the reading moves, then asserts what the session
does with it on either side of ``enforce_from`` — report and trade before,
refuse before anything is built on and after.

The last class is the "no working v2 implementation" test: every entry in
:data:`IMPLEMENTED_RULES` must import, and a test it names in this repository
must exist. Delete or rename an implementation and this suite goes red; the
private record's `carried` row then reads OUTSTANDING on the box because this
wheel no longer registers it.
"""

from __future__ import annotations

import ast
import datetime as dt
import importlib
import logging
from pathlib import Path

import pytest
import yaml
from crucible.keys import v1_carryover_key
from crucible.models import MetricRecordRow
from crucible.store import LocalStore
from ib_fakes import ctx_for

from crucible_trader.session import session_cycle
from crucible_trader.v1_rules_guard import (
    IMPLEMENTED_RULES,
    METRIC_NAME,
    RULES_KEY,
    Implementation,
    RulesRecordError,
    V1RulesUnmetError,
    enforce_v1_rules,
    parse_rules,
    read_rules,
)

REPO = Path(__file__).resolve().parents[1]
BEFORE = dt.date(2026, 11, 20)
ENFORCE = dt.date(2026, 11, 30)
AFTER = dt.date(2026, 12, 1)
HOLD_BOOK = IMPLEMENTED_RULES["hold_book_safeguard"]


def _rule(rule: str, disposition: str, *, money_path: bool = True, **extra) -> dict:
    row = {
        "rule": rule,
        "audit_row": None,
        "v1_rule": f"v1's {rule}",
        "v1_threshold": "0.15",
        "v1_source": "crucible-executor executor/x.py:f",
        "money_path": money_path,
        "disposition": disposition,
    }
    if disposition == "carried":
        row.setdefault("v2_location", HOLD_BOOK.location)
        row.setdefault("v2_test", HOLD_BOOK.test)
    else:
        row["reference"] = "alpha-engine-config-I12024"
    if disposition in ("superseded", "excluded"):
        row["reason"] = "a ruling"
    row.update(extra)
    return row


def _settled_rules() -> list[dict]:
    return [
        _rule("hold_book_safeguard", "carried"),
        _rule("ineligible_held_forced_exit", "superseded"),
        _rule("zero_entries_alarm", "pending", money_path=False),
    ]


def _ledger(*, deferred: bool = False) -> dict:
    row = {
        "v1_key": "config/executor_params.json",
        "v1_param": "stance_size_value",
        "v1_value": "1.1",
        "disposition": "deferred" if deferred else "excluded",
        "reason": "why",
        "reference": "alpha-engine-config-I9760",
    }
    return {
        "schema_version": "v1_carryover.v2",
        "arms": [
            {
                "v1_slot": "strategy",
                "v1_arm": "stock_registry",
                "disposition": "carried",
                "v2_arm": "stock_registry",
            }
        ],
        "champions": [
            {
                "v1_slot": "strategy",
                "v1_champion": "stock_registry",
                "disposition": "excluded",
                "reason": "ruled",
                "reference": "alpha-engine-config-I10961",
            }
        ],
        "parameters": [row],
    }


def _store(tmp_path, *, rules=None, ledger=None, enforce_from: str = ENFORCE.isoformat()):
    store = LocalStore(tmp_path)
    if rules is not False:
        store.put_bytes(
            RULES_KEY,
            yaml.safe_dump(
                {
                    "schema_version": "v1_trading_rules.v1",
                    "enforce_from": enforce_from,
                    "rules": _settled_rules() if rules is None else rules,
                }
            ).encode(),
        )
    if ledger is not False:
        store.put_bytes(v1_carryover_key(), yaml.safe_dump(ledger or _ledger()).encode())
    return store


def _metric(ctx) -> dict:
    (metric,) = [m for m in ctx.metrics if m["name"] == METRIC_NAME]
    MetricRecordRow.model_validate(metric)
    return metric


class TestTheReading:
    def test_a_settled_record_is_met(self, tmp_path) -> None:
        reading = read_rules(_store(tmp_path))
        assert (reading.state, reading.outstanding, reading.problem) == ("MET", (), None)

    def test_a_pending_money_path_rule_is_outstanding(self, tmp_path) -> None:
        rules = [
            *_settled_rules(),
            _rule("position_loss_floor", "pending", v2_prs=["crucible-trader-PR141"]),
        ]
        reading = read_rules(_store(tmp_path, rules=rules))
        assert reading.state == "OUTSTANDING"
        assert reading.outstanding == ("position_loss_floor: pending [crucible-trader-PR141]",)

    def test_a_pending_monitoring_rule_is_not(self, tmp_path) -> None:
        assert read_rules(_store(tmp_path)).state == "MET"  # zero_entries_alarm is pending

    def test_a_carried_rule_this_wheel_does_not_implement_is_outstanding(self, tmp_path) -> None:
        """Carried on paper, absent from the running code: the 11-03 shape."""
        reading = read_rules(_store(tmp_path), implemented={})
        assert reading.outstanding == (
            "hold_book_safeguard: carried on the record, absent from this wheel",
        )

    def test_a_carried_rule_at_another_location_is_outstanding(self, tmp_path) -> None:
        moved = {"hold_book_safeguard": Implementation("crucible-trader:x.py:f", HOLD_BOOK.test)}
        reading = read_rules(_store(tmp_path), implemented=moved)
        assert reading.state == "OUTSTANDING"
        assert "this wheel crucible-trader:x.py:f" in reading.outstanding[0]

    def test_a_deferred_executor_parameter_is_outstanding(self, tmp_path) -> None:
        """`deferred` read MET on the phase-3 clause. Here it is owed."""
        reading = read_rules(_store(tmp_path, ledger=_ledger(deferred=True)))
        assert reading.outstanding == (
            "config/executor_params.json:stance_size_value: deferred in the ledger",
        )

    def test_a_pending_rule_this_wheel_implements_is_named_ready(self, tmp_path) -> None:
        rules = [_rule("long_only", "pending", money_path=False)]
        reading = read_rules(_store(tmp_path, rules=rules))
        assert reading.state == "MET"
        assert reading.notes == ("long_only is implemented in this wheel; its row can be carried",)

    def test_no_record_is_unmeasurable_with_no_date(self, tmp_path) -> None:
        reading = read_rules(_store(tmp_path, rules=False))
        assert (reading.state, reading.enforce_from) == ("UNMEASURABLE", None)

    def test_no_ledger_is_unmeasurable_with_the_date(self, tmp_path) -> None:
        reading = read_rules(_store(tmp_path, ledger=False))
        assert (reading.state, reading.enforce_from) == ("UNMEASURABLE", ENFORCE)


class TestParse:
    def _raw(self, **change) -> bytes:
        document = {
            "schema_version": "v1_trading_rules.v1",
            "enforce_from": "2026-11-30",
            "rules": _settled_rules(),
        }
        document.update(change)
        return yaml.safe_dump(document).encode()

    def test_the_settled_record_parses(self) -> None:
        record = parse_rules(self._raw())
        assert record.enforce_from == ENFORCE
        assert [r.rule for r in record.rules] == [r["rule"] for r in _settled_rules()]

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            ({"schema_version": "v0"}, "schema_version"),
            ({"enforce_from": "soon"}, "ISO date"),
            ({"rules": []}, "non-empty"),
            ({"rules": [_rule("a", "pending", extra_field=1)]}, "unknown field"),
            ({"rules": [_rule("a", "carried", v2_test=None)]}, "v2_test"),
            ({"rules": [_rule("a", "pending", reference=None)]}, "reference"),
            ({"rules": [_rule("a", "excluded", reason=None)]}, "reason"),
            ({"rules": [_rule("a", "deferred")]}, "disposition"),
            ({"rules": [_rule("a", "pending", money_path="yes")]}, "money_path"),
            ({"rules": [_rule("a", "pending"), _rule("a", "pending")]}, "two rows"),
        ],
    )
    def test_a_malformed_record_is_refused(self, change, match) -> None:
        with pytest.raises(RulesRecordError, match=match):
            parse_rules(self._raw(**change))


class TestTheGuardByDate:
    def _owed(self, tmp_path):
        return _store(tmp_path, rules=[*_settled_rules(), _rule("broker_stop", "pending")])

    def test_before_the_date_it_reports_and_trades(self, tmp_path, caplog) -> None:
        ctx = ctx_for(self._owed(tmp_path), "trader.session", day=BEFORE)
        with caplog.at_level(logging.ERROR):
            reading = enforce_v1_rules(ctx)
        assert reading.state == "OUTSTANDING"
        metric = _metric(ctx)
        assert (metric["status"], metric["value"]) == ("BREACH", 1.0)
        assert "broker_stop: pending" in metric["status_reason"]
        assert "NOT enforced yet" in caplog.text

    @pytest.mark.parametrize("day", [ENFORCE, AFTER])
    def test_on_and_after_the_date_it_refuses(self, tmp_path, day) -> None:
        ctx = ctx_for(self._owed(tmp_path), "trader.session", day=day)
        with pytest.raises(V1RulesUnmetError, match="broker_stop: pending"):
            enforce_v1_rules(ctx)
        assert _metric(ctx)["status"] == "FAIL"

    def test_a_settled_record_never_refuses(self, tmp_path) -> None:
        ctx = ctx_for(_store(tmp_path), "trader.session", day=AFTER)
        assert enforce_v1_rules(ctx).state == "MET"
        assert _metric(ctx)["status"] == "OK"

    def test_no_record_reports_and_trades_even_late(self, tmp_path) -> None:
        """The date is in the record: with no record it is unknown, and a
        deploy-order accident must not hold a session."""
        ctx = ctx_for(_store(tmp_path, rules=False), "trader.session", day=AFTER)
        assert enforce_v1_rules(ctx).state == "UNMEASURABLE"
        assert _metric(ctx)["status"] == "unmeasurable"

    def test_an_unreadable_ledger_refuses_once_the_date_is_known_and_passed(self, tmp_path) -> None:
        ctx = ctx_for(_store(tmp_path, ledger=False), "trader.session", day=AFTER)
        with pytest.raises(V1RulesUnmetError, match="UNMEASURABLE"):
            enforce_v1_rules(ctx)


class Router:
    def __init__(self) -> None:
        self.books: list = []

    def batches(self, book):
        self.books.append(book)
        return iter(())

    def send(self, batch):  # pragma: no cover - never reached
        raise AssertionError("no batch is sent")


def test_the_session_refuses_before_resolving_the_contract(tmp_path, monkeypatch) -> None:
    """On a day it fires: nothing is resolved, built or sent."""
    store = _store(tmp_path, rules=[_rule("broker_stop", "pending")])
    monkeypatch.setattr("crucible_trader.session.running_wheel_disagreement", lambda *a: None)
    resolved: list = []
    router = Router()
    with pytest.raises(V1RulesUnmetError):
        session_cycle(
            ctx_for(store, "trader.session", day=AFTER),
            broker=object(),  # type: ignore[arg-type]
            router=router,
            installed_version="0.1.0",
            previous_weights=None,
            contract_resolver=lambda s, d: resolved.append(d),  # type: ignore[arg-type,func-returns-value]
        )
    assert (resolved, router.books) == ([], [])


# ---------------------------------------------------------------------------
# every registered implementation is real
# ---------------------------------------------------------------------------


def _split_location(location: str) -> tuple[str, str, str]:
    repo, path, symbol = location.split(":")
    return repo, path, symbol


@pytest.mark.parametrize("rule", sorted(IMPLEMENTED_RULES))
def test_every_registered_implementation_imports(rule: str) -> None:
    repo, path, symbol = _split_location(IMPLEMENTED_RULES[rule].location)
    assert repo in ("crucible", "crucible-trader")
    assert path.endswith(".py")
    module = importlib.import_module(path.removesuffix(".py").replace("/", "."))
    target = module
    for part in symbol.split("."):
        target = getattr(target, part)
    assert callable(target), f"{rule}: {IMPLEMENTED_RULES[rule].location} is not callable"


@pytest.mark.parametrize("rule", sorted(IMPLEMENTED_RULES))
def test_every_registered_test_in_this_repository_exists(rule: str) -> None:
    """A test in `crucible` cannot be resolved from here (the wheel ships no
    tests); its location still imports above, and crucible's CI runs it."""
    repo, _, node = IMPLEMENTED_RULES[rule].test.partition(":")
    path, _, name = node.partition("::")
    if repo != "crucible-trader":
        assert repo == "crucible" and path.startswith("tests/") and name
        return
    source = REPO / path
    assert source.is_file(), f"{rule}: {source} does not exist"
    defined = {
        n.name
        for n in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for part in name.split("::"):
        assert part in defined, f"{rule}: {part} is not defined in {path}"


def test_an_unquoted_enforce_date_is_refused() -> None:
    """The private record is authored by hand; a date written unquoted loads
    as a `date`, not text, and two readers could disagree about it."""
    raw = yaml.safe_dump(
        {
            "schema_version": "v1_trading_rules.v1",
            "enforce_from": ENFORCE,
            "rules": _settled_rules(),
        }
    ).encode()
    with pytest.raises(RulesRecordError, match="enforce_from"):
        parse_rules(raw)
