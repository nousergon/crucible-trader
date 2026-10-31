"""The historical export (alpha-engine-config-I12892 phase 0): stored
reconciliation documents and manifests, written here by the real job through
the real runner into a local store, read back as `exception.v1` records."""

from __future__ import annotations

import datetime as dt
import json
import runpy
import sys

import pytest
from crucible.keys import trader_broker_statement_key, trader_reconciliation_key
from crucible.store import LocalStore

from crucible_trader import exception_export
from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.exception_export import export_exceptions, main
from crucible_trader.exception_record import (
    ExceptionRecordError,
    exceptions_key,
    validate_exception_record,
)
from crucible_trader.order_intent_reconciliation import order_intent_reconciliation_key
from crucible_trader.reconciliation import CorporateActionSet
from crucible_trader.reconciliation_job import ReconciliationDiscrepancyError, run_reconciliation

ANCHOR = "2026-09-08"
CLEAN, BROKEN, LATER = "2026-09-09", "2026-09-10", "2026-09-11"


class FixedSource:
    def __init__(self, positions):
        self.positions = positions

    def read_statement(self, trading_day):
        return BrokerStatement("DU1", trading_day, self.positions, 1_000.0)


@pytest.fixture(autouse=True)
def _code_sha(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_CODE_SHA", "0123456789abcdef0123456789abcdef01234567")


@pytest.fixture
def store(tmp_path):
    store = LocalStore(tmp_path / "store")
    anchor = BrokerStatement("DU1", ANCHOR, {"AAA": 10}, 1_000.0)
    store.put_bytes(trader_broker_statement_key(ANCHOR), json.dumps(anchor.to_document()).encode())
    for day, held in ((CLEAN, 10), (BROKEN, 11), (LATER, 11)):
        try:
            run_reconciliation(
                store,
                FixedSource({"AAA": held}),
                trading_day=dt.date.fromisoformat(day),
                fills=(),
                cash_flows=(),
                corporate_actions=CorporateActionSet(frozenset({"AAA"})),
                run_mode="replay",
            )
        except ReconciliationDiscrepancyError:
            assert day == BROKEN
    return store


def intent_document(day):
    return {
        "trading_day": day,
        "generated_at": f"{day}T20:45:00.000000Z",
        "rows": [
            {
                "order_key": f"crucible-trader|{day}|AAA",
                "symbol": "AAA",
                "side": "buy",
                "intended_quantity": 4,
                "class": "unknown",
                "filled_quantity": 0.0,
                "broker_refs": [],
                "broker_statuses": [],
            }
        ],
    }


def test_every_stored_break_is_exported_with_the_jobs_own_id(store):
    filed = json.loads(store.get_bytes(exceptions_key(BROKEN)))["records"]
    exported = export_exceptions(store, since=CLEAN)
    assert [r["trading_day"] for r in exported] == [BROKEN]
    (record,) = exported
    validate_exception_record(record)
    assert record["class"] == "share_delta" and record["amount"] == 1.0
    assert record["exception_id"] == filed[0]["exception_id"]


def test_the_window_bounds_both_ends(store):
    assert export_exceptions(store, since=LATER) == []
    assert export_exceptions(store, since=ANCHOR, until=CLEAN) == []


def test_intent_breaks_are_exported_beside_the_broker_breaks(store):
    store.put_bytes(
        order_intent_reconciliation_key(BROKEN), json.dumps(intent_document(BROKEN)).encode()
    )
    classes = [r["class"] for r in export_exceptions(store, since=BROKEN, until=BROKEN)]
    assert classes == ["share_delta", "unknown"]


def test_an_unreadable_document_raises_naming_its_key(store):
    store.put_bytes(trader_reconciliation_key(BROKEN), b"[]")
    with pytest.raises(ExceptionRecordError, match=f"trader/reconciliation/{BROKEN}.json"):
        export_exceptions(store, since=CLEAN)


def test_an_intent_document_with_no_reconciliation_beside_it_raises(store):
    store.put_bytes(
        order_intent_reconciliation_key("2026-09-14"),
        json.dumps(intent_document("2026-09-14")).encode(),
    )
    with pytest.raises(ExceptionRecordError, match="2026-09-14"):
        export_exceptions(store, since=CLEAN)
    # Outside the window it is not this export's concern.
    assert len(export_exceptions(store, since=CLEAN, until=LATER)) == 1


def test_main_writes_json_lines_and_never_writes_the_store(store, tmp_path, capsys):
    root = tmp_path / "store"
    before = sorted(p.relative_to(root) for p in root.rglob("*"))
    assert main(["--store", str(root), "--since", CLEAN]) == 0
    captured = capsys.readouterr()
    (line,) = captured.out.splitlines()
    assert json.loads(line)["class"] == "share_delta"
    assert "1 exception records exported" in captured.err
    out = tmp_path / "breaks.jsonl"
    assert main(["--store", str(root), "--since", CLEAN, "--until", LATER, "--out", str(out)]) == 0
    assert out.read_text().count("\n") == 1
    assert sorted(p.relative_to(root) for p in root.rglob("*")) == before


def test_module_entry_point(store, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["exception_export", "--store", str(tmp_path / "store"), "--since", LATER]
    )
    monkeypatch.delitem(sys.modules, exception_export.__name__)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("crucible_trader.exception_export", run_name="__main__")
    assert exited.value.code == 0
    assert capsys.readouterr().out == ""
