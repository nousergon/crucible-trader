"""WORM retention on the v2 books (alpha-engine-config-I13191)."""

from __future__ import annotations

import datetime as dt
import json
import pathlib
from typing import Any

from botocore.exceptions import ClientError
from crucible.keys import trader_broker_statement_key as broker_statement_key
from crucible.models import MetricRecordRow
from crucible.runner import RunContext
from crucible.store import LocalStore, S3Store
from s3_fake import FakeS3

from crucible_trader.books_retention import (
    BOOKS_RETENTION_METRIC,
    RETENTION_MODE,
    is_books_key,
    lock_books_records,
    lock_cycle_books,
    retain_until,
)
from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.ledger import journal_key
from crucible_trader.reconciliation import CorporateActionSet, Fill
from crucible_trader.reconciliation_job import RECONCILE_JOB, reconcile_cycle

DAY = dt.date(2026, 9, 11)
UNTIL = dt.datetime(2032, 1, 1, tzinfo=dt.UTC)
NOW = "2026-09-11T21:00:00Z"


def s3_store() -> tuple[S3Store, FakeS3]:
    client = FakeS3(now=lambda: dt.datetime(2026, 9, 11, 21, 0, tzinfo=dt.UTC))
    return S3Store("bucket", "crucible", client=client), client


def ctx_for(store: Any) -> RunContext:
    return RunContext(
        run_id="01JG0000000000000000000000",
        job=RECONCILE_JOB,
        trading_day=DAY,
        calendar_date=DAY,
        store=store,
        seed=0,
        started=dt.datetime(2026, 9, 11, 21, 0, tzinfo=dt.UTC),
    )


class Refusing(FakeS3):
    def put_object_retention(self, **kw: Any) -> dict[str, Any]:
        raise ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "no"}}, "PutObjectRetention"
        )


def test_books_prefixes_only():
    assert is_books_key("trader/ledger/journal/t/2026-09-11.json")
    assert is_books_key("trader/broker_statements/2026-09-11.json")
    assert is_books_key("trader/exception_scores/current.json")
    assert not is_books_key("trader/evidence.json")
    assert not is_books_key("trader/release_pin")
    assert not is_books_key("trader/kill_switch.json")


def test_retain_until_is_fiscal_year_end_plus_five_years():
    assert retain_until(dt.date(2026, 1, 2)) == UNTIL
    assert retain_until(dt.date(2026, 12, 31)) == UNTIL


def test_locks_books_keys_in_compliance_mode_and_skips_the_rest():
    store, client = s3_store()
    keys = ["trader/ledger/journal/t/2026-09-11.json", "trader/evidence.json"]
    for k in keys:
        store.put_bytes(k, b"{}")
    locked, refused = lock_books_records(store, [*keys, keys[0]], trading_day=DAY)
    assert locked == [keys[0]] and refused == {}
    assert client.retention == {
        f"crucible/{keys[0]}": {"Mode": "COMPLIANCE", "RetainUntilDate": UNTIL}
    }
    assert RETENTION_MODE == "COMPLIANCE"


def test_a_refused_lock_is_reported_never_raised():
    client = Refusing(now=lambda: dt.datetime(2026, 9, 11, tzinfo=dt.UTC))
    store = S3Store("bucket", "crucible", client=client)
    store.put_bytes("trader/reconciliation/2026-09-11.json", b"{}")
    locked, refused = lock_books_records(
        store, ["trader/reconciliation/2026-09-11.json"], trading_day=DAY
    )
    assert locked == []
    assert "AccessDenied" in refused["trader/reconciliation/2026-09-11.json"]


def test_local_store_locks_nothing(tmp_path: pathlib.Path):
    store = LocalStore(tmp_path)
    assert lock_books_records(store, ["trader/ledger/x.json"], trading_day=DAY) == ([], {})


def test_cycle_step_records_ok_and_fail_metrics_that_validate():
    store, _ = s3_store()
    ctx = ctx_for(store)
    ctx.record_output("trader/exceptions/2026-09-11.json", b"{}")
    lock_cycle_books(ctx, now=NOW)
    (metric,) = ctx.metrics
    MetricRecordRow.model_validate(metric)
    assert metric["name"] == BOOKS_RETENTION_METRIC and metric["status"] == "OK"
    assert metric["value"] == 1.0 and "2032-01-01" in metric["status_reason"]

    client = Refusing(now=lambda: dt.datetime(2026, 9, 11, tzinfo=dt.UTC))
    ctx = ctx_for(S3Store("bucket", "crucible", client=client))
    ctx.record_output("trader/exceptions/2026-09-11.json", b"{}")
    lock_cycle_books(ctx, now=NOW)
    (metric,) = ctx.metrics
    MetricRecordRow.model_validate(metric)
    assert metric["status"] == "FAIL" and metric["value"] == 0.0
    assert "trader/exceptions/2026-09-11.json" in metric["status_reason"]


def test_cycle_step_is_silent_with_nothing_to_lock_and_never_raises(tmp_path: pathlib.Path):
    ctx = ctx_for(LocalStore(tmp_path))
    ctx.record_output("trader/ledger/x.json", b"{}")
    lock_cycle_books(ctx, now=NOW)
    assert ctx.metrics == []
    lock_cycle_books(object(), now=NOW)  # a broken context is swallowed


def test_reconcile_cycle_locks_its_books_records():
    store, client = s3_store()
    anchor = BrokerStatement("DU1", "2026-09-10", {"AAA": 10}, 1_000.0)
    store.put_bytes(broker_statement_key("2026-09-10"), json.dumps(anchor.to_document()).encode())

    class Source:
        def read_statement(self, trading_day: str) -> BrokerStatement:
            return BrokerStatement("DU1", trading_day, {"AAA": 12}, 800.0)

    ctx = ctx_for(store)
    result = reconcile_cycle(
        ctx,
        source=Source(),
        fills=(Fill("AAA", 2, -200.0),),
        cash_flows=(),
        corporate_actions=CorporateActionSet(resolved=frozenset({"AAA"})),
    )
    assert result.clean
    written = {o["key"] for o in ctx.outputs}
    assert journal_key("nous-ergon-paper", "2026-09-11") in written
    assert set(client.retention) == {f"crucible/{k}" for k in written if is_books_key(k)}
    assert {r["Mode"] for r in client.retention.values()} == {"COMPLIANCE"}
    (metric,) = [m for m in ctx.metrics if m["name"] == BOOKS_RETENTION_METRIC]
    assert metric["status"] == "OK"


def test_anything_around_the_lock_is_swallowed_too():
    store, _ = s3_store()

    def keys():
        yield "trader/ledger/x.json"
        raise RuntimeError("boom")

    locked, refused = lock_books_records(store, keys(), trading_day=DAY)
    assert locked == [] and "boom" in refused["*"]
