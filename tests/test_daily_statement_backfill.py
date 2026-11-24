"""Daily Flex recovery uses the real replay and create-only apply."""

import datetime as dt
import json

import pytest
from crucible.keys import trader_broker_statement_key, trader_reconciliation_key
from crucible.store import LocalStore
from test_reconciliation_job import ctx_for
from test_statement_backfill import CA, DAY, flex_xml, snapshot

from crucible_trader import reconciliation_job as job
from crucible_trader.broker_statement import BrokerStatement

NOW = dt.datetime(2026, 10, 3, 2, tzinfo=dt.UTC)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_CODE_SHA", "a" * 40)
    backend = LocalStore(tmp_path / "books")
    anchor = BrokerStatement("DU1", "2026-09-08", {"AAA": 10}, 1000.0)
    backend.put_bytes(
        trader_broker_statement_key("2026-09-08"), json.dumps(anchor.to_document()).encode()
    )
    flex = LocalStore(tmp_path / "flex")
    flex.put_bytes("2026-09/20260909-one.xml", flex_xml(account="DU1").encode())
    monkeypatch.setenv("CRUCIBLE_TRADER_IB_FLEX_URI", str(tmp_path / "flex"))
    return backend, flex


def run_step(backend, **kwargs):
    ctx = ctx_for(backend, dt.date(2026, 10, 2))
    job.apply_waiting_statements(ctx, now=NOW, clock=lambda: NOW, corporate_actions=CA, **kwargs)
    assert len(ctx.metrics) == 1
    return ctx.metrics[0]


def test_applies_once_and_replay_never_scans(world, monkeypatch):
    backend, flex = world
    row = run_step(backend)
    assert row["name"] == "statement_backfill_applied"
    assert row["value"] == 1 and row["status"] == "OK"
    assert DAY in row["status_reason"]
    assert backend.exists(trader_reconciliation_key(DAY))
    manifest = json.loads(backend.get_bytes(f"runs/trader.reconcile/{DAY}/run.json"))
    assert all(m["name"] != "statement_backfill_applied" for m in manifest["metrics"])
    before = snapshot(backend.root)
    assert run_step(backend)["value"] == 0
    assert snapshot(backend.root) == before


def test_present_statement_skips_even_bad_xml(world):
    backend, flex = world
    backend.put_bytes(trader_broker_statement_key(DAY), b"already filed")
    backend.put_bytes(trader_reconciliation_key(DAY), b"already reconciled")
    flex.put_bytes("2026-09/20260909-one.xml", b"malformed")
    row = run_step(backend)
    assert row["status"] == "OK" and row["value"] == 0
    assert "skipped" in row["status_reason"]


@pytest.mark.parametrize("failure", ["malformed", "reconciliation", "market"])
def test_refusals_never_write(world, failure):
    backend, flex = world
    if failure == "malformed":
        flex.put_bytes("2026-09/20260909-one.xml", b"<bad")
    if failure == "reconciliation":
        backend.put_bytes(trader_reconciliation_key(DAY), b"existing")
    before = snapshot(backend.root)
    ctx = ctx_for(backend)
    now = dt.datetime(2026, 10, 2, 15, tzinfo=dt.UTC) if failure == "market" else NOW
    job.apply_waiting_statements(ctx, now=now, clock=lambda: now, corporate_actions=CA)
    assert ctx.metrics[0]["status"] == "FAIL"
    assert DAY in ctx.metrics[0]["status_reason"]
    assert snapshot(backend.root) == before


def test_access_denied_is_fail(world, monkeypatch):
    backend, _ = world

    def denied(_uri):
        raise PermissionError("AccessDenied")

    monkeypatch.setattr(job, "open_store", denied)
    row = run_step(backend)
    assert row["status"] == "FAIL"
    assert "AccessDenied" in row["status_reason"]


def test_listing_denial_does_not_hide_other_month(world, monkeypatch):
    backend, flex = world
    original = flex.list_keys

    def listing(prefix):
        if prefix == "2026-10/":
            raise PermissionError("AccessDenied")
        return original(prefix)

    monkeypatch.setattr(flex, "list_keys", listing)
    monkeypatch.setattr(job, "open_store", lambda _uri: flex)
    row = run_step(backend)
    assert row["value"] == 1 and row["status"] == "FAIL"
    assert "2026-10: FAIL" in row["status_reason"]


@pytest.mark.parametrize(
    "key", ["2026-09/bad.xml", "2026-10/20261102-wrong.xml", "2026-10/20261003-future.xml"]
)
def test_bad_filenames_are_refused(world, key):
    backend, flex = world
    flex.put_bytes(key, b"not a statement")
    row = run_step(backend)
    assert row["value"] == 1 and row["status"] == "FAIL"
    assert key in row["status_reason"]


def test_only_two_months_and_xml_are_considered(world):
    backend, flex = world
    flex.put_bytes("2026-08/20260803-old.xml", b"bad")
    flex.put_bytes("2026-09/README.txt", b"bad")
    flex.put_bytes("2026-10/20261001-new.xml", flex_xml(day="20261001", account="DU1").encode())
    row = run_step(backend)
    assert row["status"] == "OK" and row["value"] == 2
    assert backend.exists(trader_broker_statement_key("2026-10-01"))


def test_retention_refusal_is_not_reported_as_success(world, monkeypatch):
    from crucible_trader import statement_backfill

    backend, _ = world
    monkeypatch.setattr(statement_backfill, "lock_books_records", lambda *a, **kw: ([], ["denied"]))
    row = run_step(backend)
    assert row["status"] == "FAIL" and "retention refused" in row["status_reason"]


def test_metric_failure_is_logged_without_raising(world, caplog, monkeypatch):
    backend, _ = world
    ctx = ctx_for(backend)

    def fail(_metric):
        raise OSError("metric disk full")

    monkeypatch.setattr(ctx, "record_metric", fail)
    job.apply_waiting_statements(ctx, now=NOW, corporate_actions=CA)
    assert "statement_backfill_applied metric NOT recorded" in caplog.text


def test_default_uri_is_used(world, monkeypatch):
    backend, flex = world
    monkeypatch.delenv("CRUCIBLE_TRADER_IB_FLEX_URI")

    def open_flex(uri):
        assert uri == "s3://alpha-engine-research/ops/daily-summaries/back-office/ib-flex/"
        return flex

    monkeypatch.setattr(job, "open_store", open_flex)
    assert run_step(backend)["value"] == 1


def test_empty_uri_is_fail_closed(world, monkeypatch):
    backend, _ = world
    monkeypatch.setenv("CRUCIBLE_TRADER_IB_FLEX_URI", "")
    row = run_step(backend)
    assert row["status"] == "FAIL"
    assert "empty" in row["status_reason"]


def test_delayed_cycle_never_backfills_its_own_day(world):
    backend, _ = world
    before = snapshot(backend.root)
    ctx = ctx_for(backend, dt.date(2026, 9, 9))
    job.apply_waiting_statements(ctx, now=NOW, corporate_actions=CA)
    assert not backend.exists(trader_reconciliation_key(DAY))
    assert snapshot(backend.root) == before
    assert ctx.metrics[0]["status"] == "FAIL"


def test_clock_is_refreshed_before_apply(world):
    backend, _ = world
    before = snapshot(backend.root)
    ctx = ctx_for(backend, dt.date(2026, 10, 2))
    job.apply_waiting_statements(
        ctx,
        now=dt.datetime(2026, 10, 2, 12, 59, tzinfo=dt.UTC),
        corporate_actions=CA,
        clock=lambda: dt.datetime(2026, 10, 2, 13, 0, tzinfo=dt.UTC),
    )
    assert ctx.metrics[0]["status"] == "FAIL"
    assert "09:00-16:30 ET" in ctx.metrics[0]["status_reason"]
    assert snapshot(backend.root) == before


def test_partial_previous_apply_stays_visible(world):
    backend, _ = world
    backend.put_bytes(trader_broker_statement_key(DAY), b"stored statement")
    before = snapshot(backend.root)
    row = run_step(backend)
    assert row["status"] == "FAIL" and "missing reconciliation" in row["status_reason"]
    assert snapshot(backend.root) == before


def test_replay_dependency_import_failure_is_recorded(world, monkeypatch):
    import builtins

    backend, _ = world
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "crucible_trader.statement_backfill":
            raise ImportError("replay dependency unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    row = run_step(backend)
    assert row["status"] == "FAIL" and "replay dependency unavailable" in row["status_reason"]


@pytest.mark.parametrize("outer_capture", [False, True])
def test_replay_preserves_live_dispatch_identity(world, monkeypatch, tmp_path, outer_capture):
    import os

    from crucible.llm import active_provider_ledger, begin_provider_capture, end_provider_capture

    backend, _ = world
    state = tmp_path / "dispatch"
    state.mkdir()
    marker = state / "manifest-key"
    marker.write_text("runs/trader.reconcile/2026-10-02/run.json")
    monkeypatch.setenv("CRUCIBLE_STATE_DIR", str(state))
    monkeypatch.setenv("KREPIS_RUN_ID", "live-run-id")
    end_provider_capture()
    original = begin_provider_capture() if outer_capture else None
    try:
        assert run_step(backend)["value"] == 1
        assert marker.read_text() == "runs/trader.reconcile/2026-10-02/run.json"
        assert os.environ["KREPIS_RUN_ID"] == "live-run-id"
        assert os.environ["CRUCIBLE_STATE_DIR"] == str(state)
        assert active_provider_ledger() is original
    finally:
        end_provider_capture()


def test_replay_identity_is_restored_on_interruption(monkeypatch, tmp_path):
    import os

    from crucible_trader.statement_backfill import isolated_replay

    monkeypatch.setenv("CRUCIBLE_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("CRUCIBLE_DISPATCH_ATTEMPTS", '[{"n":1,"reason":"initial"}]')
    monkeypatch.delenv("KREPIS_RUN_ID", raising=False)
    with pytest.raises(KeyboardInterrupt), isolated_replay():
        assert "CRUCIBLE_STATE_DIR" not in os.environ
        assert "CRUCIBLE_DISPATCH_ATTEMPTS" not in os.environ
        os.environ["KREPIS_RUN_ID"] = "nested"
        raise KeyboardInterrupt
    assert "KREPIS_RUN_ID" not in os.environ
    assert os.environ["CRUCIBLE_STATE_DIR"] == str(tmp_path)
    assert os.environ["CRUCIBLE_DISPATCH_ATTEMPTS"] == '[{"n":1,"reason":"initial"}]'


def test_existing_later_ledger_refuses_historical_insertion(world):
    from crucible_trader.ledger import journal_document, journal_key

    backend, _ = world
    tenant, later = "nous-ergon-paper", "2026-09-10"
    backend.put_bytes(journal_key(tenant, later), journal_document(tenant, later, ()))
    before = snapshot(backend.root)
    row = run_step(backend)
    assert row["status"] == "FAIL" and DAY in row["status_reason"]
    # The backfill refuses a day behind a live journal before its reconcile
    # runs (prior-period book, alpha-engine-config-I13232), naming the book.
    assert "is never inserted behind it" in row["status_reason"]
    assert "--book pre-v2" in row["status_reason"]
    assert snapshot(backend.root) == before


def test_replay_propagates_reclamation_without_retry(world):
    from crucible.runner import SpotInterruptionError

    from crucible_trader.statement_backfill import FiledStatementSource, isolated_replay

    backend, _ = world
    calls = []

    def interrupted(_ctx, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise SpotInterruptionError("reclaimed")

    with pytest.raises(SpotInterruptionError), isolated_replay():
        job.run_reconciliation(
            backend,
            FiledStatementSource(BrokerStatement("DU1", DAY, {"AAA": 10}, 1000.0)),
            trading_day=dt.date.fromisoformat(DAY),
            fills=(),
            cash_flows=(),
            corporate_actions=CA,
            run_mode="replay",
            body=interrupted,
        )
    assert len(calls) == 1
