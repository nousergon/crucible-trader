"""The job: I/O, outputs, MetricRecords, raises, and the two recorded harness
contract dependencies. IB is faked (`FixedSource`)."""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pytest
from crucible.manifest import ManifestValidationError, on_money_path
from crucible.models import JOB_VALUES, MetricRecordRow
from crucible.runner import RunContext
from crucible.store import LocalStore

from crucible_trader.broker_statement import BrokerStatement, BrokerStatementError
from crucible_trader.reconciliation import CorporateActionSet, Fill, reconcile
from crucible_trader.reconciliation_control import ReconciliationVoidError
from crucible_trader.reconciliation_job import (
    RECONCILE_JOB,
    ReconciliationDiscrepancyError,
    broker_statement_key,
    latest_anchor,
    previous_trading_day,
    reconcile_cycle,
    reconciliation_key,
    run_reconciliation,
)

DAY = dt.date(2026, 9, 11)
PREV = "2026-09-10"


class FixedSource:
    def __init__(self, positions, cash):
        self.positions, self.cash = positions, cash

    def read_statement(self, trading_day):
        return BrokerStatement("DU1", trading_day, self.positions, self.cash)


@pytest.fixture
def store(tmp_path: pathlib.Path) -> LocalStore:
    return LocalStore(tmp_path)


def ctx_for(store, day=DAY):
    now = dt.datetime(2026, 9, 11, 21, 0, tzinfo=dt.UTC)
    return RunContext(
        run_id="01JG0000000000000000000000",
        job=RECONCILE_JOB,
        trading_day=day,
        calendar_date=day,
        store=store,
        seed=0,
        started=now,
    )


def put_anchor(store, day=PREV, positions=None, cash=1_000.0):
    s = BrokerStatement("DU1", day, positions or {"AAA": 10}, cash)
    store.put_bytes(broker_statement_key(day), json.dumps(s.to_document()).encode())


def cycle(store, source, **kw):
    ctx = ctx_for(store)
    kw.setdefault("fills", (Fill("AAA", 2, -200.0),))
    kw.setdefault("cash_flows", ())
    kw.setdefault("corporate_actions", CorporateActionSet(resolved=frozenset({"AAA"})))
    return ctx, reconcile_cycle(ctx, source=source, **kw)


def output(store, key):
    return json.loads(store.get_bytes(key))


class TestCleanCycle:
    def test_writes_statement_result_metrics_and_records_the_anchor_input(self, store):
        put_anchor(store)
        ctx, result = cycle(store, FixedSource({"AAA": 12}, 800.0))
        assert result.clean
        assert [i["key"] for i in ctx.inputs] == [broker_statement_key(PREV)]
        assert [o["key"] for o in ctx.outputs] == [
            broker_statement_key("2026-09-11"),
            reconciliation_key("2026-09-11"),
        ]
        document = output(store, reconciliation_key("2026-09-11"))
        assert document["void"] is False and document["control_arm"]["passed"] is True
        for metric in ctx.metrics:
            MetricRecordRow.model_validate(metric)  # the harness's own row contract
        assert {m["status"] for m in ctx.metrics} == {"OK"}

    def test_tomorrow_anchors_on_todays_statement(self, store):
        put_anchor(store)
        cycle(store, FixedSource({"AAA": 12}, 800.0))
        key, _, anchor = latest_anchor(store, "2026-09-14")
        assert key == broker_statement_key("2026-09-11") and anchor.positions == {"AAA": 12}

    def test_rerun_same_day_does_not_anchor_on_itself(self, store):
        put_anchor(store)
        cycle(store, FixedSource({"AAA": 12}, 800.0))
        _, result = cycle(store, FixedSource({"AAA": 12}, 800.0))
        assert result.clean


class TestFailures:
    def test_discrepancy_raises_a_page_naming_instrument_delta_side(self, store):
        put_anchor(store)
        ctx = ctx_for(store)
        with pytest.raises(ReconciliationDiscrepancyError) as raised:
            reconcile_cycle(
                ctx,
                source=FixedSource({"AAA": 9}, 800.0),
                fills=(Fill("AAA", 2, -200.0),),
                cash_flows=(),
                corporate_actions=CorporateActionSet(frozenset({"AAA"})),
            )
        assert "share_delta: AAA" in str(raised.value) and "delta=-3" in str(raised.value)
        assert "side=broker_under" in str(raised.value)
        # Outputs are written BEFORE the raise, so the failed manifest carries them.
        assert len(ctx.outputs) == 2
        for metric in ctx.metrics:
            MetricRecordRow.model_validate(metric)
        assert {m["status"] for m in ctx.metrics} == {"FAIL", "OK"}

    def test_first_ever_run_without_genesis_fails_no_anchor(self, store):
        with pytest.raises(ReconciliationDiscrepancyError, match="no_anchor"):
            cycle(store, FixedSource({"AAA": 1}, 1.0))

    def test_declared_genesis_is_ok_with_na_metrics_not_passes(self, store):
        ctx, result = cycle(store, FixedSource({"AAA": 1}, 1.0), fills=(), genesis=True)
        assert result.genesis
        by_name = {m["name"]: m for m in ctx.metrics}
        assert by_name["broker_reconciliation_match_rate"]["status"] == "N/A-LOW-N"
        assert by_name["broker_reconciliation_match_rate"]["value"] is None
        assert (
            by_name["broker_reconciliation_position_count_delta"]["status"] == "N/A-MISSING-INPUT"
        )
        assert by_name["broker_reconciliation_control_detected"]["status"] == "OK"
        for metric in ctx.metrics:
            MetricRecordRow.model_validate(metric)

    def test_na_rate_beside_a_finding_names_it(self, store):
        put_anchor(store, positions={"AAA": 1})
        ctx = ctx_for(store)
        with pytest.raises(ReconciliationDiscrepancyError):
            reconcile_cycle(
                ctx,
                source=FixedSource({"AAA": 1}, 1_000.0),
                fills=(),
                cash_flows=(),
                corporate_actions=CorporateActionSet(frozenset()),
            )
        rate = next(m for m in ctx.metrics if m["name"] == "broker_reconciliation_match_rate")
        assert (
            rate["status"] == "N/A-LOW-N"
            and "corporate_action_check_incomplete" in rate["status_reason"]
        )

    def test_a_blinded_reconciler_voids_the_cycle_before_any_discrepancy(self, store):
        put_anchor(store)
        ctx = ctx_for(store)

        def blind(inputs):
            import dataclasses

            return dataclasses.replace(reconcile(inputs), discrepancies=())

        with pytest.raises(ReconciliationVoidError, match="VOID"):
            reconcile_cycle(
                ctx,
                source=FixedSource({"AAA": 99}, 0.0),
                fills=(),
                cash_flows=(),
                corporate_actions=CorporateActionSet(frozenset({"AAA"})),
                reconciler=blind,
            )
        assert output(store, reconciliation_key("2026-09-11"))["void"] is True
        control = next(
            m for m in ctx.metrics if m["name"] == "broker_reconciliation_control_detected"
        )
        assert control["status"] == "FAIL" and control["value"] == 0.0

    def test_a_corrupt_anchor_raises_rather_than_falling_back_to_an_older_one(self, store):
        put_anchor(store, day="2026-09-09")
        store.put_bytes(broker_statement_key(PREV), b'{"schema_version": "nope"}')
        with pytest.raises(BrokerStatementError):
            latest_anchor(store, "2026-09-11")

    def test_non_json_keys_under_the_prefix_are_not_statements(self, store):
        store.put_bytes("trader/broker_statements/README.txt", b"x")
        assert latest_anchor(store, "2026-09-11") is None


def test_previous_trading_day_walks_the_calendar():
    assert previous_trading_day(dt.date(2026, 9, 14)).isoformat() == "2026-09-11"  # weekend
    assert previous_trading_day(dt.date(2026, 9, 8)).isoformat() == "2026-09-04"  # Labor Day


class TestRecordedHarnessContractDependencies:
    """Both assertions are EXPECTED to break when `nousergon/crucible` lands its
    half; the failure is the instruction to bump the pin and replace each with a
    positive end-to-end test. They are not suppressions: each pins a gap the PR
    names, so it cannot close silently."""

    def test_the_manifest_contract_does_not_yet_admit_this_job(self, store):
        assert RECONCILE_JOB not in JOB_VALUES, (
            "crucible now admits trader.reconcile: bump the pin, delete this test, and "
            "assert run_reconciliation writes a valid runs/trader.reconcile/{day}/run.json"
        )
        put_anchor(store)
        with pytest.raises(ManifestValidationError, match="job"):
            run_reconciliation(
                store,
                FixedSource({"AAA": 12}, 800.0),
                trading_day=DAY,
                fills=(Fill("AAA", 2, -200.0),),
                cash_flows=(),
                corporate_actions=CorporateActionSet(frozenset({"AAA"})),
                run_mode="replay",
            )

    def test_the_result_is_not_yet_on_the_money_path_chain(self):
        assert not on_money_path(reconciliation_key("2026-09-11")), (
            "crucible's MONEY_PATH_PREDICATES now match the reconciliation key: the chain "
            "link is attached by the harness writer, so assert it on the written manifest"
        )
