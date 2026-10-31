"""The job: I/O, outputs, MetricRecords, raises, and the manifest the harness
writes for it through its own validator. IB is faked (`FixedSource`)."""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pytest
from crucible.keys import trader_broker_statement_key as broker_statement_key
from crucible.keys import trader_reconciliation_key as reconciliation_key
from crucible.manifest import on_money_path, read_manifest
from crucible.models import JOB_VALUES, MetricRecordRow
from crucible.runner import RunContext
from crucible.store import LocalStore

from crucible_trader.broker_statement import BrokerStatement, BrokerStatementError
from crucible_trader.exception_record import (
    CONTROL_ARM_VOID,
    exceptions_key,
    validate_exception_day_document,
)
from crucible_trader.order_intent_reconciliation import BrokerExecution, BrokerOrder
from crucible_trader.reconciliation import CorporateActionSet, Fill, reconcile
from crucible_trader.reconciliation_control import ReconciliationVoidError
from crucible_trader.reconciliation_job import (
    EXCEPTION_RECORDS_METRIC,
    RECONCILE_JOB,
    ReconciliationDiscrepancyError,
    latest_anchor,
    previous_trading_day,
    reconcile_cycle,
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
        # Outputs are written BEFORE the raise, so the failed manifest carries them:
        # the statement, the result, and the break as an exception record (I12892).
        assert [o["key"] for o in ctx.outputs][2] == exceptions_key(DAY.isoformat())
        assert len(ctx.outputs) == 3
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


class TestTheHarnessWritesThisJobsManifest:
    """crucible-PR297 registered `trader.reconcile` and put its result on the
    money-path predicates; these assert the written manifest, not the constants."""

    def _run(self, store, broker_positions):
        put_anchor(store)
        return run_reconciliation(
            store,
            FixedSource(broker_positions, 800.0),
            trading_day=DAY,
            fills=(Fill("AAA", 2, -200.0),),
            cash_flows=(),
            corporate_actions=CorporateActionSet(frozenset({"AAA"})),
            run_mode="replay",
        )

    def test_the_job_is_admitted_and_its_result_is_on_the_money_path(self):
        assert RECONCILE_JOB in JOB_VALUES
        assert on_money_path(reconciliation_key(DAY.isoformat()))

    def test_a_clean_cycle_writes_a_schema_valid_manifest_carrying_the_chain_link(self, store):
        self._run(store, {"AAA": 12})
        manifest = read_manifest(store, RECONCILE_JOB, DAY.isoformat())  # validates on read
        assert manifest["status"] == "ok" and manifest["job"] == RECONCILE_JOB
        link = manifest["money_path_link"]
        assert link["index"] == 0 and link["prev_sha256"] is None
        assert reconciliation_key(DAY.isoformat()) in link["money_path_writes"]
        assert {o["key"] for o in manifest["outputs"]} == {
            broker_statement_key(DAY.isoformat()),
            reconciliation_key(DAY.isoformat()),
        }

    def test_a_discrepancy_is_a_failed_manifest_that_still_chains(self, store):
        with pytest.raises(ReconciliationDiscrepancyError):
            self._run(store, {"AAA": 13})
        manifest = read_manifest(store, RECONCILE_JOB, DAY.isoformat())
        assert manifest["status"] == "failed"
        assert "AAA" in manifest["reason"]
        assert (
            reconciliation_key(DAY.isoformat()) in manifest["money_path_link"]["money_path_writes"]
        )


class TestThePhase4GateGradesThisReconciliation:
    """`run_reconciliation` is the producer; `broker_reconciliation_control_arm_passed`
    is the consumer. Five sessions written here through the real runner, read back
    through `crucible.gate.evaluate` as shipped in the pinned wheel
    (`alpha-engine-config-I9760`, `-I10413`)."""

    CLAUSE = "broker_reconciliation_control_arm_passed"
    # The five NYSE sessions ending 2026-09-11; 2026-09-07 is Labor Day.
    WEEK = ("2026-09-04", "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11")

    def _week(self, store, *, run_mode="live", broken=None):
        """``broken``: from that session on the broker holds one more share than
        any fill explains -- a real, persistent break, found on the day it
        appears; the next session anchors on the statement that recorded it."""
        put_anchor(store, day="2026-09-03")
        for day in self.WEEK:
            held = {"AAA": 11} if broken is not None and day >= broken else {"AAA": 10}
            try:
                run_reconciliation(
                    store,
                    FixedSource(held, 1_000.0),
                    trading_day=dt.date.fromisoformat(day),
                    fills=(),
                    cash_flows=(),
                    corporate_actions=CorporateActionSet(frozenset({"AAA"})),
                    run_mode=run_mode,
                )
            except ReconciliationDiscrepancyError:
                assert day == broken

    def test_a_trader_that_never_reconciled_is_a_graded_unmet(self, store, phase4_clause):
        clause = phase4_clause(store, self.CLAUSE, self.WEEK[-1])
        assert not clause.met and not clause.unmeasurable
        assert "no `trader.reconcile` manifest for any of the 5 sessions" in clause.detail

    def test_five_clean_live_sessions_with_the_control_arm_passing_are_met(
        self, store, phase4_clause
    ):
        self._week(store)
        clause = phase4_clause(store, self.CLAUSE, self.WEEK[-1])
        assert clause.met, clause.detail
        for day in self.WEEK:
            assert f"runs/{RECONCILE_JOB}/{day}/run.json" in clause.evidence
            assert reconciliation_key(day) in clause.evidence

    def test_a_discrepancy_on_one_session_is_unmet_naming_it(self, store, phase4_clause):
        self._week(store, broken="2026-09-09")
        clause = phase4_clause(store, self.CLAUSE, self.WEEK[-1])
        assert not clause.met
        assert f"runs/{RECONCILE_JOB}/2026-09-09/run.json: status `failed`" in clause.detail

    def test_a_replayed_week_reconciles_no_live_book(self, store, phase4_clause):
        self._week(store, run_mode="replay")
        clause = phase4_clause(store, self.CLAUSE, self.WEEK[-1])
        assert not clause.met
        assert "run_mode `replay`" in clause.detail


class TestExceptionRecords:
    """Step 5b (`alpha-engine-config-I12892` phase 0): every break is filed as an
    `exception.v1` record beside the page, and the page is exactly what it was."""

    def _discrepant(self, store, ctx, **kw):
        put_anchor(store)
        with pytest.raises(ReconciliationDiscrepancyError) as raised:
            reconcile_cycle(
                ctx,
                source=FixedSource({"AAA": 9}, 800.0),
                fills=(Fill("AAA", 2, -200.0),),
                cash_flows=(),
                corporate_actions=CorporateActionSet(frozenset({"AAA"})),
                **kw,
            )
        return str(raised.value)

    def test_a_break_is_filed_beside_an_unchanged_page(self, store, tmp_path):
        page = self._discrepant(store, ctx_for(store))
        day = output(store, exceptions_key(DAY.isoformat()))
        validate_exception_day_document(day)
        (record,) = day["records"]
        assert record["class"] == "share_delta" and record["amount"] == -3.0
        assert record["detected_by"]["manifest_key"] == f"runs/{RECONCILE_JOB}/2026-09-11/run.json"
        # The page is byte-identical to the one a run without step 5b raises.
        other = LocalStore(tmp_path / "other")
        ctx = ctx_for(other)
        ctx.store = _NoExceptionWrites(other)
        assert self._discrepant(other, ctx) == page

    def test_a_clean_cycle_files_nothing_and_adds_no_metric(self, store):
        put_anchor(store)
        ctx, _ = cycle(store, FixedSource({"AAA": 12}, 800.0))
        assert not store.exists(exceptions_key(DAY.isoformat()))
        assert EXCEPTION_RECORDS_METRIC not in {m["name"] for m in ctx.metrics}

    def test_the_metric_counts_what_was_filed(self, store):
        ctx = ctx_for(store)
        self._discrepant(store, ctx)
        (metric,) = [m for m in ctx.metrics if m["name"] == EXCEPTION_RECORDS_METRIC]
        MetricRecordRow.model_validate(metric)
        assert metric["status"] == "OK" and metric["value"] == 1.0

    def test_a_failed_write_never_masks_the_page(self, store):
        ctx = ctx_for(store)
        ctx.store = _NoExceptionWrites(store)
        page = self._discrepant(store, ctx)
        assert "share_delta: AAA" in page and "delta=-3" in page
        assert not store.exists(exceptions_key(DAY.isoformat()))
        (metric,) = [m for m in ctx.metrics if m["name"] == EXCEPTION_RECORDS_METRIC]
        MetricRecordRow.model_validate(metric)
        assert metric["status"] == "FAIL" and metric["value"] is None
        assert (
            "NOT filed" in metric["status_reason"] and "PermissionError" in metric["status_reason"]
        )

    def test_a_void_run_files_the_void_first(self, store):
        put_anchor(store)
        ctx = ctx_for(store)

        def blind(inputs):
            import dataclasses

            return dataclasses.replace(reconcile(inputs), discrepancies=())

        with pytest.raises(ReconciliationVoidError):
            reconcile_cycle(
                ctx,
                source=FixedSource({"AAA": 99}, 0.0),
                fills=(),
                cash_flows=(),
                corporate_actions=CorporateActionSet(frozenset({"AAA"})),
                reconciler=blind,
            )
        (record,) = output(store, exceptions_key(DAY.isoformat()))["records"]
        assert record["class"] == CONTROL_ARM_VOID and record["detected_by"]["void"] is True

    def test_order_intent_breaks_join_the_same_day_file(self, store):
        put_anchor(store)
        ctx = ctx_for(store)
        orphan = BrokerOrder(
            order_ref="crucible-trader|2026-09-11|ZZZ|buy|1|1.0|t", status="Filled"
        )
        reconcile_cycle(
            ctx,
            source=FixedSource({"AAA": 12}, 800.0),
            fills=(Fill("AAA", 2, -200.0),),
            cash_flows=(),
            corporate_actions=CorporateActionSet(frozenset({"AAA"})),
            order_records=([orphan], list[BrokerExecution]()),
        )
        (record,) = output(store, exceptions_key(DAY.isoformat()))["records"]
        assert record["class"] == "orphan" and record["instrument"] == "ZZZ"

    def test_a_raising_intent_step_still_files_the_broker_breaks(self, store, monkeypatch):
        import crucible_trader.reconciliation_job as job

        def boom(*_a, **_k):
            raise RuntimeError("unreadable intent")

        monkeypatch.setattr(job, "intents_for_day", boom)
        put_anchor(store)
        with pytest.raises(RuntimeError, match="unreadable intent"):
            reconcile_cycle(
                ctx_for(store),
                source=FixedSource({"AAA": 9}, 800.0),
                fills=(Fill("AAA", 2, -200.0),),
                cash_flows=(),
                corporate_actions=CorporateActionSet(frozenset({"AAA"})),
                order_records=([], []),
            )
        (record,) = output(store, exceptions_key(DAY.isoformat()))["records"]
        assert record["class"] == "share_delta"

    def test_a_failed_manifest_carries_the_record_and_still_validates(self, store):
        put_anchor(store)
        with pytest.raises(ReconciliationDiscrepancyError):
            run_reconciliation(
                store,
                FixedSource({"AAA": 13}, 800.0),
                trading_day=DAY,
                fills=(Fill("AAA", 2, -200.0),),
                cash_flows=(),
                corporate_actions=CorporateActionSet(frozenset({"AAA"})),
                run_mode="replay",
            )
        manifest = read_manifest(store, RECONCILE_JOB, DAY.isoformat())  # validates on read
        assert manifest["status"] == "failed" and "share_delta: AAA" in manifest["reason"]
        assert exceptions_key(DAY.isoformat()) in {o["key"] for o in manifest["outputs"]}


class _NoExceptionWrites:
    """The store, except that writing an exception record fails."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def put_bytes(self, key, payload):
        if key.startswith("trader/exceptions/"):
            raise PermissionError("denied")
        return self._inner.put_bytes(key, payload)
