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
from crucible_trader.exception_decision import SCORE_CURRENT_KEY, exception_score_key
from crucible_trader.exception_proposal import RULESET_VERSION, exception_proposals_key
from crucible_trader.exception_record import (
    CONTROL_ARM_VOID,
    exceptions_key,
    validate_exception_day_document,
)
from crucible_trader.ledger import journal_key, ledger_reconciliation_key
from crucible_trader.order_intent import claim_order_intent, order_intent_document
from crucible_trader.order_intent_reconciliation import (
    BrokerExecution,
    BrokerOrder,
    order_intent_reconciliation_key,
)
from crucible_trader.reconciliation import CorporateActionSet, Fill, reconcile
from crucible_trader.reconciliation_control import ReconciliationVoidError
from crucible_trader.reconciliation_job import (
    EXCEPTION_OVERDUE_METRIC,
    EXCEPTION_PROPOSALS_METRIC,
    EXCEPTION_RECORDS_METRIC,
    EXCEPTION_SCORE_METRIC,
    RECONCILE_JOB,
    ExceptionFilingError,
    ReconciliationDiscrepancyError,
    latest_anchor,
    previous_trading_day,
    reconcile_cycle,
    run_reconciliation,
    unpaged_cash_usd,
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
            # Step 5d (I13128): the ledger's journal and its reconciliation view.
            journal_key("nous-ergon-paper", "2026-09-11"),
            ledger_reconciliation_key("nous-ergon-paper", "2026-09-11"),
            # Step 5e (I13120): the back office's score, filed on a clean day too.
            exception_score_key("2026-09-11"),
            SCORE_CURRENT_KEY,
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
        # the statement, the result, the break as an exception record (I12892),
        # the shadow proposal for it (I13119), then the ledger's journal and
        # view (step 5d, I13128), then the back office's score (step 5e, I13120).
        assert [o["key"] for o in ctx.outputs][2:] == [
            exceptions_key(DAY.isoformat()),
            exception_proposals_key(DAY.isoformat()),
            journal_key("nous-ergon-paper", DAY.isoformat()),
            ledger_reconciliation_key("nous-ergon-paper", DAY.isoformat()),
            exception_score_key(DAY.isoformat()),
            SCORE_CURRENT_KEY,
        ]
        for metric in ctx.metrics:
            MetricRecordRow.model_validate(metric)
        assert {m["status"] for m in ctx.metrics} == {"FAIL", "OK", "WATCH"}

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
            journal_key("nous-ergon-paper", DAY.isoformat()),
            ledger_reconciliation_key("nous-ergon-paper", DAY.isoformat()),
            exception_score_key(DAY.isoformat()),
            SCORE_CURRENT_KEY,
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
        # The run for DAY reconciles the PREV book: its orders carry PREV.
        orphan = BrokerOrder(order_ref=f"crucible-trader|{PREV}|ZZZ|buy|1|1.0|t", status="Filled")
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

    def test_the_run_reconciles_the_book_its_session_traded(self, store):
        """alpha-engine-config-I13261: the session on DAY trades the PREV book and
        writes its intents (and orderRefs) under PREV. The run for DAY reads that
        book, once, and never DAY's own book, which the next session trades."""
        when = dt.datetime(2026, 9, 11, 14, 0, tzinfo=dt.UTC)

        def claim(book, symbol):
            ref = f"crucible-trader|{book}|{symbol}|buy|5|10.0|t"
            claim_order_intent(
                store,
                order_intent_document(
                    order_key=f"crucible-trader|{book}|{symbol}",
                    order_ref=ref,
                    trading_day=book,
                    symbol=symbol,
                    side="buy",
                    quantity=5,
                    decision_price=10.0,
                    account="DU1",
                    intended_at=when,
                ),
            )
            return ref

        ref = claim(PREV, "AAA")
        claim(DAY.isoformat(), "BBB")  # tomorrow's session: not this run's
        put_anchor(store)
        reconcile_cycle(
            ctx_for(store),
            source=FixedSource({"AAA": 12}, 800.0),
            fills=(Fill("AAA", 2, -200.0),),
            cash_flows=(),
            corporate_actions=CorporateActionSet(frozenset({"AAA"})),
            order_records=(
                [BrokerOrder(order_ref=ref, status="Filled")],
                [BrokerExecution(ref, 5.0, exec_id="e1", account="DU1", symbol="AAA", side="BOT")],
            ),
        )
        document = output(store, order_intent_reconciliation_key(DAY.isoformat()))
        assert (document["trading_day"], document["book_day"]) == (DAY.isoformat(), PREV)
        assert document["intents"] == 1 and document["counts"] == {"filled": 1}
        assert [r["symbol"] for r in document["rows"]] == ["AAA"]

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


class TestExceptionProposals:
    """Step 5c (`alpha-engine-config-I13119`, back office 1b): the shadow
    proposer runs over the records 5b filed and files its proposals, and a
    proposer failure is logged and swallowed -- every other output, the metrics
    5b and earlier wrote, and the page are exactly what they were."""

    def _discrepant(self, store, ctx):
        return TestExceptionRecords()._discrepant(store, ctx)

    def test_a_break_gets_a_filed_proposal(self, store):
        ctx = ctx_for(store)
        self._discrepant(store, ctx)
        (record,) = output(store, exceptions_key(DAY.isoformat()))["records"]
        day = output(store, exception_proposals_key(DAY.isoformat()))
        assert day["trading_day"] == DAY.isoformat()
        assert day["ruleset_version"] == RULESET_VERSION
        (proposal,) = day["proposals"]
        assert proposal["exception_id"] == record["exception_id"]
        assert proposal["tenant"] == record["tenant"] == "nous-ergon-paper"
        assert proposal["action"] == "investigate_shares"
        (metric,) = [m for m in ctx.metrics if m["name"] == EXCEPTION_PROPOSALS_METRIC]
        MetricRecordRow.model_validate(metric)
        assert metric["status"] == "OK" and metric["value"] == 1.0

    def test_a_clean_cycle_proposes_nothing(self, store):
        put_anchor(store)
        ctx, _ = cycle(store, FixedSource({"AAA": 12}, 800.0))
        assert not store.exists(exception_proposals_key(DAY.isoformat()))
        assert EXCEPTION_PROPOSALS_METRIC not in {m["name"] for m in ctx.metrics}

    @pytest.mark.parametrize("where", ["propose_day", "proposal_day_document", "write"])
    def test_a_proposer_failure_leaves_the_reconcile_unchanged(
        self, store, tmp_path, monkeypatch, caplog, where
    ):
        import crucible_trader.reconciliation_job as job

        # The job stamps `detected_at` and every metric from the wall clock, so
        # the two runs compared below must see the same clock.
        monkeypatch.setattr(job, "dt", _FrozenClock())

        # The baseline: the same day with the proposer working.
        base_ctx = ctx_for(store)
        base_page = self._discrepant(store, base_ctx)

        failing = LocalStore(tmp_path / "failing")
        ctx = ctx_for(failing)

        def boom(*_a, **_k):
            raise RuntimeError("proposer exploded")

        if where == "write":
            ctx.store = _NoProposalWrites(failing)
        else:
            monkeypatch.setattr(job, where, boom)
        with caplog.at_level("WARNING", logger=job.__name__):
            page = self._discrepant(failing, ctx)

        # The page, every output but the proposals file, and every metric but the
        # proposer's own are byte-for-byte the baseline's -- bar step 5e's score,
        # which counts the proposals and so differs when there are none.
        assert page == base_page
        proposals_key = exception_proposals_key(DAY.isoformat())
        derived = {proposals_key, exception_score_key(DAY.isoformat()), SCORE_CURRENT_KEY}
        assert [o for o in ctx.outputs if o["key"] not in derived] == [
            o for o in base_ctx.outputs if o["key"] not in derived
        ]
        assert not failing.exists(proposals_key)
        skip = {EXCEPTION_PROPOSALS_METRIC, EXCEPTION_SCORE_METRIC, EXCEPTION_OVERDUE_METRIC}
        others = [m for m in ctx.metrics if m["name"] not in skip]
        assert others == [m for m in base_ctx.metrics if m["name"] not in skip]
        # The failure is logged and reported, never raised.
        assert "were NOT filed" in caplog.text
        (metric,) = [m for m in ctx.metrics if m["name"] == EXCEPTION_PROPOSALS_METRIC]
        MetricRecordRow.model_validate(metric)
        assert metric["status"] == "FAIL" and metric["value"] is None

    def test_a_failing_metric_write_is_swallowed_too(self, store, monkeypatch):
        ctx = ctx_for(store)
        record_metric = ctx.record_metric

        def picky(metric):
            if metric["name"] == EXCEPTION_PROPOSALS_METRIC:
                raise ValueError("metric store down")
            record_metric(metric)

        monkeypatch.setattr(ctx, "record_metric", picky)
        page = self._discrepant(store, ctx)
        assert "share_delta: AAA" in page
        assert store.exists(exception_proposals_key(DAY.isoformat()))

    def test_a_failed_manifest_carries_the_proposals(self, store):
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
        assert exception_proposals_key(DAY.isoformat()) in {o["key"] for o in manifest["outputs"]}


class _FrozenClock:
    """`datetime` as the job module sees it, with `datetime.now` fixed."""

    class datetime(dt.datetime):  # noqa: N801 - stands in for `dt.datetime`
        @classmethod
        def now(cls, tz=None):  # pyright: ignore[reportImplicitOverride]
            return dt.datetime(2026, 9, 11, 21, 0, 5, tzinfo=tz)

    def __getattr__(self, name):
        return getattr(dt, name)


class _NoProposalWrites:
    """The store, except that writing a proposals file fails."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def put_bytes(self, key, payload):
        if key.startswith("trader/exception_proposals/"):
            raise PermissionError("denied")
        return self._inner.put_bytes(key, payload)


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


class TestCashTiers:
    """Brian 2026-11-02: up to 5 cents is rounding (no finding); a `cash_delta`
    up to $1 is filed and queued but does not page; above $1 pages; and the
    month's unpaged cash pages once it passes the cap."""

    NOV = dt.date(2026, 11, 4)

    def run(self, store, cash, *, positions=None, prior=None):
        put_anchor(store, day="2026-11-03")
        for day, document in (prior or {}).items():
            store.put_bytes(reconciliation_key(day), json.dumps(document).encode())
        ctx = ctx_for(store, day=self.NOV)
        result = reconcile_cycle(
            ctx,
            source=FixedSource(positions or {"AAA": 12}, cash),
            fills=(Fill("AAA", 2, -200.0),),
            cash_flows=(),
            corporate_actions=CorporateActionSet(frozenset({"AAA"})),
        )
        return ctx, result

    def test_a_low_band_gap_is_filed_and_watched_but_does_not_page(self, store):
        ctx, result = self.run(store, 800.40)
        assert not result.clean and result.discrepancies[0].klass == "cash_delta"
        assert exceptions_key(self.NOV.isoformat()) in [o["key"] for o in ctx.outputs]
        statuses = {m["name"]: m["status"] for m in ctx.metrics}
        assert statuses["broker_reconciliation_discrepancies"] == "WATCH"
        assert statuses["broker_reconciliation_unpaged_cash_mtd_usd"] == "OK"
        unpaged = next(
            m for m in ctx.metrics if m["name"] == "broker_reconciliation_unpaged_cash_mtd_usd"
        )
        assert unpaged["value"] == pytest.approx(0.40)
        for metric in ctx.metrics:
            MetricRecordRow.model_validate(metric)

    def test_a_low_band_gap_that_could_not_be_filed_pages(self, store):
        """codex 185d9463: the queue is an unpaged gap's only record, so a gap
        whose record was not written fails the run instead of succeeding."""
        put_anchor(store, day="2026-11-03")
        ctx = ctx_for(store, day=self.NOV)
        ctx.store = _NoExceptionWrites(store)
        with pytest.raises(ExceptionFilingError, match="in no queue") as raised:
            reconcile_cycle(
                ctx,
                source=FixedSource({"AAA": 12}, 800.40),
                fills=(Fill("AAA", 2, -200.0),),
                cash_flows=(),
                corporate_actions=CorporateActionSet(frozenset({"AAA"})),
            )
        assert "cash_delta" in str(raised.value) and "NOT filed" in str(raised.value)
        assert isinstance(raised.value, ReconciliationDiscrepancyError)
        # Every later step still ran: the reconciliation and the ledger are written.
        written = [o["key"] for o in ctx.outputs]
        assert reconciliation_key(self.NOV.isoformat()) in written
        assert not store.exists(exceptions_key(self.NOV.isoformat()))

    def test_a_filed_low_band_gap_still_succeeds(self, store):
        ctx, result = self.run(store, 800.40)
        assert not result.clean
        assert {m["status"] for m in ctx.metrics if m["name"] == EXCEPTION_RECORDS_METRIC} == {"OK"}

    def test_exactly_one_dollar_does_not_page_and_a_cent_more_does(self, store, tmp_path):
        self.run(store, 801.00)
        with pytest.raises(ReconciliationDiscrepancyError, match="no-page band"):
            self.run(LocalStore(tmp_path / "other"), 801.01)

    def test_a_low_band_gap_beside_any_other_finding_pages(self, store):
        with pytest.raises(ReconciliationDiscrepancyError, match="share_delta: AAA"):
            self.run(store, 800.40, positions={"AAA": 11})

    def test_the_monthly_cap_pages_on_many_small_gaps(self, store):
        prior = {
            "2026-11-02": {"cash_rounded_to_broker_usd": -0.05, "discrepancies": []},
            "2026-11-03": {
                "cash_rounded_to_broker_usd": None,
                "discrepancies": [{"klass": "cash_delta", "delta": -0.90}] * 5,
            },
            # Another month never counts toward November's cap.
            "2026-10-30": {
                "cash_rounded_to_broker_usd": 0.0,
                "discrepancies": [{"klass": "cash_delta", "delta": 0.99}] * 9,
            },
        }
        with pytest.raises(ReconciliationDiscrepancyError, match=r"5\.05 USD, over the 5\.00"):
            self.run(store, 800.50, prior=prior)

    def test_unpaged_cash_reads_stored_results_defensively(self):
        assert unpaged_cash_usd({}) == 0.0
        assert unpaged_cash_usd(
            {
                "cash_rounded_to_broker_usd": 0.03,
                "discrepancies": [
                    {"klass": "cash_delta", "delta": -0.5},
                    {"klass": "cash_delta", "delta": 4.0},
                    {"klass": "cash_delta", "delta": None},
                    {"klass": "share_delta", "delta": -0.5},
                    "not a mapping",
                ],
            }
        ) == pytest.approx(0.53)
        assert unpaged_cash_usd({"discrepancies": "not a list"}) == 0.0


class TestExceptionScore:
    """Step 5e (alpha-engine-config-I13120): the back office's score and queue."""

    def test_a_clean_day_files_an_empty_score(self, store):
        put_anchor(store)
        ctx, _ = cycle(store, FixedSource({"AAA": 12}, 800.0))
        document = output(store, exception_score_key(DAY.isoformat()))
        assert document["auto_apply"] is False
        assert document["totals"]["proposals"] == 0 and document["queue"] == []
        (metric,) = [m for m in ctx.metrics if m["name"] == EXCEPTION_SCORE_METRIC]
        MetricRecordRow.model_validate(metric)
        assert metric["status"] == "OK" and metric["value"] == 0.0
        (overdue,) = [m for m in ctx.metrics if m["name"] == EXCEPTION_OVERDUE_METRIC]
        MetricRecordRow.model_validate(overdue)
        assert overdue["status"] == "OK" and overdue["value"] == 0.0

    def test_a_break_is_queued_and_a_decision_scores_it(self, store, tmp_path):
        from crucible_trader.exception_decision import decide, record_decision

        TestExceptionRecords()._discrepant(store, ctx_for(store))
        document = output(store, exception_score_key(DAY.isoformat()))
        (item,) = document["queue"]
        assert item["status"] == "undecided" and item["age_trading_days"] == 0
        (proposal,) = output(store, exception_proposals_key(DAY.isoformat()))["proposals"]
        record_decision(
            store,
            decide(
                proposal, verdict="agree", decided_by="brian", decided_at="2026-09-11T22:00:00Z"
            ),
        )
        rerun = LocalStore(tmp_path / "rerun")
        for key in store.list_keys("trader/"):
            if key.startswith(("trader/exception_decisions/", "trader/exception_proposals/")):
                rerun.put_bytes(key, store.get_bytes(key))
        ctx = ctx_for(rerun)
        TestExceptionRecords()._discrepant(rerun, ctx)
        document = output(rerun, exception_score_key(DAY.isoformat()))
        assert document["queue"] == []
        assert document["totals"]["agreement"] == 1.0
        (metric,) = [m for m in ctx.metrics if m["name"] == EXCEPTION_SCORE_METRIC]
        assert metric["status"] == "OK" and "agreement 1.0" in metric["status_reason"]

    def test_an_overdue_item_is_watch(self, store):
        later = dt.date(2026, 9, 18)
        TestExceptionRecords()._discrepant(store, ctx_for(store))
        ctx = ctx_for(store, day=later)
        from crucible_trader.reconciliation_job import file_exception_score

        file_exception_score(ctx, now="2026-09-18T21:00:00Z")
        (metric,) = [m for m in ctx.metrics if m["name"] == EXCEPTION_SCORE_METRIC]
        assert metric["status"] == "WATCH" and "1 overdue" in metric["status_reason"]
        # alpha-engine-config-I13129: the overdue row is the page signal.
        (overdue,) = [m for m in ctx.metrics if m["name"] == EXCEPTION_OVERDUE_METRIC]
        MetricRecordRow.model_validate(overdue)
        assert overdue["status"] == "FAIL" and overdue["value"] == 1.0
        assert "(2026-09-11, 5 trading days)" in overdue["status_reason"]
        assert output(store, SCORE_CURRENT_KEY)["trading_day"] == "2026-09-18"
        # A re-run of an EARLIER day files its own score and leaves `current` alone.
        earlier = ctx_for(store)
        file_exception_score(earlier, now="2026-09-18T22:00:00Z")
        assert output(store, SCORE_CURRENT_KEY)["trading_day"] == "2026-09-18"
        assert [o["key"] for o in earlier.outputs] == [exception_score_key(DAY.isoformat())]

    def test_a_later_score_published_mid_run_is_never_overwritten(self, store, monkeypatch):
        # codex 458b09f0e5dafeb6: an earlier day's run read `current`, a later
        # day's score landed, and the earlier run's write then replaced it.
        from crucible_trader.reconciliation_job import file_exception_score

        TestExceptionRecords()._discrepant(store, ctx_for(store))
        later = json.dumps({"trading_day": "2026-09-18"}).encode()
        real_get = store.get_bytes

        def racing_get(key):
            body = real_get(key)
            if key == SCORE_CURRENT_KEY and json.loads(body)["trading_day"] != "2026-09-18":
                store.put_bytes(SCORE_CURRENT_KEY, later)  # the later run wins the race
            return body

        file_exception_score(ctx_for(store), now="2026-09-18T22:00:00Z")  # seed current
        monkeypatch.setattr(store, "get_bytes", racing_get)
        file_exception_score(ctx_for(store), now="2026-09-18T22:05:00Z")
        monkeypatch.undo()
        assert output(store, SCORE_CURRENT_KEY)["trading_day"] == "2026-09-18"

    def test_exhausted_score_races_are_a_fail_not_a_filing(self, store, monkeypatch):
        # codex 200669767300e21a: three lost compare-and-swaps fell through
        # and the step reported the score filed although `current` was not.
        from crucible.store import PointerConflictError

        from crucible_trader.reconciliation_job import file_exception_score

        TestExceptionRecords()._discrepant(store, ctx_for(store))
        ctx = ctx_for(store)

        def always_lose(*_a, **_k):
            raise PointerConflictError("raced")

        monkeypatch.setattr(ctx, "record_output_cas", always_lose)
        file_exception_score(ctx, now="2026-09-11T22:05:00Z")
        (metric,) = [m for m in ctx.metrics if m["name"] == EXCEPTION_SCORE_METRIC]
        assert metric["status"] == "FAIL" and "lost the race 3 times" in metric["status_reason"]

    def test_a_score_failure_leaves_the_reconcile_unchanged(self, store, monkeypatch, caplog):
        import crucible_trader.reconciliation_job as job

        def boom(*_a, **_k):
            raise RuntimeError("scorer exploded")

        monkeypatch.setattr(job, "score", boom)
        ctx = ctx_for(store)
        with caplog.at_level("WARNING", logger=job.__name__):
            page = TestExceptionRecords()._discrepant(store, ctx)
        assert "share_delta: AAA" in page
        assert not store.exists(exception_score_key(DAY.isoformat()))
        assert "score for 2026-09-11 was NOT filed" in caplog.text
        (metric,) = [m for m in ctx.metrics if m["name"] == EXCEPTION_SCORE_METRIC]
        MetricRecordRow.model_validate(metric)
        assert metric["status"] == "FAIL" and metric["value"] is None
        # No score, no overdue reading: the FAIL above is the only row.
        assert EXCEPTION_OVERDUE_METRIC not in [m["name"] for m in ctx.metrics]

    def test_more_overdue_items_than_the_reason_names_are_counted(self):
        from crucible_trader.reconciliation_job import _overdue_metric

        queue = [
            {
                "proposal_id": f"p{i}",
                "trading_day": "2026-09-01",
                "age_trading_days": 9,
                "overdue": True,
            }
            for i in range(7)
        ] + [
            {
                "proposal_id": "new",
                "trading_day": "2026-09-11",
                "age_trading_days": 0,
                "overdue": False,
            }
        ]
        row = _overdue_metric(
            {"queue": queue, "age_limit_trading_days": 3}, day="2026-09-11", key="k", now="t"
        )
        assert row["status"] == "FAIL" and row["value"] == 7.0
        assert "p4 (" in row["status_reason"] and "p5 (" not in row["status_reason"]
        assert row["status_reason"].endswith("; and 2 more")

    def test_a_failing_overdue_metric_write_is_swallowed(self, store, monkeypatch, caplog):
        import crucible_trader.reconciliation_job as job

        ctx = ctx_for(store)
        record_metric = ctx.record_metric

        def picky(metric):
            if metric["name"] == EXCEPTION_OVERDUE_METRIC:
                raise ValueError("metric store down")
            record_metric(metric)

        monkeypatch.setattr(ctx, "record_metric", picky)
        with caplog.at_level("WARNING", logger=job.__name__):
            page = TestExceptionRecords()._discrepant(store, ctx)
        assert "share_delta: AAA" in page
        assert "overdue metric for 2026-09-11 was NOT recorded" in caplog.text
        assert EXCEPTION_SCORE_METRIC in [m["name"] for m in ctx.metrics]

    def test_a_failing_metric_write_is_swallowed_too(self, store, monkeypatch, caplog):
        import crucible_trader.reconciliation_job as job

        ctx = ctx_for(store)
        record_metric = ctx.record_metric

        def picky(metric):
            if metric["name"] == EXCEPTION_SCORE_METRIC:
                raise ValueError("metric store down")
            record_metric(metric)

        monkeypatch.setattr(ctx, "record_metric", picky)
        with caplog.at_level("WARNING", logger=job.__name__):
            page = TestExceptionRecords()._discrepant(store, ctx)
        assert "share_delta: AAA" in page
        assert store.exists(exception_score_key(DAY.isoformat()))
        assert "score metric for 2026-09-11 was NOT recorded" in caplog.text
