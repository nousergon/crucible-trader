"""The back office's offline dry run of one month (alpha-engine-config-I13124):
a month reconciled by the real job through the real runner, then stripped of
its ledger and exception files as October is, and run end to end without one
write reaching the store."""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import runpy
import shutil
import sys

import pytest
from crucible.keys import trader_broker_statement_key, trader_reconciliation_key
from crucible.store import ETAG_ABSENT, LocalStore, open_store

from crucible_trader import back_office_dry_run
from crucible_trader.back_office_dry_run import dry_run, load_fills, main
from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.exception_decision import (
    decide,
    load_decisions,
    load_proposals,
    record_decision,
    score,
)
from crucible_trader.exception_proposal import close_proposals_key, exception_proposals_key
from crucible_trader.exception_record import PAPER_FUND_TENANT
from crucible_trader.ledger import journal_key
from crucible_trader.month_end_close import close_attempt_key
from crucible_trader.reconciliation import CorporateActionSet, Fill
from crucible_trader.reconciliation_job import ReconciliationDiscrepancyError, run_reconciliation

ANCHOR = "2026-09-08"
CLEAN, BROKEN, LATER = "2026-09-09", "2026-09-10", "2026-09-11"
NOW = "2026-11-03T05:30:00Z"
T = PAPER_FUND_TENANT


class FixedSource:
    def __init__(self, positions):
        self.positions = positions

    def read_statement(self, trading_day):
        return BrokerStatement("DU1", trading_day, self.positions, 1_000.0)


@pytest.fixture(autouse=True)
def _code_sha(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_CODE_SHA", "0123456789abcdef0123456789abcdef01234567")


@pytest.fixture
def root(tmp_path) -> pathlib.Path:
    root = tmp_path / "store"
    store = LocalStore(root)
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
    # As October is: reconciled, but before the ledger and the exception files.
    for prefix in ("ledger", "exceptions", "exception_proposals", "exception_scores"):
        shutil.rmtree(root / "trader" / prefix)
    return root


def snapshot(root: pathlib.Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class TestDryRun:
    def test_the_month_runs_end_to_end_and_writes_nothing_to_the_store(
        self, root, tmp_path, capsys
    ):
        before = snapshot(root)
        out = tmp_path / "out"
        assert (
            main(["--store", str(root), "--month", "2026-09", "--out", str(out), "--now", NOW]) == 0
        )
        assert snapshot(root) == before

        summary = json.loads((out / "summary.json").read_text())
        assert summary["days_with_reconciliation"] == [CLEAN, BROKEN, LATER]
        assert "2026-09-01" in summary["days_without_reconciliation"]
        assert summary["export_failures"] == []
        assert summary["records_by_class"]["share_delta"] == 1
        assert summary["close_run_day"] == "2026-10-01"
        assert summary["close"]["status"] == "refused"  # no statement for 09-30
        assert summary["close_metrics"][0]["name"] == "month_end_close"
        assert {row["trading_day"] for row in summary["ledger"]} == {CLEAN, BROKEN, LATER}
        for key in (
            exception_proposals_key(BROKEN),
            journal_key(T, LATER),
            close_attempt_key(T, "2026-09", "2026-10-01"),
            close_proposals_key(T, "2026-09", "2026-10-01"),
        ):
            assert key in summary["keys_written"]
            assert (out / "store" / key).is_file()

        # The queue a reviewer works from is a local store, readable as one.
        queued = load_proposals(LocalStore(out / "store"))
        assert len(queued) == summary["score_totals"]["proposals"] > 0
        assert "2026-09:" in capsys.readouterr().err

    def test_the_out_tree_carries_the_review_evidence_the_score_read(self, root, tmp_path):
        """codex 00e291d3: the score reads proposals and decisions the store
        already held, so the exported tree must hold them too, or a re-score
        from it differs from the summary's."""
        first = tmp_path / "first"
        assert (
            main(["--store", str(root), "--month", "2026-09", "--out", str(first), "--now", NOW])
            == 0
        )
        # The store already holds an earlier filing of a proposal and a
        # reviewer's decision on it, as the live store does once review starts.
        earlier = exception_proposals_key("2026-08-31")
        (root / earlier).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(first / "store" / exception_proposals_key(BROKEN), root / earlier)
        proposal = load_proposals(LocalStore(first / "store"))[0]
        decided = record_decision(
            LocalStore(root),
            decide(proposal, verdict="agree", decided_by="brian", decided_at=NOW),
        )

        out = tmp_path / "out"
        assert (
            main(["--store", str(root), "--month", "2026-09", "--out", str(out), "--now", NOW]) == 0
        )
        summary = json.loads((out / "summary.json").read_text())
        assert summary["review_evidence_keys"] == sorted([earlier, decided])
        assert summary["score_totals"]["agree"] == 1
        tree = LocalStore(out / "store")
        assert (out / "store" / decided).is_file() and (out / "store" / earlier).is_file()
        rescored = score(
            load_proposals(tree, until=summary["close_run_day"]),
            load_decisions(tree),
            as_of=summary["close_run_day"],
            generated_at=NOW,
        )
        assert rescored["totals"] == summary["score_totals"]

    def test_review_evidence_skips_what_the_run_wrote_and_what_the_score_did_not_read(self, root):
        store = LocalStore(root)
        for key in (
            exception_proposals_key("2026-09-09"),
            exception_proposals_key("2026-10-02"),
            "trader/exception_proposals/notes.txt",
            "trader/exception_decisions/2026-09-09/readme.txt",
        ):
            store.put_bytes(key, b"{}")
        assert back_office_dry_run.review_evidence_keys(
            store, until="2026-10-01", written=[exception_proposals_key("2026-09-10")]
        ) == [exception_proposals_key("2026-09-09")]
        assert (
            back_office_dry_run.review_evidence_keys(
                store, until="2026-10-01", written=[exception_proposals_key("2026-09-09")]
            )
            == []
        )

    def test_a_day_that_cannot_export_is_listed_and_the_run_goes_on(self, root):
        (root / "runs" / "trader.reconcile" / LATER / "run.json").unlink()
        summary = dry_run(open_store(str(root), dry_run=True), month="2026-09", now=NOW)
        assert [f["trading_day"] for f in summary["export_failures"]] == [LATER]
        assert summary["close"] is not None

    def test_fills_are_booked_and_fills_on_unreconciled_days_are_named(self, root, tmp_path):
        path = tmp_path / "fills.jsonl"
        rows = [
            {"trading_day": BROKEN, "ticker": "AAA", "shares": 1, "cash_usd": 0.0},
            {
                "trading_day": "2026-09-15",
                "ticker": "AAA",
                "shares": 1,
                "cash_usd": -10.0,
                "commission_usd": 1.0,
                "exec_id": "e1",
            },
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
        fills = load_fills(path)
        assert fills["2026-09-15"][0].exec_id == "e1"
        summary = dry_run(
            open_store(str(root), dry_run=True), month="2026-09", now=NOW, fills=fills
        )
        assert summary["fills_on_days_without_reconciliation"] == ["2026-09-15"]
        assert summary["fills_booked_on_a_later_day"] == {}

    def test_a_fill_on_an_unreconciled_day_is_booked_on_the_next_posted_day(self, root):
        # Without 09-10's reconciliation, 09-11 is posted against 09-09's
        # statement, so the share bought on 09-10 belongs in 09-11's posting.
        (root / trader_reconciliation_key(BROKEN)).unlink()
        fill = Fill("AAA", 1, -10.0, commission_usd=1.0, exec_id="v1:t1")

        def run(fills):
            summary = dry_run(
                open_store(str(root), dry_run=True), month="2026-09", now=NOW, fills=fills
            )
            share_gaps = [
                e["instrument"]
                for e in summary["close"]["exceptions"]
                if e["class"] == "close_suspense_shares"
            ]
            return summary, share_gaps

        summary, share_gaps = run({BROKEN: [fill], CLEAN: []})
        assert summary["days_with_reconciliation"] == [CLEAN, LATER]
        assert summary["fills_on_days_without_reconciliation"] == [BROKEN]
        assert summary["fills_booked_on_a_later_day"] == {BROKEN: LATER}
        assert share_gaps == []
        unbooked, unbooked_gaps = run({})
        assert unbooked["fills_booked_on_a_later_day"] == {}
        assert unbooked_gaps == [f"AAA@{LATER}"]

    def test_the_first_stored_day_opens_from_the_statement_before_it(self, root):
        # codex 36dcff7b: 09-09 opens from 09-08's statement, so its own fills
        # are booked rather than read as already inside its closing statement.
        store = open_store(str(root), dry_run=True)
        dry_run(store, month="2026-09", now=NOW)
        entries = json.loads(store.get_bytes(journal_key(T, CLEAN)))["entries"]
        opening = [e for e in entries if e["kind"] == "opening_balance"]
        assert [e["event_ref"] for e in opening] == [f"statement/{ANCHOR}"]

    def test_a_carried_fill_whose_posting_failed_is_not_reported_booked(self, root, monkeypatch):
        # codex d8b42301: booked_on is recorded only after post_daily_ledger succeeds.
        (root / trader_reconciliation_key(BROKEN)).unlink()
        real = back_office_dry_run.post_daily_ledger

        def post_failing_on_later(ctx, **kw):
            return None if kw["broker"].trading_day == LATER else real(ctx, **kw)

        monkeypatch.setattr(back_office_dry_run, "post_daily_ledger", post_failing_on_later)
        fill = Fill("AAA", 1, -10.0, commission_usd=1.0, exec_id="v1:t1")
        summary = dry_run(
            open_store(str(root), dry_run=True), month="2026-09", now=NOW, fills={BROKEN: [fill]}
        )
        assert summary["fills_booked_on_a_later_day"] == {}


class TestRefusals:
    def test_a_bad_fill_line_names_its_line(self, tmp_path):
        path = tmp_path / "fills.jsonl"
        path.write_text('{"trading_day": "2026-09-10"}\n')
        with pytest.raises(ValueError, match="fills.jsonl:1: not a fill"):
            load_fills(path)
        assert load_fills(None) == {}

    def test_a_store_that_could_write_is_refused(self, root):
        with pytest.raises(ValueError, match="capturing store"):
            dry_run(LocalStore(root), month="2026-09", now=NOW)

    def test_a_month_with_nothing_stored_is_refused(self, root):
        with pytest.raises(ValueError, match="no stored reconciliation"):
            dry_run(open_store(str(root), dry_run=True), month="2026-08", now=NOW)

    def test_a_used_out_directory_is_refused(self, root, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "old").write_text("x")
        with pytest.raises(SystemExit):
            main(["--store", str(root), "--month", "2026-09", "--out", str(out)])


class TestJournalsAsOf:
    """The store as the close run day saw it (`alpha-engine-config-I13124`)."""

    def test_a_month_replays_into_a_store_whose_ledger_has_moved_on(self, root):
        # As October is now: the live ledger opened on a later day, so the
        # store already holds a journal dated after the month's close run day.
        LocalStore(root).put_bytes(journal_key(T, "2026-10-02"), b"{}")
        summary = dry_run(open_store(str(root), dry_run=True), month="2026-09", now=NOW)
        assert {row["trading_day"] for row in summary["ledger"]} == {CLEAN, BROKEN, LATER}
        assert not any("NOT posted" in str(row["reason"]) for row in summary["ledger"])
        assert journal_key(T, LATER) in summary["keys_written"]
        assert summary["close"] is not None
        assert summary["close"]["status"] == "refused"  # no statement for 09-30, as without

    def test_only_journals_after_the_day_are_hidden_and_writes_pass_through(self, tmp_path):
        inner = LocalStore(tmp_path / "s")
        on_day, after = journal_key(T, "2026-10-01"), journal_key(T, "2026-10-02")
        other = trader_broker_statement_key("2026-10-02")
        for key in (on_day, after, other):
            inner.put_bytes(key, b"{}")
        view = back_office_dry_run._JournalsAsOf(inner, "2026-10-01")
        assert sorted(view.list_keys("trader/")) == sorted([on_day, other])
        assert not view.exists(after)
        assert view.exists(on_day)
        assert view.etag(after) == ETAG_ABSENT
        assert view.etag(on_day) == inner.etag(on_day)
        with pytest.raises(KeyError):
            view.get_bytes(after)
        assert view.get_bytes(other) == b"{}"
        key = journal_key(T, "2026-09-30")
        view.put_bytes(key, b"[]")
        assert inner.get_bytes(key) == b"[]"
        assert view.compare_and_swap(key, inner.etag(key), b"[1]")
        assert inner.get_bytes(key) == b"[1]"
        assert view.presigned_url(key, 60) == inner.presigned_url(key, 60)


def test_the_close_runs_on_the_next_months_first_trading_day():
    assert back_office_dry_run._next_trading_day("2026-10-30") == "2026-11-02"  # pyright: ignore[reportPrivateUsage]


def test_runs_as_a_module(root, tmp_path, monkeypatch):
    out = tmp_path / "module-out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["back_office_dry_run", "--store", str(root), "--month", "2026-09", "--out", str(out)],
    )
    sys.modules.pop("crucible_trader.back_office_dry_run", None)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("crucible_trader.back_office_dry_run", run_name="__main__")
    assert exited.value.code == 0 and (out / "summary.json").is_file()


class TestTheRunDaysOwnJournal:
    """codex facd7a3e: the run day is the next month's first trading day. Once
    the live ledger has posted it, step 3 must not see it, or every replayed
    day sits behind it and is refused; the close, run ON that day, sees it."""

    def test_a_live_journal_on_the_run_day_does_not_refuse_the_month(self, root):
        LocalStore(root).put_bytes(journal_key(T, "2026-10-01"), b"{}")
        summary = dry_run(open_store(str(root), dry_run=True), month="2026-09", now=NOW)
        assert summary["close_run_day"] == "2026-10-01"
        assert {row["trading_day"] for row in summary["ledger"]} == {CLEAN, BROKEN, LATER}
        assert not any("already exist" in str(row["reason"]) for row in summary["ledger"])
        assert journal_key(T, LATER) in summary["keys_written"]

    def test_the_run_day_is_hidden_only_when_asked(self, tmp_path):
        inner = LocalStore(tmp_path / "s")
        on_day = journal_key(T, "2026-10-01")
        inner.put_bytes(on_day, b"{}")
        booking = back_office_dry_run._JournalsAsOf(inner, "2026-10-01", include_as_of=False)
        assert not booking.exists(on_day)
        assert back_office_dry_run._JournalsAsOf(inner, "2026-10-01").exists(on_day)


class TestTheQueueCarriesItsRecords:
    """codex 09f52012: step 5 scores with each proposal's exception record, as
    the job's step 5f does, so the queue's amount, tier and citations are filled."""

    def test_step_five_passes_the_exception_records(self, root, monkeypatch):
        seen = {}
        real = back_office_dry_run.score

        def spy(*args, **kwargs):
            seen["exceptions"] = kwargs.get("exceptions")
            return real(*args, **kwargs)

        monkeypatch.setattr(back_office_dry_run, "score", spy)
        summary = dry_run(open_store(str(root), dry_run=True), month="2026-09", now=NOW)
        assert summary["records"] > 0
        exceptions = seen["exceptions"]
        assert exceptions is not None and len(exceptions) == summary["records"]

    def test_the_review_evidence_carries_the_records_the_score_read(self, root):
        from crucible_trader.exception_record import exceptions_key

        store = LocalStore(root)
        store.put_bytes(
            exception_proposals_key("2026-09-09"),
            json.dumps(
                {"trading_day": "2026-09-09", "proposals": [{"trading_day": "2026-09-09"}]}
            ).encode(),
        )
        store.put_bytes(exceptions_key("2026-09-09"), b"{}")
        keys = back_office_dry_run.review_evidence_keys(store, until="2026-10-01", written=[])
        assert exceptions_key("2026-09-09") in keys


def test_backfills_run_in_date_order_between_the_postings(root, monkeypatch):
    """codex d411bf68: each backfill runs where its day falls among the stored
    days' postings, never all of them first."""
    order: list[str] = []
    real = back_office_dry_run.post_daily_ledger

    def post(ctx, **kw):
        order.append(f"post {kw['broker'].trading_day}")
        return real(ctx, **kw)

    monkeypatch.setattr(back_office_dry_run, "post_daily_ledger", post)
    backfill = {
        day: (lambda _store, d=day: order.append(f"backfill {d}"))
        for day in ("2026-10-05", "2026-08-31", "2026-09-30")
    }
    # 09-30's backfill reconciled nothing (its reconcile failed): a day without one.
    dry_run(open_store(str(root), dry_run=True), month="2026-09", now=NOW, backfill=backfill)
    assert order == [
        "backfill 2026-08-31",
        f"post {CLEAN}",
        f"post {BROKEN}",
        f"post {LATER}",
        "backfill 2026-09-30",
        "backfill 2026-10-05",
    ]


def test_a_month_whose_only_backfill_reconciled_nothing_is_refused(tmp_path):
    with pytest.raises(ValueError, match="backfilled or stored"):
        dry_run(
            open_store(str(tmp_path / "empty"), dry_run=True),
            month="2026-09",
            now=NOW,
            backfill={"2026-09-09": lambda _store: None},
        )
