"""Back office stage 1 gate, as an automated test (`alpha-engine-config-I12892`).

nous-ergon-ops `plans/2026-11-02-back-office-implementation.md` § Schedule,
stage 1's gate: "the trial balance is zero and the roll-forward ties for every
trading day since the v2 cutover; one synthetic day goes end to end, from
break to proposal to decision to score."

Everything runs through the real job body (:func:`reconcile_cycle`, steps 5
to 5g) over a local store, with IB and the settled closes faked. Nothing here
touches AWS. The month is November 2026 so the run on 12-01 closes it:

* 11-25 genesis: 10 AAA and $1,000;
* 11-27: buy 5 BBB at $20 with a $1 commission; in the break test IB shows 50
  cents more cash than the fill explains;
* 11-30: sell 4 AAA at $12 with a $1 commission.

A reviewer's agree or correct clears the break (`alpha-engine-config-I13160`):
the next posting books the adjustment that moves it out of suspense, and the
close then finishes. A defer leaves it in suspense and the close refused. A
decision on the refused close's own suspense line clears the gap that line
reports, exactly as a decision on the reconciler's record does.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pytest
from crucible.keys import trader_reconciliation_key
from crucible.runner import RunContext
from crucible.store import LocalStore

from crucible_trader import (
    month_end_close,
    month_rebuild,
    performance_statement,
    reconciliation_job,
)
from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.exception_decision import (
    SCORE_CURRENT_KEY,
    decide,
    exception_score_key,
    record_decision,
)
from crucible_trader.exception_proposal import close_proposals_key, exception_proposals_key
from crucible_trader.exception_record import PAPER_FUND_TENANT, exceptions_key
from crucible_trader.ledger import (
    RECONCILIATION_ADJUSTMENTS,
    SUSPENSE_CASH,
    close_suspense_line_ids,
    journal_key,
    ledger_reconciliation_key,
    load_journals,
)
from crucible_trader.month_end_close import close_key
from crucible_trader.month_rebuild import rebuild_key
from crucible_trader.performance_statement import statement_key
from crucible_trader.reconciliation import CorporateActionSet, Fill
from crucible_trader.reconciliation_job import RECONCILE_JOB, reconcile_cycle

T = PAPER_FUND_TENANT
DAYS = ("2026-11-25", "2026-11-27", "2026-11-30")
ACTIONS = CorporateActionSet(resolved=frozenset({"AAA", "BBB"}))
CLOSES = {
    "2026-11-25": {"AAA": 10.0, "SPY": 500.0, "QQQ": 400.0},
    "2026-11-30": {"AAA": 11.0, "BBB": 21.0, "SPY": 510.0, "QQQ": 404.0},
}


def closes(day, tickers):
    return {t: CLOSES.get(day, {})[t] for t in tickers if t in CLOSES.get(day, {})}


def fill(ticker, shares, price, commission, exec_id):
    return Fill(
        ticker, shares, -shares * price - commission, commission_usd=commission, exec_id=exec_id
    )


class Ib:
    """IB's statement for the day the job asks about."""

    def __init__(self, books):
        self.books = books

    def read_statement(self, trading_day):
        positions, cash = self.books[trading_day]
        return BrokerStatement("DU1", trading_day, positions, cash)


@pytest.fixture
def store(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> LocalStore:
    # The settled closes the close, the statement and the rebuild read.
    for module in (month_end_close, performance_statement, month_rebuild):
        monkeypatch.setattr(module, "panel_close_source", lambda _store: closes)
    # The real fund's declared dividend entitlements are not this synthetic book's.
    monkeypatch.setattr(reconciliation_job, "load_declared_entitlements", lambda: {})
    return LocalStore(tmp_path)


def run_day(store, ib, day, fills=()):
    ctx = RunContext(
        run_id="01JG0000000000000000000000",
        job=RECONCILE_JOB,
        trading_day=dt.date.fromisoformat(day),
        calendar_date=dt.date.fromisoformat(day),
        store=store,
        seed=0,
        started=dt.datetime.fromisoformat(f"{day}T21:00:00+00:00"),
    )
    reconcile_cycle(
        ctx,
        source=ib,
        fills=tuple(fills),
        cash_flows=(),
        corporate_actions=ACTIONS,
        genesis=day == DAYS[0],
    )
    return ctx


def november(store, *, extra_cash=0.0):
    ib = Ib(
        {
            DAYS[0]: ({"AAA": 10}, 1_000.0),
            DAYS[1]: ({"AAA": 10, "BBB": 5}, 899.0 + extra_cash),
            DAYS[2]: ({"AAA": 6, "BBB": 5}, 946.0 + extra_cash),
            "2026-12-01": ({"AAA": 6, "BBB": 5}, 946.0 + extra_cash),
            "2026-12-02": ({"AAA": 6, "BBB": 5}, 946.0 + extra_cash),
        }
    )
    ctxs = {
        DAYS[0]: run_day(store, ib, DAYS[0]),
        DAYS[1]: run_day(store, ib, DAYS[1], [fill("BBB", 5, 20.0, 1.0, "e1")]),
    }
    return ib, ctxs


def read(store, key):
    return json.loads(store.get_bytes(key))


def metric(ctx, name):
    (found,) = [m for m in ctx.metrics if m["name"] == name]
    return found


def assert_every_day_balances_and_rolls_forward(store, days):
    """Gate clause 1: trial balance zero and the roll-forward ties, every day."""
    for day in days:
        view = read(store, ledger_reconciliation_key(T, day))
        assert view["trial_balance_cents"] == 0, day
        assert view["checks"]["roll_forward"]["ties"] is True, day
        assert view["checks"]["breaks"] == [], (day, view["checks"]["breaks"])
        assert view["checks"]["completeness"]["missing"] == [], day


def test_one_synthetic_break_goes_from_break_to_proposal_to_decision_to_score(store):
    ib, ctxs = november(store, extra_cash=0.5)

    # BREAK: the reconciler files the cash gap as an exception record ...
    (record,) = read(store, exceptions_key(DAYS[1]))["records"]
    assert record["class"] == "cash_delta" and record["tenant"] == T
    # ... and the ledger books it to named suspense citing that record.
    (gap,) = [e for e in load_journals(store, T, before=DAYS[2])[DAYS[1]] if e.kind == "cash_gap"]
    assert record["exception_id"] in gap.evidence["exception_ids"]
    assert gap.evidence["decision_id"] is None
    assert [p.amount_cents for p in gap.postings if p.account == SUSPENSE_CASH] == [-50]

    # PROPOSAL: the shadow proposer files one proposal for that record ...
    (proposal,) = read(store, exception_proposals_key(DAYS[1]))["proposals"]
    assert proposal["exception_id"] == record["exception_id"]
    # ... and it waits in the review queue, undecided.
    queued = read(store, SCORE_CURRENT_KEY)
    assert queued["trading_day"] == DAYS[1]
    assert [q["proposal_id"] for q in queued["queue"]] == [proposal["proposal_id"]]
    assert queued["totals"]["queue"] == 1 and queued["totals"]["agreement"] is None

    # DECISION: the reviewer, never the proposer, agrees with it.
    decision = decide(
        proposal, verdict="agree", decided_by="reviewer", decided_at="2026-11-28T15:00:00Z"
    )
    record_decision(store, decision)

    # SCORE: the next run's step 5e scores the call and empties the queue.
    ctx = run_day(store, ib, DAYS[2], [fill("AAA", -4, 12.0, 1.0, "e2")])
    scored = read(store, exception_score_key(DAYS[2]))
    assert scored == read(store, SCORE_CURRENT_KEY)
    assert scored["queue"] == []
    assert scored["totals"]["queue"] == 0 and scored["totals"]["agreement"] == 1.0
    (klass,) = [c for c in scored["classes"] if c["class"] == "cash_delta"]
    assert (klass["agree"], klass["consecutive_agree"]) == (1, 1)
    assert metric(ctx, "exception_score_filed")["status"] == "OK"

    # CLEARED: the same posting books the adjustment the decision calls for --
    # its own entry, citing the break's exception ids and the decision id.
    (adjustment,) = [
        e for e in load_journals(store, T, before="2026-12-01")[DAYS[2]] if e.kind == "adjustment"
    ]
    assert adjustment.evidence["exception_ids"] == gap.evidence["exception_ids"]
    assert adjustment.evidence["decision_id"] == decision["decision_id"]
    assert {(p.account, p.amount_cents) for p in adjustment.postings} == {
        (SUSPENSE_CASH, 50),
        (RECONCILIATION_ADJUSTMENTS, -50),
    }

    # Gate clause 1 holds through the break and its clearing.
    assert_every_day_balances_and_rolls_forward(store, DAYS)
    view = read(store, ledger_reconciliation_key(T, DAYS[2]))
    assert view["suspense"] == {"cash_cents": 0, "shares": {}} and view["clean"] is True
    assert view["checks"]["open_breaks"] == []

    # November closes, the 50 cents attributed to the decided break.
    december = run_day(store, ib, "2026-12-01")
    assert_every_day_balances_and_rolls_forward(store, (*DAYS, "2026-12-01"))
    close = read(store, close_key(T, "2026-11"))
    assert close["status"] == "closed" and close["exceptions"] == []
    assert close["pnl"]["reconciliation_adjustments_cents"] == 50
    assert close["pnl"]["suspense_cents"] == 0
    assert close["pnl"]["total_cents"] == close["nav_end_cents"] - close["nav_start_cents"]
    for name in ("month_end_close", "performance_statement", "month_rebuild"):
        assert metric(december, name)["status"] == "OK", name


def test_a_deferred_break_holds_the_close_until_an_agreement_clears_it(store):
    ib, _ = november(store, extra_cash=0.5)
    (proposal,) = read(store, exception_proposals_key(DAYS[1]))["proposals"]
    deferred = decide(
        proposal,
        verdict="defer",
        decided_by="reviewer",
        decided_at="2026-11-28T15:00:00Z",
        note="waiting for IB's activity statement",
    )
    record_decision(store, deferred)
    run_day(store, ib, DAYS[2], [fill("AAA", -4, 12.0, 1.0, "e2")])
    (gap,) = [e for e in load_journals(store, T, before=DAYS[2])[DAYS[1]] if e.kind == "cash_gap"]

    # Deferred: the gap stays in suspense and November does not close.
    view = read(store, ledger_reconciliation_key(T, DAYS[2]))
    assert view["suspense"]["cash_cents"] == -50 and view["clean"] is False
    (open_break,) = view["checks"]["open_breaks"]
    assert (open_break["entry_id"], open_break["age"]) == (gap.entry_id, 1)
    refused = run_day(store, ib, "2026-12-01")
    attempt = metric(refused, "month_end_close")
    assert attempt["status"] == "WATCH" and "close_suspense_cash" in attempt["status_reason"]
    assert not store.exists(close_key(T, "2026-11"))

    # The reviewer agrees, superseding the deferral: the next posting clears it
    # with a prior-period adjustment dated in the open month, and the retried
    # close finishes.
    agreed = decide(
        proposal,
        verdict="agree",
        decided_by="reviewer",
        decided_at="2026-12-01T22:00:00Z",
        supersedes=deferred["decision_id"],
    )
    record_decision(store, agreed)
    closed = run_day(store, ib, "2026-12-02")
    (adjustment,) = load_journals(store, T, before="2026-12-03")["2026-12-02"]
    assert adjustment.kind == "adjustment"
    assert adjustment.evidence["decision_id"] == agreed["decision_id"]
    assert_every_day_balances_and_rolls_forward(store, (*DAYS, "2026-12-01", "2026-12-02"))
    close = read(store, close_key(T, "2026-11"))
    assert close["status"] == "closed"
    assert close["suspense_cleared_after_month_end"] == [gap.entry_id]
    assert close["pnl"]["total_cents"] == close["nav_end_cents"] - close["nav_start_cents"]
    assert metric(closed, "month_end_close")["status"] == "OK"


def test_a_decision_on_the_close_line_clears_the_gap_it_reports(store):
    ib, _ = november(store, extra_cash=0.5)
    run_day(store, ib, DAYS[2], [fill("AAA", -4, 12.0, 1.0, "e2")])
    (gap,) = [e for e in load_journals(store, T, before=DAYS[2])[DAYS[1]] if e.kind == "cash_gap"]

    # Nobody decided the reconciler's record, so the close refuses on the gap
    # and files a proposal for its close_suspense_cash line ...
    refused = run_day(store, ib, "2026-12-01")
    assert metric(refused, "month_end_close")["status"] == "WATCH"
    proposals = read(store, close_proposals_key(T, "2026-11", "2026-12-01"))["proposals"]
    (line,) = [p for p in proposals if p["class"] == "close_suspense_cash"]
    # ... whose exception id is the one the ledger derives for that gap's line,
    # and is NOT among the ids the gap cites.
    assert close_suspense_line_ids(gap, "2026-11-30") == [line["exception_id"]]
    assert line["exception_id"] not in gap.evidence["exception_ids"]

    # The reviewer decides the close line, not the daily record.
    decision = decide(
        line, verdict="agree", decided_by="reviewer", decided_at="2026-12-01T22:00:00Z"
    )
    record_decision(store, decision)
    ib.books["2026-12-03"] = ib.books["2026-12-02"]
    closed = run_day(store, ib, "2026-12-02")
    run_day(store, ib, "2026-12-03")

    # The next posting clears exactly that gap, once: the adjustment cites the
    # gap's own ids (so it settles it whichever line was decided) and names the
    # decided close line; the day after books nothing more.
    journals = load_journals(store, T, before="2026-12-04")
    (adjustment,) = journals["2026-12-02"]
    assert adjustment.kind == "adjustment"
    assert adjustment.evidence["exception_ids"] == gap.evidence["exception_ids"]
    assert adjustment.evidence["decided_exception_id"] == line["exception_id"]
    assert adjustment.evidence["decision_id"] == decision["decision_id"]
    assert adjustment.evidence["decided_by"] == "reviewer"
    assert not [e for e in journals.get("2026-12-03", ()) if e.kind == "adjustment"]
    assert_every_day_balances_and_rolls_forward(
        store, (*DAYS, "2026-12-01", "2026-12-02", "2026-12-03")
    )
    view = read(store, ledger_reconciliation_key(T, "2026-12-03"))
    assert view["suspense"] == {"cash_cents": 0, "shares": {}} and view["clean"] is True

    # The retried close finishes, the gap cleared after month end.
    close = read(store, close_key(T, "2026-11"))
    assert close["status"] == "closed" and close["exceptions"] == []
    assert close["suspense_cleared_after_month_end"] == [gap.entry_id]
    assert close["pnl"]["total_cents"] == close["nav_end_cents"] - close["nav_start_cents"]
    assert metric(closed, "month_end_close")["status"] == "OK"


def test_a_clean_month_closes_files_its_statement_and_rebuilds_from_raw(store):
    ib, _ = november(store)
    run_day(store, ib, DAYS[2], [fill("AAA", -4, 12.0, 1.0, "e2")])
    ctx = run_day(store, ib, "2026-12-01")
    assert_every_day_balances_and_rolls_forward(store, (*DAYS, "2026-12-01"))
    for day in (*DAYS, "2026-12-01"):
        assert store.exists(journal_key(T, day)) and store.exists(trader_reconciliation_key(day))

    close = read(store, close_key(T, "2026-11"))
    assert close["status"] == "closed" and close["trial_balance_cents"] == 0
    assert close["pnl"]["total_cents"] == close["nav_end_cents"] - close["nav_start_cents"]
    statement = read(store, statement_key(T, "2026-11"))
    assert statement["simulated_fee_bill"]["booked"] is False

    rebuilt = read(store, rebuild_key(T, "2026-11"))
    assert rebuilt["matches"] is True and rebuilt["breaks"] == []
    assert sorted(rebuilt["documents"]) == ["close", "fee_bill", "statement"]
    for name in ("month_end_close", "performance_statement", "month_rebuild"):
        assert metric(ctx, name)["status"] == "OK", name
