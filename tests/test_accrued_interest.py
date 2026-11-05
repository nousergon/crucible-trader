"""Accrued interest in the ledger and the close (`alpha-engine-config-I13202`).

IB's NetLiquidation includes interest accrued and not yet posted to cash. The
ledger holds that accrual as a receivable, one entry per change, and the close
counts it in NAV and in the attribution. All figures are synthetic.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pytest
from crucible.keys import trader_broker_statement_key
from crucible.runner import RunContext
from crucible.store import LocalStore

from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.exception_record import PAPER_FUND_TENANT
from crucible_trader.ledger import (
    CASH,
    INTEREST_FIRST_RECORDED,
    INTEREST_INCOME,
    INTEREST_RECEIVABLE,
    OPENING,
    DayInputs,
    Entry,
    Ledger,
    book_day,
    journal_key,
    post_daily_ledger,
)
from crucible_trader.month_end_close import CloseInputs, close_month, month_trading_days
from crucible_trader.reconciliation import CorporateActionSet, ReconciliationInputs, reconcile

T = PAPER_FUND_TENANT
NONE = CorporateActionSet(resolved=frozenset({"AAA"}))
D0, D1, D2, D3 = "2026-11-24", "2026-11-25", "2026-11-27", "2026-11-30"


def stmt(day, cash, accrued=None, account="DU1"):
    return BrokerStatement(account, day, {"AAA": 10}, cash, accrued_cash_usd=accrued)


def recon(broker, anchor):
    result = reconcile(
        ReconciliationInputs(
            trading_day=broker.trading_day,
            broker=broker,
            anchor=anchor,
            fills=(),
            cash_flows=(),
            corporate_actions=NONE,
            genesis=anchor is None,
        )
    )
    return {**result.to_document(), "control_arm": {"passed": True, "missed": []}, "void": False}


def inputs(broker, anchor, opening=None):
    return DayInputs(T, broker, anchor, (), (), NONE, recon(broker, anchor), (), opening)


def kinds(posting):
    return [(e.kind, e.event_ref) for e in posting.entries]


def legs(entry: Entry) -> dict[str, int]:
    return {p.account: p.amount_cents for p in entry.postings}


class TestTheLedgerHoldsIBsAccrual:
    def test_the_opening_accrual_is_opening_equity_and_a_rise_is_income(self):
        ledger = Ledger()
        first = book_day(ledger, inputs(stmt(D1, 1_000.0, 2.50), stmt(D0, 1_000.0, 2.00)))
        assert kinds(first) == [
            ("opening_balance", f"statement/{D0}"),
            ("interest_accrual", f"accrual/opening/{D0}"),
            ("interest_accrual", f"accrual/{D1}"),
        ]
        assert legs(first.entries[1]) == {INTEREST_RECEIVABLE: 200, OPENING: -200}
        assert legs(first.entries[2]) == {INTEREST_RECEIVABLE: 50, INTEREST_INCOME: -50}
        assert first.document["interest_receivable_cents"] == 250
        assert first.document["broker_accrued_interest_cents"] == 250
        assert first.document["clean"] is True
        assert ledger.trial_balance() == 0

    def test_a_fall_is_the_posting_to_cash_and_ties_when_ib_posted_the_balance(self):
        ledger = Ledger()
        d1 = stmt(D1, 1_000.0, 2.50)
        book_day(ledger, inputs(d1, None))
        posted = stmt(D2, 1_002.10, 0.40)
        second = book_day(ledger, inputs(posted, d1))
        assert kinds(second) == [("interest_settlement", f"accrual/{D2}")]
        assert legs(second.entries[0]) == {CASH: 210, INTEREST_RECEIVABLE: -210}
        assert ledger.cents(T, "DU1", INTEREST_RECEIVABLE) == 40
        assert second.document["clean"] is True

    def test_a_posting_that_differs_from_the_fall_is_that_days_cash_gap(self):
        ledger = Ledger()
        d1 = stmt(D1, 1_000.0, 2.50)
        book_day(ledger, inputs(d1, None))
        second = book_day(ledger, inputs(stmt(D2, 1_002.50, 0.40), d1))
        assert [k for k, _ in kinds(second)] == ["interest_settlement", "cash_gap"]
        assert second.document["suspense"]["cash_cents"] == -40

    def test_genesis_opens_with_todays_accrual_and_books_no_income(self):
        ledger = Ledger()
        posting = book_day(ledger, inputs(stmt(D1, 1_000.0, 2.50), None))
        assert kinds(posting) == [
            ("opening_balance", f"statement/{D1}"),
            ("interest_accrual", f"accrual/opening/{D1}"),
        ]

    def test_an_unrecorded_opening_books_the_first_recorded_balance_on_its_own_account(self):
        ledger = Ledger()
        v1 = stmt(D1, 1_000.0)  # broker_statement.v1: accrual not recorded
        book_day(ledger, inputs(v1, None))
        unrecorded = book_day(ledger, inputs(stmt(D2, 1_000.0), v1))
        assert unrecorded.entries == ()
        first = book_day(ledger, inputs(stmt(D3, 1_000.0, 3.00), stmt(D2, 1_000.0), v1))
        assert kinds(first) == [("interest_accrual", f"accrual/first-recorded/{D3}")]
        assert legs(first.entries[0]) == {INTEREST_RECEIVABLE: 300, INTEREST_FIRST_RECORDED: -300}
        assert first.entries[0].evidence["opening_accrual"] == (
            "not recorded by the opening statement"
        )
        later = book_day(ledger, inputs(stmt("2026-12-01", 1_000.0, 3.40), stmt(D3, 1_000.0)))
        assert kinds(later) == [("interest_accrual", "accrual/2026-12-01")]

    def test_zero_accruals_book_nothing_and_an_unchanged_one_books_nothing(self):
        ledger = Ledger()
        book_day(ledger, inputs(stmt(D1, 1_000.0, 0.0), None))
        assert (
            book_day(ledger, inputs(stmt(D2, 1_000.0, 0.0), stmt(D1, 1_000.0, 0.0))).entries == ()
        )
        zero_open = Ledger()
        book_day(zero_open, inputs(stmt(D1, 1_000.0), None))
        posting = book_day(zero_open, inputs(stmt(D2, 1_000.0, 0.0), stmt(D1, 1_000.0)))
        assert posting.entries == ()
        steady = Ledger()
        d1 = stmt(D1, 1_000.0, 1.00)
        book_day(steady, inputs(d1, None))
        assert book_day(steady, inputs(stmt(D2, 1_000.0, 1.00), d1)).entries == ()


def ctx_for(store, day):
    return RunContext(
        run_id="01JG0000000000000000000000",
        job="trader.reconcile",
        trading_day=dt.date.fromisoformat(day),
        calendar_date=dt.date.fromisoformat(day),
        store=store,
        seed=0,
        started=dt.datetime(2026, 11, 30, 21, 0, tzinfo=dt.UTC),
    )


def post(store, broker, anchor):
    store.put_bytes(
        trader_broker_statement_key(broker.trading_day),
        json.dumps(broker.to_document()).encode(),
    )
    posting = post_daily_ledger(
        ctx_for(store, broker.trading_day),
        broker=broker,
        anchor=anchor,
        fills=(),
        cash_flows=(),
        corporate_actions=NONE,
        reconciliation=recon(broker, anchor),
        closes=lambda _d, tickers: {t: 1.0 for t in tickers},
        now="2026-11-30T21:00:00Z",
    )
    assert posting is not None
    return posting


@pytest.fixture
def store(tmp_path: pathlib.Path) -> LocalStore:
    return LocalStore(tmp_path)


def opened_without_interest(store, d0):
    """A journal written before the ledger booked interest: an opening only."""
    store.put_bytes(
        trader_broker_statement_key(d0.trading_day), json.dumps(d0.to_document()).encode()
    )
    ledger = Ledger()
    opening = book_day(ledger, inputs(stmt(D1, d0.cash_usd), d0)).entries[:1]
    store.put_bytes(
        journal_key(T, D1),
        json.dumps(
            {
                "schema_version": "ledger_journal.v1",
                "tenant": T,
                "trading_day": D1,
                "entries": [e.to_document() for e in opening],
            }
        ).encode(),
    )


class TestPostDailyLedgerReadsTheOpeningStatement:
    def test_a_book_opened_earlier_books_its_opening_accrual_late(self, store):
        opened_without_interest(store, stmt(D0, 1_000.0, 2.00))
        posting = post(store, stmt(D2, 1_000.0, 2.30), stmt(D1, 1_000.0, 2.10))
        # D1 posted before interest was booked: its growth is caught up first
        # (alpha-engine-config-I13234), then D2's own.
        assert kinds(posting) == [
            ("interest_accrual", f"accrual/opening/{D0}"),
            ("interest_accrual", f"accrual/{D1}"),
            ("interest_accrual", f"accrual/{D2}"),
        ]
        assert [legs(e)[INTEREST_INCOME] for e in posting.entries[1:]] == [-10, -20]
        # Once booked, the opening statement is not read again.
        assert kinds(post(store, stmt(D3, 1_000.0, 2.40), stmt(D2, 1_000.0, 2.30))) == [
            ("interest_accrual", f"accrual/{D3}")
        ]

    def test_a_late_booked_posting_day_settles_from_the_previous_statement(self, store):
        """2026-11-04 (alpha-engine-config-I13234): opened from 10-30 at a $102.51
        accrual, 11-03 posted before interest was booked (accrual $109.28), and
        11-04 IB posted the interest (accrual $3.43). The settlement is the fall
        from $109.28, not from $102.51; the $6.77 between was income, and booking
        it removes that much of the day's cash gap."""
        opened_without_interest(store, stmt(D0, 1_000.0, 102.51))
        posting = post(store, stmt(D2, 1_105.85, 3.43), stmt(D1, 1_000.0, 109.28))
        assert kinds(posting) == [
            ("interest_accrual", f"accrual/opening/{D0}"),
            ("interest_accrual", f"accrual/{D1}"),
            ("interest_settlement", f"accrual/{D2}"),
        ]
        assert legs(posting.entries[1]) == {INTEREST_RECEIVABLE: 677, INTEREST_INCOME: -677}
        assert legs(posting.entries[2]) == {CASH: 10585, INTEREST_RECEIVABLE: -10585}
        assert posting.document["clean"] is True

    def test_an_unreadable_or_foreign_opening_statement_is_an_unrecorded_opening(self, store):
        opened_without_interest(store, stmt(D0, 1_000.0, 2.00))
        store.put_bytes(trader_broker_statement_key(D0), b"not json")
        posting = post(store, stmt(D2, 1_000.0, 2.30), stmt(D1, 1_000.0, 2.10))
        assert kinds(posting) == [("interest_accrual", f"accrual/first-recorded/{D2}")]

    def test_an_opening_statement_of_another_account_is_not_the_openings(self, tmp_path):
        store = LocalStore(tmp_path)
        opened_without_interest(store, stmt(D0, 1_000.0, 2.00))
        store.put_bytes(
            trader_broker_statement_key(D0),
            json.dumps(stmt(D0, 1_000.0, 2.00, account="DU2").to_document()).encode(),
        )
        posting = post(store, stmt(D2, 1_000.0, 2.30), stmt(D1, 1_000.0, 2.10))
        assert kinds(posting) == [("interest_accrual", f"accrual/first-recorded/{D2}")]


def close_of(days):
    """Book ``days`` (statement, anchor) and close November from the journals."""
    ledger, journals, view = Ledger(), {}, None
    for broker, anchor in days:
        posting = book_day(ledger, inputs(broker, anchor))
        journals[broker.trading_day] = posting.entries
        view = posting.document
    last = days[-1][0]
    return close_month(
        CloseInputs(
            T,
            "DU1",
            "2026-11",
            "2026-12-01",
            journals,
            last,
            view,
            None,
            lambda _d, tickers: {t: 10.0 for t in tickers},
            "2026-12-01T21:00:00Z",
        )
    )


def month(accruals, opening_accrual):
    """Every November trading day at $1,000 cash, 10 AAA, and the given accrual
    (by day; a day not named repeats the previous day's)."""
    days, previous = [], stmt("2026-10-30", 1_000.0, opening_accrual)
    accrued = opening_accrual
    for day in month_trading_days("2026-11"):
        accrued = accruals.get(day, accrued)
        current = stmt(day, 1_000.0, accrued)
        days.append((current, previous))
        previous = current
    return days


class TestTheCloseCountsTheAccrual:
    def test_nav_includes_the_receivable_and_the_attribution_ties(self):
        close = close_of(month({"2026-11-02": 2.50, "2026-11-13": 3.25}, 2.00))
        assert close["status"] == "closed", close["exceptions"]
        # Opening: $1,000 cash + 10 AAA at $10 + the $2.00 accrued at the opening.
        assert close["nav_start_cents"] == 110_200
        assert close["interest_receivable_cents"] == 325
        assert close["nav_end_cents"] == 110_325
        assert close["pnl"]["interest_cents"] == 125
        assert close["pnl"]["interest_first_recorded_cents"] == 0
        assert close["pnl"]["total_cents"] == 125

    def test_an_unrecorded_opening_shows_the_first_recorded_balance_as_its_own_line(self):
        close = close_of(month({"2026-11-16": 3.00, "2026-11-20": 3.40}, None))
        assert close["status"] == "closed", close["exceptions"]
        assert close["nav_start_cents"] == 110_000
        assert close["pnl"]["interest_first_recorded_cents"] == 300
        assert close["pnl"]["interest_cents"] == 40
        assert close["nav_end_cents"] == 110_340
