"""Ledger views (aec#13753): the master ledger laid out for /page/books."""

from __future__ import annotations

import json
import pathlib

import pytest
from crucible.keys import trader_broker_statement_key, trader_reconciliation_key
from crucible.store import LocalStore

from crucible_trader.exception_record import exceptions_key
from crucible_trader.ledger import (
    CASH,
    COMMISSIONS,
    OPENING,
    PAPER_FUND_TENANT,
    SECURITIES,
    SUSPENSE_CASH,
    SUSPENSE_POSITION,
    Entry,
    LedgerError,
    Posting,
    journal_document,
    journal_key,
    ledger_reconciliation_key,
)
from crucible_trader.ledger_views import (
    LEDGER_VIEWS_SCHEMA_VERSION,
    day_views,
    ledger_views_key,
    publish_ledger_views,
    source_key,
)
from crucible_trader.order_intent_reconciliation import order_intent_reconciliation_key

T = PAPER_FUND_TENANT
A = "DU1"
D1, D2, D3 = "2026-11-03", "2026-11-04", "2026-11-05"


def entry(day, kind, ref, postings, **evidence):
    return Entry(T, A, day, kind, ref, tuple(postings), evidence)


OPEN = entry(D1, "opening_balance", "open", [Posting(CASH, 100_000), Posting(OPENING, -100_000)])
BUY = entry(
    D1,
    "fill",
    "x1",
    [Posting(CASH, -30_000), Posting(SECURITIES + "AAA", 30_000, 10)],
    source="ib_execution",
)
FEE = entry(
    D1,
    "commission",
    "x1",
    [Posting(COMMISSIONS, 100), Posting(CASH, -100)],
    source="ib_commission_report",
)
CASH_GAP = entry(
    D2,
    "cash_gap",
    "gap/USD",
    [Posting(CASH, 250), Posting(SUSPENSE_CASH, -250)],
    source="ib_statement",
    reason="broker cash differs",
    exception_ids=["ex-broker", "ex-ledger"],
    decision_id=None,
)
SHARE_GAP = entry(
    D2,
    "position_gap",
    "gap/AAA",
    [Posting(SECURITIES + "AAA", 0, 1), Posting(SUSPENSE_POSITION + "AAA", 0, -1)],
    source="ib_statement",
    exception_ids=["ex-aaa"],
)
JOURNALS = {D1: (OPEN, BUY, FEE), D2: (CASH_GAP, SHARE_GAP)}


def recon(day, **extra):
    return {
        "trading_day": day,
        "clean": not extra.get("breaks"),
        "ledger_cash_cents": 69_900,
        "broker_cash_cents": 69_900,
        "ledger_positions": {"AAA": 10},
        "broker_positions": {"AAA": 10},
        "breaks": [],
        "suspense": {"cash_cents": 0, "shares": {}},
        "checks": {"open_breaks": []},
        **extra,
    }


GAP_DAY = recon(
    D2,
    ledger_cash_cents=70_150,
    broker_cash_cents=70_150,
    ledger_positions={"AAA": 11},
    broker_positions={"AAA": 11},
    breaks=[
        {
            "class": "cash_gap",
            "instrument": "USD",
            "ledger": 69_900,
            "broker": 70_150,
            "delta": 250,
            "tier": "page",
            "exception_ids": ["ex-broker", "ex-ledger"],
        }
    ],
    suspense={"cash_cents": -250, "shares": {"AAA": -1}},
    checks={
        "open_breaks": [
            {"day": D2, "age": 0, "entry_id": CASH_GAP.entry_id, "kind": "cash_gap"},
            {"day": D2, "age": 0, "entry_id": SHARE_GAP.entry_id, "kind": "position_gap"},
        ]
    },
)


@pytest.fixture
def store(tmp_path: pathlib.Path) -> LocalStore:
    return LocalStore(tmp_path)


def write(store, journals, recons):
    for day, entries in journals.items():
        store.put_bytes(journal_key(T, day), journal_document(T, day, entries))
    for day, doc in recons.items():
        store.put_bytes(ledger_reconciliation_key(T, day), json.dumps(doc).encode())


def read(store, day):
    return json.loads(store.get_bytes(ledger_views_key(T, day)))


class TestAccounts:
    def test_the_trial_balance_by_account_sums_to_zero(self):
        views = day_views(JOURNALS, recon(D1), tenant=T, day=D1)
        assert views["schema_version"] == LEDGER_VIEWS_SCHEMA_VERSION
        assert views["trial_balance_cents"] == 0
        rows = {r["ledger_account"]: r for r in views["accounts"]}
        assert rows[CASH]["balance_cents"] == 69_900
        assert rows[CASH]["quantity"] is None
        assert rows[CASH]["postings_to_date"] == 3
        assert rows[SECURITIES + "AAA"]["quantity"] == 10
        assert sum(r["balance_cents"] for r in views["accounts"]) == 0

    def test_later_days_are_ignored(self):
        views = day_views(JOURNALS, recon(D1), tenant=T, day=D1)
        assert SUSPENSE_CASH not in {r["ledger_account"] for r in views["accounts"]}
        assert views["sources"] == {
            "journal": journal_key(T, D1),
            "ledger_reconciliation": ledger_reconciliation_key(T, D1),
            "broker_statement": trader_broker_statement_key(D1),
        }


class TestPostings:
    def test_each_posting_carries_its_running_balance_and_source(self):
        views = day_views(JOURNALS, recon(D1), tenant=T, day=D1)
        cash = [p for p in views["postings"] if p["ledger_account"] == CASH]
        assert [p["running_balance_cents"] for p in cash] == [100_000, 70_000, 69_900]
        buy = next(p for p in views["postings"] if p["ledger_account"] == SECURITIES + "AAA")
        assert buy["running_quantity"] == 10
        assert buy["source_key"] == order_intent_reconciliation_key(D1)
        assert buy["journal_key"] == journal_key(T, D1)
        assert cash[0]["running_quantity"] is None
        assert buy["entry_page"] == f"/signal/books:entry@{BUY.entry_id}"

    def test_only_the_days_entries_are_listed(self):
        views = day_views(JOURNALS, GAP_DAY, tenant=T, day=D2)
        assert {p["entry_id"] for p in views["postings"]} == {
            CASH_GAP.entry_id,
            SHARE_GAP.entry_id,
        }
        gap = next(p for p in views["postings"] if p["ledger_account"] == SUSPENSE_CASH)
        assert gap["exception_ids"] == ["ex-broker", "ex-ledger"]
        assert gap["reason"] == "broker cash differs"


class TestSourceKey:
    @pytest.mark.parametrize(
        ("source", "key"),
        [
            ("ib_statement", trader_broker_statement_key(D1)),
            ("ib_execution", order_intent_reconciliation_key(D1)),
            ("ib_commission_report", order_intent_reconciliation_key(D1)),
            ("decision", trader_reconciliation_key(D1)),
            (None, journal_key(T, D1)),
        ],
    )
    def test_the_document_an_entry_was_booked_from(self, source, key):
        evidence = {} if source is None else {"source": source}
        e = Entry(T, A, D1, "opening_balance", "o", OPEN.postings, evidence)
        assert source_key(e) == key


class TestTieout:
    def test_a_clean_day_ties(self):
        rows = day_views(JOURNALS, recon(D1), tenant=T, day=D1)["tieout"]
        assert [(r["item"], r["status"], r["delta"]) for r in rows] == [
            ("USD", "ties", 0),
            ("AAA", "ties", 0),
        ]
        assert rows[0]["unit"] == "cents"
        assert rows[1]["unit"] == "shares"

    def test_a_break_today_shows_the_ledger_before_the_gap_entry(self):
        rows = {r["item"]: r for r in day_views(JOURNALS, GAP_DAY, tenant=T, day=D2)["tieout"]}
        usd = rows["USD"]
        assert usd["status"] == "break_today"
        assert (usd["ledger"], usd["broker"], usd["delta"]) == (69_900, 70_150, 250)
        assert usd["tier"] == "page"
        # Today's break and its open suspense cite the same ids once.
        assert usd["exception_ids"] == ["ex-broker", "ex-ledger"]
        assert usd["suspense_entry_ids"] == [CASH_GAP.entry_id]
        assert rows["AAA"]["status"] == "in_suspense"
        assert rows["AAA"]["in_suspense"] == -1
        assert rows["AAA"]["exception_ids"] == ["ex-aaa"]
        assert rows["AAA"]["open_since"] == D2

    def test_a_row_links_the_first_of_its_ids_that_has_a_card(self):
        views = day_views(JOURNALS, GAP_DAY, tenant=T, day=D2, filed={"ex-ledger": D2})
        rows = {r["item"]: r for r in views["tieout"]}
        assert rows["USD"]["break_card"] == f"/signal/crucible-trading:exception@{D2}:ex-ledger"
        assert rows["AAA"]["break_card"] is None

    def test_suspense_from_an_earlier_day_stays_visible(self):
        later = recon(
            D3,
            suspense={"cash_cents": -250, "shares": {}},
            checks={
                "open_breaks": [
                    {"day": D2, "age": 1, "entry_id": CASH_GAP.entry_id, "kind": "cash_gap"}
                ]
            },
        )
        usd = day_views(JOURNALS, later, tenant=T, day=D3)["tieout"][0]
        assert (usd["status"], usd["in_suspense"], usd["open_since"]) == ("in_suspense", -250, D2)
        assert usd["tier"] is None

    def test_a_gap_whose_entry_is_not_in_the_journals_is_still_listed(self):
        orphaned = recon(
            D3,
            checks={
                "open_breaks": [
                    {"day": D1, "age": 2, "entry_id": "gone-cash", "kind": "cash_gap"},
                    {"day": D1, "age": 2, "entry_id": "gone-shares", "kind": "position_gap"},
                ]
            },
        )
        rows = {r["item"]: r for r in day_views(JOURNALS, orphaned, tenant=T, day=D3)["tieout"]}
        assert rows["USD"]["status"] == "in_suspense"
        assert rows["USD"]["suspense_entry_ids"] == ["gone-cash"]
        assert rows["?"]["status"] == "in_suspense"
        assert rows["?"]["exception_ids"] == []


class TestPublish:
    def test_every_journal_day_gets_a_views_document(self, store):
        write(store, JOURNALS, {D1: recon(D1), D2: GAP_DAY})
        assert publish_ledger_views(store, day=D2) == [
            ledger_views_key(T, D1),
            ledger_views_key(T, D2),
        ]
        assert read(store, D1)["trading_day"] == D1
        assert read(store, D2)["tieout"][0]["status"] == "break_today"

    def test_a_break_links_the_card_filed_for_it(self, store):
        write(store, JOURNALS, {D2: GAP_DAY})
        store.put_bytes(
            exceptions_key(D2),
            json.dumps(
                {"records": [{"exception_id": "ex-broker"}, {"exception_id": "x"}]}
            ).encode(),
        )
        publish_ledger_views(store, day=D2)
        usd = read(store, D2)["tieout"][0]
        assert usd["break_card"] == f"/signal/crucible-trading:exception@{D2}:ex-broker"

    def test_an_existing_earlier_day_is_not_rewritten_but_today_is(self, store):
        write(store, JOURNALS, {D1: recon(D1), D2: GAP_DAY})
        publish_ledger_views(store, day=D2)
        assert publish_ledger_views(store, day=D2) == [ledger_views_key(T, D2)]

    def test_a_day_without_its_reconciliation_view_is_skipped(self, store):
        write(store, JOURNALS, {D2: GAP_DAY})
        assert publish_ledger_views(store, day=D2) == [ledger_views_key(T, D2)]

    def test_a_journal_after_the_day_is_not_published(self, store):
        write(store, JOURNALS, {D1: recon(D1), D2: GAP_DAY})
        assert publish_ledger_views(store, day=D1) == [ledger_views_key(T, D1)]

    def test_a_failure_never_raises(self, store):
        store.put_bytes(journal_key(T, D1), b"not json")
        assert publish_ledger_views(store, day=D1) is None


def test_a_bad_tenant_has_no_views_key():
    with pytest.raises(LedgerError):
        ledger_views_key("Not A Tenant", D1)
