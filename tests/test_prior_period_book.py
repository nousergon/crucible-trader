"""The October prior-period book (alpha-engine-config-I13232, owner decision
2026-11-12): October is posted as its own book, never behind the live book's
journals; its close, rebuild and backfill run in that book; and its last close
must bridge into the live book's opening balance to the share and the cent."""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pytest
from crucible.keys import trader_broker_statement_key, trader_reconciliation_key
from crucible.runner import RunContext
from crucible.store import LocalStore

from crucible_trader import ledger as ledger_module
from crucible_trader.books_retention import is_books_key
from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.exception_proposal import propose_records
from crucible_trader.exception_record import (
    MONTH_END_CLOSE_CLASSES,
    PAPER_FUND_TENANT,
    validate_exception_record,
)
from crucible_trader.ledger import (
    PRIOR_PERIOD_BOOK,
    LedgerError,
    book_root,
    journal_key,
    ledger_reconciliation_key,
    load_journals,
    opening_balances,
    period_lock_key,
    post_daily_ledger,
    read_period_locks,
)
from crucible_trader.month_end_close import (
    close_attempt_key,
    close_key,
    close_prior_month,
)
from crucible_trader.month_rebuild import rebuild_key, verify_month_rebuild
from crucible_trader.performance_statement import file_statement, statement_key
from crucible_trader.reconciliation import (
    CorporateActionSet,
    Fill,
    ReconciliationInputs,
    reconcile,
)

T = PAPER_FUND_TENANT
BOOK = PRIOR_PERIOD_BOOK
NOW = "2026-11-04T21:00:00Z"
NONE = CorporateActionSet(resolved=frozenset({"AAA", "BBB"}))
CLOSES = {
    "2026-10-28": {"AAA": 10.0, "SPY": 500.0, "QQQ": 400.0},
    "2026-10-30": {"AAA": 11.0, "BBB": 21.0, "SPY": 510.0, "QQQ": 404.0},
}


def closes(day, tickers):
    return {t: CLOSES.get(day, {})[t] for t in tickers if t in CLOSES.get(day, {})}


def every_close(_day, tickers):
    return {t: 1.0 for t in tickers}


def stmt(day, positions, cash):
    return BrokerStatement("DU1", day, positions, cash)


def ctx_for(store, day):
    return RunContext(
        run_id="01JG0000000000000000000000",
        job="trader.reconcile",
        trading_day=dt.date.fromisoformat(day),
        calendar_date=dt.date.fromisoformat(day),
        store=store,
        seed=0,
        started=dt.datetime(2026, 11, 4, 21, 0, tzinfo=dt.UTC),
    )


def post(store, broker, anchor, fills=(), *, book=None, genesis=None):
    store.put_bytes(
        trader_broker_statement_key(broker.trading_day),
        json.dumps(broker.to_document()).encode(),
    )
    result = reconcile(
        ReconciliationInputs(
            trading_day=broker.trading_day,
            broker=broker,
            anchor=anchor,
            fills=tuple(fills),
            cash_flows=(),
            corporate_actions=NONE,
            genesis=anchor is None if genesis is None else genesis,
        )
    )
    document = {**result.to_document(), "control_arm": {"passed": True, "missed": []}}
    store.put_bytes(trader_reconciliation_key(broker.trading_day), json.dumps(document).encode())
    return post_daily_ledger(
        ctx_for(store, broker.trading_day),
        broker=broker,
        anchor=anchor,
        fills=fills,
        cash_flows=(),
        corporate_actions=NONE,
        reconciliation={**document, "void": False},
        closes=every_close,
        now="2026-11-04T21:00:00Z",
        book=book,
    )


OCT_END = stmt("2026-10-30", {"AAA": 10, "BBB": 5}, 899.0)


def live_opening(store, anchor=OCT_END):
    """The live book opens on 11-03 from IB's previous statement, as it did."""
    posting = post(store, stmt("2026-11-03", dict(anchor.positions), anchor.cash_usd), anchor)
    assert posting is not None


def october(store):
    """The v1-era month, posted into the prior-period book AFTER the live book
    opened: genesis 10-28 (10 AAA, $1,000), buy 5 BBB at 20 on 10-29 ($1
    commission), flat 10-30."""
    d1 = stmt("2026-10-28", {"AAA": 10}, 1_000.0)
    d2 = stmt("2026-10-29", {"AAA": 10, "BBB": 5}, 899.0)
    buy = Fill("BBB", 5, -101.0, commission_usd=1.0, exec_id="e1")
    for broker, anchor, fills in ((d1, None, ()), (d2, d1, (buy,)), (OCT_END, d2, ())):
        assert post(store, broker, anchor, fills, book=BOOK) is not None, broker.trading_day


@pytest.fixture
def store(tmp_path: pathlib.Path) -> LocalStore:
    return LocalStore(tmp_path)


def run_close(store, day="2026-11-04", book=BOOK):
    ctx = ctx_for(store, day)
    out = close_prior_month(
        ctx,
        day=day,
        account="DU1",
        now=NOW,
        closes=closes,
        after_close=lambda c: file_statement(ctx, c, now=NOW, closes=closes),
        book=book,
    )
    return ctx, out


# -- stage 1: book scope in ledger keys, readers and locks -------------------------


class TestBookKeys:
    def test_the_live_book_keys_are_unchanged(self):
        assert journal_key(T, "2026-11-03") == f"trader/ledger/journal/{T}/2026-11-03.json"
        assert ledger_reconciliation_key(T, "2026-11-03") == (
            f"trader/ledger/reconciliation/{T}/2026-11-03.json"
        )
        assert period_lock_key(T, "2026-11") == f"trader/ledger/periods/{T}/2026-11.json"
        assert close_key(T, "2026-11") == f"trader/ledger/close/{T}/2026-11.json"
        assert statement_key(T, "2026-11") == f"trader/ledger/statements/{T}/2026-11.json"

    def test_a_prior_period_book_has_its_own_root_under_the_books_prefix(self):
        root = f"trader/ledger/books/{BOOK}/"
        assert book_root(BOOK) == root
        keys = [
            journal_key(T, "2026-10-14", BOOK),
            ledger_reconciliation_key(T, "2026-10-14", BOOK),
            period_lock_key(T, "2026-10", BOOK),
            close_key(T, "2026-10", BOOK),
            close_attempt_key(T, "2026-10", "2026-11-02", BOOK),
            statement_key(T, "2026-10", BOOK),
            rebuild_key(T, "2026-10", BOOK),
        ]
        for key in keys:
            assert key.startswith(root), key
            # never under a prefix a live reader lists
            assert not key.startswith(("trader/ledger/journal/", "trader/ledger/periods/"))
            # and still the books: WORM-locked like the live ones
            assert is_books_key(key), key

    def test_an_undeclared_book_is_refused(self):
        with pytest.raises(LedgerError, match="not a declared prior-period book"):
            journal_key(T, "2026-10-14", "scratch")


class TestBookPosting:
    def test_october_posts_into_its_book_although_the_live_book_is_later(self, store):
        live_opening(store)
        october(store)
        assert sorted(load_journals(store, T, before="2026-12-01", book=BOOK)) == [
            "2026-10-28",
            "2026-10-29",
            "2026-10-30",
        ]
        # the live book still holds only its own journal: nothing inserted behind it
        assert sorted(load_journals(store, T, before="2026-12-01")) == ["2026-11-03"]
        for day in ("2026-10-28", "2026-10-29", "2026-10-30"):
            assert not store.exists(journal_key(T, day))
            assert store.exists(ledger_reconciliation_key(T, day, BOOK))

    def test_the_live_book_still_refuses_october_behind_its_journals(self, store):
        live_opening(store)
        ctx = ctx_for(store, "2026-10-28")
        posting = post_daily_ledger(
            ctx,
            broker=stmt("2026-10-28", {"AAA": 10}, 1_000.0),
            anchor=None,
            fills=(),
            cash_flows=(),
            corporate_actions=NONE,
            reconciliation={},
            now=NOW,
        )
        assert posting is None
        assert "never inserted behind" in ctx.metrics[0]["status_reason"]

    def test_the_guard_holds_within_the_book(self, store):
        live_opening(store)
        october(store)
        ctx = ctx_for(store, "2026-10-27")
        posting = post_daily_ledger(
            ctx,
            broker=stmt("2026-10-27", {"AAA": 10}, 1_000.0),
            anchor=None,
            fills=(),
            cash_flows=(),
            corporate_actions=NONE,
            reconciliation={},
            now=NOW,
            book=BOOK,
        )
        assert posting is None
        assert "journals after 2026-10-27 already exist (2026-10-28)" in str(
            ctx.metrics[0]["status_reason"]
        )

    def test_period_locks_are_per_book(self, store):
        live_opening(store)
        store.put_bytes(
            period_lock_key(T, "2026-10", BOOK),
            json.dumps(
                {"schema_version": "ledger_period_lock.v1", "month": "2026-10", "journals": {}}
            ).encode(),
        )
        assert read_period_locks(store, T) == {}
        assert set(read_period_locks(store, T, book=BOOK)) == {"2026-10"}
        assert post(store, stmt("2026-10-28", {"AAA": 10}, 1_000.0), None, book=BOOK) is None
        # the live book is not closed by the prior-period book's lock
        assert post(store, stmt("2026-11-04", {"AAA": 10, "BBB": 5}, 899.0), OCT_END) is not None

    def test_opening_balances_reads_what_the_live_book_opened_with(self, store):
        assert opening_balances(store, T, "DU1") is None
        live_opening(store)
        opened = opening_balances(store, T, "DU1")
        assert opened is not None
        ledger, cited = opened
        assert [e.kind for _, e in cited] == ["opening_balance"]
        assert ledger.positions(T, "DU1") == {"AAA": 10, "BBB": 5}
        assert ledger.cents(T, "DU1", "assets:cash:USD") == 89_900
        assert opening_balances(store, T, "DU2") is None  # another account's opening is not ours


# -- stage 2: the close, the rebuild and the statement run in the book ---------------


class TestBookClose:
    def test_the_book_closes_on_its_own_journals_and_files_under_its_root(self, store):
        live_opening(store)
        october(store)
        ctx, close = run_close(store)
        assert close is not None and close["status"] == "closed", close["exceptions"]
        assert close["book"] == BOOK
        assert close["inputs"]["ledger_view"] == ledger_reconciliation_key(T, "2026-10-30", BOOK)
        assert store.exists(close_key(T, "2026-10", BOOK))
        assert not store.exists(close_key(T, "2026-10"))
        assert set(read_period_locks(store, T, book=BOOK)["2026-10"]) == {
            "2026-10-28",
            "2026-10-29",
            "2026-10-30",
        }
        assert read_period_locks(store, T) == {}
        statement = json.loads(store.get_bytes(statement_key(T, "2026-10", BOOK)))
        assert statement["inputs"]["close"] == close_key(T, "2026-10", BOOK)
        assert not store.exists(statement_key(T, "2026-10"))
        assert [m["status"] for m in ctx.metrics] == ["OK", "OK"]
        # the closed book month refuses a late journal; the live book posts on
        assert post(store, OCT_END, None, book=BOOK, genesis=False) is None
        assert post(store, stmt("2026-11-04", {"AAA": 10, "BBB": 5}, 899.0), OCT_END) is not None

    def test_the_live_close_never_sees_the_book(self, store):
        live_opening(store)
        october(store)
        _, close = run_close(store, book=None)
        # the live book's oldest month is November: nothing before it to close
        assert close is None
        assert not store.exists(close_key(T, "2026-10"))

    def test_the_book_month_rebuilds_from_raw_against_the_same_bridge(self, store):
        live_opening(store)
        october(store)
        run_close(store)
        ctx = ctx_for(store, "2026-11-05")
        rebuilt = verify_month_rebuild(
            ctx, day="2026-11-05", account="DU1", now=NOW, closes=closes, book=BOOK
        )
        assert rebuilt is not None and rebuilt["matches"] is True, rebuilt
        assert rebuilt["book"] == BOOK
        assert store.exists(rebuild_key(T, "2026-10", BOOK))
        assert not store.exists(rebuild_key(T, "2026-10"))


# -- stage 3: the bridge across the cutover ------------------------------------------


class TestBridge:
    def test_a_book_that_ties_to_the_live_opening_closes(self, store):
        live_opening(store)
        october(store)
        _, close = run_close(store)
        assert close is not None
        bridge = close["bridge"]
        assert bridge["ties"] is True and bridge["differences"] == []
        assert bridge["closing_day"] == "2026-10-30"
        assert bridge["opening"]["statement_day"] == "2026-10-30"
        assert bridge["opening"]["journal"] == journal_key(T, "2026-11-03")
        lines = {line["item"]: line for line in bridge["lines"]}
        assert lines["cash_cents"] == {
            "item": "cash_cents",
            "closing": 89_900,
            "opening": 89_900,
            "difference": 0,
            "unit": "cents",
        }
        assert lines["shares:BBB"]["closing"] == lines["shares:BBB"]["opening"] == 5

    def test_every_difference_is_itemised_and_refuses_the_close(self, store):
        live_opening(store, anchor=stmt("2026-10-30", {"AAA": 10, "BBB": 4, "CCC": 2}, 899.37))
        october(store)
        ctx, close = run_close(store)
        assert close is not None and close["status"] == "refused"
        bridge = close["bridge"]
        assert bridge["ties"] is False
        assert [(d["item"], d["difference"]) for d in bridge["differences"]] == [
            ("cash_cents", 37),
            ("shares:BBB", -1),
            ("shares:CCC", 2),
        ]
        lines = [e for e in close["exceptions"] if e["class"].startswith("close_bridge")]
        assert [(e["class"], e["instrument"], e["amount"], e["unit"]) for e in lines] == [
            ("close_bridge_cash", "cash", 0.37, "usd"),
            ("close_bridge_shares", "BBB", -1.0, "shares"),
            ("close_bridge_shares", "CCC", 2.0, "shares"),
        ]
        for record in lines:
            validate_exception_record(record)
            assert record["actual"]["record"] == journal_key(T, "2026-11-03")
        # the lines wait in the review queue like any other close line
        assert {p.to_document()["action"] for p in propose_records(lines, proposed_at=NOW)} == {
            "investigate_cash",
            "investigate_shares",
        }
        assert not store.exists(close_key(T, "2026-10", BOOK))
        assert not store.exists(period_lock_key(T, "2026-10", BOOK))
        assert store.exists(close_attempt_key(T, "2026-10", "2026-11-04", BOOK))
        assert ctx.metrics[0]["status"] in {"WATCH", "FAIL"}

    def test_an_opening_from_another_day_is_a_line(self, store):
        live_opening(store, anchor=stmt("2026-10-29", {"AAA": 10, "BBB": 5}, 899.0))
        october(store)
        _, close = run_close(store)
        assert close is not None and close["status"] == "refused"
        (line,) = close["exceptions"]
        assert (line["class"], line["instrument"]) == ("close_checks_failed", "bridge")
        assert "2026-10-29" in line["detail"]
        assert close["bridge"]["ties"] is False and close["bridge"]["differences"] == []

    def test_no_live_opening_yet_is_a_missing_input(self, store):
        october(store)
        _, close = run_close(store)
        assert close is not None and close["status"] == "refused"
        (line,) = close["exceptions"]
        assert line["class"] == "close_input_missing"
        assert "no opening balance" in line["detail"]
        assert close["bridge"]["ties"] is False

    def test_only_the_books_last_month_bridges(self, store, monkeypatch):
        monkeypatch.setattr(ledger_module, "PRIOR_PERIOD_BOOKS", {BOOK: "2026-12"})
        october(store)  # no live opening at all, and October is not the last month
        _, close = run_close(store)
        assert close is not None and close["status"] == "closed", close["exceptions"]
        assert close["book"] == BOOK and close["bridge"] is None

    def test_the_bridge_classes_are_close_classes(self):
        assert {"close_bridge_cash", "close_bridge_shares"} <= MONTH_END_CLOSE_CLASSES


class TestBridgeInterest:
    def test_the_opening_accrual_is_bridged_to_the_cent(self, store):
        def accruing(broker, accrued):
            return BrokerStatement(
                "DU1",
                broker.trading_day,
                dict(broker.positions),
                broker.cash_usd,
                "ib_paper",
                accrued_cash_usd=accrued,
            )

        oct_end = accruing(OCT_END, 1.25)
        posting = post(
            store, accruing(stmt("2026-11-03", {"AAA": 10, "BBB": 5}, 899.0), 1.30), oct_end
        )
        assert posting is not None
        d1 = accruing(stmt("2026-10-28", {"AAA": 10}, 1_000.0), 1.00)
        d2 = accruing(stmt("2026-10-29", {"AAA": 10, "BBB": 5}, 899.0), 1.10)
        buy = Fill("BBB", 5, -101.0, commission_usd=1.0, exec_id="e1")
        for broker, anchor, fills in ((d1, None, ()), (d2, d1, (buy,)), (oct_end, d2, ())):
            assert post(store, broker, anchor, fills, book=BOOK) is not None
        _, close = run_close(store)
        assert close is not None
        lines = {line["item"]: line for line in close["bridge"]["lines"]}
        assert lines["interest_receivable_cents"]["closing"] == 125
        assert lines["interest_receivable_cents"]["opening"] == 125
        assert close["bridge"]["ties"] is True, close["bridge"]["differences"]
