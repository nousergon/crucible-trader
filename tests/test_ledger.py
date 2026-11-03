"""The v2 double-entry ledger (`alpha-engine-config-I13128`): balanced by
construction, one entry per event, append-only, reconciled to the broker, every
gap booked to named suspense, and never raising into the trader."""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pytest
from crucible.runner import RunContext
from crucible.store import LocalStore

from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.exception_record import exception_id
from crucible_trader.ledger import (
    CASH,
    COMMISSIONS,
    DEFAULT_TENANT,
    SECURITIES,
    SUSPENSE_CASH,
    DayInputs,
    Entry,
    Ledger,
    LedgerError,
    Posting,
    book_day,
    entry_from_document,
    journal_key,
    ledger_reconciliation_key,
    load_ledger,
    post_daily_ledger,
    to_cents,
)
from crucible_trader.reconciliation import (
    CashFlow,
    CorporateAction,
    CorporateActionSet,
    Fill,
    ReconciliationInputs,
    reconcile,
)

T = DEFAULT_TENANT
PREV, DAY, NEXT = "2026-11-02", "2026-11-03", "2026-11-04"
NONE = CorporateActionSet(resolved=frozenset({"AAA", "BBB"}))


def stmt(day, positions, cash, account="DU1"):
    return BrokerStatement(account, day, positions, cash)


def recon_doc(broker, anchor, fills=(), flows=(), actions=NONE):
    result = reconcile(
        ReconciliationInputs(
            trading_day=broker.trading_day,
            broker=broker,
            anchor=anchor,
            fills=tuple(fills),
            cash_flows=tuple(flows),
            corporate_actions=actions,
            genesis=anchor is None,
        )
    )
    return {**result.to_document(), "control_arm": {"passed": True, "missed": []}, "void": False}


def day_inputs(broker, anchor, fills=(), flows=(), actions=NONE, tenant=T):
    return DayInputs(
        tenant,
        broker,
        anchor,
        tuple(fills),
        tuple(flows),
        actions,
        recon_doc(broker, anchor, fills, flows, actions),
    )


def buy(ticker, shares, price, commission, exec_id):
    return Fill(
        ticker, shares, -shares * price - commission, commission_usd=commission, exec_id=exec_id
    )


class TestBalancedByConstruction:
    def test_cents_are_exact_decimals_rounded_half_up(self):
        assert to_cents(0.1 + 0.2) == 30
        assert to_cents(-5556.155) == -555616
        assert to_cents("1.005") == 101
        with pytest.raises(LedgerError):
            to_cents(float("nan"))

    def test_an_unbalanced_entry_cannot_be_built(self):
        with pytest.raises(LedgerError, match="does not balance"):
            Entry(T, "DU1", DAY, "fill", "x", (Posting(CASH, -100), Posting(COMMISSIONS, 99)))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tenant": "Nous Ergon"},
            {"kind": "net_change"},
            {"postings": ()},
        ],
    )
    def test_tenant_kind_and_postings_are_closed(self, kwargs):
        base = {
            "tenant": T,
            "account": "DU1",
            "trading_day": DAY,
            "kind": "fill",
            "event_ref": "x",
            "postings": (Posting(CASH, 1), Posting(COMMISSIONS, -1)),
        }
        with pytest.raises(LedgerError):
            Entry(**{**base, **kwargs})

    def test_chart_of_accounts_floats_and_shares_are_refused(self):
        with pytest.raises(LedgerError):
            Posting("assets:plug", 0)
        with pytest.raises(LedgerError):
            Posting(CASH, 1.5)  # pyright: ignore[reportArgumentType]
        with pytest.raises(LedgerError, match="carry shares"):
            Posting(CASH, 0, 10)

    def test_an_adjustment_must_cite_exception_and_decision(self):
        postings = (Posting(CASH, 3), Posting(SUSPENSE_CASH, -3))
        with pytest.raises(LedgerError, match="exception ids and decision id"):
            Entry(T, "DU1", DAY, "adjustment", "a", postings, {"exception_ids": ["e1"]})
        Entry(
            T,
            "DU1",
            DAY,
            "adjustment",
            "a",
            postings,
            {"exception_ids": ["e1"], "decision_id": "d1"},
        )

    def test_append_refuses_a_second_entry_for_the_same_event(self):
        ledger = Ledger()
        entry = Entry(T, "DU1", DAY, "fill", "e1", (Posting(CASH, 1), Posting(COMMISSIONS, -1)))
        ledger.append(entry)
        with pytest.raises(LedgerError, match="already booked"):
            ledger.append(entry)
        assert ledger.trial_balance() == 0


class TestOneEventOneEntry:
    def test_opening_fill_and_commission_are_separate_entries_and_tie_to_ib(self):
        anchor = stmt(PREV, {"AAA": 10}, 1_000.00)
        fills = (buy("AAA", 2, 100.0, 1.0, "x1"), buy("BBB", 3, 10.005, 0.35, "x2"))
        broker = stmt(DAY, {"AAA": 12, "BBB": 3}, 768.64)  # IB rounded the BBB fill the other way
        ledger = Ledger()
        posting = book_day(ledger, day_inputs(broker, anchor, fills))
        kinds = [e.kind for e in posting.entries]
        assert kinds == ["opening_balance", "fill", "commission", "fill", "commission", "cash_gap"]
        assert posting.entries[1].postings[1].quantity == 2
        assert ledger.trial_balance() == 0
        assert ledger.positions(T, "DU1") == {"AAA": 12, "BBB": 3}
        # 30.015 -> 30.02 at the fill, so the book is a cent off IB: a gap entry.
        assert [(b["class"], b["delta"]) for b in posting.breaks] == [("cash_gap", 1)]
        assert ledger.cents(T, "DU1", CASH) == to_cents(broker.cash_usd)

    def test_a_clean_day_has_no_breaks_and_no_suspense(self):
        anchor = stmt(PREV, {"AAA": 10}, 1_000.00)
        broker = stmt(DAY, {"AAA": 12}, 799.0)
        posting = book_day(Ledger(), day_inputs(broker, anchor, (buy("AAA", 2, 100.0, 1.0, "x1"),)))
        assert posting.breaks == () and posting.document["clean"] is True
        assert posting.document["trial_balance_cents"] == 0

    def test_a_fill_without_a_commission_report_is_one_entry_and_says_so(self):
        anchor = stmt(PREV, {"AAA": 10}, 1_000.00)
        broker = stmt(DAY, {"AAA": 12}, 799.0)
        posting = book_day(Ledger(), day_inputs(broker, anchor, (Fill("AAA", 2, -201.0),)))
        fill = posting.entries[1]
        assert fill.kind == "fill" and fill.evidence["commission_separated"] is False
        assert posting.document["clean"] is True

    def test_declared_flow_is_its_own_entry(self):
        anchor = stmt(PREV, {"AAA": 10}, 1_000.00)
        broker = stmt(DAY, {"AAA": 10}, 1_002.5)
        posting = book_day(Ledger(), day_inputs(broker, anchor, flows=(CashFlow("interest", 2.5),)))
        assert [e.kind for e in posting.entries] == ["opening_balance", "declared_cash_flow"]
        assert posting.document["clean"] is True

    def test_a_split_restates_quantity_at_unchanged_cents(self):
        anchor = stmt(PREV, {"AAA": 10}, 1_000.00)
        actions = CorporateActionSet(
            frozenset({"AAA"}), (CorporateAction("AAA", "split", ratio=2.0),)
        )
        broker = stmt(DAY, {"AAA": 20}, 1_000.0)
        ledger = Ledger()
        posting = book_day(ledger, day_inputs(broker, anchor, actions=actions))
        split = posting.entries[1]
        assert split.kind == "split" and split.postings == (
            Posting("assets:securities:AAA", 0, 10),
        )
        assert posting.document["clean"] is True

    def test_a_dividend_is_a_receivable_and_a_settlement(self):
        anchor = stmt(PREV, {"AAA": 10}, 1_000.00)
        actions = CorporateActionSet(
            frozenset({"AAA"}),
            (CorporateAction("AAA", "cash_dividend", amount_per_share=0.25, entitled_shares=10),),
        )
        broker = stmt(DAY, {"AAA": 10}, 1_002.5)
        ledger = Ledger()
        posting = book_day(ledger, day_inputs(broker, anchor, actions=actions))
        assert [e.kind for e in posting.entries][1:] == [
            "dividend_receivable",
            "dividend_settlement",
        ]
        assert (
            posting.document["clean"] is True
            and posting.document["dividends_receivable_cents"] == {}
        )

    def test_an_unpaid_dividend_stays_receivable_and_a_late_credit_settles_it(self):
        actions = CorporateActionSet(
            frozenset({"AAA"}),
            (CorporateAction("AAA", "cash_dividend", amount_per_share=0.25, entitled_shares=10),),
        )
        ledger = Ledger()
        anchor = stmt(PREV, {"AAA": 10}, 1_000.00)
        unpaid = stmt(DAY, {"AAA": 10}, 1_000.0)  # IB did not pay it today
        day1 = book_day(ledger, day_inputs(unpaid, anchor, actions=actions))
        assert [e.kind for e in day1.entries][1:] == ["dividend_receivable"]
        assert day1.document["dividends_receivable_cents"] == {
            "assets:dividends_receivable:AAA": 250
        }
        assert day1.breaks == ()
        late = stmt(NEXT, {"AAA": 10}, 1_002.5)
        doc = {
            **recon_doc(late, unpaid),
            "dividend_credits": [{"ticker": "AAA", "pay_date": DAY, "amount_usd": 2.5}],
        }
        day2 = book_day(ledger, DayInputs(T, late, unpaid, (), (), NONE, doc))
        assert [e.kind for e in day2.entries] == ["dividend_settlement"]
        assert day2.document["clean"] is True


class TestEveryGapIsBookedNeverAbsorbed:
    def test_a_cash_gap_is_an_entry_to_suspense_citing_its_breaks(self):
        anchor = stmt(PREV, {"AAA": 10}, 1_000.00)
        broker = stmt(DAY, {"AAA": 12}, 799.03)  # 3 cents over
        ledger = Ledger()
        posting = book_day(ledger, day_inputs(broker, anchor, (buy("AAA", 2, 100.0, 1.0, "x1"),)))
        gap = posting.entries[-1]
        assert gap.kind == "cash_gap"
        assert gap.postings == (Posting(CASH, 3), Posting(SUSPENSE_CASH, -3))
        assert gap.evidence["decision_id"] is None
        assert (
            exception_id("ledger", DAY, ("cash_gap", "DU1", "USD")) in gap.evidence["exception_ids"]
        )
        assert gap.evidence["expires"] == "2026-11-06"
        assert posting.document["suspense"]["cash_cents"] == -3
        assert posting.document["clean"] is False
        assert ledger.cents(T, "DU1", CASH) == 79_903  # ground truth is IB

    def test_a_reconciler_break_is_cited_by_its_own_exception_id(self):
        anchor = stmt(PREV, {"AAA": 10}, 1_000.00)
        broker = stmt(DAY, {"AAA": 12}, 790.0)  # $9 short: a reconciler cash_delta too
        posting = book_day(Ledger(), day_inputs(broker, anchor, (buy("AAA", 2, 100.0, 1.0, "x1"),)))
        assert len(posting.entries[-1].evidence["exception_ids"]) == 2

    def test_a_share_gap_is_an_entry_to_position_suspense(self):
        anchor = stmt(PREV, {"AAA": 10}, 1_000.00)
        broker = stmt(DAY, {"AAA": 9}, 1_000.0)
        ledger = Ledger()
        posting = book_day(ledger, day_inputs(broker, anchor))
        gap = posting.entries[-1]
        assert gap.kind == "position_gap" and gap.postings[1].quantity == 1
        assert ledger.positions(T, "DU1") == {"AAA": 9}
        assert posting.document["suspense"]["shares"] == {"AAA": 1}

    def test_suspense_standing_from_an_earlier_day_keeps_the_book_unclean(self):
        ledger = Ledger()
        book_day(
            ledger, day_inputs(stmt(DAY, {"AAA": 10}, 1_000.03), stmt(PREV, {"AAA": 10}, 1_000.0))
        )
        day2 = book_day(
            ledger, day_inputs(stmt(NEXT, {"AAA": 10}, 1_000.03), stmt(DAY, {"AAA": 10}, 1_000.03))
        )
        assert day2.breaks == () and day2.document["clean"] is False


class TestOpening:
    def test_genesis_opens_from_todays_statement_without_counting_fills_twice(self):
        broker = stmt(DAY, {"AAA": 12}, 799.0)
        ledger = Ledger()
        posting = book_day(ledger, day_inputs(broker, None, (buy("AAA", 2, 100.0, 1.0, "x1"),)))
        assert [e.kind for e in posting.entries] == ["opening_balance"]
        assert posting.entries[0].evidence["includes_this_days_events"] is True
        assert posting.document["clean"] is True

    def test_the_second_day_rolls_forward_from_the_ledger_not_the_anchor(self):
        ledger = Ledger()
        book_day(
            ledger, day_inputs(stmt(DAY, {"AAA": 10}, 1_000.0), stmt(PREV, {"AAA": 10}, 1_000.0))
        )
        # A wrong anchor tomorrow does not move the book: the ledger is the record.
        wrong_anchor = stmt(DAY, {"AAA": 99}, 5.0)
        posting = book_day(ledger, day_inputs(stmt(NEXT, {"AAA": 10}, 1_000.0), wrong_anchor))
        assert posting.entries == () and posting.document["clean"] is True

    def test_tenants_are_separate_books(self):
        ledger = Ledger()
        book_day(
            ledger, day_inputs(stmt(DAY, {"AAA": 10}, 1_000.0), stmt(PREV, {"AAA": 10}, 1_000.0))
        )
        book_day(
            ledger,
            day_inputs(
                stmt(DAY, {"BBB": 5}, 50.0), stmt(PREV, {"BBB": 5}, 50.0), tenant="client-two"
            ),
        )
        assert ledger.positions(T, "DU1") == {"AAA": 10}
        assert ledger.positions("client-two", "DU1") == {"BBB": 5}
        assert ledger.trial_balance() == 0


# ---------------------------------------------------------------- I/O step


def ctx_for(store, day):
    started = dt.datetime(2026, 11, 3, 21, 0, tzinfo=dt.UTC)
    return RunContext(
        run_id="01JG0000000000000000000000",
        job="trader.reconcile",
        trading_day=dt.date.fromisoformat(day),
        calendar_date=dt.date.fromisoformat(day),
        store=store,
        seed=0,
        started=started,
    )


def post(store, broker, anchor, fills=()):
    ctx = ctx_for(store, broker.trading_day)
    result = post_daily_ledger(
        ctx,
        broker=broker,
        anchor=anchor,
        fills=fills,
        cash_flows=(),
        corporate_actions=NONE,
        reconciliation=recon_doc(broker, anchor, fills),
        now="2026-11-03T21:00:00Z",
    )
    return ctx, result


@pytest.fixture
def store(tmp_path: pathlib.Path) -> LocalStore:
    return LocalStore(tmp_path)


class TestPostDailyLedger:
    def test_writes_the_journal_and_view_and_replays_next_day(self, store):
        ctx, posting = post(
            store,
            stmt(DAY, {"AAA": 12}, 799.0),
            stmt(PREV, {"AAA": 10}, 1_000.0),
            (buy("AAA", 2, 100.0, 1.0, "x1"),),
        )
        assert posting is not None
        assert [o["key"] for o in ctx.outputs] == [
            journal_key(T, DAY),
            ledger_reconciliation_key(T, DAY),
        ]
        assert [m["status"] for m in ctx.metrics] == ["OK"]
        replayed = load_ledger(store, T, before=NEXT)
        assert replayed.positions(T, "DU1") == {"AAA": 12} and replayed.trial_balance() == 0
        ctx2, day2 = post(store, stmt(NEXT, {"AAA": 12}, 799.0), stmt(DAY, {"AAA": 12}, 799.0))
        assert day2 is not None and day2.entries == () and day2.document["clean"] is True

    def test_an_identical_rerun_is_a_no_op_and_a_different_one_is_refused(self, store):
        broker, anchor = stmt(DAY, {"AAA": 10}, 1_000.0), stmt(PREV, {"AAA": 10}, 1_000.0)
        post(store, broker, anchor)
        before = store.get_bytes(journal_key(T, DAY))
        _, again = post(store, broker, anchor)
        assert again is not None
        ctx, refused = post(store, stmt(DAY, {"AAA": 10}, 1_000.07), anchor)
        assert refused is None and ctx.metrics[0]["status"] == "FAIL"
        assert "never rewritten" in ctx.metrics[0]["status_reason"]
        assert store.get_bytes(journal_key(T, DAY)) == before

    def test_never_raises_into_the_trader(self, store, monkeypatch):
        def boom(*_a, **_k):
            raise PermissionError("AccessDenied: s3:ListBucket")

        monkeypatch.setattr(store, "list_keys", boom)
        ctx, result = post(store, stmt(DAY, {"AAA": 10}, 1_000.0), stmt(PREV, {"AAA": 10}, 1_000.0))
        assert result is None
        assert (
            ctx.metrics[0]["status"] == "FAIL" and "AccessDenied" in ctx.metrics[0]["status_reason"]
        )
        assert ctx.outputs == []

    def test_the_stored_journal_is_balanced_and_event_addressed(self, store):
        post(
            store,
            stmt(DAY, {"AAA": 12}, 799.0),
            stmt(PREV, {"AAA": 10}, 1_000.0),
            (buy("AAA", 2, 100.0, 1.0, "x1"),),
        )
        document = json.loads(store.get_bytes(journal_key(T, DAY)))
        assert document["schema_version"] == "ledger_journal.v1" and document["tenant"] == T
        for entry in document["entries"]:
            assert sum(p["amount_cents"] for p in entry["postings"]) == 0
            assert entry["tenant"] == T


class TestRefusals:
    def test_shares_must_be_whole(self):
        with pytest.raises(LedgerError, match="must be an int"):
            Posting(SECURITIES + "AAA", 0, 1.5)  # pyright: ignore[reportArgumentType]

    def test_append_rechecks_balance_and_trial_balance(self):
        entry = Entry(T, "DU1", DAY, "fill", "e1", (Posting(CASH, 1), Posting(COMMISSIONS, -1)))
        object.__setattr__(entry, "postings", (Posting(CASH, 1),))  # a tampered entry
        ledger = Ledger()
        with pytest.raises(LedgerError, match="refusing"):
            ledger.append(entry)
        assert ledger.entries == ()

    def test_a_stored_entry_whose_id_is_not_its_events_is_refused(self):
        document = Entry(T, "DU1", DAY, "fill", "e1", (Posting(CASH, 1), Posting(COMMISSIONS, -1)))
        raw = {**document.to_document(), "entry_id": "0" * 32}
        with pytest.raises(LedgerError, match="not its event's id"):
            entry_from_document(raw)
        assert entry_from_document(document.to_document()) == document

    def test_a_bad_tenant_has_no_key(self):
        with pytest.raises(LedgerError):
            journal_key("Client One", DAY)

    def test_a_fractional_split_and_a_zero_dividend_book_nothing(self):
        anchor = stmt(PREV, {"AAA": 3}, 1_000.00)
        actions = CorporateActionSet(
            frozenset({"AAA"}),
            (
                CorporateAction("AAA", "split", ratio=1.5),
                CorporateAction("AAA", "cash_dividend", amount_per_share=0.25, entitled_shares=0),
            ),
        )
        doc = {**recon_doc(stmt(DAY, {"AAA": 3}, 1_000.0), anchor), "dividend_credits": None}
        posting = book_day(
            Ledger(), DayInputs(T, stmt(DAY, {"AAA": 3}, 1_000.0), anchor, (), (), actions, doc)
        )
        assert [e.kind for e in posting.entries] == ["opening_balance"]


def test_a_journal_of_another_schema_is_refused(store):
    store.put_bytes(journal_key(T, DAY), json.dumps({"schema_version": "x"}).encode())
    with pytest.raises(LedgerError, match="journal 2026-11-03"):
        load_ledger(store, T, before=NEXT)


def test_a_metric_that_cannot_be_recorded_does_not_raise(store):
    class _Ctx:
        def __init__(self):
            self.store = store

        def record_metric(self, _metric):
            raise RuntimeError("metric sink down")

    broker = stmt(DAY, {"AAA": 10}, 1_000.0)
    assert (
        post_daily_ledger(
            _Ctx(),
            broker=broker,
            anchor=None,
            fills=(),
            cash_flows=(),
            corporate_actions=NONE,
            reconciliation={"trading_day": DAY, "account": "DU1", "void": False},
            now="2026-11-03T21:00:00Z",
        )
        is None
    )
