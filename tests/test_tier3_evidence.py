"""The tier-3 evidence pack (`alpha-engine-config-I14134`)."""

from __future__ import annotations

import pytest

from crucible_trader.exception_decision import score
from crucible_trader.ledger import (
    CASH,
    SECURITIES,
    SHARE_ADJUSTMENTS,
    SUSPENSE_CASH,
    SUSPENSE_POSITION,
    Entry,
    Posting,
    clearing_postings,
)
from crucible_trader.tier3_evidence import break_history, cash_pack, trading_days_between, usd

TENANT = "nous-ergon-paper"
BREAK_ID = "0a35189f27a511ad324df24129696aeb"

# The stored 2026-11-19 cash_delta exception (broker over by $30.74), reduced
# to the fields the pack reads: no account identifier.
RECORD_1119 = {
    "amount": 30.74,
    "class": "cash_delta",
    "detected_by": {
        "document_key": "trader/reconciliation/2026-11-19.json",
        "manifest_key": "runs/trader.reconcile/2026-11-19/run.json",
    },
    "exception_id": BREAK_ID,
    "instrument": "USD",
    "source": "crucible-trader.broker_reconciliation",
    "tenant": TENANT,
    "trading_day": "2026-11-19",
    "unit": "usd",
}
# The stored 2026-11-19 reconciliation document predates `cash_components`
# (alpha-engine-config-I14133), so these are built from its own figures:
# expected 53569.25 and IB 53599.99 are as stored; the start balance is the
# 2026-11-18 broker statement's cash; dividends and interest are the document's
# empty `dividend_credits` and null `interest_posted_usd`; no flow is
# declared; fills is the remainder to the stored expected total, because the
# stored order-intent document carries quantities and no cash.
DOC_1119 = {
    "cash_components": {
        "anchor_trading_day": "2026-11-18",
        "start_balance_usd": 76618.65,
        "fills_usd": -23049.40,
        "flows_usd": 0.0,
        "dividends_usd": 0.0,
        "interest_usd": 0.0,
        "expected_cash_usd": 53569.25,
        "broker_cash_usd": 53599.99,
    }
}


def gap(cents: int = 3074, exception_id: str = BREAK_ID) -> Entry:
    return Entry(
        TENANT,
        "ACCT",
        "2026-11-19",
        "cash_gap",
        "gap",
        (Posting(CASH, cents), Posting(SUSPENSE_CASH, -cents)),
        {"exception_ids": [exception_id]},
    )


def share_gap() -> Entry:
    return Entry(
        TENANT,
        "ACCT",
        "2026-11-19",
        "position_gap",
        "g",
        (Posting(SUSPENSE_POSITION + "AAA", 0, 2), Posting(SECURITIES + "AAA", 0, -2)),
        {"exception_ids": [BREAK_ID]},
    )


def dollars(text: str) -> int:
    return round(float(text.replace("$", "").replace(",", "")) * 100)


class TestUsd:
    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (1234.56, "$1,234.56"),
            (-1234.56, "-$1,234.56"),
            (0, "$0.00"),
            (-0.004, "$0.00"),
            (0.005, "$0.01"),
            (53599.99, "$53,599.99"),
            (1_000_000, "$1,000,000.00"),
        ],
    )
    def test_formats(self, value, text):
        assert usd(value) == text

    @pytest.mark.parametrize("value", [None, True, "5", float("nan"), float("inf")])
    def test_no_amount_is_none(self, value):
        assert usd(value) is None


class TestTheFixture1119:
    def test_all_seven_dollar_strings_and_expected_is_the_sum(self):
        pack = cash_pack(RECORD_1119, DOC_1119, [RECORD_1119], gap())
        assert pack is not None
        names = ("start_balance", "fills", "interest", "dividends", "flows")
        assert [pack[n] for n in (*names, "expected", "broker")] == [
            "$76,618.65",
            "-$23,049.40",
            "$0.00",
            "$0.00",
            "$0.00",
            "$53,569.25",
            "$53,599.99",
        ]
        assert sum(dollars(str(pack[n])) for n in names) == dollars(str(pack["expected"]))
        assert dollars(str(pack["broker"])) - dollars(str(pack["expected"])) == 3074

    def test_each_input_names_its_raw_source(self):
        pack = cash_pack(RECORD_1119, DOC_1119, [], None)
        assert pack is not None
        assert pack["input_sources"] == [
            {"role": "start_balance", "record": "trader/broker_statements/2026-11-18.json"},
            {"role": "fills", "record": "trader/order_intent_reconciliation/2026-11-19.json"},
            {
                "role": "interest",
                "record": [
                    "trader/broker_statements/2026-11-18.json",
                    "trader/broker_statements/2026-11-19.json",
                ],
            },
            {"role": "dividends", "record": "trader/reconciliation/2026-11-19.json"},
            {"role": "flows", "record": "runs/trader.reconcile/2026-11-19/run.json"},
            {"role": "broker", "record": "trader/broker_statements/2026-11-19.json"},
        ]

    def test_the_approval_journal_is_two_equal_rows(self):
        pack = cash_pack(RECORD_1119, DOC_1119, [], gap())
        assert pack is not None
        assert pack["approval_journal"] == [
            {"account": "suspense:cash_gap", "debit": "$30.74", "credit": "$0.00"},
            {
                "account": "income:reconciliation_adjustments",
                "debit": "$0.00",
                "credit": "$30.74",
            },
        ]

    def test_a_broker_under_break_reverses_the_sides(self):
        pack = cash_pack(RECORD_1119, DOC_1119, [], gap(-3074))
        assert pack is not None
        suspense, income = pack["approval_journal"]  # type: ignore[misc]
        assert (suspense["debit"], suspense["credit"]) == ("$0.00", "$30.74")
        assert (income["debit"], income["credit"]) == ("$30.74", "$0.00")

    def test_no_gap_entry_means_a_null_journal(self):
        pack = cash_pack(RECORD_1119, DOC_1119, [], None)
        assert pack is not None and pack["approval_journal"] is None


class TestShapes:
    def test_a_share_break_has_no_pack(self):
        share = {**RECORD_1119, "unit": "shares", "instrument": "AAA"}
        assert cash_pack(share, DOC_1119, [share], None) is None

    def test_without_cash_components_every_dollar_string_is_null(self):
        pack = cash_pack(RECORD_1119, {"cash_components": None}, [], None)
        assert pack is not None
        for name in ("start_balance", "fills", "interest", "dividends", "flows", "expected"):
            assert pack[name] is None
        assert pack["broker"] is None
        # Without an anchor day the two keys that name it are left out.
        assert [s["role"] for s in pack["input_sources"]] == [  # type: ignore[attr-defined]
            "fills",
            "dividends",
            "flows",
            "broker",
        ]

    def test_a_record_without_a_manifest_key_has_no_flows_source(self):
        record = {**RECORD_1119, "detected_by": {"document_key": "x"}}
        pack = cash_pack(record, DOC_1119, [], None)
        assert pack is not None
        assert "flows" not in [s["role"] for s in pack["input_sources"]]  # type: ignore[attr-defined]

    def test_a_share_gap_shows_no_dollar_rows(self):
        shares = share_gap()
        pack = cash_pack(RECORD_1119, DOC_1119, [], shares)
        assert pack is not None and pack["approval_journal"] == []


class TestHistory:
    def test_open_days_and_tied_trading_days_oldest_first(self):
        monday = {**RECORD_1119, "trading_day": "2026-11-23", "amount": 12.5}
        other_instrument = {**RECORD_1119, "trading_day": "2026-11-20", "instrument": "AAA"}
        other_tenant = {**RECORD_1119, "trading_day": "2026-11-20", "tenant": "x"}
        rows = break_history(
            RECORD_1119, [monday, RECORD_1119, other_instrument, other_tenant], "2026-11-23"
        )
        assert rows == [
            {"day": "2026-11-19", "amount": 30.74, "status": "open"},
            {"day": "2026-11-20", "amount": 0.0, "status": "tied"},
            {"day": "2026-11-23", "amount": 12.5, "status": "open"},
        ]

    def test_cash_pack_history_runs_to_as_of(self):
        pack = cash_pack(RECORD_1119, DOC_1119, [RECORD_1119], None, as_of="2026-11-20")
        assert pack is not None
        assert [r["status"] for r in pack["break_history"]] == ["open", "tied"]  # type: ignore[attr-defined]

    def test_trading_days_skip_weekends_and_holidays(self):
        assert trading_days_between("2026-11-25", "2026-11-30") == [
            "2026-11-25",
            "2026-11-27",
            "2026-11-30",
        ]


class TestClearingPostings:
    def test_a_cash_gap_clears_suspense_into_adjustments(self):
        assert clearing_postings(gap()) == [
            Posting(SUSPENSE_CASH, 3074),
            Posting("income:reconciliation_adjustments", -3074),
        ]

    def test_a_share_gap_clears_its_shares(self):
        shares = share_gap()
        assert clearing_postings(shares) == [
            Posting(SUSPENSE_POSITION + "AAA", 0, -2),
            Posting(SHARE_ADJUSTMENTS + "AAA", 0, 2),
        ]


def test_score_attaches_a_pack_by_exception_id_and_null_otherwise():
    proposals = [
        {
            "proposal_id": f"p{i}",
            "exception_id": e,
            "tenant": TENANT,
            "trading_day": "2026-11-19",
            "source": "s",
            "class": "cash_delta",
            "ruleset_version": "1",
            "rule_id": "r",
            "nature": "defect",
            "action": "investigate_cash",
            "promotable": False,
        }
        for i, e in enumerate((BREAK_ID, "other"))
    ]
    pack = {"expected": "$1.00"}
    document = score(
        proposals,
        [],
        as_of="2026-11-19",
        generated_at="t",
        packs={BREAK_ID: pack},
    )
    by_id = {q["exception_id"]: q["tier3"] for q in document["queue"]}  # type: ignore[attr-defined]
    assert by_id == {BREAK_ID: pack, "other": None}
