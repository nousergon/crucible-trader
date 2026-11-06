"""The three-way corporate-action check: Polygon, IB and the ledger.

Back office control standard, row "Three-way check" (`alpha-engine-config-I13197`).
Every number below is synthetic. The property under test: a disagreement
between the sources is ONE finding of its own class, filed and proposed like any
other, and never resolved by preferring one source.
"""

from __future__ import annotations

from typing import Any

import pytest

from crucible_trader import reconciliation_job
from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.corporate_action_check import (
    CHECK_SCHEMA_VERSION,
    MISMATCH_CLASS,
    dividend_rows,
)
from crucible_trader.exception_proposal import propose
from crucible_trader.exception_record import records_from_broker_reconciliation
from crucible_trader.ledger import Entry, Ledger, Posting
from crucible_trader.reconciliation import (
    CorporateAction,
    CorporateActionSet,
    Fill,
    ReconciliationInputs,
    reconcile,
)
from crucible_trader.reconciliation_control import run_control_arm

ACCOUNT = "DU0000001"
DAY, PREV = "2026-09-11", "2026-09-10"


def statement(positions: dict[str, int], cash: float = 10_000.0, day: str = DAY) -> BrokerStatement:
    return BrokerStatement(account=ACCOUNT, trading_day=day, positions=positions, cash_usd=cash)


def run(
    anchor: dict[str, int],
    broker: dict[str, int],
    actions: tuple[CorporateAction, ...] = (),
    *,
    ledger: dict[str, int] | None = None,
    fills: tuple[Fill, ...] = (),
    cash: float = 10_000.0,
    resolved: frozenset[str] | None = None,
) -> Any:  # noqa: ANN401 - the result, typed by the reconciler
    tickers = set(anchor) | set(broker) | {a.ticker for a in actions} | {f.ticker for f in fills}
    return reconcile(
        ReconciliationInputs(
            trading_day=DAY,
            broker=statement(broker, cash=cash),
            anchor=statement(anchor, day=PREV),
            fills=fills,
            corporate_actions=CorporateActionSet(
                resolved=frozenset(tickers) if resolved is None else resolved,
                actions=actions,
            ),
            previous_trading_day=PREV,
            ledger_positions=ledger,
            ledger_source="test ledger" if ledger is not None else "unavailable: not supplied",
        )
    )


def classes(result: Any) -> list[str]:  # noqa: ANN401
    return [d.klass for d in result.discrepancies]


def rows(result: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    return result.to_document()["corporate_action_check"]["shares"]


SPLIT = CorporateAction("AAA", "split", ratio=2.0)


class TestSplits:
    def test_all_three_agree_is_clean_and_recorded(self):
        result = run({"AAA": 100}, {"AAA": 200}, (SPLIT,), ledger={"AAA": 100})
        assert result.clean
        check = result.to_document()["corporate_action_check"]
        assert check["schema_version"] == CHECK_SCHEMA_VERSION
        assert check["sources"]["ledger"] == "test ledger"
        assert check["mismatches"] == check["disagreements"] == 0
        (row,) = check["shares"]
        assert (row["polygon_shares"], row["ib_shares"], row["ledger_shares"]) == (100, 100, 100)
        assert row["status"] == "agree" and row["finding"] is None

    def test_ib_applying_a_different_ratio_is_one_mismatch_not_a_share_delta(self):
        result = run({"AAA": 100}, {"AAA": 300}, (SPLIT,), ledger={"AAA": 100})
        assert classes(result) == [MISMATCH_CLASS]
        (finding,) = result.discrepancies
        assert (finding.expected, finding.actual, finding.delta, finding.side) == (
            100.0,
            200.0,
            100.0,
            "broker",
        )
        assert "no source is preferred" in finding.detail
        assert rows(result)[0]["status"] == "mismatch"
        assert result.to_document()["corporate_action_check"]["mismatches"] == 1

    def test_ib_not_applying_the_split_stays_the_one_corporate_action_finding(self):
        result = run({"AAA": 100}, {"AAA": 100}, (SPLIT,), ledger={"AAA": 100})
        assert classes(result) == ["corporate_action"]
        (row,) = rows(result)
        assert (row["status"], row["finding"]) == ("ib_not_applied", "corporate_action")
        check = result.to_document()["corporate_action_check"]
        assert (check["mismatches"], check["disagreements"]) == (0, 1)

    def test_a_ledger_that_would_book_differently_is_a_ledger_side_mismatch(self):
        # IB and Polygon agree; the ledger's prior holding differs, so its
        # split entry would not match either of them.
        result = run({"AAA": 100}, {"AAA": 200}, (SPLIT,), ledger={"AAA": 60})
        assert classes(result) == [MISMATCH_CLASS]
        (finding,) = result.discrepancies
        # aec#13237 802d1ddd: the finding carries the LEDGER's change (60 -> 120,
        # +60) against Polygon's +100, not IB's matching +100.
        assert (finding.expected, finding.actual, finding.delta, finding.side) == (
            100.0,
            60.0,
            -40.0,
            "ledger",
        )
        assert rows(result)[0]["ledger_shares"] == 60

    def test_zero_ib_movement_does_not_hide_a_ledger_only_disagreement(self):
        # aec#13237 3a427069: splits compounding to 1:1 move neither IB nor
        # Polygon, but the ledger books each whole-share rebase (+100, -50).
        # IB agrees with Polygon, so this is a ledger-side mismatch, never
        # "IB has not applied it".
        undo = CorporateAction("AAA", "split", ratio=0.5)
        result = run({"AAA": 100}, {"AAA": 100}, (SPLIT, undo), ledger={"AAA": 100})
        (row,) = rows(result)
        assert (row["status"], row["ib_shares"], row["polygon_shares"]) == ("mismatch", 0, 0)
        assert classes(result) == [MISMATCH_CLASS]
        (finding,) = result.discrepancies
        assert (finding.actual, finding.side) == (50.0, "ledger")

    def test_an_unreadable_ledger_compares_the_other_two_and_says_so(self):
        result = run({"AAA": 100}, {"AAA": 200}, (SPLIT,))
        assert result.clean
        (row,) = rows(result)
        assert row["ledger_shares"] is None
        assert "ledger leg unavailable" in row["detail"]
        mismatch = run({"AAA": 100}, {"AAA": 300}, (SPLIT,))
        assert classes(mismatch) == [MISMATCH_CLASS]
        assert "n/a (unavailable)" in mismatch.discrepancies[0].detail

    def test_the_ledger_leg_books_whole_shares_only(self):
        # 3-for-2 on the ledger's 101 shares is not whole: the ledger books
        # nothing, while IB's 100 -> 150 agrees with Polygon.
        split = CorporateAction("AAA", "split", ratio=1.5)
        result = run({"AAA": 100}, {"AAA": 150}, (split,), ledger={"AAA": 101})
        assert classes(result) == [MISMATCH_CLASS]
        assert rows(result)[0]["ledger_shares"] == 0

    def test_a_split_on_a_traded_name_nets_the_executions(self):
        fill = Fill("AAA", 10, -1_000.0)
        result = run(
            {"AAA": 100}, {"AAA": 210}, (SPLIT,), ledger={"AAA": 100}, fills=(fill,), cash=9_000.0
        )
        assert result.clean
        assert rows(result)[0]["ib_shares"] == 100


class TestWhatIsNotAMismatch:
    def test_a_non_whole_split_is_not_comparable_and_stays_unhandled(self):
        split = CorporateAction("AAA", "split", ratio=1.5)
        result = run({"AAA": 101}, {"AAA": 151}, (split,), ledger={"AAA": 101})
        assert MISMATCH_CLASS not in classes(result)
        assert "unhandled_corporate_action" in classes(result)
        assert rows(result)[0]["status"] == "not_comparable"

    def test_a_malformed_split_is_not_comparable_on_every_leg(self):
        split = CorporateAction("AAA", "split", ratio=None)
        result = run({"AAA": 100}, {"AAA": 100}, (split,), ledger={"AAA": 100})
        assert MISMATCH_CLASS not in classes(result)
        (row,) = rows(result)
        assert (row["status"], row["polygon_shares"], row["ledger_shares"]) == (
            "not_comparable",
            None,
            0,
        )

    def test_an_unhandled_kind_is_not_comparable(self):
        spin = CorporateAction("AAA", "spin_off")
        result = run({"AAA": 100}, {"AAA": 100}, (spin,), ledger={"AAA": 100})
        assert MISMATCH_CLASS not in classes(result)
        (row,) = rows(result)
        assert (row["status"], row["finding"]) == ("not_comparable", "unhandled_corporate_action")
        assert row["polygon"] == ["spin_off"]

    def test_ib_moving_shares_with_no_polygon_action_is_ib_only_and_a_share_delta(self):
        result = run({"AAA": 100}, {"AAA": 50}, ledger={"AAA": 100})
        assert classes(result) == ["share_delta"]
        (row,) = rows(result)
        assert (row["status"], row["finding"], row["ib_shares"]) == ("ib_only", "share_delta", -50)
        assert (
            run({"AAA": 100}, {"AAA": 50}).to_document()["corporate_action_check"]["shares"][0][
                "ledger_shares"
            ]
            is None
        )

    def test_an_unresolved_instrument_is_not_compared(self):
        result = run({"AAA": 100}, {"AAA": 300}, (SPLIT,), resolved=frozenset())
        assert MISMATCH_CLASS not in classes(result)
        assert rows(result) == []

    def test_no_anchor_has_no_check(self):
        result = reconcile(
            ReconciliationInputs(trading_day=DAY, broker=statement({"AAA": 1}), anchor=None)
        )
        assert result.to_document()["corporate_action_check"] is None


class TestDividends:
    def test_each_paying_dividend_records_ibs_aggregate_cash_tie(self):
        assert dividend_rows({"AAA": 12.5, "ZERO": 0.0}, unpaid=(), cash_break=False) == [
            {
                "instrument": "AAA",
                "polygon_usd": 12.5,
                "ib": "agree",
                "status": "agree",
                "detail": "IB's cash ties with this dividend credited",
            }
        ]
        (not_credited,) = dividend_rows({"AAA": 12.5}, unpaid=("AAA",), cash_break=True)
        assert (not_credited["ib"], not_credited["status"]) == ("ib_not_credited", "reported")
        (broken,) = dividend_rows({"AAA": 12.5}, unpaid=(), cash_break=True)
        assert broken["ib"] == "cash_break"

    def test_the_reconciler_carries_dividend_rows(self):
        dividend = CorporateAction("AAA", "cash_dividend", amount_per_share=0.5)
        result = run({"AAA": 100}, {"AAA": 100}, (dividend,), cash=10_050.0)
        assert result.clean
        (row,) = result.to_document()["corporate_action_check"]["dividends"]
        assert (row["instrument"], row["ib"]) == ("AAA", "agree")
        assert rows(result) == []


class TestTheMismatchIsItsOwnException:
    def test_it_files_as_a_record_with_a_proposal(self):
        result = run({"AAA": 100}, {"AAA": 300}, (SPLIT,), ledger={"AAA": 100})
        document = {**result.to_document(), "void": False}
        (record,) = records_from_broker_reconciliation(document, detected_at=f"{DAY}T21:00:00Z")
        assert record.klass == MISMATCH_CLASS
        assert record.to_document()["unit"] == "shares"
        proposal = propose(record.to_document(), proposed_at=f"{DAY}T21:00:01Z").to_document()
        assert proposal["rule_id"] == "corporate_action.source_mismatch"
        assert proposal["action"] == "investigate_shares"

    def test_the_control_arm_still_passes_over_a_split_day(self):
        base = ReconciliationInputs(
            trading_day=DAY,
            broker=statement({"AAA": 200}),
            anchor=statement({"AAA": 100}, day=PREV),
            corporate_actions=CorporateActionSet(resolved=frozenset({"AAA"}), actions=(SPLIT,)),
            previous_trading_day=PREV,
            ledger_positions={"AAA": 100},
        )
        assert run_control_arm(base).passed


class _Store:
    pass


class TestTheLedgerLeg:
    def _ledger(self) -> Ledger:
        ledger = Ledger()
        ledger.append(
            Entry(
                "nous-ergon-paper",
                ACCOUNT,
                PREV,
                "opening_balance",
                f"statement/{PREV}",
                (
                    Posting("assets:cash:USD", 100),
                    Posting("equity:opening_balance", -100),
                    Posting("assets:securities:AAA", 0, 100),
                ),
                {"source": "ib_statement"},
            )
        )
        return ledger

    def test_it_reads_the_ledger_before_the_day(self, monkeypatch: pytest.MonkeyPatch):
        ledger = self._ledger()
        monkeypatch.setattr(reconciliation_job, "load_ledger", lambda *_a, **_k: ledger)
        positions, source = reconciliation_job.ledger_leg(_Store(), DAY, ACCOUNT)  # pyright: ignore[reportArgumentType]
        assert positions == {"AAA": 100}
        assert source == f"ledger journals (nous-ergon-paper) before {DAY}"

    def test_no_ledger_for_the_account_is_unavailable(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(reconciliation_job, "load_ledger", lambda *_a, **_k: Ledger())
        positions, source = reconciliation_job.ledger_leg(_Store(), DAY, ACCOUNT)  # pyright: ignore[reportArgumentType]
        assert positions is None and source.startswith("unavailable: no nous-ergon-paper ledger")

    def test_an_unreadable_ledger_never_raises(self, monkeypatch: pytest.MonkeyPatch):
        def boom(*_a: object, **_k: object) -> Ledger:
            raise OSError("store down")

        monkeypatch.setattr(reconciliation_job, "load_ledger", boom)
        positions, source = reconciliation_job.ledger_leg(_Store(), DAY, ACCOUNT)  # pyright: ignore[reportArgumentType]
        assert positions is None and source == "unavailable: OSError: store down"
