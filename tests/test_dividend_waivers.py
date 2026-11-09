"""Owed dividends an operator waived (`alpha-engine-config-I12024`).

IB's paper account never credited MU's 2026-10-29 dividend (11.85) or SPY's
2026-10-30 one (18.89), and from 2026-11-09 each failed the reconcile as owed
past `OWED_DIVIDEND_MAX_DAYS`. The negative cases are the point: a dividend
nobody waived still fails, and a waiver whose amount is not exactly what is
owed is refused, never stretched to fit.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib

import pytest
from crucible.keys import trader_broker_statement_key, trader_reconciliation_key
from crucible.runner import RunContext
from crucible.store import LocalStore

from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.dividend_entitlements import (
    DeclaredEntitlementError,
    declared_owed,
    load_declared_entitlements,
    load_declared_waivers,
)
from crucible_trader.reconciliation import (
    OWED_DIVIDEND_MAX_DAYS,
    CorporateActionSet,
    DividendWaiver,
    DividendWaiverError,
    OwedDividend,
    ReconciliationInputs,
    reconcile,
)
from crucible_trader.reconciliation_control import run_control_arm
from crucible_trader.reconciliation_job import (
    RECONCILE_JOB,
    ReconciliationDiscrepancyError,
    reconcile_cycle,
)

DAY = "2026-11-10"
PREV = "2026-11-09"
MU = OwedDividend("MU", "2026-10-29", 11.85)
SPY = OwedDividend("SPY", "2026-10-30", 18.89, "declared: v1 IB positions", True)


def waiver(owed: OwedDividend, amount: float | None = None) -> DividendWaiver:
    return DividendWaiver(
        ticker=owed.ticker,
        pay_date=owed.pay_date,
        amount_usd=owed.amount_usd if amount is None else amount,
        ruling="test ruling",
        ruled_on="2026-11-09",
        evidence=("IB paper account did not credit",),
    )


def inputs(owed=(MU, SPY), waivers=(), cash=1_000.0, anchor=True):
    return ReconciliationInputs(
        trading_day=DAY,
        broker=BrokerStatement("DU1", DAY, {"AAA": 10}, cash),
        anchor=BrokerStatement("DU1", PREV, {"AAA": 10}, 1_000.0) if anchor else None,
        corporate_actions=CorporateActionSet(frozenset({"AAA"}), ()),
        previous_trading_day=PREV,
        genesis=not anchor,
        owed_dividends=tuple(owed),
        dividend_waivers=tuple(waivers),
    )


def corporate_action_findings(result):
    return [d for d in result.discrepancies if d.klass == "corporate_action"]


class TestTheReconciler:
    def test_an_unwaived_owed_dividend_past_the_limit_still_fails(self):
        # Both are past OWED_DIVIDEND_MAX_DAYS on DAY; neither is waived.
        assert (dt.date.fromisoformat(DAY) - dt.date(2026, 10, 30)).days > OWED_DIVIDEND_MAX_DAYS
        result = reconcile(inputs())
        assert {d.instrument for d in corporate_action_findings(result)} == {"MU", "SPY"}
        assert result.dividends_waived == ()

    def test_waiving_one_dividend_leaves_the_other_failing(self):
        result = reconcile(inputs(waivers=(waiver(MU),)))
        assert [d.instrument for d in corporate_action_findings(result)] == ["SPY"]
        assert result.dividends_owed == (SPY,)
        assert result.dividends_waived == (waiver(MU),)

    def test_a_waived_dividend_is_not_owed_raises_nothing_and_is_recorded(self):
        result = reconcile(inputs(waivers=(waiver(MU), waiver(SPY))))
        assert result.clean
        assert result.dividends_owed == ()
        document = result.to_document()
        assert [
            (r["ticker"], r["pay_date"], r["amount_usd"]) for r in document["dividends_waived"]
        ] == [
            ("MU", "2026-10-29", 11.85),
            ("SPY", "2026-10-30", 18.89),
        ]
        assert document["dividends_waived"][0]["evidence"] == ["IB paper account did not credit"]
        json.dumps(document)  # the stored document must serialise

    @pytest.mark.parametrize("amount", [11.84, 11.86, 118.5, 11.0])
    def test_a_waiver_whose_amount_differs_is_refused(self, amount):
        with pytest.raises(DividendWaiverError, match="declares .* but 11.85 is owed"):
            inputs(waivers=(waiver(MU, amount),))

    def test_a_second_waiver_for_one_key_is_refused(self):
        with pytest.raises(DividendWaiverError, match="a second waiver"):
            inputs(waivers=(waiver(MU), waiver(MU)))

    def test_a_waiver_for_a_dividend_not_owed_is_inert_and_not_recorded(self):
        result = reconcile(inputs(owed=(MU,), waivers=(waiver(SPY),)))
        assert result.dividends_waived == ()
        assert [d.instrument for d in corporate_action_findings(result)] == ["MU"]

    def test_a_waived_dividend_absorbs_no_later_credit(self):
        # If IB pays after all, the cash is unexplained: the waiver is wrong
        # and must be withdrawn, never matched silently.
        result = reconcile(inputs(waivers=(waiver(MU), waiver(SPY)), cash=1_000.0 + 11.85))
        assert [d.klass for d in result.discrepancies] == ["cash_delta"]
        assert result.dividend_credits == ()

    def test_a_genesis_day_records_its_waivers(self):
        result = reconcile(inputs(waivers=(waiver(MU),), anchor=False))
        assert result.dividends_owed == (SPY,)
        assert result.dividends_waived == (waiver(MU),)

    def test_the_control_arm_still_sees_every_plant_with_waivers(self):
        base = inputs(waivers=(waiver(MU), waiver(SPY)))
        verdict = run_control_arm(base, reconciler=reconcile)
        assert verdict.passed, verdict.reason()


class TestTheDeclaredFile:
    def test_the_shipped_waivers_are_exactly_mu_and_spy(self):
        waivers = load_declared_waivers()
        assert [(w.ticker, w.pay_date, w.amount_usd) for w in waivers] == [
            ("MU", "2026-10-29", 11.85),
            ("SPY", "2026-10-30", 18.89),
        ]
        for w in waivers:
            assert w.ruling and w.ruled_on == "2026-11-09"
            assert "IB paper account did not credit" in w.evidence

    def test_the_spy_waiver_matches_its_declared_entitlement_to_the_cent(self):
        spy = next(w for w in load_declared_waivers() if w.ticker == "SPY")
        owed = declared_owed(load_declared_entitlements()[("SPY", "2026-10-30")])
        assert round(owed.amount_usd * 100) == round(spy.amount_usd * 100)

    def base_entry(self):
        return {
            "ticker": "MU",
            "pay_date": "2026-10-29",
            "amount_usd": 11.85,
            "ruling": "r",
            "ruled_on": "2026-11-09",
            "evidence": ["IB paper account did not credit"],
        }

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"evidence": []}, "evidence"),
            ({"evidence": [""]}, "evidence"),
            ({"ruling": ""}, "ruling"),
            ({"amount_usd": 0}, "amount_usd"),
            ({"amount_usd": True}, "amount_usd"),
            ({"amount_usd": 11.855}, "whole cents"),
        ],
    )
    def test_a_malformed_waiver_is_refused(self, tmp_path, change, message):
        path = tmp_path / "w.json"
        path.write_text(
            json.dumps({"schema_version": 1, "waivers": [{**self.base_entry(), **change}]})
        )
        with pytest.raises(DeclaredEntitlementError, match=message):
            load_declared_waivers(path)

    @pytest.mark.parametrize(
        ("document", "message"),
        [
            ([], "not a JSON object"),
            ({"schema_version": 2, "waivers": []}, "schema_version 1"),
            ({"schema_version": 1, "waivers": {}}, "not a list"),
            ({"schema_version": 1, "waivers": ["MU"]}, "not an object"),
        ],
    )
    def test_a_malformed_file_is_refused(self, tmp_path, document, message):
        path = tmp_path / "w.json"
        path.write_text(json.dumps(document))
        with pytest.raises(DeclaredEntitlementError, match=message):
            load_declared_waivers(path)

    def test_a_duplicate_key_is_refused(self, tmp_path):
        path = tmp_path / "w.json"
        entry = self.base_entry()
        path.write_text(json.dumps({"schema_version": 1, "waivers": [entry, entry]}))
        with pytest.raises(DeclaredEntitlementError, match="a second waiver"):
            load_declared_waivers(path)


class TestTheJob:
    """2026-11-10 replayed: MU owed from the stored 2026-10-29 result, SPY from
    its declared entitlement, and IB's cash showing neither."""

    def run(self, tmp_path: pathlib.Path, **kwargs):
        store = LocalStore(tmp_path)
        anchor = BrokerStatement("DU1", PREV, {"AAA": 10}, 1_000.0)
        store.put_bytes(
            trader_broker_statement_key(PREV), json.dumps(anchor.to_document()).encode()
        )
        legacy = {
            "discrepancies": [
                {
                    "klass": "corporate_action",
                    "instrument": "MU",
                    "detail": "cash is short exactly the MU dividend (11.85); "
                    "it is not applied on the broker side",
                }
            ]
        }
        store.put_bytes(trader_reconciliation_key("2026-10-29"), json.dumps(legacy).encode())

        class Source:
            def read_statement(self, trading_day):
                return BrokerStatement("DU1", trading_day, {"AAA": 10}, 1_000.0)

        ctx = RunContext(
            run_id="01JG0000000000000000000000",
            job=RECONCILE_JOB,
            trading_day=dt.date(2026, 11, 10),
            calendar_date=dt.date(2026, 11, 10),
            store=store,
            seed=0,
            started=dt.datetime(2026, 11, 10, 21, 0, tzinfo=dt.UTC),
        )
        result = reconcile_cycle(
            ctx,
            source=Source(),
            fills=(),
            cash_flows=(),
            corporate_actions=CorporateActionSet(frozenset({"AAA"})),
            **kwargs,
        )
        return result, json.loads(store.get_bytes(trader_reconciliation_key(DAY)))

    def test_without_waivers_both_dividends_page(self, tmp_path):
        with pytest.raises(ReconciliationDiscrepancyError, match="corporate_action: MU"):
            self.run(tmp_path, dividend_waivers=())

    def test_the_shipped_waivers_clear_both_and_record_them(self, tmp_path):
        result, document = self.run(tmp_path)
        assert result.clean
        assert document["dividends_owed"] == []
        assert [(r["ticker"], r["amount_usd"]) for r in document["dividends_waived"]] == [
            ("MU", 11.85),
            ("SPY", 18.89),
        ]

    def test_a_shipped_waiver_off_by_a_cent_fails_the_run(self, tmp_path):
        wrong = tuple(
            dataclasses.replace(w, amount_usd=11.84) if w.ticker == "MU" else w
            for w in load_declared_waivers()
        )
        with pytest.raises(DividendWaiverError):
            self.run(tmp_path, dividend_waivers=wrong)
