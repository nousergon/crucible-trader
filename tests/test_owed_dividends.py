"""Dividends the broker credits after their pay date, and declared entitlements.

`alpha-engine-config-I12376`. IB's paper account credited MU's 2026-10-29
dividend late, and SPY's 2026-10-30 dividend has a record session (2026-09-17)
before the trader's first stored statement. The negative cases are the point: a
credit that does not agree with what was owed stays a `cash_delta`, and an
entitlement that is derived is marked estimated.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pytest
from crucible.keys import trader_reconciliation_key as reconciliation_key
from crucible.runner import RunContext
from crucible.store import LocalStore

from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.dividend_entitlements import (
    DeclaredEntitlementError,
    declared_owed,
    load_declared_entitlements,
    owed_dividends,
)
from crucible_trader.reconciliation import (
    CorporateAction,
    CorporateActionSet,
    OwedDividend,
    ReconciliationInputs,
    reconcile,
)
from crucible_trader.reconciliation_control import run_control_arm
from crucible_trader.reconciliation_job import (
    RECONCILE_JOB,
    ReconciliationDiscrepancyError,
    reconcile_cycle,
    stored_results,
)

DAY = "2026-11-02"
PREV = "2026-10-30"
MU = OwedDividend("MU", "2026-10-29", 11.85)
SPY = OwedDividend("SPY", "2026-10-30", 18.89, "declared: v1 IB positions", True)


def run(broker_cash, owed=(), day=DAY, actions=(), resolved=("AAA",)):
    inputs = ReconciliationInputs(
        trading_day=day,
        broker=BrokerStatement("DU1", day, {"AAA": 10}, broker_cash),
        anchor=BrokerStatement("DU1", PREV, {"AAA": 10}, 1_000.0),
        corporate_actions=CorporateActionSet(frozenset(resolved), tuple(actions)),
        previous_trading_day=PREV,
        owed_dividends=tuple(owed),
    )
    return inputs, reconcile(inputs)


class TestLateCredit:
    def test_a_credit_equal_to_what_was_owed_is_explained(self):
        _, result = run(1_000.0 + 11.85 + 18.89, owed=(MU, SPY))
        assert result.clean
        assert {c.ticker for c in result.dividend_credits} == {"MU", "SPY"}
        doc = result.to_document()
        assert {r["ticker"] for r in doc["dividend_credits"]} == {"MU", "SPY"}
        assert [r["estimated"] for r in doc["dividend_credits"] if r["ticker"] == "SPY"] == [True]

    def test_one_of_two_credited_matches_only_that_one(self):
        _, result = run(1_000.0 + 18.89, owed=(MU, SPY))
        assert result.clean
        assert [c.ticker for c in result.dividend_credits] == ["SPY"]

    def test_nothing_credited_within_the_window_is_not_a_finding(self):
        _, result = run(1_000.0, owed=(MU, SPY))
        assert result.clean
        assert result.dividend_credits == ()

    def test_a_credit_that_disagrees_with_the_derived_entitlement_stays_a_cash_delta(self):
        # 11 shares' worth of SPY, not the 10 derived: the credit never confirms itself.
        _, result = run(1_000.0 + round(11 * 1.888834, 2), owed=(SPY,))
        assert [d.klass for d in result.discrepancies] == ["cash_delta"]
        assert result.dividend_credits == ()

    def test_an_owed_dividend_past_the_window_is_a_finding(self):
        _, result = run(1_000.0, owed=(MU,), day="2026-11-09")
        assert [d.klass for d in result.discrepancies] == ["corporate_action"]
        assert "still not credited after 11 days" in result.discrepancies[0].detail

    def test_the_control_arm_still_catches_every_plant(self):
        inputs, _ = run(1_000.0 + 11.85, owed=(MU, SPY))
        assert run_control_arm(inputs).passed


class TestUnpaidToday:
    def test_an_unpaid_dividend_is_recorded_as_owed_with_its_entitlement(self):
        action = CorporateAction(
            "AAA",
            "cash_dividend",
            amount_per_share=1.5,
            entitled_shares=10,
            entitlement_source="declared: x",
            estimated=True,
        )
        _, result = run(1_000.0, day=DAY, actions=(action,))
        assert [d.klass for d in result.discrepancies] == ["corporate_action"]
        assert result.dividends_unpaid == (OwedDividend("AAA", DAY, 15.0, "declared: x", True),)


def legacy_mu_result():
    return {
        "discrepancies": [
            {
                "klass": "corporate_action",
                "instrument": "MU",
                "detail": "cash is short exactly the MU dividend (11.85); "
                "it is not applied on the broker side",
            }
        ]
    }


class TestOwedFromStoredResults:
    def test_a_legacy_result_row_is_read_as_owed(self):
        owed = owed_dividends({"2026-10-29": legacy_mu_result()}, today=DAY, declared={})
        assert owed == (MU,)

    def test_a_credited_dividend_is_no_longer_owed(self):
        results = {
            "2026-10-29": legacy_mu_result(),
            "2026-10-30": {"dividend_credits": [{"ticker": "MU", "pay_date": "2026-10-29"}]},
        }
        assert owed_dividends(results, today=DAY, declared={}) == ()

    def test_results_before_the_bootstrap_are_not_read(self):
        assert owed_dividends({"2026-10-28": legacy_mu_result()}, today=DAY, declared={}) == ()

    def test_a_declared_entitlement_is_owed_after_its_pay_date_only(self):
        declared = load_declared_entitlements()
        assert owed_dividends({}, today="2026-10-30", declared=declared) == ()
        (spy,) = owed_dividends({}, today=DAY, declared=declared)
        assert (spy.ticker, spy.pay_date, spy.amount_usd, spy.estimated) == (
            "SPY",
            "2026-10-30",
            18.89,
            True,
        )
        assert spy.entitlement_source.startswith("declared: ")


class TestDeclaredFile:
    def test_the_shipped_spy_entitlement_is_ten_shares_and_estimated(self):
        entry = load_declared_entitlements()[("SPY", "2026-10-30")]
        assert entry["entitled_shares"] == 10
        assert entry["record_session"] == "2026-09-17"
        assert declared_owed(entry).amount_usd == 18.89

    @pytest.mark.parametrize(
        ("field", "value"),
        [("estimated", False), ("entitled_shares", 2.5), ("amount_per_share", 0), ("ruling", "")],
    )
    def test_a_malformed_entry_is_refused(self, tmp_path: pathlib.Path, field, value):
        entry = {
            "ticker": "SPY",
            "pay_date": "2026-10-30",
            "record_session": "2026-09-17",
            "entitled_shares": 10,
            "amount_per_share": 1.888834,
            "estimated": True,
            "source": "s",
            "ruling": "r",
            field: value,
        }
        path = tmp_path / "d.json"
        path.write_text(json.dumps({"schema_version": 1, "entitlements": [entry]}))
        with pytest.raises(DeclaredEntitlementError):
            load_declared_entitlements(path)


class TestTheJobCarriesOwedDividends:
    def ctx(self, store, day):
        return RunContext(
            run_id="01JG0000000000000000000000",
            job=RECONCILE_JOB,
            trading_day=day,
            calendar_date=day,
            store=store,
            seed=0,
            started=dt.datetime(2026, 11, 2, 21, 0, tzinfo=dt.UTC),
        )

    def source(self, cash):
        class Source:
            def read_statement(self, trading_day):
                return BrokerStatement("DU1", trading_day, {"AAA": 10}, cash)

        return Source()

    def seed(self, store):
        from crucible.keys import trader_broker_statement_key

        anchor = BrokerStatement("DU1", PREV, {"AAA": 10}, 1_000.0)
        store.put_bytes(
            trader_broker_statement_key(PREV), json.dumps(anchor.to_document()).encode()
        )
        store.put_bytes(reconciliation_key("2026-10-29"), json.dumps(legacy_mu_result()).encode())

    def test_a_late_credit_of_mu_and_spy_reconciles_clean(self, tmp_path):
        store = LocalStore(tmp_path)
        self.seed(store)
        ctx = self.ctx(store, dt.date(2026, 11, 2))
        result = reconcile_cycle(
            ctx,
            source=self.source(1_000.0 + 11.85 + 18.89),
            fills=(),
            cash_flows=(),
            corporate_actions=CorporateActionSet(frozenset({"AAA"})),
        )
        assert result.clean
        assert {c.ticker for c in result.dividend_credits} == {"MU", "SPY"}

    def test_a_wrong_credit_still_pages(self, tmp_path):
        store = LocalStore(tmp_path)
        self.seed(store)
        ctx = self.ctx(store, dt.date(2026, 11, 2))
        with pytest.raises(ReconciliationDiscrepancyError, match="cash_delta"):
            reconcile_cycle(
                ctx,
                source=self.source(1_000.0 + 25.00),
                fills=(),
                cash_flows=(),
                corporate_actions=CorporateActionSet(frozenset({"AAA"})),
            )


class TestDeclaredFileShape:
    @pytest.mark.parametrize(
        "document",
        [
            [],
            {"schema_version": 2, "entitlements": []},
            {"schema_version": 1, "entitlements": {}},
            {"schema_version": 1, "entitlements": ["SPY"]},
        ],
    )
    def test_a_malformed_file_is_refused(self, tmp_path: pathlib.Path, document):
        path = tmp_path / "d.json"
        path.write_text(json.dumps(document))
        with pytest.raises(DeclaredEntitlementError):
            load_declared_entitlements(path)

    def test_a_second_entitlement_for_the_same_dividend_is_refused(self, tmp_path):
        entry = load_declared_entitlements()[("SPY", "2026-10-30")]
        path = tmp_path / "d.json"
        path.write_text(json.dumps({"schema_version": 1, "entitlements": [entry, entry]}))
        with pytest.raises(DeclaredEntitlementError, match="second entitlement"):
            load_declared_entitlements(path)


def test_a_current_result_is_read_from_its_dividends_unpaid_field():
    _, result = run(
        1_000.0, actions=(CorporateAction("AAA", "cash_dividend", amount_per_share=1.5),)
    )
    document = result.to_document()
    assert owed_dividends({DAY: document}, today="2026-11-03", declared={}) == (
        OwedDividend("AAA", DAY, 15.0),
    )


def test_non_json_keys_under_the_results_prefix_are_ignored(tmp_path):
    store = LocalStore(tmp_path)
    store.put_bytes(reconciliation_key("2026-10-29") + ".tmp", b"not json")
    assert stored_results(store, DAY) == {}
