"""`fills_from_ib`, `live_statement_source` and `live_corporate_actions`: the
seam supplying `reconcile_cycle`'s broker-derived arguments (`alpha-engine-
config-I11071`). Fills are read from a fake shaped to the documented
`ib_async` `Fill`/`Execution`/`CommissionReport` fields; there is no live
gateway in this session to exercise the real SDK against."""

from __future__ import annotations

import datetime as dt

import pytest
from ib_fakes import as_requested, broker_fill

from crucible_trader.reconcile_entrypoint import (
    COMMISSION_REPORT_WAIT_S,
    MissingCommissionReportError,
    UnrecognizedFillSideError,
    fills_from_ib,
    live_corporate_actions,
    live_statement_source,
)
from crucible_trader.reconciliation import CorporateActionSet, Fill

DAY = "2026-09-11"
T = dt.datetime(2026, 9, 11, 15, 0, tzinfo=dt.UTC)
OTHER_DAY = dt.datetime(2026, 9, 10, 15, 0, tzinfo=dt.UTC)


class _Client:
    """`reqExecutions()` answers as `ib_async` 2.1.0 does after its startup
    sync (fresh records, unattached default reports); `fills()` is the
    session store, whose reports the SDK attaches. ``late`` reports land in
    the store only once the session's event loop runs (`sleep`)."""

    def __init__(self, records, *, late=(), answer=as_requested):
        self._store = list(records)
        self._answer = [answer(r) for r in records]
        self._late = list(late)
        self.slept = 0.0

    def reqExecutions(self):  # noqa: N802 - SDK spelling
        return list(self._answer)

    def fills(self):
        return list(self._store)

    def sleep(self, seconds):
        self.slept += seconds
        for record in self._late:
            for i, held in enumerate(self._store):
                if held.execution.execId == record.execution.execId:
                    self._store[i] = record
        self._late = []


class TestFillsFromIb:
    def test_a_buy_is_signed_positive_and_costs_notional_plus_commission(self):
        client = _Client(
            [broker_fill("AAA", side="BOT", shares=10, avg_price=20.0, commission=1.0, time=T)]
        )
        (fill,) = fills_from_ib(client, DAY)
        assert fill == Fill(ticker="AAA", shares=10, cash_usd=-201.0)  # -(200) - 1

    def test_a_sell_is_signed_negative_and_credits_notional_minus_commission(self):
        client = _Client(
            [broker_fill("AAA", side="SLD", shares=10, avg_price=20.0, commission=1.0, time=T)]
        )
        (fill,) = fills_from_ib(client, DAY)
        assert fill == Fill(ticker="AAA", shares=-10, cash_usd=199.0)

    def test_fills_on_another_day_are_excluded(self):
        client = _Client(
            [
                broker_fill("AAA", side="BOT", shares=1, avg_price=1.0, time=T),
                broker_fill("BBB", side="BOT", shares=1, avg_price=1.0, time=OTHER_DAY),
            ]
        )
        (fill,) = fills_from_ib(client, DAY)
        assert fill.ticker == "AAA"

    def test_another_api_clients_fills_are_the_accounts_fills(self):
        """alpha-engine-config-I12376: the paper account is shared with the v1
        executor (client ids 1-3). On a master session its executions arrive
        beside the trader's own, and each is a fill of the account."""
        client = _Client(
            [
                broker_fill(
                    "UAL",
                    side="BOT",
                    shares=768,
                    avg_price=50.0,
                    commission=1.0,
                    time=T,
                    client_id=1,
                ),
                broker_fill(
                    "HL", side="SLD", shares=100, avg_price=5.0, commission=1.0, time=T, client_id=2
                ),
                broker_fill(
                    "AAA", side="BOT", shares=1, avg_price=1.0, commission=0.0, time=T, client_id=41
                ),
            ]
        )
        assert fills_from_ib(client, DAY) == (
            Fill(ticker="UAL", shares=768, cash_usd=-38_401.0),
            Fill(ticker="HL", shares=-100, cash_usd=499.0),
            Fill(ticker="AAA", shares=1, cash_usd=-1.0),
        )

    def test_no_fills_today_is_an_empty_tuple(self):
        assert fills_from_ib(_Client([]), DAY) == ()

    def test_an_unrecognized_side_raises_rather_than_guessing_a_sign(self):
        client = _Client([broker_fill("AAA", side="XYZ", shares=1, avg_price=1.0, time=T)])
        with pytest.raises(UnrecognizedFillSideError, match="XYZ"):
            fills_from_ib(client, DAY)

    def test_a_missing_commission_report_raises_rather_than_costing_it_as_zero(self):
        client = _Client(
            [broker_fill("AAA", side="BOT", shares=1, avg_price=1.0, commission=None, time=T)]
        )
        with pytest.raises(MissingCommissionReportError, match="AAA"):
            fills_from_ib(client, DAY)


class TestCommissionComesFromTheExecutionsOwnReport:
    """alpha-engine-config-I12376, the 2026-10-21 reconcile: one v1 fill
    (DUOL BOT 39 @ 142.44, commission 1.00) and a `cash_delta` of exactly
    -1.00. `ib_async` 2.1.0 answers an explicit `reqExecutions()` with new
    `Fill`s whose `CommissionReport()` is the SDK default (`commission=0.0`,
    not None), so every fill was costed at zero without a word."""

    def test_the_2026_10_21_duol_fill_is_costed_with_its_commission(self):
        duol = broker_fill("DUOL", side="BOT", shares=39, avg_price=142.44, commission=1.0, time=T)
        (fill,) = fills_from_ib(_Client([duol]), DAY)
        assert fill == Fill(ticker="DUOL", shares=39, cash_usd=pytest.approx(-5_556.16))

    def test_a_default_report_is_never_read_as_a_zero_commission(self):
        unattached = as_requested(
            broker_fill("AAA", side="BOT", shares=1, avg_price=1.0, commission=None, time=T)
        )
        client = _Client([unattached], answer=as_requested)
        with pytest.raises(MissingCommissionReportError, match="AAA"):
            fills_from_ib(client, DAY)

    def test_a_report_still_in_flight_is_waited_for(self):
        pending = broker_fill("AAA", side="SLD", shares=10, avg_price=20.0, commission=None, time=T)
        landed = broker_fill("AAA", side="SLD", shares=10, avg_price=20.0, commission=1.0, time=T)
        client = _Client([pending], late=[landed])
        (fill,) = fills_from_ib(client, DAY)
        assert fill == Fill(ticker="AAA", shares=-10, cash_usd=199.0)
        assert 0 < client.slept < COMMISSION_REPORT_WAIT_S

    def test_the_wait_for_a_report_is_bounded(self):
        never = broker_fill("AAA", side="BOT", shares=1, avg_price=1.0, commission=None, time=T)
        client = _Client([never])
        with pytest.raises(MissingCommissionReportError, match="no commission report"):
            fills_from_ib(client, DAY, report_wait_s=2.0, poll_s=0.5)
        assert client.slept == 2.0

    def test_a_report_for_another_execution_is_not_this_ones(self):
        record = broker_fill("AAA", side="BOT", shares=1, avg_price=1.0, commission=5.0, time=T)
        record.commissionReport.execId = "someone-else"
        with pytest.raises(MissingCommissionReportError, match="AAA"):
            fills_from_ib(_Client([record]), DAY, report_wait_s=0.0)

    @pytest.mark.parametrize("value", [None, float("nan"), float("inf"), 1.7976931348623157e308])
    def test_an_attached_report_without_a_number_is_no_commission(self, value):
        record = broker_fill("AAA", side="BOT", shares=1, avg_price=1.0, commission=1.0, time=T)
        record.commissionReport.commission = value
        with pytest.raises(MissingCommissionReportError, match="AAA"):
            fills_from_ib(_Client([record]), DAY, report_wait_s=0.0)

    def test_the_answers_own_attached_report_is_used(self):
        record = broker_fill("AAA", side="BOT", shares=2, avg_price=10.0, commission=1.0, time=T)
        client = _Client([record], answer=lambda r: r)
        assert fills_from_ib(client, DAY) == (Fill(ticker="AAA", shares=2, cash_usd=-21.0),)
        assert client.slept == 0.0

    def test_an_execution_answered_twice_is_one_fill(self):
        record = broker_fill("AAA", side="BOT", shares=2, avg_price=10.0, commission=1.0, time=T)
        client = _Client([record])
        client._answer = client._answer * 2
        assert fills_from_ib(client, DAY) == (Fill(ticker="AAA", shares=2, cash_usd=-21.0),)


def test_live_corporate_actions_resolves_nothing_by_design():
    """No live source is wired: an empty `resolved` is the reconciler's own
    documented fail-loud state for "cannot positively say nothing happened",
    never a fabricated clean resolution."""
    actions = live_corporate_actions()
    assert actions == CorporateActionSet(resolved=frozenset())


def test_live_statement_source_reads_through_the_read_only_view():
    from types import SimpleNamespace

    from crucible_trader.broker_session import ReadOnlyBroker
    from crucible_trader.broker_statement import IbPaperStatementSource

    cash_row = SimpleNamespace(account="DU1", tag="TotalCashValue", currency="USD", value="1.0")

    class Client:
        def managedAccounts(self):
            return ["DU1"]

        def positions(self):
            return []

        def accountSummary(self):
            return [cash_row]

    source = live_statement_source(Client())
    assert isinstance(source, IbPaperStatementSource)
    assert isinstance(source._client, ReadOnlyBroker)
