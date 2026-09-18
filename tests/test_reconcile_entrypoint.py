"""`fills_from_ib`, `live_statement_source` and `live_corporate_actions`: the
seam supplying `reconcile_cycle`'s broker-derived arguments (`alpha-engine-
config-I11071`). Fills are read from a fake shaped to the documented
`ib_async` `Fill`/`Execution`/`CommissionReport` fields; there is no live
gateway in this session to exercise the real SDK against."""

from __future__ import annotations

import datetime as dt

import pytest
from ib_fakes import broker_fill

from crucible_trader.reconcile_entrypoint import (
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
    def __init__(self, records):
        self._records = records

    def fills(self):
        return list(self._records)


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
