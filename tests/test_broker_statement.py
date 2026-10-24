"""The broker's statement and the IB paper reader. IB is FAKED here: the
reader takes an `ib_insync.IB`-shaped client, and these fakes implement exactly
the three methods it calls. The live read is verified on paper (see the PR)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from crucible_trader.broker_statement import (
    BROKER_STATEMENT_SCHEMA_VERSION,
    BrokerStatement,
    BrokerStatementError,
    IbPaperStatementSource,
    statement_from_document,
)

DAY = "2026-09-11"


def pos(symbol, qty, *, account="DU1", sec="STK"):
    return NS(account=account, position=qty, contract=NS(symbol=symbol, secType=sec))


def cash(value, *, account="DU1", tag="TotalCashValue", currency="USD"):
    return NS(account=account, tag=tag, currency=currency, value=str(value))


class FakeIB:
    def __init__(self, accounts=("DU1",), positions=(), summary=None):
        self._accounts, self._positions = list(accounts), positions
        self._summary = (cash(1000.5),) if summary is None else summary

    def managedAccounts(self):  # noqa: N802
        return self._accounts

    def positions(self):
        return list(self._positions)

    def accountSummary(self):  # noqa: N802
        return list(self._summary)


class TestStatement:
    def test_round_trip(self):
        s = BrokerStatement("DU1", DAY, {"BBB": 2, "AAA": -3}, 5.0)
        assert list(s.positions) == ["AAA", "BBB"]
        assert statement_from_document(s.to_document()) == s

    def test_a_ledger_reconstruction_cannot_be_filed_as_the_broker(self):
        with pytest.raises(BrokerStatementError, match="20 positions against a real 7"):
            BrokerStatement("DU1", DAY, {}, 0.0, source="ledger_replay")  # type: ignore[arg-type]

    @pytest.mark.parametrize("positions", [{"AAA": 0}, {"AAA": 1.5}, {"": 1}, {"AAA": True}])
    def test_malformed_positions(self, positions):
        with pytest.raises(BrokerStatementError):
            BrokerStatement("DU1", DAY, positions, 0.0)

    def test_non_finite_cash(self):
        with pytest.raises(BrokerStatementError, match="finite"):
            BrokerStatement("DU1", DAY, {}, float("nan"))

    def test_document_with_wrong_version_or_extra_fields(self):
        good = BrokerStatement("DU1", DAY, {}, 0.0).to_document()
        with pytest.raises(BrokerStatementError, match="schema_version"):
            statement_from_document({**good, "schema_version": "x"})
        with pytest.raises(BrokerStatementError, match="fields"):
            statement_from_document({**good, "nav": 1})
        assert good["schema_version"] == BROKER_STATEMENT_SCHEMA_VERSION


class TestIbPaperReader:
    def test_reads_positions_and_cash_for_the_one_paper_account(self):
        ib = FakeIB(positions=[pos("AAA", 10.0), pos("BBB", 0.0), pos("ZZZ", 5, account="DU2")])
        s = IbPaperStatementSource(ib).read_statement(DAY)
        assert (s.account, s.positions, s.cash_usd, s.trading_day) == (
            "DU1",
            {"AAA": 10},
            1000.5,
            DAY,
        )

    @pytest.mark.parametrize(
        ("ib", "match"),
        [
            (FakeIB(accounts=()), "0 accounts"),
            (FakeIB(accounts=("DU1", "DU2")), "2 accounts"),
            (FakeIB(accounts=("U1234",)), "not an IB paper account"),
            (FakeIB(positions=[pos("AAA", 1, sec="OPT")]), "OPT"),
            (FakeIB(positions=[pos("AAA", 1.5)]), "fractional"),
            (FakeIB(positions=[pos("AAA", 1), pos("AAA", 2)]), "twice"),
            (FakeIB(summary=()), "found 0"),
            (FakeIB(summary=(cash(1), cash(2))), "found 2"),
            (FakeIB(summary=(cash(1, currency="EUR"), cash(1, tag="NetLiquidation"))), "found 0"),
        ],
    )
    def test_refusals(self, ib, match):
        with pytest.raises(BrokerStatementError, match=match):
            IbPaperStatementSource(ib).read_statement(DAY)


class TestAccruedCash:
    """`alpha-engine-config-I11076` phase 0: the accrual is recorded, softly."""

    def test_v2_round_trip_carries_the_accrual(self):
        s = BrokerStatement("DU1", DAY, {"AAA": 1}, 5.0, accrued_cash_usd=12.34)
        doc = s.to_document()
        assert doc["schema_version"] == "broker_statement.v2"
        assert doc["accrued_cash_usd"] == 12.34
        assert statement_from_document(doc) == s

    def test_a_v1_statement_still_reads_as_accrual_not_recorded(self):
        v1 = {
            "schema_version": "broker_statement.v1",
            "account": "DU1",
            "trading_day": DAY,
            "source": "ib_paper",
            "positions": {"AAA": 1},
            "cash_usd": 5.0,
        }
        s = statement_from_document(v1)
        assert s.accrued_cash_usd is None
        assert s.positions == {"AAA": 1}

    def test_each_version_has_an_exact_field_set(self):
        v2 = BrokerStatement("DU1", DAY, {}, 0.0).to_document()
        with pytest.raises(BrokerStatementError, match="fields"):
            statement_from_document({**v2, "schema_version": "broker_statement.v1"})
        missing = {k: v for k, v in v2.items() if k != "accrued_cash_usd"}
        with pytest.raises(BrokerStatementError, match="fields"):
            statement_from_document(missing)

    def test_a_non_finite_accrual_cannot_be_constructed(self):
        with pytest.raises(BrokerStatementError, match="accrued_cash_usd"):
            BrokerStatement("DU1", DAY, {}, 0.0, accrued_cash_usd=float("inf"))

    def test_the_reader_records_the_one_usd_accrued_cash_row(self):
        ib = FakeIB(
            summary=(
                cash(1000.5),
                cash(7.25, tag="AccruedCash"),
                cash(9.0, tag="AccruedCash", currency="BASE"),
                cash(3.0, tag="AccruedCash", account="DU2"),
            )
        )
        assert IbPaperStatementSource(ib).read_statement(DAY).accrued_cash_usd == 7.25

    @pytest.mark.parametrize(
        "rows",
        [
            (),
            (cash(1.0, tag="AccruedCash"), cash(2.0, tag="AccruedCash")),
            (cash("", tag="AccruedCash"),),
            (cash("nan", tag="AccruedCash"),),
        ],
        ids=["absent", "duplicated", "unparsable", "non-finite"],
    )
    def test_an_unreadable_accrual_is_none_and_never_stops_the_read(self, rows):
        """This reader also serves the kill switch and the order router."""
        ib = FakeIB(summary=(cash(1000.5), *rows))
        s = IbPaperStatementSource(ib).read_statement(DAY)
        assert s.accrued_cash_usd is None
        assert s.cash_usd == 1000.5

    def test_settled_cash_is_still_strict(self):
        ib = FakeIB(summary=(cash(7.25, tag="AccruedCash"),))
        with pytest.raises(BrokerStatementError, match="TotalCashValue"):
            IbPaperStatementSource(ib).read_statement(DAY)

    def test_lineage_records_the_version_the_stored_bytes_declare(self):
        import json

        from crucible_trader.broker_statement import statement_schema_version

        v1 = json.dumps({"schema_version": "broker_statement.v1"}).encode()
        assert statement_schema_version(v1) == "broker_statement.v1"
        with pytest.raises(BrokerStatementError):
            statement_schema_version(json.dumps({"schema_version": "x"}).encode())
