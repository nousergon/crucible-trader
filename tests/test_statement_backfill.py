"""Backfill one missing reconcile day from an IB activity statement
(alpha-engine-config-I13232): the statement parsed strictly, the real reconcile
run against a capturing view, and a create-only, window-guarded apply."""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
import runpy
import shutil
import sys

import pytest
from crucible.keys import trader_broker_statement_key, trader_reconciliation_key
from crucible.store import LocalStore, S3Store
from s3_fake import FakeS3

from crucible_trader import statement_backfill
from crucible_trader.books_retention import RETENTION_MODE, is_books_key, retain_until
from crucible_trader.broker_statement import BrokerStatement, BrokerStatementError
from crucible_trader.ledger import journal_key
from crucible_trader.reconciliation import CorporateActionSet, Fill
from crucible_trader.statement_backfill import (
    BackfillRefused,
    FiledStatementSource,
    apply_writes,
    backfill_day,
    main,
    parse_activity_statement,
    parse_flex_statement,
    parse_statement,
)

ANCHOR, DAY = "2026-09-08", "2026-09-09"
#: Sunday 2026-11-08 02:00Z: outside the window. Monday 15:00Z is 10:00 ET, inside.
OUTSIDE = dt.datetime(2026, 11, 8, 2, 0, tzinfo=dt.UTC)
INSIDE = dt.datetime(2026, 11, 9, 15, 0, tzinfo=dt.UTC)
CA = CorporateActionSet(frozenset({"AAA", "BBB"}))


@pytest.fixture(autouse=True)
def _code_sha(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_CODE_SHA", "0123456789abcdef0123456789abcdef01234567")


def csv_text(
    *,
    period: str = "September 9, 2026",
    account: str = "DU1234567",
    positions: tuple[tuple[str, str, str], ...] = (("AAA", "USD", "10"),),
    cash: tuple[tuple[str, str], ...] = (("Base Currency Summary", "1000"), ("USD", "1000")),
    extra: str = "",
) -> str:
    lines = [
        "Statement,Header,Field Name,Field Value",
        "Statement,Data,BrokerName,Interactive Brokers LLC",
        f'Statement,Data,Period,"{period}"',
        "Account Information,Header,Field Name,Field Value",
        f"Account Information,Data,Account,{account}",
        "Account Information,Data,Base Currency,USD",
        "Cash Report,Header,Currency Summary,Currency,Total,Securities,Futures",
    ]
    for currency, total in cash:
        lines.append(f'Cash Report,Data,Starting Cash,{currency},"{total}",{total},0')
        lines.append(f'Cash Report,Data,Ending Cash,{currency},"{total}",{total},0')
    lines.append(
        "Open Positions,Header,DataDiscriminator,Asset Category,Currency,Symbol,Quantity,Mult,"
        "Cost Price,Close Price,Value"
    )
    for symbol, currency, qty in positions:
        lines.append(f'Open Positions,Data,Summary,Stocks,{currency},{symbol},"{qty}",1,1,1,1')
        lines.append(f'Open Positions,Data,Lot,Stocks,{currency},{symbol},"{qty}",1,1,1,1')
    lines.append("Open Positions,Total,,Stocks,USD,,,,,,10")
    return "\n".join(lines) + "\n" + extra


def flex_xml(
    *,
    day: str = "20260909",
    to_day: str | None = None,
    account: str = "DU1234567",
    info_account: str | None = None,
    base: str = "USD",
    positions: tuple[tuple[str, str, str], ...] = (("AAA", "USD", "10"),),
    cash: tuple[tuple[str, str], ...] = (("BASE_SUMMARY", "1000"),),
    cash_account: str | None = None,
    cash_day: str | None = None,
    asset: str = "STK",
    position_account: str | None = None,
    position_day: str | None = None,
    statements: int = 1,
    prolog: str = "",
    root: str = "FlexQueryResponse",
) -> str:
    """The shape of a Flex Web Service Activity statement (query type AF), cut to the
    sections and attributes the parser reads plus the ones it must ignore. Every
    identifier is synthetic: this repository is public, and a real statement's
    account, name, address and balances never belong in a fixture."""
    lines = []
    for _ in range(statements):
        lines.append(
            f'<FlexStatement accountId="{account}" fromDate="{day}" toDate="{to_day or day}" '
            'period="" whenGenerated="20260910;010203">'
        )
        lines.append(
            f'<AccountInformation accountId="{info_account or account}" currency="{base}" '
            'accountType="Individual" name="" street="" dateOfBirth="" />'
        )
        lines.append("<EquitySummaryInBase>")
        lines.append(f'<EquitySummaryByReportDateInBase reportDate="{day}" total="1" cash="1" />')
        lines.append("</EquitySummaryInBase><CashReport>")
        for currency, ending in cash:
            lines.append(
                f'<CashReportCurrency accountId="{cash_account or account}" currency="{currency}" '
                f'levelOfDetail="BaseCurrency" startingCash="1" endingCash="{ending}" '
                f'endingSettledCash="1" fromDate="{cash_day or day}" toDate="{cash_day or day}" />'
            )
        lines.append("</CashReport><OpenPositions>")
        for symbol, currency, qty in positions:
            for level in ("SUMMARY", "LOT"):
                lines.append(
                    f'<OpenPosition accountId="{position_account or account}" '
                    f'assetCategory="{asset}" symbol="{symbol}" currency="{currency}" '
                    f'position="{qty}" side="Long" levelOfDetail="{level}" '
                    f'reportDate="{position_day or day}" subCategory="COMMON" fxRateToBase="1" />'
                )
        lines.append("</OpenPositions><Trades>")
        lines.append(
            f'<Trade accountId="{account}" symbol="AAA" buySell="BUY" quantity="3" '
            f'tradeDate="{day}" levelOfDetail="EXECUTION" />'
        )
        lines.append("</Trades><CorporateActions /><CashTransactions /><InterestAccruals>")
        lines.append(
            f'<InterestAccrualsCurrency accountId="{account}" currency="BASE_SUMMARY" '
            f'fromDate="{day}" toDate="{day}" endingAccrualBalance="2.5" />'
        )
        lines.append("</InterestAccruals></FlexStatement>")
    return (
        f'{prolog}<{root} queryName="daily-activity" type="AF">'
        f'<FlexStatements count="{statements}">' + "".join(lines) + f"</FlexStatements></{root}>"
    )


# -- 1. the statement ---------------------------------------------------------------


class TestParse:
    def test_positions_and_ending_cash_become_a_v2_paper_statement(self):
        st = parse_activity_statement(
            csv_text(
                positions=(("AAA", "USD", "1,200"), ("BBB", "USD", "-5")),
                cash=(("USD", "-1,234.56"),),
            ),
            trading_day=DAY,
        )
        assert st == BrokerStatement("DU1234567", DAY, {"AAA": 1200, "BBB": -5}, -1234.56)
        doc = st.to_document()
        assert doc["schema_version"] == "broker_statement.v2" and doc["source"] == "ib_paper"
        assert doc["accrued_cash_usd"] is None  # not recorded, never guessed

    def test_a_bom_and_a_re_declared_header_are_read(self):
        extra = (
            "Open Positions,Header,DataDiscriminator,Asset Category,Currency,Symbol,Quantity\n"
            "Open Positions,Data,Summary,Stocks,USD,BBB,3\n"
        )
        st = parse_activity_statement("﻿" + csv_text(extra=extra), trading_day=DAY)
        assert st.positions == {"AAA": 10, "BBB": 3}

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"account": "U1234567"}, "not an IB paper account"),
            ({"positions": (("AAA", "CAD", "10"),)}, "not USD"),
            ({"cash": (("USD", "1"), ("EUR", "2"))}, "not USD"),
            ({"cash": (("Base Currency Summary", "1"),)}, "exactly one USD Ending Cash"),
            ({"period": "September 10, 2026"}, "not 2026-09-09"),
            ({"period": "September 1, 2026 - September 9, 2026"}, "not one day"),
            ({"positions": (("AAA", "USD", "10.5"),)}, "not whole shares"),
            ({"positions": (("AAA", "USD", "10"), ("AAA", "USD", "2"))}, "appears twice"),
            ({"positions": (("AAA", "USD", "0"),)}, "is zero"),
        ],
    )
    def test_refuses_anything_that_is_not_one_paper_usd_day(self, kwargs, message):
        with pytest.raises(BrokerStatementError, match=message):
            parse_activity_statement(csv_text(**kwargs), trading_day=DAY)

    def test_refuses_a_non_stock_position(self):
        text = csv_text().replace(
            "Summary,Stocks,USD,AAA", "Summary,Equity and Index Options,USD,AAA"
        )
        with pytest.raises(BrokerStatementError, match="only Stocks"):
            parse_activity_statement(text, trading_day=DAY)

    def test_the_source_answers_its_one_day_only(self):
        src = FiledStatementSource(BrokerStatement("DU1", DAY, {}, 1.0))
        assert src.read_statement(DAY).trading_day == DAY
        with pytest.raises(BrokerStatementError, match="holds 2026-09-09 only"):
            src.read_statement(ANCHOR)


class TestParseFlex:
    def test_positions_and_ending_cash_become_a_v2_paper_statement(self):
        st = parse_flex_statement(
            flex_xml(
                positions=(("AAA", "USD", "1200"), ("BBB", "USD", "-5")),
                cash=(("BASE_SUMMARY", "-1234.56"),),
            ),
            trading_day=DAY,
        )
        assert st == BrokerStatement("DU1234567", DAY, {"AAA": 1200, "BBB": -5}, -1234.56)
        doc = st.to_document()
        assert doc["schema_version"] == "broker_statement.v2" and doc["source"] == "ib_paper"
        assert doc["accrued_cash_usd"] is None  # not recorded, never guessed

    def test_it_is_the_same_statement_the_csv_path_files(self):
        csv = parse_activity_statement(csv_text(account="DU1234567"), trading_day=DAY)
        assert parse_flex_statement(flex_xml(), trading_day=DAY) == csv

    def test_lot_rows_trades_and_accruals_do_not_leak_into_the_statement(self):
        st = parse_flex_statement(flex_xml(positions=(("AAA", "USD", "10"),)), trading_day=DAY)
        assert st.positions == {"AAA": 10}  # the LOT row restates it; the trade is not a position

    def test_a_per_currency_usd_row_is_preferred_and_a_bom_is_read(self):
        text = flex_xml(cash=(("USD", "7"), ("BASE_SUMMARY", "999")))
        st = parse_flex_statement("\ufeff\n" + text, trading_day=DAY)
        assert st.cash_usd == 7.0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"account": "U1234567"}, "not an IB paper account"),
            ({"info_account": "DU7654321"}, "exactly one AccountInformation"),
            ({"base": "EUR"}, "base currency"),
            ({"positions": (("AAA", "CAD", "10"),)}, "not USD"),
            ({"asset": "OPT"}, "only STK"),
            ({"positions": (("AAA", "USD", "10.5"),)}, "not whole shares"),
            ({"positions": (("AAA", "USD", "ten"),)}, "not a number"),
            ({"positions": (("AAA", "USD", "10"), ("AAA", "USD", "2"))}, "appears twice"),
            ({"positions": (("AAA", "USD", "0"),)}, "is zero"),
            ({"position_account": "DU7654321"}, "different account"),
            ({"position_day": "20260908"}, "not as of 2026-09-09"),
            ({"position_day": "soon"}, "not a YYYYMMDD date"),
            ({"cash": (("BASE_SUMMARY", "1"), ("EUR", "2"))}, "not USD"),
            ({"cash": ()}, "exactly one USD endingCash row"),
            ({"cash": (("USD", "1"), ("USD", "2"))}, "exactly one USD endingCash row"),
            ({"cash": (("BASE_SUMMARY", "n/a"),)}, "not a number"),
            ({"cash_account": "DU7654321"}, "different account"),
            ({"cash_day": "20260908"}, "cash report fromDate is not"),
            ({"day": "20260910"}, "not 2026-09-09"),
            ({"day": "20260908", "to_day": "20260909"}, "not one day"),
            ({"statements": 2}, "exactly one FlexStatement"),
            ({"statements": 0}, "exactly one FlexStatement"),
            ({"root": "FlexStatementResponse"}, "not 'FlexQueryResponse'"),
            (
                {"prolog": '<!DOCTYPE x [<!ENTITY a "b">]>'},
                "DTD or entity",
            ),
        ],
    )
    def test_refuses_anything_that_is_not_one_paper_usd_day(self, kwargs, message):
        with pytest.raises(BrokerStatementError, match=message):
            parse_flex_statement(flex_xml(**kwargs), trading_day=DAY)

    def test_a_truncated_document_is_refused(self):
        with pytest.raises(BrokerStatementError, match="not well-formed"):
            parse_flex_statement(flex_xml()[:-30], trading_day=DAY)

    def test_parse_statement_reads_either_form_by_content(self):
        assert parse_statement(flex_xml(), trading_day=DAY).positions == {"AAA": 10}
        assert parse_statement(csv_text(), trading_day=DAY).positions == {"AAA": 10}
        assert parse_statement("\ufeff  " + flex_xml(), trading_day=DAY).cash_usd == 1000.0

    def test_the_fixture_carries_no_personal_attribute_values(self):
        """A guard on the fixture itself: public repo, so the identifying attributes IB
        emits stay empty here."""
        text = flex_xml()
        for attr in ("name", "street", "dateOfBirth"):
            assert f'{attr}=""' in text
        assert "DU1234567" in text and not re.search(r"\bDU(?!1234567)\d+", text)


# -- 2 and 3. the captured cycle and the apply ------------------------------------------


@pytest.fixture
def root(tmp_path) -> pathlib.Path:
    root = tmp_path / "store"
    anchor = BrokerStatement("DU1", ANCHOR, {"AAA": 10}, 1_000.0)
    LocalStore(root).put_bytes(
        trader_broker_statement_key(ANCHOR), json.dumps(anchor.to_document()).encode()
    )
    return root


def snapshot(root: pathlib.Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _statement(held: int = 10, cash: float = 1_000.0) -> BrokerStatement:
    return BrokerStatement("DU1", DAY, {"AAA": held}, cash)


class TestBackfill:
    def test_the_dry_run_runs_the_real_reconcile_and_writes_nothing(self, root):
        before = snapshot(root)
        view, summary = backfill_day(LocalStore(root), _statement(), corporate_actions=CA)
        assert snapshot(root) == before
        assert summary["reconcile"] == "ok" and summary["has_reconciliation"]
        keys = summary["keys_to_write"]
        for key in (
            trader_broker_statement_key(DAY),
            trader_reconciliation_key(DAY),
            f"runs/trader.reconcile/{DAY}/run.json",
            journal_key("nous-ergon-paper", DAY),
        ):
            assert key in keys, key
        # the live queue pointer is the latest live run's, never a backfill's
        assert summary["keys_not_applied"] == ["trader/exception_scores/current.json"]
        assert not any(k.endswith("/current.json") for k in keys)
        doc = json.loads(view.get_bytes(trader_reconciliation_key(DAY)))
        assert doc["discrepancies"] == []

    def test_apply_writes_exactly_the_captured_objects_create_only(self, root):
        backend = LocalStore(root)
        view, summary = backfill_day(backend, _statement(), corporate_actions=CA)
        written = apply_writes(backend, view, summary, now=OUTSIDE)
        assert written == summary["keys_to_write"]
        for key in written:
            assert backend.get_bytes(key) == view.get_bytes(key)
        assert not backend.exists("trader/exception_scores/current.json")
        # the day is no longer missing, so a second backfill refuses up front
        with pytest.raises(BackfillRefused, match="not a missing day"):
            backfill_day(backend, _statement(), corporate_actions=CA)

    def test_apply_refuses_before_the_first_write_if_any_key_exists(self, root):
        backend = LocalStore(root)
        view, summary = backfill_day(backend, _statement(), corporate_actions=CA)
        taken = journal_key("nous-ergon-paper", DAY)
        backend.put_bytes(taken, b"{}")
        before = snapshot(root)
        with pytest.raises(BackfillRefused, match="create-only"):
            apply_writes(backend, view, summary, now=OUTSIDE)
        assert snapshot(root) == before

    def test_apply_refuses_inside_the_trading_window(self, root):
        backend = LocalStore(root)
        view, summary = backfill_day(backend, _statement(), corporate_actions=CA)
        before = snapshot(root)
        with pytest.raises(BackfillRefused, match="09:00-16:30 ET"):
            apply_writes(backend, view, summary, now=INSIDE)
        assert snapshot(root) == before

    def test_a_paging_day_is_reported_not_raised_and_still_applies(self, root):
        backend = LocalStore(root)
        view, summary = backfill_day(backend, _statement(held=11), corporate_actions=CA)
        assert summary["reconcile"] == "failed"
        assert "ReconciliationDiscrepancyError" in summary["error"]
        assert summary["has_reconciliation"]
        apply_writes(backend, view, summary, now=OUTSIDE)
        manifest = json.loads(backend.get_bytes(f"runs/trader.reconcile/{DAY}/run.json"))
        assert manifest["status"] == "failed" and manifest["run_mode"] == "replay"

    def test_fills_reach_the_reconciler(self, root):
        _, summary = backfill_day(
            LocalStore(root),
            _statement(held=12, cash=980.0),
            fills=[Fill("AAA", 2, -20.0)],
            corporate_actions=CA,
        )
        assert summary["reconcile"] == "ok" and summary["fills"] == 1

    def test_a_cycle_that_wrote_no_reconciliation_applies_nothing(self, root):
        def boom(*_a, **_k):
            raise RuntimeError("broker unreadable")

        backend = LocalStore(root)
        view, summary = backfill_day(backend, _statement(), corporate_actions=CA, runner=boom)
        assert summary["reconcile"] == "failed" and not summary["has_reconciliation"]
        with pytest.raises(BackfillRefused, match="no reconciliation"):
            apply_writes(backend, view, summary, now=OUTSIDE)

    def test_a_cycle_that_wrote_no_journal_applies_nothing(self, root):
        # codex 29e6af2958a0c547: a reconciliation with a refused ledger
        # posting applied anyway, leaving the day with no books.
        def reconciliation_only(view, *_a, **_k):
            view.put_bytes(trader_reconciliation_key(DAY), b"{}")

        backend = LocalStore(root)
        view, summary = backfill_day(
            backend, _statement(), corporate_actions=CA, runner=reconciliation_only
        )
        assert summary["has_reconciliation"] and not summary["has_ledger"]
        before = snapshot(root)
        with pytest.raises(BackfillRefused, match="no journal and ledger view"):
            apply_writes(backend, view, summary, now=OUTSIDE)
        assert snapshot(root) == before

    def test_a_key_written_after_the_scan_is_refused_not_overwritten(self, root, monkeypatch):
        # codex 9b1df526e391319d: exists() then an unconditional put let a
        # writer landing between the two be overwritten.
        backend = LocalStore(root)
        view, summary = backfill_day(backend, _statement(), corporate_actions=CA)
        taken = journal_key("nous-ergon-paper", DAY)
        backend.put_bytes(taken, b"{}")
        monkeypatch.setattr(backend, "exists", lambda _key: False)  # the scan saw nothing
        with pytest.raises(BackfillRefused, match="written during the apply"):
            apply_writes(backend, view, summary, now=OUTSIDE)
        assert backend.get_bytes(taken) == b"{}"


class TestCli:
    def _args(self, root, tmp_path, *extra, text=None):
        path = tmp_path / "stmt.csv"
        path.write_text(text or csv_text(account="DU1"), encoding="utf-8")
        return ["--store", str(root), "--day", DAY, "--statement", str(path), *extra]

    def test_dry_run_by_default(self, root, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(statement_backfill, "live_corporate_action_source", lambda: CA)
        before = snapshot(root)
        assert main(self._args(root, tmp_path)) == 0
        assert snapshot(root) == before
        out = json.loads(capsys.readouterr().out)
        assert out["mode"] == "dry-run" and "written" not in out

    def test_apply_writes_outside_the_window(self, root, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(statement_backfill, "live_corporate_action_source", lambda: CA)
        assert main(self._args(root, tmp_path, "--apply"), clock=lambda: OUTSIDE) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["written"] and LocalStore(root).exists(trader_reconciliation_key(DAY))

    def test_a_flex_xml_statement_is_backfilled_through_the_cli(
        self, root, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.setattr(statement_backfill, "live_corporate_action_source", lambda: CA)
        path = tmp_path / "stmt.xml"
        path.write_text(flex_xml(account="DU1"), encoding="utf-8")
        args = ["--store", str(root), "--day", DAY, "--statement", str(path)]
        before = snapshot(root)
        assert main(args) == 0
        assert snapshot(root) == before
        out = json.loads(capsys.readouterr().out)
        assert out["account"] == "DU1" and out["has_reconciliation"] and out["cash_usd"] == 1000.0

    def test_a_bad_statement_is_a_no_go(self, root, tmp_path, capsys):
        assert main(self._args(root, tmp_path, text=csv_text(account="U9"))) == 1
        assert "NO-GO" in capsys.readouterr().err


def test_nothing_on_the_trading_path_imports_the_backfill():
    """Failure isolation: an operator command, never reachable from the trader."""
    pkg = pathlib.Path(statement_backfill.__file__).parent
    for path in pkg.glob("*.py"):
        if path.name == "statement_backfill.py":
            continue
        assert not re.search(r"\bstatement_backfill\b", path.read_text(encoding="utf-8")), path.name


class TestEdges:
    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("Notes\nX,Data,Field Name,Field Value\n", "Data row before its Header"),
            (csv_text().replace("Account Information,Data,Account,DU1234567\n", ""), "'Account'"),
            (csv_text().replace("Base Currency,USD", "Base Currency,EUR"), "base currency"),
            (csv_text(positions=(("AAA", "USD", "ten"),)), "not a number"),
        ],
    )
    def test_unreadable_statements_are_refused(self, text, message):
        with pytest.raises(BrokerStatementError, match=message):
            parse_activity_statement(text, trading_day=DAY)

    def test_an_object_lock_write_cannot_be_replayed(self, root):
        backend = LocalStore(root)
        view, summary = backfill_day(backend, _statement(), corporate_actions=CA)
        summary["locked_writes"] = ["trader/x.json"]
        with pytest.raises(BackfillRefused, match="Object Lock"):
            apply_writes(backend, view, summary, now=OUTSIDE)

    def test_module_entry_point(self, root, tmp_path, monkeypatch):
        path = tmp_path / "stmt.csv"
        path.write_text(csv_text(account="U9"), encoding="utf-8")
        args = ["x", "--store", str(root), "--day", DAY, "--statement", str(path)]
        monkeypatch.setattr(sys, "argv", args)
        sys.modules.pop("crucible_trader.statement_backfill", None)
        with pytest.raises(SystemExit) as exited:
            runpy.run_module("crucible_trader.statement_backfill", run_name="__main__")
        assert exited.value.code == 1


# -- WORM retention over S3 (alpha-engine-config-I13237) ---------------------------------


class _RecordingS3(FakeS3):
    """Records every ``PutObjectRetention`` ATTEMPT, refused or not."""

    def __init__(self) -> None:
        super().__init__(now=lambda: dt.datetime(2026, 9, 9, 21, 0, tzinfo=dt.UTC))
        self.retention_calls: list[str] = []

    def put_object_retention(self, **kw):
        self.retention_calls.append(kw["Key"])
        return super().put_object_retention(**kw)


def _s3_backend() -> tuple[S3Store, _RecordingS3]:
    client = _RecordingS3()
    store = S3Store("bucket", "crucible", client=client)
    anchor = BrokerStatement("DU1", ANCHOR, {"AAA": 10}, 1_000.0)
    store.put_bytes(trader_broker_statement_key(ANCHOR), json.dumps(anchor.to_document()).encode())
    return store, client


class TestRetention:
    def test_the_dry_run_makes_no_write_and_no_retention_call(self):
        backend, client = _s3_backend()
        before = dict(client.objects)
        _view, summary = backfill_day(backend, _statement(), corporate_actions=CA)
        assert summary["has_reconciliation"]
        assert client.objects == before
        assert client.retention_calls == []

    def test_apply_locks_the_books_it_writes_as_the_cycle_does(self):
        backend, client = _s3_backend()
        view, summary = backfill_day(backend, _statement(), corporate_actions=CA)
        written = apply_writes(backend, view, summary, now=OUTSIDE)
        until = retain_until(dt.date.fromisoformat(DAY))
        books = [k for k in written if is_books_key(k)]
        assert trader_reconciliation_key(DAY) in books
        assert client.retention == {
            f"crucible/{k}": {"Mode": RETENTION_MODE, "RetainUntilDate": until} for k in books
        }

    def test_a_refused_lock_keeps_the_writes_and_exits_2(self, tmp_path, capsys, monkeypatch):
        backend, client = _s3_backend()

        def refuse(**kw):
            raise RuntimeError("AccessDenied")

        monkeypatch.setattr(client, "put_object_retention", refuse)
        monkeypatch.setattr(statement_backfill, "live_corporate_action_source", lambda: CA)
        path = tmp_path / "stmt.csv"
        path.write_text(csv_text(account="DU1"), encoding="utf-8")
        args = ["--day", DAY, "--statement", str(path), "--apply"]
        assert main(args, opener=lambda _uri: backend, clock=lambda: OUTSIDE) == 2
        out = json.loads(capsys.readouterr().out)
        assert backend.exists(trader_reconciliation_key(DAY))
        assert out["locked"] == [] and trader_reconciliation_key(DAY) in out["lock_refused"]


# -- the close rehearsal: backfilled days and the month's close in one capture ----

LATER = "2026-09-10"


@pytest.fixture
def gap_root(root) -> pathlib.Path:
    """ANCHOR and LATER reconciled, DAY between them missing, as 10-14 is."""
    from crucible_trader.reconciliation_job import (
        ReconciliationDiscrepancyError,
        run_reconciliation,
    )

    later = BrokerStatement("DU1", LATER, {"AAA": 10}, 1_000.0)
    # the live run that day paged stale_anchor (DAY was never reconciled), but filed
    with pytest.raises(ReconciliationDiscrepancyError, match="stale_anchor"):
        run_reconciliation(
            LocalStore(root),
            statement_backfill.FiledStatementSource(later),
            trading_day=dt.date.fromisoformat(LATER),
            fills=(),
            cash_flows=(),
            corporate_actions=CA,
            run_mode="replay",
        )
    assert LocalStore(root).exists(trader_reconciliation_key(LATER))
    # As October is: reconciled, but before the ledger and the exception files.
    for prefix in ("ledger", "exceptions", "exception_proposals", "exception_scores"):
        shutil.rmtree(root / "trader" / prefix, ignore_errors=True)
    return root


def _mentions(summary, day):
    close = summary["close"] or {}
    return [
        e
        for e in close.get("exceptions", [])
        if e["class"] == "close_input_missing" and day in str(e["detail"])
    ]


class TestRehearseClose:
    def test_the_close_sees_the_backfilled_day_and_nothing_is_written(self, gap_root):
        from crucible.store import capturing

        from crucible_trader.back_office_dry_run import dry_run

        before = snapshot(gap_root)
        baseline = dry_run(
            capturing(LocalStore(gap_root)), month="2026-09", now="2026-11-03T05:30:00Z"
        )
        assert DAY in baseline["days_without_reconciliation"]
        view, summary = statement_backfill.rehearse_close(
            LocalStore(gap_root),
            [_statement()],
            month="2026-09",
            now="2026-11-03T05:30:00Z",
            corporate_actions=CA,
        )
        assert snapshot(gap_root) == before
        assert summary["backfill"][0]["has_reconciliation"]
        assert DAY in summary["days_with_reconciliation"]
        # the four inputs the close's completeness check reads for a day
        for key in (
            trader_broker_statement_key(DAY),
            trader_reconciliation_key(DAY),
            journal_key("nous-ergon-paper", DAY),
            f"trader/ledger/reconciliation/nous-ergon-paper/{DAY}.json",
        ):
            assert key in summary["keys_written"] and view.exists(key), key
        posted = {row["trading_day"]: row["status"] for row in summary["ledger"]}
        assert DAY not in {row["trading_day"] for row in baseline["ledger"]}
        assert posted[DAY] == "OK"
        assert not _mentions(summary, DAY)

    def test_cli_close_writes_only_to_out(self, gap_root, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(statement_backfill, "live_corporate_action_source", lambda: CA)
        stmt = tmp_path / "stmt.csv"
        stmt.write_text(csv_text(account="DU1"), encoding="utf-8")
        before = snapshot(gap_root)
        out = tmp_path / "out"
        args = ["--store", str(gap_root), "--day", DAY, "--statement", str(stmt)]
        assert (
            main([*args, "--close", "2026-09", "--out", str(out), "--now", "2026-11-03T05:30:00Z"])
            == 0
        )
        assert snapshot(gap_root) == before
        summary = json.loads((out / "summary.json").read_text())
        assert summary["backfill"][0]["trading_day"] == DAY
        assert (out / "store" / trader_reconciliation_key(DAY)).exists()
        assert "close_input_missing line(s)" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            (("--close", "2026-09", "--apply"), "give --book"),
            (("--close", "2026-09"), "needs --out"),
            (("--day", "2026-09-10"), "one --statement per --day"),
        ],
    )
    def test_cli_close_refusals(self, gap_root, tmp_path, capsys, extra, message):
        stmt = tmp_path / "stmt.csv"
        stmt.write_text(csv_text(account="DU1"), encoding="utf-8")
        args = ["--store", str(gap_root), "--day", DAY, "--statement", str(stmt), *extra]
        with pytest.raises(SystemExit):
            main(args)
        assert message in capsys.readouterr().err

    def test_two_days_without_close_is_refused(self, gap_root, tmp_path, capsys):
        stmt = tmp_path / "stmt.csv"
        stmt.write_text(csv_text(account="DU1"), encoding="utf-8")
        args = ["--store", str(gap_root)]
        args += ["--day", DAY, "--statement", str(stmt)] * 2
        with pytest.raises(SystemExit):
            main(args)
        assert "needs --close" in capsys.readouterr().err

    def test_cli_close_refuses_a_non_empty_out(self, gap_root, tmp_path, capsys):
        stmt = tmp_path / "stmt.csv"
        stmt.write_text(csv_text(account="DU1"), encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir()
        (out / "earlier.json").write_text("{}", encoding="utf-8")
        args = ["--store", str(gap_root), "--day", DAY, "--statement", str(stmt)]
        with pytest.raises(SystemExit):
            main([*args, "--close", "2026-09", "--out", str(out)])
        assert "is not empty" in capsys.readouterr().err

    def test_cli_close_bad_statement_is_a_no_go_and_writes_nothing(
        self, gap_root, tmp_path, capsys
    ):
        stmt = tmp_path / "stmt.csv"
        stmt.write_text(csv_text(account="U9"), encoding="utf-8")
        out = tmp_path / "out"
        args = ["--store", str(gap_root), "--day", DAY, "--statement", str(stmt)]
        assert main([*args, "--close", "2026-09", "--out", str(out)]) == 1
        assert "NO-GO" in capsys.readouterr().err
        assert not out.exists()


# -- codex 40108050: a refusal part-way locks what it wrote, and a re-run resumes ----


class TestPartialApply:
    def _conflict_on(self, backend, monkeypatch, target):
        from crucible.store import PointerConflictError

        real = backend.compare_and_swap

        def cas(key, expected, payload):
            if key == target:
                raise PointerConflictError(key)
            return real(key, expected, payload)

        monkeypatch.setattr(backend, "compare_and_swap", cas)
        return real

    def test_the_books_written_before_a_late_conflict_are_locked(self, monkeypatch):
        backend, client = _s3_backend()
        view, summary = backfill_day(backend, _statement(), corporate_actions=CA)
        self._conflict_on(backend, monkeypatch, trader_reconciliation_key(DAY))
        with pytest.raises(BackfillRefused, match="already written and locked"):
            apply_writes(backend, view, summary, now=OUTSIDE)
        written = summary["written"]
        assert journal_key("nous-ergon-paper", DAY) in written
        assert trader_reconciliation_key(DAY) not in written
        assert sorted(summary["locked"]) == sorted(k for k in written if is_books_key(k))
        assert {f"crucible/{k}" for k in summary["locked"]} == set(client.retention)

    def test_a_re_run_resumes_an_apply_stopped_before_the_reconciliation(self, monkeypatch):
        backend, client = _s3_backend()
        view, summary = backfill_day(backend, _statement(), corporate_actions=CA)
        real = self._conflict_on(backend, monkeypatch, trader_reconciliation_key(DAY))
        with pytest.raises(BackfillRefused):
            apply_writes(backend, view, summary, now=OUTSIDE)
        first = list(summary["written"])
        monkeypatch.setattr(backend, "compare_and_swap", real)
        view, summary = backfill_day(backend, _statement(), corporate_actions=CA)
        written = apply_writes(backend, view, summary, now=OUTSIDE)
        assert backend.exists(trader_reconciliation_key(DAY))
        for key in written:
            assert backend.get_bytes(key) == view.get_bytes(key)
        # The journal the stopped apply wrote is verified, not re-written; the
        # rest it wrote replays byte for byte and is skipped. All of it is locked.
        assert journal_key("nous-ergon-paper", DAY) in first
        assert set(first) - {journal_key("nous-ergon-paper", DAY)} <= set(summary["written"])
        books = {f"crucible/{k}" for k in [*first, *written] if is_books_key(k)}
        assert set(client.retention) == books

    def test_a_concurrent_identical_write_is_the_same_write(self, root, monkeypatch):
        backend = LocalStore(root)
        view, summary = backfill_day(backend, _statement(), corporate_actions=CA)
        taken = journal_key("nous-ergon-paper", DAY)
        backend.put_bytes(taken, view.get_bytes(taken))
        monkeypatch.setattr(backend, "exists", lambda _key: False)  # the scan saw nothing
        assert apply_writes(backend, view, summary, now=OUTSIDE) == summary["keys_to_write"]
        assert taken in summary["written"]


class TestRehearseSeesTheRunDaysJournals:
    """crucible-trader#147 review: the backfill's own ledger posting reads the
    store as of the close run day, as the dry run's postings do. A live journal
    dated after the run day would otherwise refuse the missing day's posting."""

    @pytest.mark.parametrize("live", ["2026-10-01", "2026-11-03"])
    def test_a_later_live_journal_does_not_refuse_the_backfilled_day(self, gap_root, live):
        LocalStore(gap_root).put_bytes(journal_key("nous-ergon-paper", live), b"{}")
        view, summary = statement_backfill.rehearse_close(
            LocalStore(gap_root),
            [_statement()],
            month="2026-09",
            now="2026-11-03T05:30:00Z",
            corporate_actions=CA,
        )
        assert summary["backfill"][0]["has_ledger"]
        assert journal_key("nous-ergon-paper", DAY) in summary["keys_written"]
        rows = {row["trading_day"]: row for row in summary["ledger"]}
        assert rows[DAY]["status"] == "OK"
        assert view.exists(journal_key("nous-ergon-paper", live))  # hidden only while booking


class TestRehearseInDateOrder:
    """codex d411bf68: a backfilled day is booked between the stored days, so an
    earlier stored day with no journal is not refused behind it."""

    def test_a_stored_day_before_the_backfilled_one_is_still_booked(self, gap_root):
        after = BrokerStatement("DU1", "2026-09-11", {"AAA": 10}, 1_000.0)
        _view, summary = statement_backfill.rehearse_close(
            LocalStore(gap_root),
            [after],
            month="2026-09",
            now="2026-11-03T05:30:00Z",
            corporate_actions=CA,
        )
        rows = {row["trading_day"]: row for row in summary["ledger"]}
        assert LATER in rows and "already exist" not in str(rows[LATER]["reason"])
        assert rows[LATER]["status"] == "OK"
        assert summary["backfill"][0]["trading_day"] == "2026-09-11"

    def test_one_statement_per_day(self, gap_root):
        with pytest.raises(BackfillRefused, match="one statement per day"):
            statement_backfill.rehearse_close(
                LocalStore(gap_root),
                [_statement(), _statement()],
                month="2026-09",
                now="2026-11-03T05:30:00Z",
                corporate_actions=CA,
            )


# -- the prior-period book (owner decision 2026-11-12 on aec#13232) ----------------

BOOK = "pre-v2"


class TestPriorPeriodBook:
    """A day before the live book opened is posted into its declared
    prior-period book, never behind the live book's journals."""

    def test_a_day_behind_the_live_book_is_refused_naming_the_book(self, root):
        LocalStore(root).put_bytes(journal_key("nous-ergon-paper", "2026-11-03"), b"{}")
        with pytest.raises(BackfillRefused, match="--book pre-v2"):
            backfill_day(LocalStore(root), _statement(), corporate_actions=CA)

    def test_the_day_posts_into_the_book_and_applies_there(self, root):
        backend = LocalStore(root)
        backend.put_bytes(journal_key("nous-ergon-paper", "2026-11-03"), b"{}")
        view, summary = backfill_day(backend, _statement(), corporate_actions=CA, book=BOOK)
        assert summary["book"] == BOOK and summary["has_ledger"], summary
        keys = summary["keys_to_write"]
        assert journal_key("nous-ergon-paper", DAY, BOOK) in keys
        assert f"trader/ledger/books/{BOOK}/reconciliation/nous-ergon-paper/{DAY}.json" in keys
        assert not [k for k in keys if k.startswith("trader/ledger/journal/")]
        apply_writes(backend, view, summary, now=OUTSIDE)
        assert backend.exists(journal_key("nous-ergon-paper", DAY, BOOK))
        assert not backend.exists(journal_key("nous-ergon-paper", DAY))
        assert backend.get_bytes(journal_key("nous-ergon-paper", "2026-11-03")) == b"{}"

    def test_the_rehearsal_books_and_closes_the_month_in_the_book(self, gap_root):
        live = journal_key("nous-ergon-paper", "2026-11-03")
        LocalStore(gap_root).put_bytes(live, b"{}")
        before = snapshot(gap_root)
        view, summary = statement_backfill.rehearse_close(
            LocalStore(gap_root),
            [_statement()],
            month="2026-09",
            now="2026-11-03T05:30:00Z",
            corporate_actions=CA,
            book=BOOK,
        )
        assert snapshot(gap_root) == before
        assert summary["book"] == BOOK and summary["backfill"][0]["has_ledger"]
        written = summary["keys_written"]
        assert {journal_key("nous-ergon-paper", d, BOOK) for d in (DAY, LATER)} <= set(written)
        assert not [
            k for k in written if k.startswith(("trader/ledger/journal/", "trader/ledger/close/"))
        ]
        assert any(k.startswith(f"trader/ledger/books/{BOOK}/close/") for k in written)
        rows = {row["trading_day"]: row["status"] for row in summary["ledger"]}
        assert rows[DAY] == "OK"  # LATER books too; it lacks only a settled close, as live
        assert "already exist" not in str(summary["ledger"])
        assert view.get_bytes(live) == b"{}"

    def test_cli_rehearsal_reports_the_bridge_for_the_books_last_month(
        self, gap_root, tmp_path, capsys, monkeypatch
    ):
        from crucible_trader import ledger

        monkeypatch.setattr(ledger, "PRIOR_PERIOD_BOOKS", {BOOK: "2026-09"})
        monkeypatch.setattr(statement_backfill, "live_corporate_action_source", lambda: CA)
        stmt = tmp_path / "stmt.csv"
        stmt.write_text(csv_text(account="DU1"), encoding="utf-8")
        out = tmp_path / "out"
        args = ["--store", str(gap_root), "--day", DAY, "--statement", str(stmt), "--book", BOOK]
        assert main([*args, "--close", "2026-09", "--out", str(out), "--now", NOW_STAMP]) == 0
        err = capsys.readouterr().err
        assert "bridge to the live opening does NOT tie" in err
        summary = json.loads((out / "summary.json").read_text())
        assert summary["close"]["bridge"]["ties"] is False
        assert "close_input_missing" in {e["class"] for e in summary["close"]["exceptions"]}

    def test_a_single_day_apply_ahead_of_unbooked_days_is_refused(
        self, gap_root, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.setattr(statement_backfill, "live_corporate_action_source", lambda: CA)
        day = "2026-09-11"
        stmt = tmp_path / "stmt.csv"
        stmt.write_text(csv_text(account="DU1", period="September 11, 2026"), encoding="utf-8")
        args = ["--store", str(gap_root), "--day", day, "--statement", str(stmt), "--book", BOOK]
        before = snapshot(gap_root)
        assert main([*args, "--apply"], clock=lambda: OUTSIDE) == 1
        assert f"no journal yet for ['{LATER}']" in capsys.readouterr().err
        assert snapshot(gap_root) == before

    def test_an_undeclared_book_is_a_no_go(self, root, tmp_path, capsys):
        stmt = tmp_path / "stmt.csv"
        stmt.write_text(csv_text(account="DU1"), encoding="utf-8")
        args = ["--store", str(root), "--day", DAY, "--statement", str(stmt), "--book", "x"]
        assert main(args) == 1
        assert "not a declared prior-period book" in capsys.readouterr().err


NOW_STAMP = "2026-11-03T05:30:00Z"


# -- codex review of crucible-trader#160 ---------------------------------------------


class TestBookEnd:
    """A prior-period book holds nothing past its declared final month."""

    def test_a_backfill_past_the_books_end_is_refused(self, root, monkeypatch):
        from crucible_trader import ledger

        monkeypatch.setattr(ledger, "PRIOR_PERIOD_BOOKS", {BOOK: "2026-08"})
        with pytest.raises(BackfillRefused, match="after 2026-08, the last month"):
            backfill_day(LocalStore(root), _statement(), corporate_actions=CA, book=BOOK)


SEP11 = "2026-09-11"


class TestWholeBookApply:
    """A month with a stored reconciliation (LATER) between missing days (DAY,
    09-11) is applied into the book from its rehearsal, create-only and in
    date order: the stored day cannot be backfilled (it is not missing) and a
    missing day alone ahead of it is refused."""

    def _args(self, gap_root, tmp_path, *extra):
        tmp_path.mkdir(parents=True, exist_ok=True)
        args = ["--store", str(gap_root), "--book", BOOK]
        for day, period in ((DAY, "September 9, 2026"), (SEP11, "September 11, 2026")):
            path = tmp_path / f"{day}.csv"
            path.write_text(csv_text(account="DU1", period=period), encoding="utf-8")
            args += ["--day", day, "--statement", str(path)]
        out = tmp_path / "out"
        return [*args, "--close", "2026-09", "--out", str(out), "--now", NOW_STAMP, *extra], out

    def _rehearse(self, gap_root):
        return statement_backfill.rehearse_close(
            LocalStore(gap_root),
            [_statement(), BrokerStatement("DU1", SEP11, {"AAA": 10}, 1_000.0)],
            month="2026-09",
            now=NOW_STAMP,
            corporate_actions=CA,
            book=BOOK,
        )

    def test_stored_and_missing_days_book_in_date_order(
        self, gap_root, tmp_path, capsys, monkeypatch
    ):
        from crucible_trader.exception_proposal import exception_proposals_key

        monkeypatch.setattr(statement_backfill, "live_corporate_action_source", lambda: CA)
        args, out = self._args(gap_root, tmp_path, "--apply")
        assert main(args, clock=lambda: OUTSIDE) == 0, capsys.readouterr().err
        assert "applied" in capsys.readouterr().err
        backend = LocalStore(gap_root)
        journals = [journal_key("nous-ergon-paper", d, BOOK) for d in (DAY, LATER, SEP11)]
        for key in journals:
            assert backend.exists(key), key
        for day in (DAY, LATER, SEP11):
            assert not backend.exists(journal_key("nous-ergon-paper", day))
        assert backend.exists(trader_reconciliation_key(DAY))
        assert backend.exists(trader_reconciliation_key(SEP11))
        applied = json.loads((out / "summary.json").read_text())["applied"]
        at = [applied.index(k) for k in journals]
        assert at == sorted(at)
        assert applied.index(trader_reconciliation_key(DAY)) < at[1]
        # the dry run's own review-queue rehearsal is not applied
        assert exception_proposals_key(LATER) not in applied
        assert not backend.exists(exception_proposals_key(LATER))
        # the applied days are no longer missing: a re-run is refused, nothing written
        before = snapshot(gap_root)
        args, _ = self._args(gap_root, tmp_path / "again", "--apply")
        assert main(args, clock=lambda: OUTSIDE) == 1
        assert "not a missing day" in capsys.readouterr().err
        assert snapshot(gap_root) == before

    def test_create_only_refuses_before_the_first_write(
        self, gap_root, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.setattr(statement_backfill, "live_corporate_action_source", lambda: CA)
        taken = (
            f"trader/ledger/books/{BOOK}/close/nous-ergon-paper/2026-09/attempts/2026-10-01.json"
        )
        LocalStore(gap_root).put_bytes(taken, b"{}")
        before = snapshot(gap_root)
        args, out = self._args(gap_root, tmp_path, "--apply")
        assert main(args, clock=lambda: OUTSIDE) == 1
        assert "create-only" in capsys.readouterr().err
        assert snapshot(gap_root) == before
        assert (out / "summary.json").exists()  # the rehearsal is kept

    def test_a_refused_lock_exits_2(self, gap_root, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(statement_backfill, "live_corporate_action_source", lambda: CA)
        monkeypatch.setattr(
            statement_backfill, "lock_books_records", lambda *_a, **_k: ([], {"k": "denied"})
        )
        args, _ = self._args(gap_root, tmp_path, "--apply")
        assert main(args, clock=lambda: OUTSIDE) == 2
        assert "NOT under WORM retention" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("spoil", "message"),
        [
            (lambda s: s["backfill"][0].update(has_ledger=False), "no reconciliation, journal"),
            (lambda s: s["backfill"][0].update(locked_writes=["x"]), "Object Lock"),
            (lambda s: s["days_with_reconciliation"].append("2026-09-30"), "no pre-v2 journal"),
        ],
    )
    def test_refusals_write_nothing(self, gap_root, spoil, message):
        view, summary = self._rehearse(gap_root)
        spoil(summary)
        before = snapshot(gap_root)
        with pytest.raises(BackfillRefused, match=message):
            statement_backfill.apply_book(
                LocalStore(gap_root), view, summary, book=BOOK, now=OUTSIDE
            )
        assert snapshot(gap_root) == before

    @pytest.mark.parametrize(
        ("book", "now", "message"), [(None, OUTSIDE, "give --book"), (BOOK, INSIDE, "09:00-16:30")]
    )
    def test_no_book_or_inside_the_window_is_refused(self, gap_root, book, now, message):
        view, summary = self._rehearse(gap_root)
        with pytest.raises(BackfillRefused, match=message):
            statement_backfill.apply_book(LocalStore(gap_root), view, summary, book=book, now=now)


class TestWholeBookResume:
    """Codex re-review of crucible-trader#160: an apply stopped part-way is
    finished by re-running it, with no --day once every missing day has its
    reconciliation; and the book's close proposals reach the review queue."""

    def _stop_at(self, backend, monkeypatch, target):
        """The apply (not the rehearsal before it: the capturing view copies
        the backend's attributes) meets a writer at ``target`` and stops."""
        from crucible.store import PointerConflictError

        real_apply, real_cas = statement_backfill.apply_book, backend.compare_and_swap

        def cas(key, expected, payload):
            if key == target:
                raise PointerConflictError(key)
            return real_cas(key, expected, payload)

        def apply_book(*a, **k):
            monkeypatch.setattr(backend, "compare_and_swap", cas)
            try:
                return real_apply(*a, **k)
            finally:
                monkeypatch.setattr(backend, "compare_and_swap", real_cas)
                monkeypatch.setattr(statement_backfill, "apply_book", real_apply)

        monkeypatch.setattr(statement_backfill, "apply_book", apply_book)

    def _main(self, backend, tmp_path, *days):
        args = ["--book", BOOK]
        tmp_path.mkdir(parents=True, exist_ok=True)
        for day in days:
            path = tmp_path / f"{day}.csv"
            period = dt.date.fromisoformat(day).strftime("%B %-d, %Y")
            path.write_text(csv_text(account="DU1", period=period), encoding="utf-8")
            args += ["--day", day, "--statement", str(path)]
        out = tmp_path / "out"
        args += ["--close", "2026-09", "--out", str(out), "--now", NOW_STAMP, "--apply"]
        code = main(args, opener=lambda _uri: backend, clock=lambda: OUTSIDE)
        return code, out

    ATTEMPT = f"trader/ledger/books/{BOOK}/close/nous-ergon-paper/2026-09/attempts/2026-10-01.json"

    def test_resume_with_no_missing_day_left_finishes_the_close(
        self, gap_root, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.setattr(statement_backfill, "live_corporate_action_source", lambda: CA)
        backend = LocalStore(gap_root)
        self._stop_at(backend, monkeypatch, self.ATTEMPT)
        code, _ = self._main(backend, tmp_path / "first", DAY, SEP11)
        assert code == 1 and "written during the apply" in capsys.readouterr().err
        assert backend.exists(trader_reconciliation_key(SEP11))
        assert not backend.exists(self.ATTEMPT)
        code, _ = self._main(backend, tmp_path / "resume")
        assert code == 0, capsys.readouterr().err
        assert backend.exists(self.ATTEMPT)
        for day in (DAY, LATER, SEP11):
            assert backend.exists(journal_key("nous-ergon-paper", day, BOOK))

    def test_resume_after_a_stop_between_days(self, gap_root, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(statement_backfill, "live_corporate_action_source", lambda: CA)
        backend = LocalStore(gap_root)
        self._stop_at(backend, monkeypatch, journal_key("nous-ergon-paper", LATER, BOOK))
        code, _ = self._main(backend, tmp_path / "first", DAY, SEP11)
        assert code == 1
        assert backend.exists(trader_reconciliation_key(DAY))
        assert not backend.exists(journal_key("nous-ergon-paper", LATER, BOOK))
        assert not backend.exists(trader_reconciliation_key(SEP11))
        code, _ = self._main(backend, tmp_path / "resume", SEP11)  # DAY is no longer missing
        assert code == 0, capsys.readouterr().err
        for day in (DAY, LATER, SEP11):
            assert backend.exists(journal_key("nous-ergon-paper", day, BOOK))
        assert backend.exists(self.ATTEMPT)

    def test_the_close_proposals_are_applied_with_the_close(
        self, gap_root, tmp_path, capsys, monkeypatch
    ):
        from crucible_trader.exception_proposal import close_proposals_key

        monkeypatch.setattr(statement_backfill, "live_corporate_action_source", lambda: CA)
        backend = LocalStore(gap_root)
        code, out = self._main(backend, tmp_path, DAY, SEP11)
        assert code == 0, capsys.readouterr().err
        key = close_proposals_key("nous-ergon-paper", "2026-09", "2026-10-01")
        applied = json.loads((out / "summary.json").read_text())["applied"]
        assert key in applied and backend.exists(key)
        assert applied.index(key) > applied.index(self.ATTEMPT)
        assert json.loads(backend.get_bytes(key))["proposals"]


def test_a_single_day_run_needs_its_day(root, capsys):
    with pytest.raises(SystemExit):
        main(["--store", str(root)])
    assert "give the missing --day" in capsys.readouterr().err
