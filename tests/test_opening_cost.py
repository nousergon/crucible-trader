"""The opening-cost evidence pack (`alpha-engine-config-I14137`).

Every identifier, ticker and amount here is synthetic: the repository is public.
The fixture has the shape of the real one, a 12-name opening statement booked at
0 cents and later sales of some names, but none of its values."""

from __future__ import annotations

import io
import json
import runpy
import sys
from contextlib import redirect_stderr, redirect_stdout

import pandas as pd
import pytest
from crucible.keys import data_panel_key, trader_broker_statement_key
from crucible.store import LocalStore

from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.ledger import (
    CASH,
    OPENING,
    REALIZED_GAINS,
    SECURITIES,
    Entry,
    Posting,
    journal_document,
    journal_key,
    ledger_from,
    load_journals,
)
from crucible_trader.opening_cost import (
    OpeningCostError,
    _entry,  # pyright: ignore[reportPrivateUsage]
    main,
    opening_cost_key,
    opening_cost_pack,
    write_pack,
)

T, ACCT = "paper-fund", "DU1234567"
STATEMENT_DAY, AS_OF = "2026-10-30", "2026-11-16"
NAMES = [f"N{i:02d}" for i in range(1, 13)]
SHARES = {n: 10 * (i + 1) for i, n in enumerate(NAMES)}
# N03 closes at a fractional cent product (30 x 12.345) to pin the rounding.
CLOSES = {n: 20.0 + i for i, n in enumerate(NAMES)} | {"N03": 12.345}
SOLD_OUT = NAMES[:5]  # sold in full before AS_OF
PART_SOLD = "N06"  # sold in part: stays open, untouched but for its cost
SOLD_LATER = "N07"  # sold in full AFTER AS_OF: must still count as held


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "trader")


@pytest.fixture
def flex(tmp_path):
    return LocalStore(tmp_path / "flex")


def sale(day, ticker, shares, price_cents):
    proceeds = shares * price_cents
    return Entry(
        T,
        ACCT,
        day,
        "fill",
        f"sell/{ticker}",
        (Posting(CASH, proceeds), Posting(SECURITIES + ticker, -proceeds, -shares)),
    )


def seed(store, *, closes=CLOSES, statement=True):
    positions = dict(SHARES)
    if statement:
        doc = BrokerStatement(ACCT, STATEMENT_DAY, positions, 1000.0, accrued_cash_usd=0.0)
        store.put_bytes(
            trader_broker_statement_key(STATEMENT_DAY), json.dumps(doc.to_document()).encode()
        )
    if closes is not None:
        frame = pd.DataFrame(
            {
                "trading_day": [STATEMENT_DAY] * len(closes),
                "ticker": list(closes),
                "close_raw": list(closes.values()),
            }
        )
        store.put_bytes(data_panel_key(STATEMENT_DAY), frame.to_parquet())
    opening = Entry(
        T,
        ACCT,
        "2026-11-03",
        "opening_balance",
        f"statement/{STATEMENT_DAY}",
        (Posting(CASH, 100_000), Posting(OPENING, -100_000))
        + tuple(Posting(SECURITIES + t, 0, q) for t, q in sorted(positions.items())),
        {"statement_day": STATEMENT_DAY},
    )
    days = {
        "2026-11-03": [opening],
        "2026-11-04": [sale("2026-11-04", t, SHARES[t], 2500) for t in SOLD_OUT],
        "2026-11-16": [sale("2026-11-16", PART_SOLD, 5, 2600)],
        "2026-11-17": [sale("2026-11-17", SOLD_LATER, SHARES[SOLD_LATER], 2700)],
    }
    for day, entries in days.items():
        store.put_bytes(journal_key(T, day), journal_document(T, day, entries))


def flex_xml(costs, *, day="20261030", positions=None, level="SUMMARY", lots=False):
    rows = "".join(
        f'<OpenPosition accountId="{ACCT}" assetCategory="STK" symbol="{t}" currency="USD" '
        f'reportDate="{day}" position="{q}" costBasisMoney="{costs[t]}" levelOfDetail="{level}" />'
        for t, q in (positions or SHARES).items()
    )
    if lots:  # lot rows restate the summary row, here with a different cost
        rows += rows.replace("SUMMARY", "LOT").replace('costBasisMoney="', 'costBasisMoney="9')
    return (
        "<FlexQueryResponse><FlexStatements>"
        f'<FlexStatement accountId="{ACCT}" fromDate="{day}" toDate="{day}">'
        f'<AccountInformation accountId="{ACCT}" currency="USD" />'
        "<CashReport>"
        f'<CashReportCurrency accountId="{ACCT}" currency="BASE_SUMMARY" '
        'levelOfDetail="BaseCurrency" '
        f'endingCash="1000" fromDate="{day}" toDate="{day}" />'
        f"</CashReport><OpenPositions>{rows}</OpenPositions>"
        "</FlexStatement></FlexStatements></FlexQueryResponse>"
    )


FLEX_COSTS = {t: f"{(i + 1) * 1000}.505" for i, t in enumerate(NAMES)}


def put_flex(flex, text=None, name="2026-10/20261030-a.xml"):
    flex.put_bytes(name, (text or flex_xml(FLEX_COSTS)).encode())


def sums_to_zero(entry):
    return sum(p["amount_cents"] for p in entry["postings"]) == 0


class TestThePack:
    def test_the_settled_close_pack_balances_and_clears_every_closed_name(self, store, flex):
        seed(store)
        pack = opening_cost_pack(store, T, ACCT, AS_OF, flex_store=flex)
        assert pack["schema_version"] == "opening_cost_pack.v1"
        assert pack["statement_day"] == STATEMENT_DAY
        lines = {ln["ticker"]: ln for ln in pack["lines"]}
        assert sorted(lines) == NAMES
        assert {ln["source"] for ln in lines.values()} == {"settled_close"}
        assert {ln["source_record"] for ln in lines.values()} == {STATEMENT_DAY}
        # 30 shares x 12.345 = 370.35 exactly; every other name is whole dollars.
        assert lines["N03"]["cost_cents"] == 37_035
        assert lines["N01"]["cost_cents"] == 10 * 2000

        assert sums_to_zero(pack["entry_1"]) and sums_to_zero(pack["entry_2"])
        assert pack["entry_1"]["kind"] == "adjustment"
        total = sum(ln["cost_cents"] for ln in lines.values())
        opening = [p for p in pack["entry_1"]["postings"] if p["account"] == OPENING]
        assert opening == [{"account": OPENING, "amount_cents": -total, "quantity": None}]
        # Entry 2 touches exactly the names with no shares left on AS_OF.
        moved = {
            p["account"].removeprefix(SECURITIES)
            for p in pack["entry_2"]["postings"]
            if p["account"].startswith(SECURITIES)
        }
        assert moved == set(SOLD_OUT)
        assert pack["trial_balance_after_cents"] == 0

        ledger = ledger_from(load_journals(store, T, before="2026-11-17"))
        for t in NAMES:
            account = SECURITIES + t
            assert pack["before"][account] == ledger.cents(T, ACCT, account)
            if ledger.shares(T, ACCT, account) == 0:
                assert pack["after"][account] == 0
        for t in SOLD_OUT:
            before_one = -SHARES[t] * 2500 + lines[t]["cost_cents"]
            assert pack["after"][REALIZED_GAINS + t] == before_one
        assert (
            pack["after"][SECURITIES + PART_SOLD]
            == pack["before"][SECURITIES + PART_SOLD] + lines[PART_SOLD]["cost_cents"]
        )
        assert pack["after"][SECURITIES + SOLD_LATER] == lines[SOLD_LATER]["cost_cents"]

    def test_a_flex_statement_for_the_day_prices_the_names(self, store, flex):
        seed(store)
        put_flex(flex)
        pack = opening_cost_pack(store, T, ACCT, AS_OF, flex_store=flex)
        lines = {ln["ticker"]: ln for ln in pack["lines"]}
        assert {ln["source"] for ln in lines.values()} == {"ib_flex_cost_basis"}
        assert {ln["source_record"] for ln in lines.values()} == {"2026-10/20261030-a.xml"}
        assert lines["N01"]["cost_cents"] == 100_051  # 1000.505, half-up
        assert sums_to_zero(pack["entry_1"]) and sums_to_zero(pack["entry_2"])
        assert pack["trial_balance_after_cents"] == 0

    def test_a_flex_file_for_another_day_does_not_count(self, store, flex):
        seed(store)
        put_flex(flex, name="2026-10/20261029-a.xml")
        flex.put_bytes("2026-10/20261030-notes.txt", b"x")
        pack = opening_cost_pack(store, T, ACCT, AS_OF, flex_store=flex)
        assert {ln["source"] for ln in pack["lines"]} == {"settled_close"}

    def test_lot_rows_in_the_flex_file_do_not_restate_the_summary(self, store, flex):
        seed(store)
        put_flex(flex, flex_xml(FLEX_COSTS, lots=True))
        pack = opening_cost_pack(store, T, ACCT, AS_OF, flex_store=flex)
        assert pack["lines"][0]["cost_cents"] == 100_051

    def test_the_default_flex_location_is_opened_read_only(self, store, monkeypatch, flex):
        seed(store)
        opened = []

        def fake_open(uri, *, dry_run=False):
            opened.append((uri, dry_run))
            return flex

        monkeypatch.setattr("crucible_trader.opening_cost.open_store", fake_open)
        monkeypatch.setenv("CRUCIBLE_TRADER_IB_FLEX_URI", "s3://example/flex/")
        opening_cost_pack(store, T, ACCT, AS_OF)
        assert opened == [("s3://example/flex/", True)]

    def test_a_closed_name_that_was_never_costed_zero_is_still_paired(self, store, flex):
        seed(store, closes={n: 0.0 for n in NAMES})
        pack = opening_cost_pack(store, T, ACCT, AS_OF, flex_store=flex)
        assert {ln["cost_cents"] for ln in pack["lines"]} == {0}
        assert sums_to_zero(pack["entry_2"])


class TestRefusals:
    def test_a_missing_close_names_the_ticker_and_writes_nothing(self, store, flex):
        seed(store, closes={n: c for n, c in CLOSES.items() if n not in ("N04", "N09")})
        with pytest.raises(OpeningCostError, match="N04, N09"):
            opening_cost_pack(store, T, ACCT, AS_OF, flex_store=flex)
        buffer = io.StringIO()
        with pytest.raises(OpeningCostError, match="N04"), redirect_stdout(buffer):
            main(
                ["--store", "x", "--tenant", T, "--account", ACCT, "--as-of", AS_OF, "--write"],
                opener=lambda uri, write: store,
                flex_opener=lambda uri: flex,
            )
        assert buffer.getvalue() == ""
        assert not store.exists(opening_cost_key(T, AS_OF))

    def test_no_opening_entry_is_refused(self, store, flex):
        with pytest.raises(OpeningCostError, match="no opening_balance"):
            opening_cost_pack(store, T, ACCT, AS_OF, flex_store=flex)

    def test_the_opening_entry_of_another_account_is_not_this_accounts(self, store, flex):
        seed(store)
        with pytest.raises(OpeningCostError, match="no opening_balance"):
            opening_cost_pack(store, T, "DU7654321", AS_OF, flex_store=flex)

    def test_an_unreadable_statement_is_refused(self, store, flex):
        seed(store, statement=False)
        with pytest.raises(OpeningCostError, match="unreadable"):
            opening_cost_pack(store, T, ACCT, AS_OF, flex_store=flex)

    def test_a_statement_for_another_account_is_refused(self, store, flex):
        seed(store)
        doc = BrokerStatement("DU7654321", STATEMENT_DAY, dict(SHARES), 1.0, accrued_cash_usd=0.0)
        store.put_bytes(
            trader_broker_statement_key(STATEMENT_DAY), json.dumps(doc.to_document()).encode()
        )
        with pytest.raises(OpeningCostError, match="different account"):
            opening_cost_pack(store, T, ACCT, AS_OF, flex_store=flex)

    def test_two_flex_files_for_the_day_are_refused(self, store, flex):
        seed(store)
        put_flex(flex)
        put_flex(flex, name="2026-10/20261030-b.xml")
        with pytest.raises(OpeningCostError, match="refusing to pick"):
            opening_cost_pack(store, T, ACCT, AS_OF, flex_store=flex)

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("<FlexQueryResponse>", "not well-formed"),
            ('<!DOCTYPE x [<!ENTITY a "b">]>' + flex_xml(FLEX_COSTS), "DTD or entity"),
            (flex_xml(FLEX_COSTS).replace("FlexQueryResponse", "Reply"), "not one FlexStatement"),
            (flex_xml(FLEX_COSTS, day="20261029"), "not one FlexStatement"),
            (flex_xml(FLEX_COSTS).replace(ACCT, "DU7654321", 1), "not one FlexStatement"),
            (flex_xml(FLEX_COSTS).replace('position="10"', 'position="ten"'), "N01 position"),
        ],
    )
    def test_a_flex_file_that_is_not_the_statements_is_refused(self, store, flex, text, message):
        seed(store)
        put_flex(flex, text)
        with pytest.raises(OpeningCostError, match=message):
            opening_cost_pack(store, T, ACCT, AS_OF, flex_store=flex)

    def test_a_flex_file_that_disagrees_with_the_statement_is_refused(self, store, flex):
        seed(store)
        put_flex(flex, flex_xml(FLEX_COSTS, positions=dict(SHARES) | {"N01": 11}))
        with pytest.raises(OpeningCostError, match="disagree"):
            opening_cost_pack(store, T, ACCT, AS_OF, flex_store=flex)

    @pytest.mark.parametrize("bad", ["", "n/a", "NaN"])
    def test_an_unreadable_cost_basis_is_refused(self, store, flex, bad):
        seed(store)
        put_flex(flex, flex_xml(FLEX_COSTS | {"N02": bad}))
        with pytest.raises(OpeningCostError, match="N02 position or cost"):
            opening_cost_pack(store, T, ACCT, AS_OF, flex_store=flex)

    def test_an_unbalanced_entry_is_refused(self):
        with pytest.raises(OpeningCostError, match="does not balance: 5 cents"):
            _entry([Posting(CASH, 5)], "entry_x")


class TestWriteAndCli:
    def test_write_is_create_only(self, store, flex):
        seed(store)
        pack = opening_cost_pack(store, T, ACCT, AS_OF, flex_store=flex)
        key = write_pack(store, pack)
        assert key == f"trader/ledger/reconciliation/{T}/opening_cost/{AS_OF}.json"
        assert json.loads(store.get_bytes(key)) == json.loads(json.dumps(pack))
        with pytest.raises(OpeningCostError, match="already exists"):
            write_pack(store, pack)

    def test_the_cli_prints_the_pack_and_writes_only_on_request(self, store, flex):
        seed(store)
        argv = ["--tenant", T, "--account", ACCT, "--as-of", AS_OF, "--flex-uri", "unused"]
        opened = []

        def opener(uri, write):
            opened.append(write)
            return store

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            assert main(argv, opener=opener, flex_opener=lambda uri: flex) == 0
        assert json.loads(out.getvalue())["trial_balance_after_cents"] == 0
        assert err.getvalue() == "" and not store.exists(opening_cost_key(T, AS_OF))

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            assert main([*argv, "--write"], opener=opener, flex_opener=lambda uri: flex) == 0
        assert store.exists(opening_cost_key(T, AS_OF)) and "wrote trader/ledger" in err.getvalue()
        with (
            pytest.raises(OpeningCostError, match="already exists"),
            redirect_stdout(io.StringIO()),
        ):
            main([*argv, "--write"], opener=opener, flex_opener=lambda uri: flex)
        assert opened == [False, True, True]

    def test_the_cli_defaults_open_the_stores_without_writing_unless_asked(
        self, monkeypatch, store, flex
    ):
        seed(store)
        calls = []

        def fake_open(uri, *, dry_run=False):
            calls.append((uri, dry_run))
            return flex if uri == "flex://" else store

        monkeypatch.setattr("crucible_trader.opening_cost.open_store", fake_open)
        argv = ["--store", "s://x", "--tenant", T, "--account", ACCT, "--as-of", AS_OF]
        with redirect_stdout(io.StringIO()):
            main([*argv, "--flex-uri", "flex://"])
        monkeypatch.delenv("CRUCIBLE_TRADER_IB_FLEX_URI", raising=False)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            main([*argv, "--flex-uri", "flex://", "--write"])
        assert calls == [("s://x", True), ("flex://", True), ("s://x", False), ("flex://", True)]


def test_runs_as_a_module(tmp_path, monkeypatch):
    root = tmp_path / "trader"
    seed(LocalStore(root))
    (tmp_path / "flex").mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["opening_cost", "--store", str(root), "--tenant", T, "--account", ACCT]
        + ["--as-of", AS_OF, "--flex-uri", str(tmp_path / "flex")],
    )
    sys.modules.pop("crucible_trader.opening_cost", None)
    out = io.StringIO()
    with pytest.raises(SystemExit) as exited, redirect_stdout(out):
        runpy.run_module("crucible_trader.opening_cost", run_name="__main__")
    assert exited.value.code == 0 and json.loads(out.getvalue())["trial_balance_after_cents"] == 0
    assert not LocalStore(root).exists(opening_cost_key(T, AS_OF))
