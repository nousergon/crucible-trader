"""The Polygon corporate-action scan as the reconciler's source
(alpha-engine-config-I11076, option A).

The property: a present, complete scan RESOLVES the book and turns paid
dividends and splits into the actions the broker should show; anything the
scan cannot settle stays unresolved, and no scan resolves nothing.
"""

from __future__ import annotations

import io
import json
import pathlib

import pytest
from crucible.keys import trader_broker_statement_key as broker_statement_key
from crucible.manifest import read_manifest
from crucible.store import LocalStore

from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.corporate_action_feed import (
    SCAN_SCHEMA_VERSION,
    CorporateActionScanError,
    actions_from_scan,
    ib_symbol,
    read_scan,
    scan_provider,
    scans_uri_from_env,
)
from crucible_trader.reconciliation import (
    CorporateAction,
    CorporateActionSet,
    Fill,
)
from crucible_trader.reconciliation_control import ReconciliationVoidError
from crucible_trader.reconciliation_job import (
    RECONCILE_JOB,
    ReconciliationDiscrepancyError,
    previous_trading_day,
    reconcile_cycle,
    run_reconciliation,
)
from tests.test_reconciliation_job import DAY, PREV, FixedSource, ctx_for

BASE = "s3://research/corporate_actions/scans"
DAY_ISO = DAY.isoformat()


def scan(day=DAY_ISO, **fields):
    doc = {
        "schema": SCAN_SCHEMA_VERSION,
        "trading_day": day,
        "complete": True,
        "splits": [],
        "dividends_paid": [],
        "dividends_ex": [],
        "declared": [],
    }
    doc.update(fields)
    return doc


def dividend(ticker, amount, ex_date, currency="USD"):
    return {
        "ticker": ticker,
        "cash_amount": amount,
        "currency": currency,
        "ex_dividend_date": ex_date,
        "record_date": ex_date,
        "pay_date": DAY.isoformat(),
    }


class FakeS3:
    def __init__(self, objects):
        self.objects = objects

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys):
        if self.objects == "denied":
            err = Exception("AccessDenied")
            err.response = {"Error": {"Code": "AccessDenied"}}
            raise err
        keys = [k for b, k in self.objects if b == Bucket and k.startswith(Prefix)]
        return {"Contents": [{"Key": k} for k in keys[:MaxKeys]]}

    def get_object(self, *, Bucket, Key):
        try:
            body = self.objects[(Bucket, Key)]
        except KeyError:
            err = Exception("NoSuchKey")
            err.response = {"Error": {"Code": "NoSuchKey"}}
            raise err from None
        return {"Body": io.BytesIO(body)}


def s3_with(doc, day=DAY_ISO):
    return FakeS3({("research", f"corporate_actions/scans/{day}.json"): json.dumps(doc).encode()})


def resolve(doc, universe, holdings):
    return actions_from_scan(
        doc,
        day=DAY.isoformat(),
        universe=universe,
        holdings_on=holdings.get,
        previous_trading_day=previous_trading_day,
    )


def test_polygon_share_classes_map_to_ib_symbols():
    assert ib_symbol("BRK.B") == "BRK B"
    assert ib_symbol("AAPL") == "AAPL"


def test_a_quiet_day_resolves_the_whole_book():
    got = resolve(scan(), {"AAA", "BBB"}, {})
    assert got == CorporateActionSet(resolved=frozenset({"AAA", "BBB"}), actions=())


def test_a_split_on_a_held_name_becomes_a_ratio():
    doc = scan(
        splits=[
            {"ticker": "AAA", "execution_date": "2026-09-11", "split_from": 1, "split_to": 4},
            {"ticker": "ZZZ", "execution_date": "2026-09-11", "split_from": 1, "split_to": 2},
        ]
    )
    got = resolve(doc, {"AAA"}, {})
    assert got.actions == (CorporateAction("AAA", "split", ratio=4.0),)


def test_a_dividend_is_paid_on_the_record_date_holding_even_if_sold_since():
    # Ex 09-08 (a Tuesday after Labor Day) -> record session 09-04.
    doc = scan(dividends_paid=[dividend("CCC", 0.5, "2026-09-08")])
    got = resolve(doc, {"AAA"}, {"2026-09-04": {"CCC": 100, "AAA": 10}})
    assert got.resolved == frozenset({"AAA"})
    assert got.actions == (
        CorporateAction("CCC", "cash_dividend", amount_per_share=0.5, entitled_shares=100),
    )


def test_an_unheld_dividend_is_no_action():
    doc = scan(dividends_paid=[dividend("DDD", 0.5, "2026-09-08")])
    assert resolve(doc, {"AAA"}, {"2026-09-04": {"AAA": 10}}).actions == ()


@pytest.mark.parametrize(
    "row, holdings",
    [
        (dividend("AAA", 0.5, "2026-09-08"), {}),  # no stored record-date statement
        (dividend("AAA", 0.5, "2026-09-08", currency="CAD"), {"2026-09-04": {"AAA": 10}}),
        ({**dividend("AAA", 0.5, "2026-09-08"), "ex_dividend_date": None}, {}),
    ],
)
def test_an_unsettleable_dividend_leaves_the_held_name_unresolved(row, holdings):
    got = resolve(scan(dividends_paid=[row]), {"AAA", "BBB"}, holdings)
    assert got.resolved == frozenset({"BBB"})
    assert got.actions == ()


def test_a_declared_spinoff_is_carried_so_the_reconciler_refuses_it():
    doc = scan(declared=[{"kind": "spinoff", "ticker": "AAA", "ex_date": "2026-09-11"}])
    assert resolve(doc, {"AAA"}, {}).actions == (CorporateAction("AAA", "spinoff"),)


@pytest.mark.parametrize(
    "doc",
    [
        scan(schema="other.v1"),
        scan(day="2026-09-10"),
        scan(complete=False),
        scan(splits=None),
    ],
)
def test_a_scan_that_is_not_trustworthy_raises(doc):
    with pytest.raises(CorporateActionScanError):
        resolve(doc, {"AAA"}, {})


def test_a_missing_scan_is_unknown_not_empty():
    read = read_scan(BASE, DAY.isoformat(), s3_client=FakeS3({}))
    assert read.document is None and "does not exist" in read.reason


def test_an_unreadable_scan_is_unknown_and_names_the_error():
    read = read_scan(BASE, DAY_ISO, s3_client=FakeS3("denied"))
    assert read.document is None and "AccessDenied" in read.reason


def test_the_env_var_is_required_and_has_no_default():
    assert scans_uri_from_env({}) is None
    assert scans_uri_from_env({"CRUCIBLE_TRADER_CORPORATE_ACTION_SCANS_URI": BASE}) == BASE


# --- through the job --------------------------------------------------------


@pytest.fixture
def store(tmp_path: pathlib.Path) -> LocalStore:
    return LocalStore(tmp_path)


def put_statement(store, day, positions, cash):
    s = BrokerStatement("DU1", day, positions, cash)
    store.put_bytes(broker_statement_key(day), json.dumps(s.to_document()).encode())


def test_a_paid_dividend_on_a_name_sold_before_pay_date_reconciles_clean(store):
    # Held 100 CCC at the 09-04 record session, sold it 09-10, paid 09-11.
    put_statement(store, "2026-09-04", {"AAA": 10, "CCC": 100}, 1_000.0)
    put_statement(store, PREV, {"AAA": 10}, 1_000.0)
    doc = scan(dividends_paid=[dividend("CCC", 0.5, "2026-09-08")])
    ctx = ctx_for(store)
    result = reconcile_cycle(
        ctx,
        source=FixedSource({"AAA": 10}, 1_050.0),
        fills=(),
        cash_flows=(),
        corporate_actions=scan_provider(BASE, s3_client=s3_with(doc)),
    )
    assert result.clean
    assert result.match_rate == 1.0
    assert any(i["key"] == f"{BASE}/{DAY.isoformat()}.json" for i in ctx.inputs)


def test_an_unpaid_dividend_is_a_corporate_action_finding(store):
    put_statement(store, "2026-09-04", {"AAA": 10}, 1_000.0)
    put_statement(store, PREV, {"AAA": 10}, 1_000.0)
    doc = scan(dividends_paid=[dividend("AAA", 0.5, "2026-09-08")])
    # The control arm's planted cash break can absorb this finding and void the
    # cycle instead; either way the run fails, naming the unpaid dividend.
    with pytest.raises(
        (ReconciliationDiscrepancyError, ReconciliationVoidError), match="corporate_action.*AAA"
    ):
        reconcile_cycle(
            ctx_for(store),
            source=FixedSource({"AAA": 10}, 1_000.0),
            fills=(),
            cash_flows=(),
            corporate_actions=scan_provider(BASE, s3_client=s3_with(doc)),
        )


def test_no_scan_leaves_every_instrument_unresolved(store):
    put_statement(store, PREV, {"AAA": 10}, 1_000.0)
    with pytest.raises(ReconciliationDiscrepancyError, match="corporate_action_check_incomplete"):
        reconcile_cycle(
            ctx_for(store),
            source=FixedSource({"AAA": 10}, 1_000.0),
            fills=(),
            cash_flows=(),
            corporate_actions=scan_provider(BASE, s3_client=FakeS3({})),
        )


def test_the_manifest_records_the_scan_as_an_input_and_validates(store):
    put_statement(store, PREV, {"AAA": 10}, 1_000.0)
    run_reconciliation(
        store,
        FixedSource({"AAA": 12}, 800.0),
        trading_day=DAY,
        fills=(Fill("AAA", 2, -200.0),),
        cash_flows=(),
        corporate_actions=scan_provider(BASE, s3_client=s3_with(scan())),
        run_mode="replay",
    )
    manifest = read_manifest(store, RECONCILE_JOB, DAY.isoformat())  # validates on read
    assert manifest["status"] == "ok"
    scan_inputs = [i for i in manifest["inputs"] if i["schema_version"] == SCAN_SCHEMA_VERSION]
    assert [i["key"] for i in scan_inputs] == [f"{BASE}/{DAY.isoformat()}.json"]


def test_unchanged_dividend_fallback_still_uses_the_anchor():
    # A CorporateAction with no entitled_shares keeps the old anchored behaviour.
    assert CorporateAction("AAA", "cash_dividend", amount_per_share=1.0).entitled_shares is None


def test_a_non_s3_base_uri_is_refused():
    with pytest.raises(CorporateActionScanError, match="s3://"):
        read_scan("/local/path", DAY_ISO, s3_client=FakeS3({}))


def test_a_present_scan_that_is_not_json_raises():
    s3 = FakeS3({("research", f"corporate_actions/scans/{DAY_ISO}.json"): b"{not json"})
    with pytest.raises(CorporateActionScanError, match="not JSON"):
        read_scan(BASE, DAY_ISO, s3_client=s3)


def test_the_live_read_builds_its_own_s3_client(monkeypatch):
    import sys
    import types

    fake = types.SimpleNamespace(client=lambda name: s3_with(scan()))
    monkeypatch.setitem(sys.modules, "boto3", fake)
    assert read_scan(BASE, DAY_ISO).document["trading_day"] == DAY_ISO


def test_an_unconfigured_source_resolves_nothing(store):
    provide = scan_provider(None)
    assert provide(ctx_for(store), DAY_ISO, frozenset({"AAA"})) == CorporateActionSet(
        resolved=frozenset()
    )


def test_a_missing_record_date_statement_leaves_the_name_unresolved(store):
    doc = scan(dividends_paid=[dividend("AAA", 0.5, "2026-09-08")])
    provide = scan_provider(BASE, s3_client=s3_with(doc))
    got = provide(ctx_for(store), DAY_ISO, frozenset({"AAA", "BBB"}))
    assert got.resolved == frozenset({"BBB"})
