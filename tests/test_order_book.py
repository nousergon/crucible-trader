"""The day's order book (Brian 2026-11-05): every name considered, kept."""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest
from crucible.features import DEFAULT_FEATURE_VERSION
from crucible.keys import features_key
from crucible.slots.strategy import BookUniverse
from crucible.store import LocalStore

from crucible_trader.books_retention import is_books_key
from crucible_trader.construction import TargetBook
from crucible_trader.order_book import (
    ORDER_BOOK_METRIC,
    ShareBasis,
    order_book_document,
    order_book_key,
    record_order_book,
    sector_labels,
)

NOW = dt.datetime(2026, 11, 5, 15, 0, tzinfo=dt.UTC)


def _book() -> TargetBook:
    tickers = ("AAA", "BBB", "CCC", "DDD", "SPY", "$CASH")
    universe = BookUniverse(
        tickers=tickers, sectors=("t", "t", "f", "f", "idx", "cash"), benchmark_idx=4, cash_idx=5
    )
    return TargetBook(
        trading_day="2026-11-05",
        s_champion="s:arm:x",
        m_champion="m:arm:y",
        u_champion=None,
        universe=universe,
        weights={"AAA": 0.6, "BBB": 0.0, "CCC": 0.3, "DDD": 0.0, "SPY": 0.0, "$CASH": 0.1},
        turnover_one_way_ratio=0.5,
        cost_bps=3.0,
        portfolio_notional_usd=100_000.0,
        evidence={},
        metric={},
        session_inputs_key="k",
        alpha={"AAA": 0.02, "BBB": 0.01, "CCC": 0.015, "DDD": -0.01, "SPY": 0.0, "$CASH": 0.0},
        eligible={"AAA": True, "BBB": True, "CCC": True, "DDD": False, "SPY": False, "$CASH": True},
    )


class _Ctx:
    def __init__(self, store) -> None:
        self.store = store
        self.metrics: list[dict] = []
        self.outputs: list[dict] = []

    def record_output(self, key, payload, schema_version="v1") -> None:
        self.store.put_bytes(key, payload)
        self.outputs.append({"key": key, "schema_version": schema_version})

    def record_metric(self, metric) -> None:
        self.metrics.append(metric)


def test_every_name_is_a_row_with_what_the_book_does_with_it() -> None:
    doc = order_book_document(
        _book(), previous_weights={"BBB": 0.2, "AAA": 0.6}, written_at_utc="t"
    )
    actions = {r["ticker"]: r["action"] for r in doc["rows"]}
    assert actions == {
        "AAA": "hold",
        "BBB": "exit",
        "CCC": "enter",
        "DDD": "ineligible",
        "SPY": "benchmark",
        "$CASH": "cash",
    }
    aaa = next(r for r in doc["rows"] if r["ticker"] == "AAA")
    assert aaa["target_notional_usd"] == 60_000.0 and aaa["alpha_rank"] == 1
    assert aaa["target_weight_pct"] == 60.0 and aaa["book_position"] == 1
    assert doc["summary"]["positions"] == 2 and doc["previous_book"] == "given"
    assert [r["ticker"] for r in doc["rows"]][:2] == ["AAA", "CCC"]  # biggest target first


def test_an_eligible_name_with_no_weight_is_not_selected() -> None:
    doc = order_book_document(_book(), previous_weights=None, written_at_utc="t")
    assert {r["ticker"]: r["action"] for r in doc["rows"]}["BBB"] == "not_selected"
    assert doc["previous_book"] == "all_cash"
    # codex ac64b346aa382999: an all-cash previous book records cash at 1.0.
    cash = next(r for r in doc["rows"] if r["ticker"] == "$CASH")
    assert cash["previous_weight"] == 1.0 and cash["weight_change"] == pytest.approx(-0.9)


def test_the_book_is_written_under_its_day_and_is_a_book_of_record(tmp_path) -> None:
    ctx = _Ctx(LocalStore(tmp_path))
    key = record_order_book(ctx, _book(), previous_weights=None, now=NOW)
    assert key == order_book_key("2026-11-05") == "trader/order_books/2026-11-05.json"
    assert is_books_key(key)
    doc = json.loads(ctx.store.get_bytes(key))
    assert doc["schema_version"] == "order_book.v1" and len(doc["rows"]) == 6
    assert ctx.metrics[-1]["name"] == ORDER_BOOK_METRIC and ctx.metrics[-1]["status"] == "OK"


def test_a_failed_write_never_stops_the_session(tmp_path) -> None:
    class Refusing(_Ctx):
        def record_output(self, *a, **k) -> None:
            raise PermissionError("AccessDenied")

    ctx = Refusing(LocalStore(tmp_path))
    assert record_order_book(ctx, _book(), previous_weights=None, now=NOW) is None
    assert (
        ctx.metrics[-1]["status"] == "FAIL" and "AccessDenied" in ctx.metrics[-1]["status_reason"]
    )


def test_a_weight_moved_up_or_down_is_increase_or_reduce() -> None:
    doc = order_book_document(
        _book(), previous_weights={"AAA": 0.4, "CCC": 0.5}, written_at_utc="t"
    )
    actions = {r["ticker"]: r["action"] for r in doc["rows"]}
    assert actions["AAA"] == "increase" and actions["CCC"] == "reduce"


def test_a_refused_lock_is_a_fail_metric_but_the_book_stays_written(tmp_path, monkeypatch) -> None:
    import crucible_trader.order_book as ob

    monkeypatch.setattr(
        ob, "lock_books_records", lambda store, keys, **k: ([], {keys[0]: "no lock"})
    )
    ctx = _Ctx(LocalStore(tmp_path))
    assert record_order_book(ctx, _book(), previous_weights=None, now=NOW) is not None
    assert ctx.metrics[-1]["status"] == "FAIL" and "no lock" in ctx.metrics[-1]["status_reason"]


def test_a_metric_that_cannot_be_recorded_does_not_raise(tmp_path) -> None:
    class NoMetrics(_Ctx):
        def record_metric(self, metric) -> None:
            raise RuntimeError("manifest closed")

    assert record_order_book(
        NoMetrics(LocalStore(tmp_path)), _book(), previous_weights=None, now=NOW
    )


def _write_sectors(store, day: str, labels: dict[str, object]) -> str:
    key = features_key(DEFAULT_FEATURE_VERSION, day)
    frame = pd.DataFrame(
        {
            "trading_day": [day] * len(labels),
            "ticker": list(labels),
            "sector_raw": list(labels.values()),
        }
    )
    store.put_bytes(key, frame.to_parquet(index=False))
    return key


def test_each_row_carries_its_gics_sector_label_beside_the_engine_sector(tmp_path) -> None:
    """alpha-engine-config-I13578: Brian read every holding as `__unsectored__`.
    The label is display only; ``sector`` stays the engine's own value."""
    ctx = _Ctx(LocalStore(tmp_path))
    source = _write_sectors(
        ctx.store,
        "2026-11-05",
        {"AAA": "Information Technology", "BBB": " Financials ", "CCC": None, "SPY": None},
    )
    key = record_order_book(ctx, _book(), previous_weights=None, now=NOW)
    doc = json.loads(ctx.store.get_bytes(key))
    labels = {r["ticker"]: r["sector_label"] for r in doc["rows"]}
    assert labels == {
        "AAA": "Information Technology",
        "BBB": "Financials",
        "CCC": None,  # the feature layer left it unlabelled
        "DDD": None,  # not in the feature artifact
        "SPY": None,
        "$CASH": None,
    }
    assert {r["ticker"]: r["sector"] for r in doc["rows"]}["AAA"] == "t"
    assert doc["sector_label_source"] == source


def test_no_feature_artifact_leaves_the_sector_labels_null_and_the_book_written(tmp_path) -> None:
    ctx = _Ctx(LocalStore(tmp_path))
    key = record_order_book(ctx, _book(), previous_weights=None, now=NOW)
    doc = json.loads(ctx.store.get_bytes(key))
    assert {r["sector_label"] for r in doc["rows"]} == {None}
    assert doc["sector_label_source"] is None and ctx.metrics[-1]["status"] == "OK"


def test_an_unreadable_feature_artifact_never_costs_the_book(tmp_path) -> None:
    ctx = _Ctx(LocalStore(tmp_path))
    ctx.store.put_bytes(features_key(DEFAULT_FEATURE_VERSION, "2026-11-05"), b"not parquet")
    key = record_order_book(ctx, _book(), previous_weights=None, now=NOW)
    assert key is not None and ctx.metrics[-1]["status"] == "OK"
    doc = json.loads(ctx.store.get_bytes(key))
    assert {r["sector_label"] for r in doc["rows"]} == {None}


def test_each_held_row_carries_the_shares_the_session_sized(tmp_path) -> None:
    ctx = _Ctx(LocalStore(tmp_path))
    basis = ShareBasis(
        positions={"AAA": 500, "BBB": 100},
        prices={"AAA": 110.0, "BBB": 50.0, "CCC": 33.0},
    )
    key = record_order_book(
        ctx,
        _book(),
        previous_weights={"AAA": 0.55, "BBB": 0.05},
        now=NOW,
        share_basis=lambda: basis,
    )
    doc = json.loads(ctx.store.get_bytes(key))
    rows = {r["ticker"]: r for r in doc["rows"]}
    # AAA 0.6 * 100k / 110 = 545.45 -> 545; CCC 0.3 * 100k / 33 = 909.09 -> 909.
    assert (rows["AAA"]["previous_shares"], rows["AAA"]["target_shares"]) == (500, 545)
    assert rows["AAA"]["share_price_usd"] == 110.0
    assert (rows["BBB"]["previous_shares"], rows["BBB"]["target_shares"]) == (100, 0)
    assert (rows["CCC"]["previous_shares"], rows["CCC"]["target_shares"]) == (0, 909)
    assert (rows["DDD"]["previous_shares"], rows["DDD"]["target_shares"]) == (0, 0)
    for sentinel in ("SPY", "$CASH"):
        assert rows[sentinel]["target_shares"] is None
        assert rows[sentinel]["previous_shares"] is None
    assert doc["share_basis"] == "router_plan"


def test_without_a_share_basis_the_share_columns_are_null() -> None:
    doc = order_book_document(_book(), previous_weights=None, written_at_utc="t")
    assert {(r["previous_shares"], r["target_shares"]) for r in doc["rows"]} == {(None, None)}
    assert doc["share_basis"] is None


def test_a_failing_share_basis_never_costs_the_book(tmp_path) -> None:
    ctx = _Ctx(LocalStore(tmp_path))

    def broken() -> ShareBasis:
        raise RuntimeError("router state unreadable")

    key = record_order_book(ctx, _book(), previous_weights=None, now=NOW, share_basis=broken)
    assert key is not None and ctx.metrics[-1]["status"] == "OK"
    doc = json.loads(ctx.store.get_bytes(key))
    assert doc["share_basis"] is None


def test_a_weighted_name_with_no_usable_price_has_no_target_shares() -> None:
    basis = ShareBasis(positions={}, prices={"AAA": float("nan")})
    assert basis.target_shares("AAA", 0.5, 100_000.0) is None
    assert basis.target_shares("BBB", 0.5, 100_000.0) is None
    assert basis.target_shares("BBB", 0.0, 100_000.0) == 0


def test_a_feature_artifact_without_a_sector_column_labels_nothing(tmp_path) -> None:
    store = LocalStore(tmp_path)
    frame = pd.DataFrame({"trading_day": ["2026-11-05"], "ticker": ["AAA"]})
    store.put_bytes(
        features_key(DEFAULT_FEATURE_VERSION, "2026-11-05"), frame.to_parquet(index=False)
    )
    assert sector_labels(store, "2026-11-05", ["AAA"]) == ({}, None)
